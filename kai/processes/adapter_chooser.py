import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from kai.processes.base import BaseProcess
from kai.inference import get_model_pricing, get_model_response
from kai.schemas import (
    AdapterChooserInput,
    AdapterChooserOutput,
    AdapterSelection,
    Framework,
    Language,
    MasterContext,
)

# Mapping: language -> frameworks
LANGUAGE_FRAMEWORK_MAP: Dict[Language, list[Framework]] = {
    Language.SOLIDITY: [Framework.FOUNDRY],
    Language.JAVASCRIPT: [Framework.NODE],
    Language.RUST: [Framework.CARGO],
    Language.CPP: [Framework.CMAKE],
}

# Mapping: framework -> adapter class name (or None if not needed)
FRAMEWORK_ADAPTER_MAP: Dict[Framework, Optional[str]] = {
    Framework.FOUNDRY: "SolidityAdapter",
    Framework.NODE: None,
    # Setup-only (no dependency adapter yet)
    Framework.CARGO: None,
    Framework.CMAKE: None,
}


class AdapterChooserProcess(BaseProcess[AdapterChooserInput, AdapterChooserOutput]):
    """
    Single-turn process that asks the model to pick languages, then maps them to
    frameworks and adapters using enums/registries.
    """

    PROMPT_TEMPLATE = (
        "You are a language detector for a code repository.\n"
        "Pick ONLY from the allowed languages: {languages}.\n"
        "Return a single JSON object with keys:\n"
        '{{"languages": ["<language1>", "<language2>", ...], '
        '"reason": "<short reason>"}}\n'
        "Do not mention frameworks or adapters; only list languages."
    )

    def __init__(self, context: MasterContext):
        super().__init__(context)

    async def execute(self, input_data: AdapterChooserInput) -> AdapterChooserOutput:
        repo_root = Path(self.context.root_path)

        readme_text = self._read_readme(repo_root)
        file_listing = self._list_files(repo_root, depth=2)

        allowed_languages = list(Language)
        system_prompt = self.PROMPT_TEMPLATE.format(
            languages=", ".join(sorted({lang.value for lang in allowed_languages}))
        )
        user_message = self._build_user_message(
            repo_root=repo_root,
            readme_text=readme_text,
            file_listing=file_listing,
        )

        usage_data: Dict[str, int] = {}
        response_text: Optional[str] = None
        model_error: Optional[str] = None

        try:
            response_text, usage_data = await get_model_response(
                message=user_message,
                system_prompt=system_prompt,
                model=input_data.model_name,
                use_openai=input_data.use_openai,
            )
        except Exception as e:
            model_error = str(e)

        # Always run deterministic detection as a fallback/augmentation
        detected_languages = self._detect_languages(repo_root)

        if response_text:
            model_languages = self._parse_languages(response_text)
            detected_languages.update(model_languages)

        if not detected_languages:
            error_msg = model_error or "No languages detected"
            return AdapterChooserOutput(
                choice=None,
                raw_response=response_text,
                estimated_cost=0.0,
                total_tokens=usage_data,
                success=False,
                error_message=error_msg,
            )

        selection = self._build_selection(detected_languages)
        pricing = get_model_pricing(input_data.model_name, input_data.use_openai)
        estimated_cost = (
            usage_data.get("prompt_tokens", 0) * pricing["prompt"]
            + usage_data.get("completion_tokens", 0) * pricing["completion"]
        )

        return AdapterChooserOutput(
            choice=selection,
            raw_response=response_text,
            estimated_cost=estimated_cost,
            total_tokens=usage_data,
            success=True,
            error_message=model_error,
        )

    def _build_user_message(
        self,
        repo_root: Path,
        readme_text: str,
        file_listing: str,
    ) -> str:
        return (
            f"Repository root: {repo_root}\n"
            "README.md contents:\n"
            f"{readme_text}\n\n"
            "File tree (depth=2):\n"
            f"{file_listing}\n"
            "List the languages used by this repository."
        )

    def _read_readme(self, repo_root: Path) -> str:
        candidates = [
            repo_root / "README.md",
            repo_root / "README.MD",
            repo_root / "README",
        ]
        for path in candidates:
            if path.exists() and path.is_file():
                try:
                    return path.read_text(encoding="utf-8")
                except Exception:
                    return f"Failed to read README at {path}"
        return "README not found in repository root."

    def _list_files(self, repo_root: Path, depth: int) -> str:
        if not repo_root.exists():
            return f"Repository root not found: {repo_root}"

        entries: List[str] = []
        for current, dirs, files in os.walk(repo_root):
            rel_path = Path(current).relative_to(repo_root)
            current_depth = len(rel_path.parts)
            if current_depth > depth:
                dirs[:] = []
                continue

            indent = "  " * current_depth
            label = "." if rel_path == Path(".") else str(rel_path)
            entries.append(f"{indent}{label}/")
            for f in sorted(files):
                entries.append(f"{indent}  {f}")

            if len(entries) > 400:
                entries.append("... (truncated)")
                break

        return "\n".join(entries) if entries else "No files found."

    def _extract_json(self, text: str) -> Optional[Dict]:
        cleaned = text.strip()

        if "```" in cleaned:
            # Try to pull JSON from first fenced block
            start = cleaned.find("```")
            end = cleaned.find("```", start + 3)
            if end != -1:
                cleaned = cleaned[start + 3 : end].strip()
                # Drop optional language hint
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1].strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Fallback: attempt to locate first/last braces
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = cleaned[first : last + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
        return None

    def _parse_languages(self, response_text: str) -> Set[Language]:
        data = self._extract_json(response_text)
        if not isinstance(data, dict):
            return set()
        langs = data.get("languages")
        if not isinstance(langs, list):
            return set()
        detected: Set[Language] = set()
        for item in langs:
            val = (item or "").strip().lower()
            try:
                detected.add(Language(val))
            except ValueError:
                continue
        return detected

    def _detect_languages(self, repo_root: Path) -> Set[Language]:
        detected: Set[Language] = set()

        # Heuristic: master context frameworks
        for fw in self.context.frameworks or []:
            fw_lower = str(fw).lower()
            if fw_lower in {"foundry", "forge"}:
                detected.add(Language.SOLIDITY)
            if fw_lower in {"cargo"}:
                detected.add(Language.RUST)
            if fw_lower in {"cmake"}:
                detected.add(Language.CPP)

        # File-based heuristics
        for current, dirs, files in os.walk(repo_root):
            # Limit depth to avoid huge walks
            depth = len(Path(current).relative_to(repo_root).parts)
            if depth > 3:
                dirs[:] = []
                continue

            lowered_files = [f.lower() for f in files]
            if "foundry.toml" in lowered_files:
                detected.add(Language.SOLIDITY)
            if any(f.endswith(".sol") for f in lowered_files):
                detected.add(Language.SOLIDITY)
            if "package.json" in lowered_files or "yarn.lock" in lowered_files:
                detected.add(Language.JAVASCRIPT)
            if any(f.endswith((".js", ".ts", ".cjs", ".mjs")) for f in lowered_files):
                detected.add(Language.JAVASCRIPT)

            if "cargo.toml" in lowered_files or "cargo.lock" in lowered_files:
                detected.add(Language.RUST)
            if any(f.endswith(".rs") for f in lowered_files):
                detected.add(Language.RUST)

            if (
                "cmakelists.txt" in lowered_files
                or "compile_commands.json" in lowered_files
            ):
                detected.add(Language.CPP)
            if any(
                f.endswith((".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"))
                for f in lowered_files
            ):
                detected.add(Language.CPP)

            # Early exit once we have a reasonable multi-language set
            if (
                Language.SOLIDITY in detected
                and Language.JAVASCRIPT in detected
                and Language.RUST in detected
                and Language.CPP in detected
            ):
                break

        return detected

    def _build_selection(self, languages: Set[Language]) -> AdapterSelection:
        frameworks: List[Framework] = []
        adapters: List[str | None] = []

        for lang in languages:
            for fw in LANGUAGE_FRAMEWORK_MAP.get(lang, []):
                if fw not in frameworks:
                    frameworks.append(fw)
                    adapters.append(FRAMEWORK_ADAPTER_MAP.get(fw))

        return AdapterSelection(
            languages=sorted(languages, key=lambda lang: lang.value),
            frameworks=frameworks,
            adapters=adapters,
        )
