"""
InvariantProcess: Grounded invariant generation from ActorMatrix + DependencyGraph.

This process generates invariants by:
1. Building vocab tables from the graph (functions, vars, files)
2. Chunking vocab to fit LLM context
3. Per-chunk LLM invariant generation with strict ID constraints
4. Validation to ensure all IDs exist in vocab
5. Merging/deduplication across chunks

LLM can only reference IDs from the provided vocabulary - cannot hallucinate locations.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kai.inference import create_openai_client, get_model_pricing
from kai.processes.base import BaseProcess
from kai.schemas import (
    ActorMatrix,
    FileVocabEntry,
    FunctionVocabEntry,
    Invariant,
    InvariantProcessInput,
    InvariantProcessOutput,
    InvariantType,
    ProtocolManifesto,
    VarVocabEntry,
    VocabChunk,
)
from kai.utils.dependency.adapters import DomainAdapter, get_adapter
from kai.utils.dependency.analysis import FileSourceLoader, GraphQueryEngine
from kai.utils.dependency.models import EdgeKind

# Load prompt template
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "invariant_generation.txt"
INVARIANT_PROMPT = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else ""


class InvariantProcess(BaseProcess[InvariantProcessInput, InvariantProcessOutput]):
    """
    Process to generate grounded invariants from ActorMatrix + DependencyGraph.

    Uses vocab tables to constrain LLM to valid IDs only.
    """

    async def execute(
        self, input_data: InvariantProcessInput
    ) -> InvariantProcessOutput:
        ctx = input_data.master_context
        graph = input_data.dependency_graph
        actor_matrix = input_data.actor_matrix
        manifesto = input_data.protocol_manifesto

        if graph is None:
            return InvariantProcessOutput(
                success=False,
                error_message="No dependency graph provided",
            )

        # Get adapter
        try:
            self.adapter: DomainAdapter = get_adapter(ctx.adapter)
        except ValueError as e:
            return InvariantProcessOutput(
                success=False,
                error_message=str(e),
            )

        # Build query engine
        source_loader = FileSourceLoader(ctx.root_path)
        engine = GraphQueryEngine(
            graph=graph, adapter=self.adapter, source_loader=source_loader
        )

        try:
            # Step 1: Build vocab tables
            self.logger.info("Building vocabulary tables from graph...")
            vocab = self._build_vocab(engine, actor_matrix)
            self.logger.info(
                f"Vocab: {len(vocab['functions'])} functions, "
                f"{len(vocab['vars'])} vars, {len(vocab['files'])} files"
            )

            if not vocab["functions"]:
                return InvariantProcessOutput(
                    success=True,
                    invariants=[],
                    stats={
                        "total_generated": 0,
                        "validated": 0,
                        "dropped": 0,
                        "merged": 0,
                    },
                )

            # Step 2: Chunk vocab
            chunks = self._chunk_vocab(vocab, input_data.max_chunk_functions)
            self.logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Per-chunk LLM generation
            raw_invariants: List[Invariant] = []
            total_cost = 0.0
            total_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

            for chunk in chunks:
                chunk_invs, cost, tokens = await self._generate_invariants_for_chunk(
                    chunk=chunk,
                    manifesto=manifesto,
                    actor_matrix=actor_matrix,
                    model_name=input_data.model_name,
                    use_openai=input_data.use_openai,
                )
                raw_invariants.extend(chunk_invs)
                total_cost += cost
                total_tokens["prompt_tokens"] += tokens.get("prompt_tokens", 0)
                total_tokens["completion_tokens"] += tokens.get("completion_tokens", 0)

            self.logger.info(f"Generated {len(raw_invariants)} raw invariants")

            # Step 4: Validate
            valid_set = self._build_valid_id_sets(vocab)
            validated, dropped = self._validate_invariants(raw_invariants, valid_set)
            self.logger.info(f"Validated: {len(validated)}, Dropped: {dropped}")

            # Step 5: Merge/dedupe
            final_invariants = self._merge_invariants(validated)
            merged_count = len(validated) - len(final_invariants)
            self.logger.info(
                f"Final: {len(final_invariants)} invariants (merged {merged_count})"
            )

            return InvariantProcessOutput(
                invariants=final_invariants,
                success=True,
                estimated_cost=total_cost,
                total_tokens=total_tokens,
                stats={
                    "total_generated": len(raw_invariants),
                    "validated": len(validated),
                    "dropped": dropped,
                    "merged": merged_count,
                },
            )

        except Exception as e:
            self.logger.error(f"InvariantProcess failed: {e}", exc_info=True)
            return InvariantProcessOutput(
                success=False,
                error_message=str(e),
            )

    def _build_vocab(
        self,
        engine: GraphQueryEngine,
        actor_matrix: ActorMatrix,
    ) -> Dict[str, List[Any]]:
        """
        Build vocabulary tables from graph + ActorMatrix.

        Returns: {functions: [...], vars: [...], files: [...]}
        """
        # Build role lookup from ActorMatrix
        func_to_role: Dict[str, Tuple[str, str]] = {}  # func_id -> (role_name, trust)
        for role in actor_matrix.roles:
            for priv in role.privileges:
                func_to_role[priv.id] = (role.name, role.trust)

        # Get protocol entrypoints
        entrypoints = engine.protocol_entrypoints()

        functions: List[FunctionVocabEntry] = []
        vars_map: Dict[str, VarVocabEntry] = {}  # var_id -> entry
        files_map: Dict[str, Set[str]] = defaultdict(set)  # file -> contracts

        for ep in entrypoints:
            # Get reads/writes
            reads = engine.neighbors(ep.id, [EdgeKind.READS], "out")
            writes = engine.neighbors(ep.id, [EdgeKind.WRITES], "out")

            role_name, trust = func_to_role.get(ep.id, ("", ""))

            functions.append(
                FunctionVocabEntry(
                    id=ep.id,
                    name=ep.name,
                    container=ep.container or "",
                    file=ep.file or "",
                    role=role_name,
                    trust=trust,
                    reads=[r.name for r in reads],
                    writes=[w.name for w in writes],
                )
            )

            # Track file -> contracts
            if ep.file and ep.container:
                files_map[ep.file].add(ep.container)

            # Build var entries
            for var_ref in reads + writes:
                if var_ref.id not in vars_map:
                    vars_map[var_ref.id] = VarVocabEntry(
                        id=var_ref.id,
                        name=var_ref.name,
                        container=var_ref.container or "",
                        file=var_ref.file or "",
                        writers=[],
                        readers=[],
                    )

                if var_ref in writes:
                    if ep.id not in vars_map[var_ref.id].writers:
                        vars_map[var_ref.id].writers.append(ep.id)
                if var_ref in reads:
                    if ep.id not in vars_map[var_ref.id].readers:
                        vars_map[var_ref.id].readers.append(ep.id)

        files: List[FileVocabEntry] = [
            FileVocabEntry(id=f, contracts=sorted(contracts))
            for f, contracts in files_map.items()
        ]

        return {
            "functions": functions,
            "vars": list(vars_map.values()),
            "files": files,
        }

    def _chunk_vocab(
        self,
        vocab: Dict[str, List[Any]],
        max_functions: int,
    ) -> List[VocabChunk]:
        """
        Split vocab into chunks, each with at most max_functions functions.

        Includes related vars and files for each chunk.
        """
        functions = vocab["functions"]
        vars_by_id = {v.id: v for v in vocab["vars"]}
        files_by_id = {f.id: f for f in vocab["files"]}

        chunks: List[VocabChunk] = []

        for i in range(0, len(functions), max_functions):
            chunk_funcs = functions[i : i + max_functions]
            chunk_id = f"chunk_{i // max_functions}"

            # Collect related vars
            var_ids_needed: Set[str] = set()
            file_ids_needed: Set[str] = set()

            for func in chunk_funcs:
                # Find var IDs from reads/writes (need to look up by name)
                for var in vocab["vars"]:
                    if var.name in func.reads or var.name in func.writes:
                        var_ids_needed.add(var.id)

                if func.file:
                    file_ids_needed.add(func.file)

            chunk_vars = [
                vars_by_id[vid] for vid in var_ids_needed if vid in vars_by_id
            ]
            chunk_files = [
                files_by_id[fid] for fid in file_ids_needed if fid in files_by_id
            ]

            chunks.append(
                VocabChunk(
                    chunk_id=chunk_id,
                    functions=chunk_funcs,
                    vars=chunk_vars,
                    files=chunk_files,
                )
            )

        return chunks

    async def _generate_invariants_for_chunk(
        self,
        chunk: VocabChunk,
        manifesto: Optional[ProtocolManifesto],
        actor_matrix: ActorMatrix,
        model_name: str,
        use_openai: bool,
    ) -> Tuple[List[Invariant], float, Dict[str, int]]:
        """
        Generate invariants for a single vocab chunk.

        Returns: (invariants, cost, tokens)
        """
        # Format vocab for prompt
        functions_vocab = json.dumps(
            [f.model_dump() for f in chunk.functions], indent=2
        )
        vars_vocab = json.dumps([v.model_dump() for v in chunk.vars], indent=2)
        files_vocab = json.dumps([f.model_dump() for f in chunk.files], indent=2)

        # Actor matrix summary
        actor_summary_lines = []
        for role in actor_matrix.roles:
            funcs = [p.name for p in role.privileges[:5]]
            if len(role.privileges) > 5:
                funcs.append(f"... +{len(role.privileges) - 5} more")
            actor_summary_lines.append(
                f"- {role.name} (trust: {role.trust}): {', '.join(funcs)}"
            )
        actor_matrix_summary = "\n".join(actor_summary_lines)

        # Minimal, language-agnostic protocol context
        protocol_context = ""
        if manifesto:
            lines: List[str] = []
            if getattr(manifesto, "name", None):
                lines.append(f"Protocol: {manifesto.name}")
            if getattr(manifesto, "purpose", None):
                purpose = manifesto.purpose or ""
                lines.append(f"Purpose: {purpose[:160]}")
            if getattr(manifesto, "domain", None):
                dom = manifesto.domain
                if dom:
                    lines.append(f"Domain: {dom}")
            # Summarize without heavy details
            users = list(getattr(manifesto, "intended_users", []) or [])
            if users:
                lines.append("Users: " + ", ".join(users[:3]))
            concepts = list((getattr(manifesto, "key_concepts", {}) or {}).keys())
            if concepts:
                lines.append("Key Concepts: " + ", ".join(concepts[:3]))
            features = [
                getattr(f, "name", "")
                for f in (getattr(manifesto, "key_features", []) or [])
            ]
            features = [f for f in features if f]
            if features:
                lines.append("Features: " + ", ".join(features[:2]))
            # Languages are informational only; keep very short
            langs = list(getattr(manifesto, "programming_languages", []) or [])
            if langs:
                lines.append("Languages: " + ", ".join(langs[:2]))
            protocol_context = "\n".join(lines)

        # Format prompt
        prompt = INVARIANT_PROMPT.format(
            protocol_context=protocol_context,
            functions_vocab=functions_vocab,
            vars_vocab=vars_vocab,
            files_vocab=files_vocab,
            actor_matrix_summary=actor_matrix_summary,
        )

        # Call LLM
        client = create_openai_client(use_openai=use_openai)

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Slight creativity but mostly deterministic
        )

        content = response.choices[0].message.content or ""

        # Parse response
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        invariants: List[Invariant] = []
        try:
            data = json.loads(json_str)
            inv_list = data.get("invariants", [])

            for item in inv_list:
                try:
                    inv_type = InvariantType(item.get("type", "other").lower())
                except ValueError:
                    inv_type = InvariantType.OTHER

                invariants.append(
                    Invariant(
                        label=item.get("label", item.get("id", f"INV_{len(invariants)}")),
                        type=inv_type,
                        rule=item.get("rule", ""),
                        explanation=item.get("explanation", ""),
                        target_function_ids=item.get("target_function_ids", []),
                        target_var_ids=item.get("target_var_ids", []),
                        target_file_ids=item.get("target_file_ids", []),
                        confidence=item.get("confidence", 0.5),
                        source="llm",
                        chunk_id=chunk.chunk_id,
                    )
                )
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse LLM response for {chunk.chunk_id}")

        # Calculate tokens and cost
        usage = response.usage
        tokens = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

        # Get pricing and calculate cost
        pricing = get_model_pricing(model_name, use_openai)
        cost = (
            tokens["prompt_tokens"] * pricing["prompt"]
            + tokens["completion_tokens"] * pricing["completion"]
        )

        return invariants, cost, tokens

    def _build_valid_id_sets(
        self,
        vocab: Dict[str, List[Any]],
    ) -> Dict[str, Set[str]]:
        """Build sets of valid IDs for validation."""
        return {
            "functions": {f.id for f in vocab["functions"]},
            "vars": {v.id for v in vocab["vars"]},
            "files": {f.id for f in vocab["files"]},
        }

    def _validate_invariants(
        self,
        invariants: List[Invariant],
        valid_sets: Dict[str, Set[str]],
    ) -> Tuple[List[Invariant], int]:
        """
        Validate invariants - drop those with invalid IDs or empty targets.

        Returns: (valid_invariants, dropped_count)
        """
        valid: List[Invariant] = []
        dropped = 0

        for inv in invariants:
            # Check all function IDs are valid
            invalid_funcs = [
                fid
                for fid in inv.target_function_ids
                if fid not in valid_sets["functions"]
            ]
            # Check all var IDs are valid
            invalid_vars = [
                vid for vid in inv.target_var_ids if vid not in valid_sets["vars"]
            ]
            # Check all file IDs are valid
            invalid_files = [
                fid for fid in inv.target_file_ids if fid not in valid_sets["files"]
            ]

            # Drop if any invalid IDs
            if invalid_funcs or invalid_vars or invalid_files:
                self.logger.debug(
                    f"Dropping {inv.label}: invalid Labels - "
                    f"funcs={invalid_funcs}, vars={invalid_vars}, files={invalid_files}"
                )
                dropped += 1
                continue

            # Drop if all target lists are empty
            if (
                not inv.target_function_ids
                and not inv.target_var_ids
                and not inv.target_file_ids
            ):
                self.logger.debug(f"Dropping {inv.label}: no targets")
                dropped += 1
                continue

            valid.append(inv)

        return valid, dropped

    def _merge_invariants(
        self,
        invariants: List[Invariant],
    ) -> List[Invariant]:
        """
        Merge/dedupe invariants with similar rules and targets.

        Simple approach: group by normalized rule + target set, keep highest confidence.
        """
        # Group by (normalized_rule, frozenset of targets)
        groups: Dict[Tuple[str, frozenset], List[Invariant]] = defaultdict(list)

        for inv in invariants:
            # Normalize rule for comparison
            norm_rule = inv.rule.lower().strip()
            target_key = frozenset(
                inv.target_function_ids + inv.target_var_ids + inv.target_file_ids
            )
            groups[(norm_rule, target_key)].append(inv)

        # Keep best from each group
        merged: List[Invariant] = []
        for group in groups.values():
            # Sort by confidence descending, take first
            group.sort(key=lambda x: -x.confidence)
            merged.append(group[0])

        return merged
