from enum import Enum
import importlib
import inspect
import os
import pathspec
import re
from typing import Optional

from kai.agents import settings


class AgentType(Enum):
    FINDER = "finder"
    NON_DUPLICATE_VERIFIER = "non_duplicate_verifier"
    TEST_GENERATOR = "test_generator"
    SETUP = settings.SETUP_AGENT_PROMPT_PATH
    PROFILER = settings.PROFILER_AGENT_PROMPT_PATH
    BLACKBOX = settings.BLACKBOX_AGENT_PROMPT_PATH
    INVARIANT_SYNTHESIZER = settings.INVARIANT_SYNTHESIZER_AGENT_PROMPT_PATH
    STATE = settings.STATE_AGENT_PROMPT_PATH
    QUANT = settings.QUANT_AGENT_PROMPT_PATH
    VERIFIER = settings.VERIFIER_AGENT_PROMPT_PATH
    WORKSPACE_VALIDATION = settings.WORKSPACE_VALIDATION_AGENT_PROMPT_PATH
    FIXER = settings.FIXER_AGENT_PROMPT_PATH
    GAMIFIED = settings.GAMIFIED_AGENT_PROMPT_PATH


def agent_type_to_kind(agent_type: AgentType) -> str:
    """Convert AgentType enum to kind string for database storage."""
    if agent_type == AgentType.SETUP:
        return "setup"
    if agent_type == AgentType.PROFILER:
        return "profiler"
    if agent_type == AgentType.BLACKBOX:
        return "blackbox"
    if agent_type == AgentType.INVARIANT_SYNTHESIZER:
        return "invariant_synthesizer"
    if agent_type == AgentType.STATE:
        return "state"
    if agent_type == AgentType.QUANT:
        return "quant"
    if agent_type == AgentType.WORKSPACE_VALIDATION:
        return "workspace_validation"
    if agent_type == AgentType.VERIFIER:
        return "verifier"
    if agent_type == AgentType.FIXER:
        return "fixer"
    if agent_type == AgentType.GAMIFIED:
        return "gamified"
    return "unknown"


def load_system_prompt(
    agent_type: AgentType,
    is_sub_agent: bool = False,
    scope_path: str = "",
    task_description: str = "",
    max_turns: int = settings.MAX_TOOL_TURNS,
    depth: int = 0,
    tools_schema: str | None = None,
) -> str:
    """
    Load the system prompt from the file for supported Kai v2 agent types.
    """
    if agent_type not in {
        AgentType.SETUP,
        AgentType.PROFILER,
        AgentType.BLACKBOX,
        AgentType.INVARIANT_SYNTHESIZER,
        AgentType.STATE,
        AgentType.QUANT,
        AgentType.VERIFIER,
        AgentType.WORKSPACE_VALIDATION,
        AgentType.FIXER,
        AgentType.GAMIFIED,
    }:
        raise ValueError(f"Unsupported agent type for Kai v2 scope: {agent_type}")

    prompt_path = agent_type.value
    try:
        with open(prompt_path, "r") as f:
            system_prompt = f.read()
            system_prompt = system_prompt.replace("{{max_tool_turns}}", str(max_turns))

            if is_sub_agent:
                system_prompt = system_prompt.replace("{{scope_path}}", scope_path)
                system_prompt = system_prompt.replace(
                    "{{task_description}}", task_description
                )
                system_prompt = system_prompt.replace("{{max_turns}}", str(max_turns))
                system_prompt = system_prompt.replace("{{depth}}", str(depth))

            if tools_schema:
                system_prompt = (
                    system_prompt + "\n\n## Available Tools\n" + tools_schema.strip()
                )

            return system_prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found at {prompt_path}")


def generate_tool_schema(tools_module: str) -> str:
    """
    Generate a tool schema for a tools module, formatted as Python stubs with docstrings.
    """
    try:
        module = importlib.import_module(tools_module)
    except Exception as e:
        return f"Error importing tools module {tools_module}: {e}"

    lines: list[str] = ["```python"]

    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue

        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(...)"

        is_async = inspect.iscoroutinefunction(obj)
        header = f"async def {name}{sig}:" if is_async else f"def {name}{sig}:"

        doc = inspect.getdoc(obj) or ""
        if doc:
            doc = inspect.cleandoc(doc)
            doc_lines = (
                ['    """'] + [f"    {line}" for line in doc.splitlines()] + ['    """']
            )
        else:
            doc_lines = []

        lines.append(header)
        lines.extend(doc_lines)
        lines.append("    pass")
        lines.append("")  # blank line between functions

    if len(lines) == 1:  # only header
        return f"No public functions found in {tools_module}"

    lines.append("```")
    return "\n".join(lines)


def generate_openai_tools(tools_module: str, adapter=None) -> list[dict]:
    """
    Generate OpenAI-style tool definitions from a Python module.

    Converts Python functions with type hints and docstrings into OpenAI's
    function calling format.

    Args:
        tools_module: Fully qualified module name (e.g., "kai.agents.tools.state_tools")
        adapter: Optional ToolAdapter for framework-specific tool descriptions.
                 If provided, tools in ADAPTER_DESCRIBED_TOOLS will use
                 adapter.get_tool_description() instead of docstrings.

    Returns:
        List of tool definitions in OpenAI format.
    """
    import typing
    from typing import get_type_hints, get_origin, get_args, Union
    from kai.agents.tools.tools import get_tool_description, ADAPTER_DESCRIBED_TOOLS

    try:
        module = importlib.import_module(tools_module)
    except Exception as e:
        print(f"Error importing tools module {tools_module}: {e}")
        return []

    tools = []

    def python_type_to_json_schema(py_type) -> dict:
        """Convert Python type hints to JSON schema types."""
        if py_type is None or py_type is type(None):
            return {"type": "null"}

        # Handle Optional/Union types
        origin = get_origin(py_type)
        if origin is Union:
            args = get_args(py_type)
            # Optional[X] is Union[X, None]
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return python_type_to_json_schema(non_none_args[0])
            # Multiple types - use anyOf
            return {"anyOf": [python_type_to_json_schema(a) for a in non_none_args]}

        # Handle List types
        if origin is list:
            args = get_args(py_type)
            if args:
                return {"type": "array", "items": python_type_to_json_schema(args[0])}
            return {"type": "array"}

        # Handle Dict types
        if origin is dict:
            return {"type": "object"}

        # Handle Literal types
        if origin is typing.Literal:
            args = get_args(py_type)
            return {"type": "string", "enum": list(args)}

        # Basic types
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        # Handle string type annotations
        if isinstance(py_type, str):
            return {"type": "string"}

        return type_map.get(py_type, {"type": "string"})

    def parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
        """Parse docstring to extract description and parameter descriptions."""
        if not docstring:
            return "", {}

        lines = docstring.strip().split("\n")
        description_lines = []
        param_descs = {}
        current_param = None
        in_args_section = False

        for line in lines:
            stripped = line.strip()

            # Check for Args section
            if stripped.lower().startswith("args:"):
                in_args_section = True
                continue

            # Check for other sections that end Args
            if stripped.lower().startswith(
                ("returns:", "raises:", "examples:", "example:")
            ):
                in_args_section = False
                current_param = None
                continue

            if in_args_section:
                # Check for parameter line (name: description)
                if ": " in stripped and not stripped.startswith(" "):
                    parts = stripped.split(": ", 1)
                    current_param = parts[0].strip()
                    if len(parts) > 1:
                        param_descs[current_param] = parts[1].strip()
                elif current_param and stripped:
                    # Continuation of previous parameter description
                    param_descs[current_param] = (
                        param_descs.get(current_param, "") + " " + stripped
                    )
            else:
                if stripped:
                    description_lines.append(stripped)

        return " ".join(description_lines), param_descs

    for name, obj in vars(module).items():
        # Skip private functions and non-functions
        if name.startswith("_"):
            continue
        if not inspect.isfunction(obj):
            continue

        # Get function signature
        try:
            sig = inspect.signature(obj)
        except Exception:
            continue

        # Get type hints
        try:
            hints = get_type_hints(obj)
        except Exception:
            hints = {}

        # Get description - use adapter if available for adapter-described tools
        if adapter is not None and name in ADAPTER_DESCRIBED_TOOLS:
            description = get_tool_description(obj, adapter)
            # Parse docstring just for parameter descriptions
            docstring = inspect.getdoc(obj) or ""
            _, param_descs = parse_docstring(docstring)
        else:
            # Parse docstring for both description and param descriptions
            docstring = inspect.getdoc(obj) or ""
            description, param_descs = parse_docstring(docstring)

        # Build parameters schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip *args, **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get type
            param_type = hints.get(param_name, str)
            param_schema = python_type_to_json_schema(param_type)

            # Add description from docstring
            if param_name in param_descs:
                param_schema["description"] = param_descs[param_name]

            properties[param_name] = param_schema

            # Check if required (no default value)
            if param.default is param.empty:
                required.append(param_name)

        # Build tool definition
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"Function {name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        tools.append(tool)

    return tools


def extract_python_code(response: str) -> str:
    """
    Extract the python code from the response and format it with Black.
    """
    if "<python>" in response and "</python>" in response:
        response = response.split("<python>")[1].split("</python>")[0]
        if "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        return code
    else:
        return ""


def extract_thoughts(response: str) -> str:
    """
    Extract the thoughts from the response.
    """
    if "<think>" in response and "</think>" in response:
        return response.split("<think>")[1].split("</think>")[0]
    else:
        return ""


def extract_decision(response: str) -> bool:
    """
    Extract the decision from the response.
    """
    if "<decision>" in response and "</decision>" in response:
        return response.split("<decision>")[1].split("</decision>")[0] == "True"
    else:
        return False


def extract_test_script(response: str) -> str:
    """
    Extract the test script from the response.
    """
    if "<test_script>" in response and "</test_script>" in response:
        return response.split("<test_script>")[1].split("</test_script>")[0]
    else:
        return ""


def extract_suggest_fix(response: str) -> str:
    """
    Extract the suggested fix from the response.
    """
    if "<suggest_fix>" in response and "</suggest_fix>" in response:
        return response.split("<suggest_fix>")[1].split("</suggest_fix>")[0]

    # Fallback: capture the first ```diff block even if the agent forgot the wrapper
    diff_match = re.search(r"```diff[\\s\\S]+?```", response)
    if diff_match:
        return diff_match.group(0)

    return ""


def check_done(response: str) -> bool:
    """
    Check if the response contains the <done> tag.
    """
    if "<done>" in response and "</done>" in response:
        return True
    else:
        return False


def format_results_and_remaining_turns(
    results: dict, error_msg: str = "", remaining_turns: int = 0
) -> str:
    """
    Format the results into a string and add the remaining turns to the string.
    """
    return (
        "<result>\\n("
        + str(results)
        + ", {"
        + error_msg
        + "})\\n</result>\\n<remaining_turns>\\n"
        + str(remaining_turns)
        + "\\n</remaining_turns>"
        if error_msg
        else "<result>\\n"
        + str(results)
        + "\\n</result>\\n<remaining_turns>\\n"
        + str(remaining_turns)
        + "\\n</remaining_turns>"
    )


def load_gitignore_spec(directory: str) -> Optional[pathspec.PathSpec]:
    """
    Load and parse .gitignore file from a directory.
    """
    gitignore_path = os.path.join(directory, ".gitignore")
    if not os.path.exists(gitignore_path):
        return None

    try:
        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()
        return pathspec.PathSpec.from_lines(
            "gitwildmatch", gitignore_content.splitlines()
        )
    except Exception:
        return None


def should_ignore_path(
    path: str, root_dir: str, gitignore_spec: Optional[pathspec.PathSpec]
) -> bool:
    """
    Check if a path should be ignored based on gitignore patterns.
    """
    if gitignore_spec is None:
        return False

    try:
        rel_path = os.path.relpath(path, root_dir)
        return gitignore_spec.match_file(rel_path)
    except Exception:
        return False
