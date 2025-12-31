from kai.inference import (
    get_model_response_with_tools,
    create_openai_client,
    create_vllm_client,
    get_model_pricing,
)
from kai.agents.utils import (
    load_system_prompt,
    AgentType,
    generate_openai_tools,
)
from kai.agents.settings import (
    MAX_TOOL_TURNS,
    MAIN_DEFAULT_MODEL,
    VLLM_HOST,
    VLLM_PORT,
)
from kai.schemas import ChatMessage, Role, AgentResponse
import asyncio
from logger import logger

from collections.abc import Callable
from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod

import os
from bson import ObjectId


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        max_tool_turns: Optional[int] = None,
        repo_path: Optional[str] = None,
        use_vllm: bool = False,
        model: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        use_openai: bool = False,
        scope_paths: Optional[list[str]] = None,  # Restrict file access to these paths
        parent_agent_id: Optional[str] = None,  # Track hierarchy
        depth: int = 0,  # Depth in hierarchy (for logging)
        system_prompt_tools_schema: str | None = None,
    ):
        if agent_type is None:
            raise ValueError("agent_type must be provided")
        self.agent_type: AgentType = agent_type

        # Agent identification and hierarchy
        self.agent_id = str(ObjectId())  # Convert to string for JSON serialization
        self.parent_agent_id = parent_agent_id
        self.depth = depth

        # For sub-agents, store execution_id
        self.execution_id = None  # Will be set for sub-agents

        # Scope restriction (NEW)
        self.scope_paths = scope_paths
        self.repo_path = os.path.abspath(repo_path) if repo_path else os.getcwd()

        if scope_paths:
            self.restricted_scope = True
            # Convert scope paths to absolute paths
            self.allowed_paths = []
            for p in scope_paths:
                # If path is already absolute and within repo_path, use it as-is
                if os.path.isabs(p):
                    abs_p = os.path.abspath(p)
                    if self.repo_path and abs_p.startswith(self.repo_path):
                        self.allowed_paths.append(abs_p)
                    else:
                        # Absolute path outside repo - shouldn't happen but handle it
                        self.allowed_paths.append(abs_p)
                else:
                    # Relative path - join with repo_path
                    # Remove any leading repo directory names to avoid duplication
                    # e.g. if p is "repos/xxx/programs/store/" and repo_path ends with "repos/xxx"
                    clean_p = p
                    repo_name = (
                        os.path.basename(self.repo_path) if self.repo_path else ""
                    )
                    if clean_p.startswith("repos/"):
                        # Strip everything up to and including the repo name
                        parts = clean_p.split("/")
                        if repo_name in parts:
                            idx = parts.index(repo_name)
                            clean_p = "/".join(parts[idx + 1 :])
                    self.allowed_paths.append(
                        os.path.abspath(os.path.join(self.repo_path, clean_p))
                    )
            # For sub-agents with single scope, use that as working directory
            # This allows tools to work with relative paths naturally
            if len(scope_paths) == 1:
                self.working_dir = self.allowed_paths[0]
            else:
                self.working_dir = self.repo_path
        else:
            self.restricted_scope = False
            self.allowed_paths = []
            self.working_dir = self.repo_path

        # Track sub-agents spawned by this agent (NEW)
        self.sub_agent_reports: list = []  # Will hold SubAgentReport objects

        # Configure tool turn budget early for downstream consumers
        self.max_tool_turns = (
            max_tool_turns if max_tool_turns is not None else MAX_TOOL_TURNS
        )

        # Set exploits.json path based on working directory (dynamic per agent)
        self.exploits_path = os.path.join(self.working_dir, "exploits.json")

        # Load the system prompt and add it to the conversation history
        # Use condensed prompt for sub-agents
        is_sub_agent = depth > 0
        scope_path_str = scope_paths[0] if scope_paths else ""
        self.system_prompt = load_system_prompt(
            self.agent_type,
            is_sub_agent=is_sub_agent,
            scope_path=scope_path_str,
            task_description="",  # Will be set in delegation instruction
            max_turns=self.max_tool_turns,
            depth=depth,
            tools_schema=system_prompt_tools_schema,
        )
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]

        # Fixer-specific format enforcement state
        self._pending_feedback: Optional[str] = None
        self._fixer_completion_warnings = 0
        self._max_completion_warnings = 3

        # Set the maximum number of tool turns and use_vllm flag
        self.use_vllm = use_vllm
        self.use_openai = use_openai

        # Budget tracking
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self.estimated_cost = 0.0

        # Time tracking
        self.time_spent = 0.0  # Time spent by this agent only (seconds)
        self.start_time = None  # Will be set when chat() starts

        # Set model: use provided model, or fallback to MAIN_DEFAULT_MODEL
        self.model = model if model else MAIN_DEFAULT_MODEL

        # Each Agent instance gets its own clients to avoid bottlenecks
        if use_vllm:
            self._client = create_vllm_client(host=VLLM_HOST, port=VLLM_PORT)
        else:
            self._client = create_openai_client(use_openai=use_openai)

    def calculate_cost(self, usage_data: Dict[str, int]) -> float:
        """
        Calculate cost from usage data using dynamic pricing.

        Args:
            usage_data: Dict with prompt_tokens and completion_tokens

        Returns:
            Cost in dollars
        """
        pricing = get_model_pricing(self.model, self.use_openai)
        prompt_cost = usage_data.get("prompt_tokens", 0) * pricing["prompt"]
        completion_cost = usage_data.get("completion_tokens", 0) * pricing["completion"]
        return prompt_cost + completion_cost

    def update_budget(self, usage_data: Dict[str, int]):
        """
        Update token usage and estimated cost.

        Args:
            usage_data: Dict with prompt_tokens, completion_tokens, total_tokens
        """
        self.total_tokens["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
        self.total_tokens["completion_tokens"] += usage_data.get("completion_tokens", 0)
        cost = self.calculate_cost(usage_data)
        self.estimated_cost += cost

    def _get_remaining_turns(self) -> int:
        """Get the number of remaining turns for this agent."""
        # Count assistant messages with python code to determine turns used
        turns_used = sum(
            1
            for msg in self.messages
            if msg.role == Role.ASSISTANT and "<python>" in msg.content
        )
        return self.max_tool_turns - turns_used

    def _create_tool_executor(self):
        """
        Create a tool executor for native OpenAI tool calling.

        Returns a function that takes (tool_name, args_dict) and returns the result.
        """
        import importlib
        from kai.agents.tools.tools import set_current_agent

        module = importlib.import_module(self.get_tools_module())

        def executor(name: str, args: dict):
            # Set current agent via contextvars (async-safe and reliable)
            set_current_agent(self)
            # Also keep stack variable for backwards compatibility
            _agent_instance = self  # noqa: F841
            func = getattr(module, name, None)
            if func is None:
                return {"error": f"Unknown tool: {name}"}
            try:
                result = func(**args)
                return result
            except Exception as e:
                return {"error": str(e)}

        return executor

    async def chat_with_tools(
        self,
        message: str,
        tools: Optional[list] = None,
        tool_executor: Optional[Callable[[str, dict], Any]] = None,
    ) -> AgentResponse:
        """
        Chat using native OpenAI tool calling (no XML parsing).

        Args:
            message: Initial user message
            tools: Optional list of OpenAI-format tool definitions.
                   If not provided, generates from get_tools_module().
            tool_executor: Optional custom tool executor.
                           If not provided, uses _create_tool_executor().

        Returns:
            AgentResponse with the final result.
        """
        import time

        # Start timing
        if self.start_time is None:
            self.start_time = time.time()

        # Generate tools if not provided
        if tools is None:
            adapter = self.get_tool_adapter()
            tools = generate_openai_tools(self.get_tools_module(), adapter=adapter)

        # Create executor if not provided
        if tool_executor is None:
            tool_executor = self._create_tool_executor()

        # Add user message
        self._add_message(ChatMessage(role=Role.USER, content=message))

        # Agent description for logging
        agent_type_str = (
            self.agent_type.name.title()
            if hasattr(self.agent_type, "name")
            else str(self.agent_type).title()
        )
        if "." in agent_type_str:
            agent_type_str = agent_type_str.split(".")[-1].title()
        agent_desc = f"({agent_type_str} {self.agent_id} at depth {self.depth})"

        # Tool calling loop
        remaining_turns = self.max_tool_turns
        final_response = ""
        tool_calls_made = []
        no_observation_nudges = 0

        while remaining_turns > 0:
            current_turn = self.max_tool_turns - remaining_turns + 1
            logger.info(f"{current_turn}/{self.max_tool_turns} - {agent_desc}")
            # Lightweight per-round budget visibility for the model (informational only).
            # Keep this as a short USER "meta" line to minimize interference with tool calling.
            self._add_message(
                ChatMessage(
                    role=Role.USER,
                    content=(
                        f"<meta>Remaining turns: {remaining_turns}/{self.max_tool_turns}</meta>"
                    ),
                )
            )

            # Call model with tools
            (
                response,
                calls_made,
                usage_data,
                messages_payload,
            ) = await get_model_response_with_tools(
                messages=self.messages,
                tools=tools,
                tool_executor=tool_executor,
                model=self.model,
                client=self._client,
                use_vllm=self.use_vllm,
                use_openai=self.use_openai,
                max_tool_rounds=1,  # One round at a time for fine control
            )

            # Update budget
            self.update_budget(usage_data)
            tool_calls_made.extend(calls_made)

            # Replace message history with the tool-calling payload so subsequent rounds
            # include tool_call_id-linked tool outputs.
            try:
                self.messages = [ChatMessage(**m) for m in messages_payload]
            except Exception:
                # If tool-call metadata shape is unexpected, fall back to minimal assistant text.
                if response:
                    self._add_message(
                        ChatMessage(role=Role.ASSISTANT, content=response)
                    )

            if response:
                final_response = response

            # If the blackbox agent is actively using tools but has not recorded any observations yet:
            # - early on: don't force an observation (it can stall exploration)
            # - later: ensure at least one observation gets recorded with evidence
            if (
                self.agent_type == AgentType.BLACKBOX
                and calls_made
                and not getattr(self, "blackbox_observations", [])
                and remaining_turns > 1
                and no_observation_nudges < 2
            ):
                self._add_message(
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            "BLACKBOX: Continue investigating. "
                            "Make your next step a concrete tool call based on the latest tool output "
                            "(e.g., read targeted files/symbols, query the graph, or run an experiment). "
                            "Do NOT emit <done>."
                        ),
                    )
                )
                no_observation_nudges += 1
            elif (
                self.agent_type == AgentType.BLACKBOX
                and calls_made
                and len(getattr(self, "blackbox_observations", []) or []) == 1
                and current_turn >= max(6, self.max_tool_turns // 2)
                and remaining_turns > 1
                and no_observation_nudges < 4
            ):
                self._add_message(
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            "BLACKBOX: Keep recording observations as you go. "
                            "Prefer multiple focused observations (one experiment/hypothesis each) "
                            "instead of bundling everything into a single summary. "
                            "Include concrete evidence from tool outputs (negative results are OK). "
                            "Do NOT emit <done>."
                        ),
                    )
                )
                no_observation_nudges += 1

            # Check if model stopped calling tools (finished)
            if not calls_made:
                # Blackbox must keep going until budget is exhausted.
                if self.agent_type == AgentType.BLACKBOX and remaining_turns > 1:
                    self._add_message(
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                "BLACKBOX: Continue investigating. "
                                "Call at least one tool. Prefer concrete exploration "
                                "(graph queries, reading targeted code) and experiments "
                                "(write_campaign_file + run_forge_campaign) when useful. "
                                "Do NOT emit <done>."
                            ),
                        )
                    )
                    remaining_turns -= 1
                    continue
                if self.agent_type == AgentType.BLACKBOX and remaining_turns == 1:
                    # We already spent the last model round; mark budget exhausted.
                    remaining_turns = 0

                logger.info(f"{agent_desc} - Completed (no more tool calls)")
                break

            # Turn accounting:
            # - Blackbox: count every model round as a turn (even if it didn't call tools).
            # - Others: count only rounds where tools were called.
            if self.agent_type == AgentType.BLACKBOX:
                remaining_turns -= 1
            else:
                remaining_turns -= 1

            # Small delay for event loop
            await asyncio.sleep(0.01)

        if remaining_turns == 0:
            logger.info(f"{agent_desc} - Completed (max turns)")

        # Update time
        if self.start_time is not None:
            self.time_spent = time.time() - self.start_time

        # Extract final result using the standard method
        return self.extract_final_result("", "", final_response)

    @abstractmethod
    def check_termination(self, response: str, python_code: str) -> bool:
        """
        Check if the agent should terminate based on the response.

        Args:
            response: The full response from the model.
            python_code: The extracted python code from the response.

        Returns:
            True if the agent should terminate, False otherwise.
        """
        pass

    @abstractmethod
    def get_tools_module(self) -> str:
        """
        Get the tools module name for execute_sandboxed_code.

        Returns:
            The module name to import for tools.
        """
        pass

    def get_tool_adapter(self):
        """
        Get the tool adapter for framework-specific tool descriptions.

        Checks master_context.frameworks for a supported tool framework.
        Note: master_context.adapter is for language/domain (solidity, rust),
        while frameworks contains the tooling (foundry, hardhat, anchor, cargo).

        Returns:
            ToolAdapter instance
        """
        from kai.utils.tool_adapters import get_tool_adapter, get_supported_frameworks

        master_context = getattr(self, "master_context", None)
        if master_context:
            # Check frameworks list for a supported tool framework
            frameworks = getattr(master_context, "frameworks", None) or []
            supported = set(get_supported_frameworks())
            for fw in frameworks:
                fw_lower = fw.lower()
                if fw_lower in supported:
                    return get_tool_adapter(fw_lower)

        # Default to foundry if no recognized framework found
        return get_tool_adapter("foundry")

    @abstractmethod
    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result from the agent's work.

        Args:
            thoughts: The extracted thoughts.
            python_code: The extracted python code.
            response: The full response.

        Returns:
            An AgentResponse object with the final result.
        """
        pass

    def _add_message(self, message: Union[ChatMessage, dict]):
        """Add a message to the conversation history."""
        if isinstance(message, dict):
            self.messages.append(ChatMessage(**message))
        elif isinstance(message, ChatMessage):
            self.messages.append(message)
        else:
            raise ValueError("Invalid message type")

    async def close(self):
        """
        Clean up resources used by the agent, including closing the HTTP client.
        Should be called when the agent is no longer needed.
        """
        try:
            client = getattr(self, "_client", None)
            if client is not None:
                import inspect

                aclose = getattr(client, "aclose", None)
                if callable(aclose):
                    result = aclose()
                    if inspect.isawaitable(result):
                        await result
                self._client = None
        except Exception:
            pass  # Ignore errors during cleanup
