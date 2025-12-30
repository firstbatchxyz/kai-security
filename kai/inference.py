from openai import AsyncOpenAI
from typing import Optional, Union, Dict, Tuple, List, Any, Callable

import json
import requests

from kai.agents.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MAIN_DEFAULT_MODEL,
    OPENAI_API_KEY,
)
from kai.schemas import ChatMessage, Role

# Cache for model pricing to avoid repeated API calls
_pricing_cache: Dict[str, Dict[str, float]] = {}


def create_openai_client(use_openai: bool = False) -> AsyncOpenAI:
    """Create a new async OpenAI client instance."""
    if use_openai:
        return AsyncOpenAI(
            api_key=OPENAI_API_KEY,
        )
    else:
        return AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )


def create_vllm_client(host: str = "0.0.0.0", port: int = 8000) -> AsyncOpenAI:
    """Create a new async vLLM client instance (OpenAI-compatible)."""
    return AsyncOpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="EMPTY",  # vLLM doesn't require a real API key
    )


def get_model_pricing(model_name: str, use_openai: bool = False) -> Dict[str, float]:
    """
    Get pricing for a model from OpenRouter or use defaults for OpenAI.

    Returns:
        Dict with 'prompt' and 'completion' keys (cost per token in dollars)
    """
    # Check cache first
    if model_name in _pricing_cache:
        return _pricing_cache[model_name]

    if use_openai:
        # Default OpenAI pricing (approximate, will be updated dynamically)
        default_pricing = {
            "prompt": 0.00003,  # $30/1M tokens
            "completion": 0.00006,  # $60/1M tokens
        }
        _pricing_cache[model_name] = default_pricing
        return default_pricing

    # Fetch from OpenRouter API
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=5,
        )
        if response.status_code == 200:
            models = response.json().get("data", [])
            for model in models:
                if model.get("id") == model_name:
                    pricing = model.get("pricing", {})
                    result = {
                        "prompt": float(pricing.get("prompt", 0)),
                        "completion": float(pricing.get("completion", 0)),
                    }
                    _pricing_cache[model_name] = result
                    return result
    except Exception:
        pass  # Fall through to default

    # Default fallback pricing
    default_pricing = {
        "prompt": 0.000003,  # $3/1M tokens (reasonable default)
        "completion": 0.000015,  # $15/1M tokens
    }
    _pricing_cache[model_name] = default_pricing
    return default_pricing


def _as_dict(msg: Union[ChatMessage, dict]) -> dict:
    """
    Accept either ChatMessage or raw dict and return the raw dict.

    Args:
        msg: A ChatMessage object or a raw dict.

    Returns:
        A raw dict.
    """
    return msg if isinstance(msg, dict) else msg.model_dump()


async def get_model_response(
    messages: Optional[list[ChatMessage]] = None,
    message: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: str = MAIN_DEFAULT_MODEL,
    client: Optional[AsyncOpenAI] = None,
    use_vllm: bool = False,
    use_openai: bool = False,
) -> Tuple[str, Dict[str, int]]:
    """
    Get a response from a model using OpenRouter or vLLM (async).

    Args:
        messages: A list of ChatMessage objects (optional).
        message: A single message string (optional).
        system_prompt: A system prompt for the model (optional).
        model: The model to use.
        client: Optional AsyncOpenAI client to use. If None, creates a new one.
        use_vllm: Whether to use vLLM backend instead of OpenRouter.
        use_openai: Whether to use OpenAI API directly.

    Returns:
        Tuple of (response_text, usage_dict) where usage_dict contains:
        - prompt_tokens: int
        - completion_tokens: int
        - total_tokens: int
    """
    if messages is None and message is None:
        raise ValueError("Either 'messages' or 'message' must be provided.")

    # Use provided client or create a new one
    if client is None:
        if use_vllm:
            client = create_vllm_client()
        else:
            client = create_openai_client(use_openai=use_openai)

    # Build message history
    messages_payload: list[dict] = []
    if messages is None:
        if system_prompt:
            messages_payload.append(
                _as_dict(ChatMessage(role=Role.SYSTEM, content=system_prompt))
            )
        if message is None:
            raise ValueError("Message cannot be None if messages list is None")

        messages_payload.append(_as_dict(ChatMessage(role=Role.USER, content=message)))
    else:
        messages_payload = [_as_dict(m) for m in messages]

    try:
        completion = await client.chat.completions.create(
            model=model, messages=messages_payload
        )

        response_text = completion.choices[0].message.content

        # Extract usage data
        usage = completion.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        return response_text, usage_dict

    except Exception as e:
        error_msg = str(e)

        # Check if this is a context length error
        if any(
            keyword in error_msg.lower()
            for keyword in ["context length", "maximum context", "tokens"]
        ):
            # Calculate approximate token count from messages
            total_chars = sum(len(str(m)) for m in messages_payload)
            approx_tokens = total_chars // 4  # Rough approximation

            raise Exception(
                f"Context length exceeded: approximately {approx_tokens} tokens.\n"
                f"This usually means the conversation history is too long.\n"
                f"Original error: {error_msg}"
            )
        else:
            # Re-raise the original exception
            raise


async def get_model_response_with_tools(
    messages: list[ChatMessage],
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], Any],
    model: str = MAIN_DEFAULT_MODEL,
    client: Optional[AsyncOpenAI] = None,
    use_vllm: bool = False,
    use_openai: bool = False,
    max_tool_rounds: int = 1,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """
    Get a response from a model using OpenAI's native tool calling.

    Args:
        messages: A list of ChatMessage objects.
        tools: List of OpenAI-format tool definitions.
        tool_executor: Function that executes tools: (name, args) -> result
        model: The model to use.
        client: Optional AsyncOpenAI client.
        use_vllm: Whether to use vLLM backend.
        use_openai: Whether to use OpenAI API directly.
        max_tool_rounds: Maximum rounds of tool calling (default 1).

    Returns:
        Tuple of (final_response_text, tool_calls_made, usage_dict, messages_payload)
    """
    if client is None:
        if use_vllm:
            client = create_vllm_client()
        else:
            client = create_openai_client(use_openai=use_openai)

    messages_payload = [_as_dict(m) for m in messages]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_calls_made = []

    def _json_default(obj: Any):
        """
        Best-effort conversion to JSON-serializable types.

        Tool results often contain:
        - dataclasses (e.g., NodeRef/ContextSlice/EvidencePack)
        - Pydantic models
        - Enums
        which need conversion before json encoding.
        """
        # Enums (e.g., NodeKind)
        try:
            from enum import Enum as _Enum

            if isinstance(obj, _Enum):
                return obj.value
        except Exception:
            pass

        # Dataclasses (e.g., NodeRef, ContextSlice, EvidencePack)
        try:
            import dataclasses

            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
        except Exception:
            pass

        try:
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                # mode="json" coerces enums and other non-primitive values.
                return obj.model_dump(mode="json")
        except Exception:
            pass
        try:
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                return obj.dict()
        except Exception:
            pass
        # Fallback: string representation (better than crashing tool calling).
        return str(obj)

    last_content: str = ""

    for _ in range(max_tool_rounds):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages_payload,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )
        except Exception as e:
            error_msg = str(e)
            if any(
                keyword in error_msg.lower()
                for keyword in ["context length", "maximum context", "tokens"]
            ):
                total_chars = sum(len(str(m)) for m in messages_payload)
                approx_tokens = total_chars // 4
                raise Exception(
                    f"Context length exceeded: approximately {approx_tokens} tokens.\n"
                    f"Original error: {error_msg}"
                )
            raise

        # Defensive checks: some OpenAI-compatible backends can return malformed/partial
        # responses that deserialize but lack choices. Fail with a clear error instead
        # of crashing with "NoneType is not subscriptable".
        try:
            choices = getattr(completion, "choices", None)
            if not choices:
                raise Exception(
                    f"Tool-calling completion missing choices (model={model}). "
                    f"completion={repr(completion)}"
                )
        except Exception:
            # Re-raise as a regular exception for upstream handling/logging.
            raise

        # Update usage
        if completion.usage:
            total_usage["prompt_tokens"] += completion.usage.prompt_tokens
            total_usage["completion_tokens"] += completion.usage.completion_tokens
            total_usage["total_tokens"] += completion.usage.total_tokens

        message = completion.choices[0].message
        last_content = message.content or ""

        # Check for tool calls
        if message.tool_calls:
            # Capture reasoning that accompanies tool calls
            reasoning = message.content or ""

            # Add assistant message with tool calls to history
            messages_payload.append(
                {
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Execute each tool call
            first_in_batch = True
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                # Record the tool call (include reasoning on first call of batch)
                tool_entry = {
                    "id": tool_call.id,
                    "name": func_name,
                    "arguments": func_args,
                }
                if first_in_batch and reasoning:
                    tool_entry["reasoning"] = reasoning
                    first_in_batch = False
                tool_calls_made.append(tool_entry)

                # Execute the tool
                try:
                    result = tool_executor(func_name, func_args)
                    result_str = (
                        json.dumps(result, default=_json_default)
                        if not isinstance(result, str)
                        else result
                    )
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})

                # Add tool result to messages
                messages_payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

            # Continue to next round to let model process results
            continue

        # No tool calls - return the final response
        messages_payload.append(
            {
                "role": "assistant",
                "content": last_content,
            }
        )
        return last_content, tool_calls_made, total_usage, messages_payload

    # Max rounds reached - return last response
    if last_content:
        messages_payload.append(
            {
                "role": "assistant",
                "content": last_content,
            }
        )
    return last_content, tool_calls_made, total_usage, messages_payload
