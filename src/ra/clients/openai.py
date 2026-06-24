import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any

import httpx
import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

# Use Langfuse-wrapped OpenAI client when available for automatic tracing
_LangfuseOpenAI = None
_LangfuseAsyncOpenAI = None
if os.environ.get("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.openai import OpenAI as _LangfuseOpenAI, AsyncOpenAI as _LangfuseAsyncOpenAI
    except ImportError:
        pass

from ra.clients.base_lm import BaseLM
from ra.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

_log = logging.getLogger(__name__)


# Exceptions that warrant a retry. Network blips, malformed payloads from
# the proxy (e.g. OpenRouter occasionally returns a body that fails JSON
# parsing), rate limits, and 5xx errors are all transient. Hard 4xx errors
# (auth, bad request, model-not-found) are NOT retried.
def _build_retryable_tuple() -> tuple[type[BaseException], ...]:
    """Assemble the retryable-exception tuple at import time.

    Kept as a builder so the ``_TransientLLMError`` class (defined
    below) can be appended without ordering games.
    """
    return (
        json.JSONDecodeError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
        httpx.RequestError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
    )


_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = _build_retryable_tuple()

# Status codes (when an APIStatusError surfaces) that should be retried.
# 429 is rate-limit; 5xx are server-side blips.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

_DEFAULT_MAX_RETRIES = int(os.environ.get("KAI_LLM_MAX_RETRIES", "4"))
_DEFAULT_BASE_BACKOFF_S = float(os.environ.get("KAI_LLM_BACKOFF_S", "1.5"))
# Per-request timeout (seconds) passed to the OpenAI SDK so a single
# completion can't hang a worker indefinitely. Override with
# KAI_LLM_REQUEST_TIMEOUT_S.
_DEFAULT_LLM_REQUEST_TIMEOUT_S = float(
    os.environ.get("KAI_LLM_REQUEST_TIMEOUT_S", "540")
)


class _TransientLLMError(RuntimeError):
    """Raised by ``_extract_text`` when the LLM response is structurally
    malformed (no choices, empty content) so the retry layer can treat
    it as transient without forging an ``openai.APIError`` (which
    requires a real ``httpx.Request``)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _TransientLLMError):
        return True
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
            return True
    return False


def _sleep_for_retry(attempt: int) -> float:
    """Exponential backoff with jitter; attempt is 1-indexed."""
    base = _DEFAULT_BASE_BACKOFF_S * (2 ** (attempt - 1))
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def _call_with_retry(call_fn, *, model: str, log_prefix: str) -> Any:
    """Call ``call_fn`` synchronously with retry on transient errors."""
    attempt = 0
    last_exc: BaseException | None = None
    while attempt < _DEFAULT_MAX_RETRIES:
        attempt += 1
        try:
            return call_fn()
        except BaseException as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt >= _DEFAULT_MAX_RETRIES:
                break
            delay = _sleep_for_retry(attempt)
            _log.warning(
                "%s transient LLM failure (%s) on attempt %d/%d; "
                "retrying in %.1fs (model=%s)",
                log_prefix,
                type(exc).__name__,
                attempt,
                _DEFAULT_MAX_RETRIES,
                delay,
                model,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _acall_with_retry(coro_factory, *, model: str, log_prefix: str) -> Any:
    """Async variant of ``_call_with_retry``."""
    attempt = 0
    last_exc: BaseException | None = None
    while attempt < _DEFAULT_MAX_RETRIES:
        attempt += 1
        try:
            return await coro_factory()
        except BaseException as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt >= _DEFAULT_MAX_RETRIES:
                break
            delay = _sleep_for_retry(attempt)
            _log.warning(
                "%s transient LLM failure (%s) on attempt %d/%d; "
                "retrying in %.1fs (model=%s)",
                log_prefix,
                type(exc).__name__,
                attempt,
                _DEFAULT_MAX_RETRIES,
                delay,
                model,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _extract_text(response: Any) -> str:
    """Pull the message text out of a chat-completions response, raising
    ``_TransientLLMError`` when the payload is structurally bad so the
    retry layer can treat it as transient.
    """
    try:
        choices = response.choices
    except AttributeError as exc:
        raise _TransientLLMError("response has no .choices field") from exc
    if not choices:
        raise _TransientLLMError("response.choices is empty")
    content = getattr(choices[0].message, "content", None)
    if content is None or content == "":
        raise _TransientLLMError("response.choices[0].message.content is empty")
    return content


# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"

OPENROUTER_APP_URL = os.getenv(
    "OPENROUTER_APP_URL",
    "https://kai.dria.co/",
)
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "kai-security")
OPENROUTER_APP_CATEGORIES = os.getenv(
    "OPENROUTER_APP_CATEGORIES",
    "cli-agent,programming-app",
)
_OPENROUTER_HEADERS = {
    "HTTP-Referer": OPENROUTER_APP_URL,
    "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
}
if OPENROUTER_APP_CATEGORIES:
    _OPENROUTER_HEADERS["X-OpenRouter-Categories"] = OPENROUTER_APP_CATEGORIES


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY

        # For vLLM, set base_url to local vLLM server address.
        extra_headers = (
            _OPENROUTER_HEADERS if base_url == "https://openrouter.ai/api/v1" else None
        )
        _ClientCls = _LangfuseOpenAI or openai.OpenAI
        self.client = _ClientCls(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
        )
        self._async_client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "default_headers": extra_headers,
        }
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = self._build_extra_body()

        def _do_call() -> tuple[ChatCompletion, str]:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                extra_body=extra_body,
                timeout=_DEFAULT_LLM_REQUEST_TIMEOUT_S,
            )
            return response, _extract_text(response)

        response, text = _call_with_retry(
            _do_call, model=model, log_prefix="sync completion:"
        )
        self._track_cost(response, model)
        return text

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = self._build_extra_body()

        _AsyncCls = _LangfuseAsyncOpenAI or openai.AsyncOpenAI
        async with _AsyncCls(
            **self._async_client_kwargs,
        ) as client:

            async def _do_call() -> tuple[ChatCompletion, str]:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    extra_body=extra_body,
                    timeout=_DEFAULT_LLM_REQUEST_TIMEOUT_S,
                )
                return response, _extract_text(response)

            response, text = await _acall_with_retry(
                _do_call, model=model, log_prefix="async completion:"
            )
        self._track_cost(response, model)
        return text

    def _build_extra_body(self) -> dict[str, Any]:
        """Per-call ``extra_body`` for chat completions.

        Prompt caching stays enabled by default (matching upstream
        behavior). Set ``KAI_DISABLE_PROMPT_CACHE=1`` to turn off
        OpenRouter's response cache for stricter per-task independence
        on long benchmark sweeps.
        """
        extra_body: dict[str, Any] = {}
        base_url = str(self.client.base_url or "")
        if base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}
        if (
            "openrouter" in base_url.lower()
            and os.environ.get("KAI_DISABLE_PROMPT_CACHE") == "1"
        ):
            extra_body["cache"] = False
        return extra_body

    def _track_cost(self, response: ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
