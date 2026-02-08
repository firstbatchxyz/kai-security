import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent model settings
SETUP_DEFAULT_MODEL = "openai/gpt-5.2-codex"
MAIN_DEFAULT_MODEL = "google/gemini-3-flash-preview"  # stepfun/step-3.5-flash:free x-ai/grok-code-fast-1 google/gemini-3-flash-preview
GAMIFIED_DEFAULT_MODEL = "google/gemini-3-flash-preview"
VERIFIER_DEFAULT_MODEL = "openai/gpt-5.2-codex"
INVARIANT_DEFAULT_MODEL = "openai/gpt-5.2-codex"
DEDUPE_DEFAULT_MODEL = "openai/gpt-5.2-codex"
FIXER_DEFAULT_MODEL = "openai/gpt-5.2-codex"

# Fallback model (used when primary model fails after retries)
FALLBACK_MODEL = "google/gemini-3-flash-preview"

# Agent turn limits (centralized)
DEFAULT_MAX_TURNS = 32
SETUP_MAX_TURNS = 32  # Setup agent
PROFILER_MAX_TURNS = 24  # Profiler agent
VERIFIER_MAX_TURNS = 16  # Verifier needs fewer turns
VALIDATION_MAX_TURNS = 8  # Workspace validation is quick
INVARIANT_SYNTH_MAX_TURNS = 8  # Invariant synthesizer per observation

# Legacy alias (deprecated - use DEFAULT_MAX_TURNS)
MAX_TOOL_TURNS = DEFAULT_MAX_TURNS

# Dispatcher settings
MAX_CONCURRENT_AGENTS = 2
MAX_CONCURRENT_FIXERS = 4

# Python workspace settings
PRE_INSTALL_PACKAGES: list[str] = ["pytest", "requests"]

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "https://kai.dria.co")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Kai")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# vLLM
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# Engine
SANDBOX_TIMEOUT = 3600  # 1 hour

# Path settings
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SETUP_AGENT_PROMPT_PATH = PROMPTS_DIR / "setup_agent_prompt.txt"
PROFILER_AGENT_PROMPT_PATH = PROMPTS_DIR / "profiler_agent_prompt.txt"
BLACKBOX_AGENT_PROMPT_PATH = PROMPTS_DIR / "blackbox_agent_prompt.txt"
INVARIANT_SYNTHESIZER_AGENT_PROMPT_PATH = (
    PROMPTS_DIR / "invariant_synthesizer_agent_prompt.txt"
)
STATE_AGENT_PROMPT_PATH = PROMPTS_DIR / "state_agent_prompt.txt"
QUANT_AGENT_PROMPT_PATH = PROMPTS_DIR / "quant_agent_prompt.txt"
VERIFIER_AGENT_PROMPT_PATH = PROMPTS_DIR / "verifier_agent_prompt.txt"
WORKSPACE_VALIDATION_AGENT_PROMPT_PATH = (
    PROMPTS_DIR / "workspace_validation_agent_prompt.txt"
)
FIXER_AGENT_PROMPT_PATH = PROMPTS_DIR / "fixer_agent_prompt.txt"
GAMIFIED_AGENT_PROMPT_PATH = PROMPTS_DIR / "gamified_agent_prompt.txt"
BUCKETING_AGENT_PROMPT_PATH = PROMPTS_DIR / "bucketing_agent_prompt.txt"
HTTP_AGENT_PROMPT_PATH = PROMPTS_DIR / "http_agent_prompt.txt"

# Conversation defaults
SAVE_CONVERSATION_PATH = "output/conversations/"
EXPLOITS_PATH = "exploits.json"
TEST_SCRIPTS_PATH = "test/"

# Tool output truncation settings
# These help prevent context overflow when tool outputs (e.g., npm errors) are very large
TOOL_OUTPUT_MAX_LENGTH = 50_000  # Max characters per tool output (default: 50k chars ~12.5k tokens)
TOOL_OUTPUT_TRUNCATION_MESSAGE = "\n\n... [OUTPUT TRUNCATED - exceeded {max_len} characters] ..."

# Logging settings
MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
MONGO_DB_NAME: str = "kai"
