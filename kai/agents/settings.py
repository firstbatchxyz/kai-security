import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent model settings
SETUP_DEFAULT_MODEL = "google/gemini-3-flash-preview"
MAIN_DEFAULT_MODEL = "openai/gpt-5.2"
GAMIFIED_DEFAULT_MODEL = "anthropic/claude-opus-4.5"
VERIFIER_DEFAULT_MODEL = "anthropic/claude-opus-4.5"

# Agent turn limits (centralized)
DEFAULT_MAX_TURNS = (
    32  # Default for most agents (state, quant, blackbox, gamified, fixer)
)
SETUP_MAX_TURNS = 24  # Setup agent needs fewer turns
PROFILER_MAX_TURNS = 12  # Profiler agent
VERIFIER_MAX_TURNS = 16  # Verifier needs fewer turns
VALIDATION_MAX_TURNS = 8  # Workspace validation is quick
INVARIANT_SYNTH_MAX_TURNS = 8  # Invariant synthesizer per observation

# Legacy alias (deprecated - use DEFAULT_MAX_TURNS)
MAX_TOOL_TURNS = DEFAULT_MAX_TURNS

# Dispatcher settings
MAX_CONCURRENT_AGENTS = 2

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

# Conversation defaults
SAVE_CONVERSATION_PATH = "output/conversations/"
EXPLOITS_PATH = "exploits.json"
TEST_SCRIPTS_PATH = "test/"

# Logging settings
MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
MONGO_DB_NAME: str = "kai"
