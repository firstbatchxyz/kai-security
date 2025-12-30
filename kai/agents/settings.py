import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent settings (aligned with legacy run_scaffold defaults)
DEFAULT_TURNS = 32
SETUP_DEFAULT_MODEL = "z-ai/glm-4.7"
MAIN_DEFAULT_MODEL = "z-ai/glm-4.7"
MAX_TOOL_TURNS = 24

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

# Conversation defaults
SAVE_CONVERSATION_PATH = "output/conversations/"
EXPLOITS_PATH = "exploits.json"
TEST_SCRIPTS_PATH = "test/"

# Logging settings
MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
MONGO_DB_NAME: str = "kai"
