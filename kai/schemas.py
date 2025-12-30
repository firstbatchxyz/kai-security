from enum import Enum
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator

from kai.agents import settings

# Adapter type literal for structured output validation
AdapterType = Literal["solidity"]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: str
    # Optional tool-calling metadata (for native OpenAI tool calling).
    # These are ignored in python-block mode but allow us to persist tool call traces.
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Language(str, Enum):
    SOLIDITY = "solidity"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    CPP = "cpp"


class Framework(str, Enum):
    FOUNDRY = "foundry"
    NODE = "node"
    CARGO = "cargo"
    CMAKE = "cmake"


class AdapterSelection(BaseModel):
    """Result of selecting adapters based on detected languages."""

    languages: list[Language] = Field(default_factory=list)
    frameworks: list[Framework] = Field(default_factory=list)
    adapters: list[str | None] = Field(default_factory=list)
    reason: Optional[str] = None


class MasterContext(BaseModel):
    """
    Immutable view of the built repository used by downstream agents.
    """

    model_config = ConfigDict(extra="forbid")

    root_path: str
    frameworks: Optional[list[str]] = None
    artifacts_path: Optional[str] = None
    src_path: Optional[str] = None
    lib_path: Optional[str] = None
    test_path: Optional[str] = None
    compile_success: bool
    # Setup output: store both the path (relative to root_path) and the full contents.
    # Prefer scripts over command lists so downstream can execute consistently.
    build_script_path: Optional[str] = None
    build_script: Optional[str] = None
    test_script_path: Optional[str] = None
    test_script: Optional[str] = None
    adapter: AdapterType = "solidity"  # Domain adapter for dependency graph analysis


class AgentResponse(BaseModel):
    thoughts: str
    python_block: Optional[str] = None
    test_script: Optional[str] = None
    suggest_fix: Optional[str] = None
    master_context: Optional[MasterContext] = None
    # Profiler-specific optional payload
    protocol_manifesto: Optional["ProtocolManifesto"] = None

    def __str__(self):
        return (
            f"Thoughts: {self.thoughts}\n"
            f"Python block:\n {self.python_block}\n"
            f"Test script:\n {self.test_script}\n"
            f"Suggest fix:\n {self.suggest_fix}\n"
            f"Master context:\n {self.master_context}"
        )


class ExploitSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------
# Agent Record (for Agent Tracking in MongoDB)
# ---------------------------


class AgentRecord(BaseModel):
    """
    Record of an agent execution for MongoDB tracking.
    
    Created when an agent starts, updated when it completes.
    The _id in MongoDB equals the worker_id used in exploits collection.
    
    MongoDB fields use camelCase (executionId, agentType, completedAt)
    """
    
    agent_id: str = Field(serialization_alias="agentId")  # Same as worker_id in exploits
    execution_id: Optional[str] = Field(default=None, serialization_alias="executionId")
    agent_type: str = Field(serialization_alias="agentType")  # AgentType enum value
    created_at: Optional[Any] = Field(default=None, serialization_alias="createdAt")  # datetime
    completed_at: Optional[Any] = Field(default=None, serialization_alias="completedAt")  # datetime


# ---------------------------
# Invariant Types (for InvariantProcess → Dispatcher)
# ---------------------------


class InvariantType(str, Enum):
    """Categories of invariants - just labels, not constraints."""

    ACCESS = "access"  # Only X can call Y
    SOLVENCY = "solvency"  # totalAssets >= totalLiabilities
    CONSERVATION = "conservation"  # Sum invariants (balances, supply)
    LIVENESS = "liveness"  # Function must be callable
    FEE_BOUND = "fee_bound"  # Fee constraints
    REENTRANCY = "reentrancy"  # No reentrant state corruption
    ORDERING = "ordering"  # State transition ordering
    OTHER = "other"  # Anything else


class Invariant(BaseModel):
    """
    A grounded invariant rule used by Dispatcher to schedule missions.

    All target fields reference node IDs from the DependencyGraph.
    Output of InvariantProcess, consumed by Dispatcher and Workers.
    
    MongoDB fields use camelCase (targetFunctionIds, targetVarIds, etc.)
    """

    id: str  # e.g., "INV_SUPPLY_CONSERVATION", "INV_ADMIN_UPGRADE"
    type: InvariantType
    rule: str  # Human-readable invariant statement
    explanation: str = ""  # LLM's reasoning for this invariant

    # Grounded targets - all must be valid graph node IDs
    target_function_ids: List[str] = Field(
        default_factory=list, serialization_alias="targetFunctionIds"
    )
    target_var_ids: List[str] = Field(
        default_factory=list, serialization_alias="targetVarIds"
    )
    target_file_ids: List[str] = Field(
        default_factory=list, serialization_alias="targetFileIds"
    )

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "llm"  # "llm", "pattern", "docs"
    chunk_id: Optional[str] = Field(
        default=None, serialization_alias="chunkId"
    )  # Which chunk this came from


# Vocab table entries for LLM context
class FunctionVocabEntry(BaseModel):
    """Function entry in vocab table for LLM."""

    id: str  # Node ID from graph
    name: str
    container: str  # Contract name
    file: str
    role: str = ""  # From ActorMatrix (e.g., "Admin", "User")
    trust: str = ""  # From ActorMatrix (e.g., "high", "none")
    reads: List[str] = Field(default_factory=list)  # Var names
    writes: List[str] = Field(default_factory=list)  # Var names


class VarVocabEntry(BaseModel):
    """State variable entry in vocab table for LLM."""

    id: str  # Node ID from graph
    name: str
    container: str  # Contract name
    file: str
    writers: List[str] = Field(default_factory=list)  # Function IDs that write
    readers: List[str] = Field(default_factory=list)  # Function IDs that read


class FileVocabEntry(BaseModel):
    """File entry in vocab table for LLM."""

    id: str  # File path
    contracts: List[str] = Field(default_factory=list)


class VocabChunk(BaseModel):
    """A chunk of vocabulary for one LLM call."""

    chunk_id: str
    functions: List[FunctionVocabEntry] = Field(default_factory=list)
    vars: List[VarVocabEntry] = Field(default_factory=list)
    files: List[FileVocabEntry] = Field(default_factory=list)


class InvariantProcessInput(BaseModel):
    """Input for InvariantProcess."""

    master_context: "MasterContext"
    dependency_graph: Any  # DependencyGraph object
    actor_matrix: "ActorMatrix"
    protocol_manifesto: Optional["ProtocolManifesto"] = None
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = False
    max_chunk_functions: int = 25  # Max functions per chunk


class InvariantProcessOutput(BaseModel):
    """Output of InvariantProcess."""

    invariants: List[Invariant] = Field(default_factory=list)
    success: bool
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    total_tokens: Dict[str, int] = Field(default_factory=dict)
    stats: Dict[str, int] = Field(default_factory=dict)
    # stats: {total_generated, validated, dropped, merged}


class Observation(BaseModel):
    """
    Intermediate finding from non-invariant workers (BlackBox, Gamified).

    Gets refined by LLM into a tentative Invariant.
    
    MongoDB fields use camelCase (workerId, missionId, affectedFunctions, etc.)
    """

    worker_id: str = Field(serialization_alias="workerId")
    mission_id: str = Field(serialization_alias="missionId")
    description: str  # "Function X always reverts when called by any actor"
    affected_functions: List[str] = Field(default_factory=list, serialization_alias="affectedFunctions")
    affected_files: List[str] = Field(default_factory=list, serialization_alias="affectedFiles")
    logs: List[str] = Field(default_factory=list)  # Raw output/traces
    anomaly_type: Optional[str] = Field(default=None, serialization_alias="anomalyType")  # "always_reverts", "unexpected_state", etc.

    # --- Grounded blackbox fields (optional; backwards compatible) ---
    repro_command: Optional[str] = Field(default=None, serialization_alias="reproCommand")
    seed: Optional[int] = None


class ExploitCandidate(BaseModel):
    """
    A potential exploit from invariant-dependent agents.

    Sent to Verifier for confirmation.
    
    MongoDB fields use camelCase (missionId, workerId, invariantId, etc.)
    """

    mission_id: str = Field(serialization_alias="missionId")
    worker_id: str = Field(serialization_alias="workerId")
    invariant_id: str = Field(serialization_alias="invariantId")  # Which invariant this claims to violate
    mechanism: str  # "reentrancy", "access_control_bypass", etc.
    poc_code: str = Field(serialization_alias="pocCode")  # The exploit contract/test code
    target_file: str = Field(serialization_alias="targetFile")
    target_function: str = Field(serialization_alias="targetFunction")
    description: str
    compiled: bool = False  # Did it compile in agent's workspace?
    logs: List[str] = Field(default_factory=list)
    # Verdict fields (populated after verification)
    severity: Optional[str] = None
    verdict: Optional[Dict[str, Any]] = None  # {isValid: bool}
    fixes: List["Fix"] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_shapes(cls, values: Any) -> Any:
        """
        Backwards compatibility:
        - agent_id -> worker_id
        """
        if not isinstance(values, dict):
            return values
        data = dict(values)
        if "worker_id" not in data and "agent_id" in data:
            data["worker_id"] = data.get("agent_id")
        return data


class VerdictSeverity(str, Enum):
    """Severity levels for verified exploits."""

    CRITICAL = "critical"  # Direct fund loss, no user action required
    HIGH = "high"  # Fund loss possible with some conditions
    MEDIUM = "medium"  # Limited impact or requires specific circumstances
    LOW = "low"  # Minimal impact, edge cases only
    INFORMATIONAL = "informational"  # Known limitation, not a bug


class Verdict(BaseModel):
    """
    Verifier's judgment on an ExploitCandidate.

    Determines if the finding is valid, its severity, and economic feasibility.
    
    MongoDB fields use camelCase (missionId, invariantId, workerId, isValid, etc.)
    """

    # Reference to original finding
    mission_id: str = Field(serialization_alias="missionId")
    invariant_id: str = Field(serialization_alias="invariantId")
    worker_id: str = Field(serialization_alias="workerId")

    # Core verdict
    is_valid: bool = Field(serialization_alias="isValid")  # Is this a real, exploitable vulnerability?
    severity: VerdictSeverity

    # Analysis flags
    uses_mock_contracts: bool = Field(default=False, serialization_alias="usesMockContracts")  # Did PoC use fake/hostile contracts?
    is_economically_feasible: bool = Field(default=True, serialization_alias="isEconomicallyFeasible")  # Is attack profitable or at least cheap?
    is_known_limitation: bool = Field(default=False, serialization_alias="isKnownLimitation")  # Known design tradeoff vs actual bug?
    targets_real_implementation: bool = Field(default=True, serialization_alias="targetsRealImplementation")  # Tests actual code, not just interface?

    # Economic analysis
    attack_cost_estimate: Optional[str] = Field(default=None, serialization_alias="attackCostEstimate")  # e.g., "$1M donation to grief $1"
    attacker_profit_estimate: Optional[str] = Field(default=None, serialization_alias="attackerProfitEstimate")  # e.g., "Can extract $X"
    cost_benefit_ratio: Optional[str] = Field(default=None, serialization_alias="costBenefitRatio")  # e.g., "1000:1 loss ratio"

    # Classification
    vulnerability_class: str = Field(default="", serialization_alias="vulnerabilityClass")  # "reentrancy", "donation_attack", "access_control"

    # Reasoning
    reasoning: str  # Full analysis and justification
    rejection_reason: Optional[str] = Field(default=None, serialization_alias="rejectionReason")  # If invalid, specific reason

    # Original PoC reference
    poc_path: Optional[str] = Field(default=None, serialization_alias="pocPath")
    test_passed: bool = Field(default=False, serialization_alias="testPassed")  # Did the PoC test pass?

    # Fixes (populated after fixer runs)
    fixes: List["Fix"] = Field(default_factory=list)


class Fix(BaseModel):
    """
    A code fix for a verified exploit.

    Generated by FixerAgent after Verifier confirms a vulnerability.
    
    MongoDB fields use camelCase (fixId, missionId, invariantId, etc.)
    """

    # Unique identifier
    fix_id: str = Field(serialization_alias="fixId")

    # References to original finding (for DB linking)
    mission_id: str = Field(serialization_alias="missionId")
    invariant_id: str = Field(serialization_alias="invariantId")
    verdict_id: Optional[str] = Field(default=None, serialization_alias="verdictId")  # If verdicts get IDs

    # Fix content
    summary: str  # One-paragraph summary of what was fixed
    reasoning: str  # Why this fix addresses the vulnerability
    canonical_diff: str = Field(serialization_alias="canonicalDiff")  # Unified diff string (can be multi-file)
    files_changed: List[str] = Field(default_factory=list, serialization_alias="filesChanged")

    # Validation status
    compiled: bool = False  # Did the fix compile?
    tests_passed: bool = Field(default=False, serialization_alias="testsPassed")  # Did tests pass after applying fix?


class EnvironmentSetupInput(BaseModel):
    repo_url: str
    num_turns: int = settings.MAX_TOOL_TURNS
    model_name: str = settings.SETUP_DEFAULT_MODEL
    use_openai: bool = False
    execution_id: Optional[str] = None
    repo_path_override: Optional[str] = None


class EnvironmentSetupOutput(BaseModel):
    response: Optional[AgentResponse]
    master_context: Optional[MasterContext]
    estimated_cost: float
    total_tokens: Dict[str, int]
    success: bool
    error_message: Optional[str]
    master_repo_path: str
    repo_slug: str


# ---------------------------
# Workspace Validation Schemas
# ---------------------------


class WorkspaceValidationInput(BaseModel):
    """
    Input for WorkspaceValidationProcess.

    Validates that a provisioned agent workspace can:
    - accept a minimal test file write
    - compile successfully (adapter-based)
    - run a targeted smoke test (adapter-based)
    """

    master_context: MasterContext
    presets: List["WorkspacePreset"] = Field(default_factory=list)
    timeout_compile_s: int = 120
    timeout_test_s: int = 120


class WorkspaceValidationResult(BaseModel):
    preset: "WorkspacePreset"
    workspace_path: str
    smoke_test_relpath: str
    framework: str
    compiled: bool = False
    compile_errors: List[str] = Field(default_factory=list)
    test_success: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    raw_output: str = ""
    error: Optional[str] = None


class WorkspaceValidationOutput(BaseModel):
    success: bool
    results: List[WorkspaceValidationResult] = Field(default_factory=list)
    error_message: Optional[str] = None


# ---------------------------
# Actor Analysis Schemas
# ---------------------------


class SuspiciousFunction(BaseModel):
    """Function flagged by heuristic scan for potential access control issues."""

    function_name: str
    function_id: str
    contract_name: Optional[str] = None
    file_path: Optional[str] = None
    visibility: str
    patterns_matched: List[str] = Field(default_factory=list)
    has_modifier: bool = False
    reason: str
    writes_state: bool = False


class PrivilegeChain(BaseModel):
    """Cross-contract privilege chain (e.g., Keeper -> Vault -> Strategy)."""

    source_contract: str
    source_function: str
    source_role: Optional[str] = None
    target_contract: str
    target_function: str
    call_path: List[str] = Field(default_factory=list)
    can_send_eth: bool = False
    edge_count: int = 1


class Actor(BaseModel):
    """Actor with roles and privileges (enhanced ActorRole)."""

    role: str
    trust_level: str  # "High", "Medium", "Low", "None", "N/A"
    modifier_patterns: List[str] = Field(default_factory=list)
    direct_privileges: List[str] = Field(default_factory=list)
    indirect_privileges: List[str] = Field(default_factory=list)
    function_count: int = 0
    contracts: List[str] = Field(default_factory=list)


class LLMReviewResult(BaseModel):
    """Result from LLM review of a suspicious function."""

    function_name: str
    contract_name: Optional[str] = None
    is_vulnerability: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issue_type: Optional[str] = None
    description: str
    recommendation: Optional[str] = None


class ActorAnalysisInput(BaseModel):
    """Input for ActorAnalysisProcess."""

    graph: Any  # DependencyGraph object
    slither: Optional[Any] = None
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = True
    enable_llm_review: bool = True
    max_suspicious_for_llm: int = 200


class ActorReport(BaseModel):
    """Output of ActorAnalysisProcess."""

    actors: List[Actor] = Field(default_factory=list)
    privilege_chains: List[PrivilegeChain] = Field(default_factory=list)
    guard_issues: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_functions: List[SuspiciousFunction] = Field(default_factory=list)
    llm_review_results: List[LLMReviewResult] = Field(default_factory=list)
    llm_invoked: bool = False
    llm_tokens_used: Dict[str, int] = Field(default_factory=dict)
    llm_cost_estimate: float = 0.0
    total_public_functions: int = 0
    protected_functions: int = 0
    unprotected_functions: int = 0
    suspicious_count: int = 0
    confirmed_issues: int = 0


class Feature(BaseModel):
    name: str
    description: str
    actors: list[str]


class ProtocolManifesto(BaseModel):
    name: str
    purpose: str
    description: str
    domain: str = ""
    programming_languages: list[str]

    intended_users: list[str] = Field(default_factory=list)
    # ["depositors", "borrowers", "admins"] - semantic roles, not modifier names

    # === Domain concepts ===
    key_concepts: dict[str, str] = Field(default_factory=dict)
    # {"health_factor": "ratio determining liquidation eligibility",
    #  "utilization": "borrowed / total liquidity"}

    # === Key Features ===
    key_features: list[Feature]

    @model_validator(mode="before")
    @classmethod
    def coerce_key_concepts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Allow key_concepts provided as a list of strings; coerce to {k: k}
        if not isinstance(values, dict):
            return values
        kc = values.get("key_concepts")
        if isinstance(kc, list):
            coerced = {}
            for item in kc:
                if isinstance(item, str):
                    coerced[item] = item
            values["key_concepts"] = coerced
        return values


class ProfilerInput(BaseModel):
    master_context: MasterContext
    dependency_graph: Any | None = None  # DependencyGraph object (optional, for reuse)
    num_turns: int
    model_name: str
    use_openai: bool = False
    execution_id: Optional[str] = None


class ProfilerOutput(BaseModel):
    response: Optional[AgentResponse]
    protocol_manifesto: Optional[ProtocolManifesto]
    estimated_cost: float
    total_tokens: Dict[str, int]
    success: bool
    error_message: Optional[str]
    repo_path: str


# ---------------------------
# Actor Matrix Schemas (Grounded)
# ---------------------------


class Privilege(BaseModel):
    """A single privilege (entrypoint) with full grounding info."""

    id: str  # Node ID (e.g., "Vault.withdraw(uint256)")
    name: str  # Function name
    container: str  # Contract name
    signature: Optional[str] = None  # Full signature
    file: Optional[str] = None  # Source file path
    writes_state: bool = False  # Does it write state variables?
    write_targets: List[str] = Field(default_factory=list)  # State var names (display)
    write_target_ids: List[str] = Field(
        default_factory=list
    )  # State var node IDs (matching)


class RoleEvidence(BaseModel):
    """Evidence anchoring a role assignment to code."""

    function_id: str
    modifiers: List[str] = Field(
        default_factory=list
    )  # Modifier names from ACCEPTS edges
    snippet_file: Optional[str] = None
    snippet_lines: Optional[List[int]] = None  # [start, end]


class ActorMatrixRole(BaseModel):
    """A role in the ActorMatrix with grounded privileges and evidence."""

    name: str  # Role name (e.g., "Owner", "User")
    trust: str  # "high", "medium", "low", "none"
    reasoning: str = ""  # LLM's reasoning for trust assignment
    access_signature: List[str] = Field(default_factory=list)  # Normalized modifiers
    privileges: List[Privilege] = Field(default_factory=list)
    evidence: List[RoleEvidence] = Field(default_factory=list)
    risk_score: int = 0  # Aggregate of state writes


class RoleAssignment(BaseModel):
    """LLM's trust assignment for a single role cluster."""

    signature_key: (
        str  # The cluster key (e.g., "Ownable.onlyOwner" or "__unprotected__")
    )
    name: str  # Human-readable role name
    trust: Literal["high", "medium", "low", "none"]
    reasoning: str  # Brief explanation of why this trust level


class ActorMatrix(BaseModel):
    """
    Grounded ActorMatrix output.

    All privileges are anchored to node IDs, not bare names.
    Evidence links roles to actual code locations.
    """

    roles: List[ActorMatrixRole] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)
    # stats: {total_entrypoints, unprotected_count, review_required_count}


class ActorMatrixInput(BaseModel):
    """Input for ActorProcess."""

    master_context: "MasterContext"
    dependency_graph: Any  # DependencyGraph object
    protocol_manifesto: Optional["ProtocolManifesto"] = None
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = False


class ActorMatrixOutput(BaseModel):
    """Output of ActorProcess."""

    actor_matrix: Optional[ActorMatrix] = None
    success: bool
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    total_tokens: Dict[str, int] = Field(default_factory=dict)


# Resolve forward references for models that refer to ProtocolManifesto
AgentResponse.model_rebuild()
# Resolve forward reference for Verdict -> Fix
Verdict.model_rebuild()


class AdapterChooserInput(BaseModel):
    model_name: str
    use_openai: bool = False
    available_frameworks: Optional[list[str]] = None


class AdapterChooserOutput(BaseModel):
    choice: Optional[AdapterSelection]
    raw_response: Optional[str] = None
    estimated_cost: float = 0.0
    total_tokens: Dict[str, int] = Field(default_factory=dict)
    success: bool
    error_message: Optional[str] = None


# Dispatcher and Campaign Schemas


class MissionAgentType(str, Enum):
    """
    Agent types for missions.

    Invariant-dependent:
    - QUANT: Break numeric invariants, equations (solvency, conservation)
    - STATE: Break state machine invariants (call sequences, access control)

    Non-dependent (exploration):
    - BLACKBOX: Anomaly detection, surfaces observations → new invariants
    - GAMIFIED: Game-theoretic exploitation (maximize attacker payoff)
    """

    QUANT = "quant"
    STATE = "state"
    BLACKBOX = "blackbox"
    GAMIFIED = "gamified"


class CampaignMode(str, Enum):
    """Campaign execution mode."""

    INVARIANT_BOUNDED = "invariant_bounded"  # Test specific invariants
    EXPLORATORY = "exploratory"  # BlackBox anomaly detection
    GAME = "game"  # Gamified payoff maximization


class WorkspacePreset(str, Enum):
    """Workspace provisioning presets."""

    CLEAN = "clean"  # src/test COPY, lib SYMLINK
    WRITEABLE = "writeable"  # src/lib/test COPY (writeable)
    SANDBOX = "sandbox"  # FULL COPY + extras (world/, mocks/, actors/)
    LIGHTWEIGHT = "lightweight"  # Minimal forge project with remappings to parent


class RewardModel(str, Enum):
    """Reward model for gamified agents."""

    NONE = "none"  # No reward model (invariant-based)
    PROFIT = "profit"  # Maximize attacker profit
    GRIEF = "grief"  # Maximize griefing effect
    COVERAGE = "coverage"  # Maximize code coverage


class EntrypointPolicy(BaseModel):
    """Policy for filtering/ordering entrypoints."""

    max_sequence_len: int = 6
    prefer: List[str] = Field(
        default_factory=list
    )  # ["writes_state", "external_calls"]
    exclude_patterns: List[str] = Field(default_factory=list)


class EntrypointSubset(BaseModel):
    """Filtered entrypoints with policy."""

    ids: List[str] = Field(default_factory=list)
    policy: EntrypointPolicy = Field(default_factory=EntrypointPolicy)


class CampaignScope(BaseModel):
    """Scope definition for a campaign."""

    entrypoints_subset: EntrypointSubset = Field(default_factory=EntrypointSubset)
    primary_var_ids: List[str] = Field(default_factory=list)
    file_ids: List[str] = Field(default_factory=list)
    actor_roles: List[str] = Field(default_factory=list)


class CampaignObjectives(BaseModel):
    """Objectives for a campaign."""

    reward_model: RewardModel = RewardModel.NONE
    notes: str = ""


class CampaignBudget(BaseModel):
    """Resource budget for a campaign.
    
    MongoDB fields use camelCase (maxMissions, maxAgents, maxTurnsPerAgent)
    """

    max_missions: int = Field(default=6, serialization_alias="maxMissions")
    max_agents: int = Field(default=3, serialization_alias="maxAgents")
    max_turns_per_agent: int = Field(default=20, serialization_alias="maxTurnsPerAgent")


class InvariantCluster(BaseModel):
    """
    Intermediate grouping of related invariants.

    Clustering rule: invariants sharing vars OR functions
    in the same container/file → same cluster.
    """

    cluster_id: str
    invariant_ids: List[str] = Field(default_factory=list)
    primary_var_ids: List[str] = Field(default_factory=list)
    primary_function_ids: List[str] = Field(default_factory=list)
    primary_container: Optional[str] = None
    dominant_type: Optional[InvariantType] = None


class CampaignBrief(BaseModel):
    """
    A campaign brief with full execution context for agents.

    Self-contained: agents can execute with only this + workspace.
    
    MongoDB storage excludes: scope, master_context (stored in S3)
    MongoDB fields use camelCase (campaignId, agentTypes, workspacePreset, etc.)
    """

    campaign_id: str = Field(serialization_alias="campaignId")
    mode: CampaignMode = CampaignMode.INVARIANT_BOUNDED
    agent_types: List[MissionAgentType] = Field(default_factory=list, serialization_alias="agentTypes")
    framework: Optional[str] = None
    workspace_preset: WorkspacePreset = Field(default=WorkspacePreset.CLEAN, serialization_alias="workspacePreset")
    # Scope (derived from cluster + ActorMatrix + Graph) - excluded from MongoDB
    scope: CampaignScope = Field(default_factory=CampaignScope, exclude=True)
    # Invariants to test (only IDs stored in MongoDB)
    invariants: List[Invariant] = Field(default_factory=list)
    # Objectives
    objectives: CampaignObjectives = Field(default_factory=CampaignObjectives, exclude=True)
    # Budget
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    # MasterContext for agent independence - excluded from MongoDB (stored in S3)
    master_context: Optional[MasterContext] = Field(default=None, exclude=True)
    # Priority: 0=verification, 1=narrow, 2=broad
    priority: int = 1


class Mission(BaseModel):
    """
    A single mission spawned from a campaign.

    What agents actually execute.
    
    MongoDB fields use camelCase (missionId, campaignId, invariantId, agentType, etc.)
    """

    mission_id: str = Field(serialization_alias="missionId")
    campaign_id: str = Field(serialization_alias="campaignId")
    # Target invariant (None for exploratory/game modes)
    invariant_id: Optional[str] = Field(default=None, serialization_alias="invariantId")
    invariant: Optional[Invariant] = None  # Excluded from MongoDB storage
    # Agent assignment
    agent_type: MissionAgentType = Field(serialization_alias="agentType")
    # Inherited from campaign - excluded from MongoDB storage
    scope: CampaignScope = Field(default_factory=CampaignScope, exclude=True)
    workspace_preset: WorkspacePreset = Field(default=WorkspacePreset.CLEAN, exclude=True, serialization_alias="workspacePreset")
    objectives: CampaignObjectives = Field(default_factory=CampaignObjectives, exclude=True)
    # Budget for this mission
    max_turns: int = Field(default=20, serialization_alias="maxTurns")
    # State
    status: str = "pending"  # pending, in_progress, completed, failed


class DispatcherProcessInput(BaseModel):
    """Input for DispatcherProcess."""

    master_context: "MasterContext"
    dependency_graph: Any  # DependencyGraph object
    actor_matrix: "ActorMatrix"
    invariants: List[Invariant]
    # Config
    max_invariants_per_cluster: int = 5
    max_campaigns: int = 10
    default_budget: CampaignBudget = Field(default_factory=CampaignBudget)
    include_exploration: bool = True  # Add exploratory/game campaigns


class DispatcherProcessOutput(BaseModel):
    """Output of DispatcherProcess."""

    clusters: List[InvariantCluster] = Field(default_factory=list)
    campaigns: List[CampaignBrief] = Field(default_factory=list)
    missions: List[Mission] = Field(default_factory=list)
    success: bool
    error_message: Optional[str] = None
    stats: Dict[str, int] = Field(default_factory=dict)


class EntrypointsPolicy(BaseModel):
    """Controls how entrypoints are sequenced/selected for a campaign."""

    max_sequence_len: int = Field(default=4, ge=1, le=50)
    prefer: List[str] = Field(default_factory=list)


class EntrypointsSubset(BaseModel):
    """Subset of allowed entrypoints for a campaign, with optional sequencing policy."""

    ids: List[str] = Field(default_factory=list)
    policy: Optional[EntrypointsPolicy] = None


class BlackboxBrief(BaseModel):
    """
    Full briefing for the Blackbox worker (v2).

    Note: `master_context` is optional here and may be provided as `mastercontext`.
    """

    # --- Core campaign identity ---
    campaign_id: str
    kind: str

    # --- Targeting ---
    invariant_ids: List[str] = Field(default_factory=list)
    primary_var_ids: List[str] = Field(default_factory=list)
    actor_roles: List[str] = Field(default_factory=list)

    entrypoints_subset: EntrypointsSubset = Field(default_factory=EntrypointsSubset)

    # --- Execution controls ---
    worker_types: List[str] = Field(default_factory=list)
    workspace_preset: str = "writeable"
    priority: int = 1

    class Budget(BaseModel):
        max_missions: int = Field(default=1, ge=1, le=1000)
        max_workers: int = Field(default=1, ge=1, le=1000)
        max_turns_per_worker: int = Field(default=32, ge=1, le=1000)

    budget: Budget = Field(default_factory=Budget)

    # --- Optional attached context (backwards compatible / transitional) ---
    master_context: Optional[MasterContext] = None
    protocol_manifesto: Optional[ProtocolManifesto] = None
    actor_matrix: Optional[ActorMatrix] = None


class BlackboxInput(BaseModel):
    campaign_brief: CampaignBrief
    num_turns: int
    model_name: str
    use_openai: bool = False
    execution_id: Optional[str] = None


class BlackboxOutput(BaseModel):
    response: Optional[AgentResponse]
    observations: List[Observation] = Field(default_factory=list)
    estimated_cost: float
    total_tokens: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    repo_path: str


class InvariantSynthesizerInput(BaseModel):
    observations: List[Observation]
    master_context: MasterContext
    dependency_graph: Any
    protocol_manifesto: Optional[ProtocolManifesto] = None
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = False
    max_turns_per_observation: int = 8


class InvariantSynthesizerOutput(BaseModel):
    invariants: List[Invariant] = Field(default_factory=list)
    success: bool
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    total_tokens: Dict[str, int] = Field(default_factory=dict)
    stats: Dict[str, int] = Field(
        default_factory=lambda: {
            "seen": 0,
            "converted": 0,
            "no_invariant": 0,
            "unresolved_targets": 0,
            "llm_failed": 0,
        }
    )


class VerifierProcessInput(BaseModel):
    """Input for VerifierProcess."""

    exploit_candidate: "ExploitCandidate"
    invariant: "Invariant"
    master_context: "MasterContext"
    dependency_graph: Any = None  # DependencyGraph object
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = False
    max_turns: int = 16


class VerifierProcessOutput(BaseModel):
    """Output of VerifierProcess."""

    verdict: Optional["Verdict"] = None
    success: bool
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    total_tokens: Dict[str, int] = Field(default_factory=dict)


# ---------------------------
# Fixer Schemas
# ---------------------------


class FixerInput(BaseModel):
    """
    Input shape for a Fixer workflow.

    Note: Fixer is not currently wired into the Kai v2 dispatcher, but this schema
    is provided for future compatibility and for tool outputs that reference these types.
    """

    exploit_candidate: ExploitCandidate
    verdict: Verdict
    master_context: Optional[MasterContext] = None
    model_name: str = settings.MAIN_DEFAULT_MODEL
    use_openai: bool = False
