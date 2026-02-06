from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, TYPE_CHECKING

from ..models import Node

if TYPE_CHECKING:
    from ..graph import DependencyGraph


@dataclass
class LensDefinition:
    """
    Domain-agnostic lens definition for invariant generation.

    Each lens focuses on a specific security concern. The LLM uses
    the description and prompt_template to categorize functions and
    generate focused invariants.
    """

    name: str
    description: str
    invariant_types: List[str]

    # Lens-specific prompt template for LLM invariant generation
    prompt_template: str = ""

    # Mandatory checklist items - used for coverage verification
    checklist: List[str] = field(default_factory=list)


class DomainAdapter(ABC):
    """
    Defines how generic graph nodes map to language-specific security concepts.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_domain_mapping(self) -> Dict[str, str]:
        """Map generic NodeKinds to domain terms (e.g. 'Contract' --> CONTAINER)."""
        pass

    @abstractmethod
    def is_public_entrypoint(self, node: Node) -> bool:
        """
        Is this node callable by an external attacker?
        Solidity: public/external visibility.
        Rust: pub fn in generic module / instruction handler.
        """
        pass

    @abstractmethod
    def is_state_variable(self, node: Node) -> bool:
        """Does this node represent persistent state?"""
        pass

    @abstractmethod
    def is_test_file(self, file_path: str) -> bool:
        """Is this a test/mock file?"""
        pass

    @abstractmethod
    def is_library_file(self, file_path: str) -> bool:
        """
        Is this file from an external library (not protocol code)?
        E.g., OpenZeppelin, Solmate, forge-std, node_modules, lib/
        """
        pass

    @abstractmethod
    def resolve_symbol(
        self, name: str, context_graph: "DependencyGraph", scope: Optional[str] = None
    ) -> List[str]:
        """
        Resolve a user-provided string (e.g. "deposit") to Node IDs.
        Handles fuzzy matching or language-specific overloading.

        Args:
            name: Symbol name to resolve
            context_graph: The dependency graph to search
            scope: Optional container ID to limit search scope
        """
        pass

    @abstractmethod
    def is_non_auth_guard(self, modifier_name: str) -> bool:
        """
        Is this modifier a non-auth guard (e.g., reentrancy, pause)?
        These should be filtered out when clustering by access control.
        """
        pass

    @abstractmethod
    def get_trust_for_modifiers(self, modifier_names: List[str]) -> str:
        """
        Determine trust level based on modifier patterns.
        Returns: "high", "medium", "low", "none", or "review_required"
        """
        pass

    # =========================================================================
    # Lens-based invariant generation methods
    # =========================================================================

    @abstractmethod
    def get_lens_definitions(self) -> List[LensDefinition]:
        """
        Return lens definitions for this domain.

        Each lens defines a focused security concern with:
        - name: Identifier for the lens
        - description: What security concerns this lens covers
        - invariant_types: Which InvariantType values this lens generates
        - prompt_template: Lens-specific LLM prompt
        - checklist: Mandatory coverage items

        The LLM uses these definitions to:
        1. Categorize functions into lens buckets
        2. Generate focused invariants per lens

        Returns:
            List of LensDefinition for this domain
        """
        pass

    @abstractmethod
    def get_function_metadata_extractors(self) -> Dict[str, Callable]:
        """
        Return extractors for function metadata.

        Each extractor populates domain-specific metadata on functions.
        This metadata is passed to the LLM to help with bucketing and
        invariant generation.

        Returns:
            Dict mapping field_name -> callable(node, graph) -> value

        Example for Solidity:
            {
                "is_payable": lambda node, graph: node.meta.get("is_payable", False),
                "visibility": lambda node, graph: node.meta.get("visibility", ""),
            }
        """
        pass

    # =========================================================================
    # Lifecycle/Temporal and Economic detection (language-agnostic interface)
    # =========================================================================
    # These methods enable lifecycle/temporal and economic lenses by detecting
    # time-bounded state transitions and value flows in a generic way.
    # Language adapters override to provide language-specific patterns.
    # =========================================================================

    def get_time_var_patterns(self) -> List[str]:
        """
        Return case-insensitive substrings indicating time-like variables.

        These patterns are used to detect functions that read/write time-related
        state, enabling lifecycle/temporal invariant generation.

        Default patterns work for most languages. Override to add language-specific
        patterns or adjust for domain conventions.

        Returns:
            List of lowercase substrings to match against variable names.
        """
        return [
            "deadline",
            "expiry",
            "expires",
            "endtime",
            "starttime",
            "lastclaim",
            "lastupdate",
            "lastaction",
            "graceperiod",
            "duration",
            "cooldown",
            "timeout",
            "epoch",
            "roundend",
            "roundstart",
            "period",
            "windowend",
            "windowstart",
        ]

    def get_time_source_patterns(self) -> List[str]:
        """
        Return patterns for time source access in this language.

        Each language has different ways to get current time:
        - Solidity: block.timestamp, block.number
        - JavaScript: Date.now(), new Date()
        - Python: time.time(), datetime.now()
        - Rust: SystemTime::now(), Instant::now()

        Returns:
            List of lowercase patterns to detect time source usage in code.
        """
        return []  # Abstract - each adapter must implement

    def get_reset_patterns(self) -> List[str]:
        """
        Return patterns that indicate state reset operations.

        Used to detect time_reset role (functions that reinitialize round/epoch state).

        Returns:
            List of patterns to match in code snippets (case-insensitive).
        """
        return ["= false", "= 0", "reset", "= true"]  # Base defaults

    def get_accumulator_var_patterns(self) -> List[str]:
        """
        Return patterns for accumulator/treasury variables.

        Accumulators collect value over time (pot, treasury, reserve, totalStaked).
        Used to detect economic flow patterns.

        Returns:
            List of lowercase substrings to match against variable names.
        """
        return [
            "pot",
            "treasury",
            "reserve",
            "pool",
            "totalstaked",
            "totaldeposit",
            "balance",
            "collected",
            "accumulated",
        ]

    def get_obligation_var_patterns(self) -> List[str]:
        """
        Return patterns for obligation/pending payout variables.

        Obligations represent amounts owed to users (pending withdrawals, rewards).
        Used to detect economic completeness issues.

        Returns:
            List of lowercase substrings to match against variable names.
        """
        return [
            "pending",
            "withdrawable",
            "outstanding",
            "owed",
            "claimable",
            "reward",
            "payout",
            "refund",
        ]

    def classify_timerish(
        self,
        fn_name: str,
        code_snippet: str,
        read_var_names: List[str],
        write_var_names: List[str],
        is_view_or_pure: bool,
    ) -> Tuple[int, List[str]]:
        """
        Heuristic classification for timer-related functions.

        Analyzes function characteristics to determine if it participates in
        time-bounded lifecycle patterns (auctions, epochs, rounds, etc.).

        Args:
            fn_name: Function name
            code_snippet: Function source code
            read_var_names: Variables read by function
            write_var_names: Variables written by function
            is_view_or_pure: Whether function is read-only

        Returns:
            Tuple of (score, role_tags) where:
            - score: 0-10 indicating likelihood of timer involvement
            - role_tags: List of detected roles:
                - "time_view": reads timers, returns delta/countdown
                - "time_update": writes timer vars on participation
                - "time_guard_mutation": uses time guard and mutates state
                - "time_reset": resets end flags/accumulators for new round
        """
        code_l = (code_snippet or "").lower()
        reads_l = [v.lower() for v in (read_var_names or [])]
        writes_l = [v.lower() for v in (write_var_names or [])]

        time_var_pats = [p.lower() for p in self.get_time_var_patterns()]
        time_src_pats = [p.lower() for p in self.get_time_source_patterns()]
        reset_pats = [p.lower() for p in self.get_reset_patterns()]

        def _has_time_var(names: List[str]) -> bool:
            return any(any(p in n for p in time_var_pats) for n in names)

        def _has_time_src() -> bool:
            return any(p in code_l for p in time_src_pats)

        # Feature scoring (0-10 scale)
        score = 0
        if _has_time_var(reads_l):
            score += 2
        if _has_time_var(writes_l):
            score += 2
        if _has_time_src():
            score += 3

        # Detect time guards (comparisons involving time)
        comparison_ops = [">=", "<=", ">", "<"]
        has_comparison = any(op in code_l for op in comparison_ops)
        has_guard = has_comparison and (_has_time_src() or _has_time_var(reads_l))
        if has_guard:
            score += 3

        # Role detection
        roles: List[str] = []

        # time_view: read-only function that reads timer vars
        if is_view_or_pure and _has_time_var(reads_l):
            roles.append("time_view")

        # time_update: writes timer vars (participation resets timer)
        time_write_patterns = ["last", "start", "update"]
        if any(
            any(p in n for p in time_write_patterns) for n in writes_l
        ) and _has_time_var(writes_l):
            roles.append("time_update")

        # time_guard_mutation: has time guard and writes non-timer state
        if has_guard and writes_l and not all(_has_time_var([w]) for w in writes_l):
            roles.append("time_guard_mutation")

        # time_reset: resets end/finalized flags
        end_flags = ["ended", "finalized", "closed", "active", "started"]
        writes_end_flag = any(any(ef in w for ef in end_flags) for w in writes_l)
        has_reset_pattern = any(rp in code_l for rp in reset_pats)
        if writes_end_flag and has_reset_pattern:
            roles.append("time_reset")

        return score, roles

    def classify_economic(
        self,
        fn_name: str,
        code_snippet: str,
        read_var_names: List[str],
        write_var_names: List[str],
        is_payable: bool,
    ) -> Tuple[int, List[str]]:
        """
        Heuristic classification for economic/value flow functions.

        Analyzes function characteristics to determine if it participates in
        economic patterns (value transfers, fee calculations, distributions).

        Args:
            fn_name: Function name
            code_snippet: Function source code
            read_var_names: Variables read by function
            write_var_names: Variables written by function
            is_payable: Whether function accepts value (msg.value, etc.)

        Returns:
            Tuple of (score, role_tags) where:
            - score: 0-10 indicating likelihood of economic involvement
            - role_tags: List of detected roles:
                - "participation_entry": accepts value, writes holder/leader vars
                - "touches_accumulator": reads/writes pot/treasury/reserve
                - "touches_obligation": reads/writes pending/withdrawable
                - "distribution": sends value to multiple recipients
        """
        code_l = (code_snippet or "").lower()
        reads_l = [v.lower() for v in (read_var_names or [])]
        writes_l = [v.lower() for v in (write_var_names or [])]
        all_vars = reads_l + writes_l

        acc_pats = [p.lower() for p in self.get_accumulator_var_patterns()]
        obl_pats = [p.lower() for p in self.get_obligation_var_patterns()]

        def _has_accumulator(names: List[str]) -> bool:
            return any(any(p in n for p in acc_pats) for n in names)

        def _has_obligation(names: List[str]) -> bool:
            return any(any(p in n for p in obl_pats) for n in names)

        # Feature scoring
        score = 0
        if is_payable:
            score += 3
        if _has_accumulator(all_vars):
            score += 2
        if _has_obligation(all_vars):
            score += 2

        # Detect holder/leader pattern writes
        holder_patterns = [
            "holder",
            "leader",
            "current",
            "owner",
            "king",
            "winner",
            "bidder",
        ]
        writes_holder = any(any(p in w for p in holder_patterns) for w in writes_l)
        if writes_holder:
            score += 2

        # Detect value transfer patterns in code
        transfer_patterns = ["transfer", "send", "call{value", ".call("]
        has_transfer = any(tp in code_l for tp in transfer_patterns)
        if has_transfer:
            score += 1

        # Role detection
        roles: List[str] = []

        # participation_entry: payable or writes holder-like vars
        if is_payable or writes_holder:
            roles.append("participation_entry")

        # touches_accumulator: reads/writes pot/treasury
        if _has_accumulator(all_vars):
            roles.append("touches_accumulator")

        # touches_obligation: reads/writes pending/withdrawable
        if _has_obligation(all_vars):
            roles.append("touches_obligation")

        # distribution: has transfer and reads accumulator or obligation
        if has_transfer and (_has_accumulator(reads_l) or _has_obligation(reads_l)):
            roles.append("distribution")

        return score, roles
