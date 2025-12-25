"""
Abstract state manager for Kai persistence operations.

Implementations can handle local file storage, MongoDB/S3, or other backends.
Passed to Dispatcher to decouple business logic from persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal

from kai.schemas import (
    ActorMatrix,
    ExploitCandidate,
    Fix,
    Invariant,
    MasterContext,
    Mission,
    Observation,
    ProtocolManifesto,
    Verdict,
    CampaignBrief,
)


class KaiStateManager(ABC):
    """
    Abstract state manager for Kai persistence operations.

    Implementations handle where/how data is stored (local files, MongoDB, S3, etc.).
    The Dispatcher and agents call these methods at appropriate points without
    needing to know the storage details.

    Example implementations:
        - LocalStateManager: Saves to JSON files in output/
        - MongoStateManager: Saves to MongoDB collections + S3 for large blobs
    """

    def __init__(self, execution_id: str):
        """
        Args:
            execution_id: The execution identifier this manager is bound to
        """
        self.execution_id = execution_id

    @abstractmethod
    async def update_state(
        self, state: Literal["setup", "profiler", "invariant"]
    ) -> None:
        pass

    @abstractmethod
    async def save_campaigns(self, campaigns: List[CampaignBrief]) -> bool:
        """
        Save campaign briefs.

        Args:
            campaigns: List of campaign briefs to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_dependency_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Save the dependency graph.

        Args:
            graph_data: Serialized dependency graph data

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_actor_matrix(self, actor_matrix: ActorMatrix) -> bool:
        """
        Save the actor matrix.

        Args:
            actor_matrix: The actor matrix to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_protocol_manifesto(self, manifesto: ProtocolManifesto) -> bool:
        """
        Save the protocol manifesto.

        Args:
            manifesto: The protocol manifesto to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_master_context(self, context: MasterContext) -> bool:
        """
        Save the master context.

        Args:
            context: The master context to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_invariants(self, invariants: List[Invariant]) -> bool:
        """
        Save discovered invariants.

        Args:
            invariants: List of invariants to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_missions(self, missions: List[Mission]) -> bool:
        """
        Save missions.

        Args:
            missions: List of missions to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def update_mission_status(
        self,
        mission_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update a mission's status.

        Args:
            mission_id: The mission identifier
            status: New status (pending, in_progress, completed, failed)
            error: Optional error message if failed

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_exploit_candidate(self, candidate: ExploitCandidate) -> bool:
        """
        Save an exploit candidate.

        Args:
            candidate: The exploit candidate to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_verdict(self, verdict: Verdict) -> bool:
        """
        Save a verification verdict.

        Args:
            verdict: The verdict to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_fix(self, fix: Fix) -> bool:
        """
        Save a code fix for a verified exploit.

        Args:
            fix: The Fix object to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_observations(self, observations: List[Observation]) -> bool:
        """
        Save observations from blackbox exploration.

        Args:
            observations: List of observations to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def save_conversation(
        self,
        agent_id: str,
        agent_type: str,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Save an agent's conversation history.

        Args:
            agent_id: The agent's unique identifier
            agent_type: Type of agent (state, quant, blackbox, verifier, etc.)
            messages: List of conversation messages
            metadata: Additional metadata (tokens, cost, time, etc.)

        Returns:
            Path or URI where conversation was saved, or None if failed
        """
        pass
