"""
Dispatcher mission planning.
"""

from __future__ import annotations

import secrets
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from kai.schemas import (
    ActorMatrix,
    CampaignBrief,
    CampaignMode,
    CampaignObjectives,
    CampaignScope,
    EntrypointPolicy,
    EntrypointSubset,
    Invariant,
    InvariantCluster,
    InvariantType,
    MasterContext,
    Mission,
    MissionAgentType,
    RewardModel,
    WorkspacePreset,
)
from kai.utils.dependency.graph import DependencyGraph
from kai.utils.dependency.models import EdgeKind, NodeKind


def generate_mission_id() -> str:
    """Generate a 24-character hex ID (MongoDB ObjectId compatible)."""
    return secrets.token_hex(12)


# Agent type mapping based on invariant semantics
# NOTE: Only invariant-dependent agents (STATE, QUANT) are mapped here.
# BLACKBOX and GAMIFIED are non-invariant-dependent - they explore freely
# and generate Observations that may become new invariants.
INVARIANT_TO_AGENTS: Dict[InvariantType, List[MissionAgentType]] = {
    InvariantType.ACCESS: [MissionAgentType.STATE],
    InvariantType.SOLVENCY: [MissionAgentType.QUANT, MissionAgentType.STATE],
    InvariantType.CONSERVATION: [MissionAgentType.QUANT, MissionAgentType.STATE],
    InvariantType.LIVENESS: [MissionAgentType.STATE],
    InvariantType.FEE_BOUND: [MissionAgentType.QUANT],
    InvariantType.REENTRANCY: [MissionAgentType.STATE],
    InvariantType.ORDERING: [MissionAgentType.STATE],
    InvariantType.OTHER: [MissionAgentType.STATE, MissionAgentType.QUANT],
}


class MissionPlanner:
    def __init__(
        self,
        *,
        dependency_graph: DependencyGraph,
        actor_matrix: ActorMatrix,
        max_invariants_per_cluster: int,
        max_campaigns: int,
        include_exploration: bool,
        default_budget,
        master_context: MasterContext,
    ) -> None:
        self.graph = dependency_graph
        self.actor_matrix = actor_matrix
        self.max_invariants_per_cluster = max_invariants_per_cluster
        self.max_campaigns = max_campaigns
        self.include_exploration = include_exploration
        self.default_budget = default_budget
        self.master_context = master_context

    def plan(
        self,
        *,
        invariants: List[Invariant],
        base_mission_index: int = 0,
    ) -> Tuple[List[CampaignBrief], List[Mission]]:
        """
        Build STATE/QUANT campaigns and missions from invariants.

        Returns (campaigns, missions).
        """
        clusters = self._cluster_invariants(invariants)
        campaigns = self._build_campaigns(clusters, invariants)

        missions: List[Mission] = []
        for campaign in campaigns:
            missions.extend(
                self._spawn_missions_from_campaign(
                    campaign, base_id=base_mission_index + len(missions)
                )
            )

        return campaigns, missions

    def _cluster_invariants(
        self, invariants: List[Invariant]
    ) -> List[InvariantCluster]:
        """Cluster invariants by overlapping vars OR functions in same container."""
        inv_by_id = {inv.id: inv for inv in invariants}

        def get_container(node_id: str) -> Optional[str]:
            if node_id in self.graph._nodes:
                node = self.graph._nodes[node_id]
                if node.parent_id:
                    parent = self.graph._nodes.get(node.parent_id)
                    if parent and parent.kind == NodeKind.CONTAINER:
                        return parent.name
            return None

        inv_context: Dict[str, Set[str]] = {}
        for inv in invariants:
            context = set(inv.target_var_ids) | set(inv.target_function_ids)
            for vid in inv.target_var_ids:
                container = get_container(vid)
                if container:
                    context.add(f"container:{container}")
            for fid in inv.target_function_ids:
                container = get_container(fid)
                if container:
                    context.add(f"container:{container}")
            inv_context[inv.id] = context

        # Union-find
        parent: Dict[str, str] = {inv.id: inv.id for inv in invariants}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        inv_ids = list(inv_context.keys())
        for i in range(len(inv_ids)):
            for j in range(i + 1, len(inv_ids)):
                if inv_context[inv_ids[i]] & inv_context[inv_ids[j]]:
                    union(inv_ids[i], inv_ids[j])

        cluster_groups: Dict[str, List[str]] = defaultdict(list)
        for inv_id in inv_ids:
            cluster_groups[find(inv_id)].append(inv_id)

        clusters: List[InvariantCluster] = []
        max_per = self.max_invariants_per_cluster
        for _, group_inv_ids in cluster_groups.items():
            for chunk_start in range(0, len(group_inv_ids), max_per):
                chunk = group_inv_ids[chunk_start : chunk_start + max_per]
                clusters.append(
                    self._build_cluster(
                        f"CLS_{len(clusters)}",
                        chunk,
                        inv_by_id,
                    )
                )
        return clusters

    def _build_cluster(
        self,
        cluster_id: str,
        inv_ids: List[str],
        inv_by_id: Dict[str, Invariant],
    ) -> InvariantCluster:
        all_vars: Set[str] = set()
        all_funcs: Set[str] = set()
        type_counts: Counter = Counter()
        containers: Set[str] = set()

        for inv_id in inv_ids:
            inv = inv_by_id[inv_id]
            all_vars.update(inv.target_var_ids)
            all_funcs.update(inv.target_function_ids)
            type_counts[inv.type] += 1

            for fid in inv.target_function_ids:
                if fid in self.graph._nodes:
                    node = self.graph._nodes[fid]
                    if node.parent_id and node.parent_id in self.graph._nodes:
                        parent = self.graph._nodes[node.parent_id]
                        if parent.kind == NodeKind.CONTAINER:
                            containers.add(parent.name)

        dominant_type = type_counts.most_common(1)[0][0] if type_counts else None

        return InvariantCluster(
            cluster_id=cluster_id,
            invariant_ids=inv_ids,
            primary_var_ids=sorted(all_vars),
            primary_function_ids=sorted(all_funcs),
            primary_container=next(iter(containers)) if len(containers) == 1 else None,
            dominant_type=dominant_type,
        )

    def _build_campaigns(
        self, clusters: List[InvariantCluster], invariants: List[Invariant]
    ) -> List[CampaignBrief]:
        inv_by_id = {inv.id: inv for inv in invariants}
        campaigns: List[CampaignBrief] = []

        sorted_clusters = sorted(clusters, key=lambda c: -len(c.invariant_ids))
        for cluster in sorted_clusters[: self.max_campaigns]:
            campaigns.append(self._build_campaign_from_cluster(cluster, inv_by_id))
        return campaigns

    def _build_campaign_from_cluster(
        self, cluster: InvariantCluster, inv_by_id: Dict[str, Invariant]
    ) -> CampaignBrief:
        cluster_invariants = [
            inv_by_id[inv_id] for inv_id in cluster.invariant_ids if inv_id in inv_by_id
        ]

        actor_roles = self._derive_actor_roles(cluster)
        entrypoints = self._derive_entrypoints(cluster)
        file_ids = self._derive_file_ids(cluster)
        agent_types = self._select_agent_types(cluster_invariants)
        workspace_preset = self._select_workspace_preset(agent_types)
        notes = "; ".join(inv.rule for inv in cluster_invariants[:3])

        scope = CampaignScope(
            entrypoints_subset=EntrypointSubset(
                ids=entrypoints,
                policy=EntrypointPolicy(max_sequence_len=6, prefer=["writes_state"]),
            ),
            primary_var_ids=cluster.primary_var_ids,
            file_ids=file_ids,
            actor_roles=actor_roles,
        )

        framework = None
        if self.master_context and self.master_context.frameworks:
            framework = self.master_context.frameworks[0]

        return CampaignBrief(
            label=f"CMP_{cluster.cluster_id}",
            mode=CampaignMode.INVARIANT_BOUNDED,
            agent_types=agent_types,
            framework=framework,
            workspace_preset=workspace_preset,
            scope=scope,
            invariants=cluster_invariants,
            objectives=CampaignObjectives(reward_model=RewardModel.NONE, notes=notes),
            budget=self.default_budget,
            master_context=self.master_context,
            priority=1,
        )

    def _derive_actor_roles(self, cluster: InvariantCluster) -> List[str]:
        roles: Set[str] = set()
        target_funcs = set(cluster.primary_function_ids)
        target_vars = set(cluster.primary_var_ids)

        for role in self.actor_matrix.roles:
            for priv in role.privileges:
                if priv.id in target_funcs:
                    roles.add(role.name)
                    break
                if target_vars and set(priv.write_target_ids) & target_vars:
                    roles.add(role.name)
                    break
        return sorted(roles)

    def _derive_entrypoints(
        self, cluster: InvariantCluster, max_depth: int = 3
    ) -> List[str]:
        targets: Set[str] = set(cluster.primary_function_ids) | set(
            cluster.primary_var_ids
        )
        if not targets:
            return []

        reachable = self.graph.bfs(
            start=targets,
            max_hops=max_depth,
            edge_kinds={EdgeKind.CALLS, EdgeKind.WRITES, EdgeKind.READS},
            direction="in",
        )
        public_eps = set(self.graph.public_entrypoints())
        return sorted(reachable & public_eps)

    def _derive_file_ids(self, cluster: InvariantCluster) -> List[str]:
        files: Set[str] = set()
        for func_id in cluster.primary_function_ids:
            if func_id in self.graph._nodes:
                node = self.graph._nodes[func_id]
                if node.span and node.span.file:
                    files.add(f"file:{self.graph.norm_path(node.span.file)}")
        return sorted(files)

    def _select_agent_types(
        self, invariants: List[Invariant]
    ) -> List[MissionAgentType]:
        agents: Set[MissionAgentType] = set()
        for inv in invariants:
            agents.update(INVARIANT_TO_AGENTS.get(inv.type, [MissionAgentType.STATE]))
        return sorted(agents, key=lambda a: a.value)

    def _select_workspace_preset(
        self, agent_types: List[MissionAgentType]
    ) -> WorkspacePreset:
        if (
            MissionAgentType.GAMIFIED in agent_types
            or MissionAgentType.BLACKBOX in agent_types
        ):
            return WorkspacePreset.SANDBOX
        if MissionAgentType.STATE in agent_types:
            return WorkspacePreset.WRITEABLE
        return WorkspacePreset.CLEAN

    def build_blackbox_campaign(self) -> Tuple[CampaignBrief, List[Mission]]:
        """
        Build blackbox campaign and missions.

        Called as a separate phase before state/quant to discover new invariants.
        """
        all_eps = self.graph.public_entrypoints()
        all_roles = [role.name for role in self.actor_matrix.roles]

        framework = None
        if self.master_context and self.master_context.frameworks:
            framework = self.master_context.frameworks[0]

        campaign = CampaignBrief(
            label="CMP_BLACKBOX_GLOBAL",
            mode=CampaignMode.EXPLORATORY,
            agent_types=[MissionAgentType.BLACKBOX],
            framework=framework,
            workspace_preset=WorkspacePreset.SANDBOX,
            scope=CampaignScope(
                entrypoints_subset=EntrypointSubset(
                    ids=all_eps, policy=EntrypointPolicy()
                ),
                actor_roles=all_roles,
            ),
            invariants=[],
            objectives=CampaignObjectives(
                reward_model=RewardModel.COVERAGE, notes="Anomaly detection"
            ),
            budget=self.default_budget,
            master_context=self.master_context,
            priority=0,  # Runs first
        )

        missions = self._spawn_missions_from_campaign(campaign, base_id=0)
        return campaign, missions

    def _spawn_missions_from_campaign(
        self, campaign: CampaignBrief, *, base_id: int
    ) -> List[Mission]:
        missions: List[Mission] = []

        if campaign.mode == CampaignMode.INVARIANT_BOUNDED:
            for inv in campaign.invariants:
                for agent_type in campaign.agent_types:
                    missions.append(
                        Mission(
                            mission_id=generate_mission_id(),
                            campaign_id=campaign.id,
                            invariant_id=inv.id,
                            invariant=inv,
                            agent_type=agent_type,
                            scope=campaign.scope,
                            workspace_preset=campaign.workspace_preset,
                            objectives=campaign.objectives,
                            max_turns=campaign.budget.max_turns_per_agent,
                            status="pending",
                        )
                    )
        else:
            # Non-invariant-bounded modes (EXPLORATORY, GAME)
            for agent_type in campaign.agent_types:
                # For GAMIFIED agents, set invariant_cluster to the full cluster
                invariant_cluster = None
                if agent_type == MissionAgentType.GAMIFIED and campaign.invariants:
                    invariant_cluster = campaign.invariants

                missions.append(
                    Mission(
                        mission_id=generate_mission_id(),
                        campaign_id=campaign.id,
                        invariant_id=None,
                        invariant=None,
                        invariant_cluster=invariant_cluster,
                        agent_type=agent_type,
                        scope=campaign.scope,
                        workspace_preset=campaign.workspace_preset,
                        objectives=campaign.objectives,
                        max_turns=campaign.budget.max_turns_per_agent,
                        status="pending",
                    )
                )

        return missions

    def create_missions_for_invariant(
        self, invariant: Invariant, base_id: int
    ) -> List[Mission]:
        """Create missions for a single dynamically discovered invariant."""
        agent_types = self._select_agent_types([invariant])
        missions: List[Mission] = []

        for agent_type in agent_types:
            missions.append(
                Mission(
                    mission_id=generate_mission_id(),
                    campaign_id="CMP_DYNAMIC",
                    invariant_id=invariant.id,
                    invariant=invariant,
                    agent_type=agent_type,
                    scope=CampaignScope(),
                    workspace_preset=WorkspacePreset.WRITEABLE,
                    objectives=CampaignObjectives(notes=f"Dynamic: {invariant.rule}"),
                    max_turns=self.default_budget.max_turns_per_agent,
                    status="pending",
                )
            )

        return missions

    def build_gamified_campaigns(
        self, invariants: List[Invariant]
    ) -> Tuple[List[CampaignBrief], List[Mission]]:
        """
        Build gamified campaigns and missions from invariant clusters.

        Called during phase 2 of two-phase execution, after state/quant missions complete.
        Creates one gamified campaign per cluster, using the same clustering logic.

        Args:
            invariants: Full list of invariants to cluster

        Returns:
            Tuple of (campaigns, missions) for gamified agents
        """
        # Re-cluster invariants (or use cached clusters if available)
        clusters = self._cluster_invariants(invariants)
        inv_by_id = {inv.id: inv for inv in invariants}

        framework = None
        if self.master_context and self.master_context.frameworks:
            framework = self.master_context.frameworks[0]

        campaigns: List[CampaignBrief] = []
        missions: List[Mission] = []

        for cluster in clusters:
            # Get invariants for this cluster
            cluster_invariants = [
                inv_by_id[inv_id]
                for inv_id in cluster.invariant_ids
                if inv_id in inv_by_id
            ]

            if not cluster_invariants:
                continue

            # Derive scope from cluster
            entrypoints = self._derive_entrypoints(cluster)
            actor_roles = self._derive_actor_roles(cluster)

            campaign = CampaignBrief(
                label=f"CMP_GAMIFIED_{cluster.cluster_id}",
                mode=CampaignMode.GAME,
                agent_types=[MissionAgentType.GAMIFIED],
                framework=framework,
                workspace_preset=WorkspacePreset.SANDBOX,
                scope=CampaignScope(
                    entrypoints_subset=EntrypointSubset(
                        ids=entrypoints, policy=EntrypointPolicy()
                    ),
                    primary_var_ids=cluster.primary_var_ids,
                    actor_roles=actor_roles,
                ),
                invariants=cluster_invariants,
                objectives=CampaignObjectives(
                    reward_model=RewardModel.PROFIT,
                    notes=f"Gap exploitation for {cluster.cluster_id}",
                ),
                budget=self.default_budget,
                master_context=self.master_context,
                priority=2,
            )
            campaigns.append(campaign)

            # Spawn mission using existing logic
            campaign_missions = self._spawn_missions_from_campaign(
                campaign, base_id=len(missions)
            )
            missions.extend(campaign_missions)

        return campaigns, missions
