"""
Agent factories for Dispatcher.

Factory functions create properly configured agent instances for missions.
Each factory handles agent-specific setup (prompts, workspace paths, etc.).
"""

from typing import Optional, Dict, Any

from kai.agents import settings
from kai.schemas import (
    ActorMatrix,
    Invariant,
    MasterContext,
    Mission,
    MissionAgentType,
)
from kai.utils.dependency.graph import DependencyGraph


def filter_actor_context(
    actor_matrix: Optional[ActorMatrix],
    invariant: Optional[Invariant],
) -> str:
    """
    Filter actor matrix to roles relevant to the invariant's targets.

    Returns a formatted string for embedding in agent prompts.

    Args:
        actor_matrix: The full ActorMatrix from preprocessing
        invariant: The target invariant (may be None for exploration missions)

    Returns:
        Formatted string describing relevant roles and privileges
    """
    if not actor_matrix:
        return "No actor matrix available."

    if not invariant:
        # For exploration missions, return all roles summary
        lines = ["All protocol roles:\n"]
        for role in actor_matrix.roles:
            lines.append(f"**{role.name}** (trust: {role.trust})")
            if role.access_signature:
                lines.append(f"  - Access via: {', '.join(role.access_signature)}")
            lines.append(f"  - Privileges: {len(role.privileges)} functions")
            lines.append("")
        return "\n".join(lines)

    # Get target function IDs and variable IDs from invariant
    target_func_ids = set(invariant.target_function_ids or [])
    target_var_ids = set(invariant.target_var_ids or [])

    # Also match by function name (in case IDs don't match exactly)
    target_func_names = set()
    for fid in target_func_ids:
        # Extract function name from ID like "Contract.funcName(args)"
        if "." in fid:
            name_part = fid.split(".")[-1].split("(")[
                0
            ]  # TODO: check if works on other languages
            target_func_names.add(name_part)

    relevant_roles = []
    for role in actor_matrix.roles:
        role_name = role.name
        trust = role.trust
        privileges = role.privileges
        access_sig = role.access_signature

        # Check if any privilege touches our targets
        relevant_privs = []
        for priv in privileges:
            priv_id = priv.id
            priv_name = priv.name
            write_target_ids = set(priv.write_target_ids or [])

            # Match by function ID, function name, or written variable IDs
            if (
                priv_id in target_func_ids
                or priv_name in target_func_names
                or write_target_ids & target_var_ids
            ):
                relevant_privs.append(priv)

        if relevant_privs:
            relevant_roles.append(
                {
                    "name": role_name,
                    "trust": trust,
                    "access_signature": access_sig,
                    "privileges": relevant_privs,
                }
            )

    if not relevant_roles:
        return "No roles directly touch the target functions/variables."

    # Format output
    lines = ["Roles relevant to this invariant:\n"]
    for role in relevant_roles:
        lines.append(f"**{role['name']}** (trust: {role['trust']})")
        if role["access_signature"]:
            lines.append(f"  - Access via: {', '.join(role['access_signature'])}")
        lines.append("  - Can call:")
        for priv in role["privileges"]:
            sig = priv.signature or priv.name
            container = priv.container or ""
            writes = priv.write_targets or []
            write_str = f" -> writes: {', '.join(writes)}" if writes else ""
            lines.append(f"    - {container}.{sig}{write_str}")
        lines.append("")

    return "\n".join(lines)


def create_state_agent(
    mission: Mission,
    workspace_path: str,
    master_context: MasterContext,
    dependency_graph: Optional[DependencyGraph] = None,
    actor_matrix: Optional[ActorMatrix] = None,
    model: str = settings.MAIN_DEFAULT_MODEL,
    use_openai: bool = False,
    execution_id: Optional[str] = None,
):
    """
    Factory function to create a properly configured StateAgent.
    Args:
        mission: The mission to execute
        workspace_path: Path to the provisioned workspace
        master_context: MasterContext from preprocessing
        dependency_graph: Optional DependencyGraph for code analysis tools
        actor_matrix: Optional ActorMatrix for actor context filtering
        model: Model to use for inference
        use_openai: Whether to use OpenAI API directly
        execution_id: Optional execution ID for logging

    Returns:
        Configured StateAgent ready for chat_with_tools()
    """
    from kai.agents.agent_types.state_agent import StateAgent

    # Create agent
    agent = StateAgent(
        mission=mission,
        master_context=master_context,
        dependency_graph=dependency_graph,
        max_tool_turns=mission.max_turns,
        repo_path=workspace_path,
        model=model,
        use_openai=use_openai,
        execution_id=execution_id,
    )

    # Set workspace path for tools
    agent.workspace_path = workspace_path

    # Set up toolcalling prompt with invariant context
    if mission.invariant:
        actor_context = filter_actor_context(actor_matrix, mission.invariant)
        agent.set_toolcalling_prompt(
            invariant=mission.invariant,
            actor_context=actor_context,
        )

    return agent


def create_quant_agent(
    mission: Mission,
    workspace_path: str,
    master_context: MasterContext,
    dependency_graph: Optional[DependencyGraph] = None,
    actor_matrix: Optional[ActorMatrix] = None,
    model: str = settings.MAIN_DEFAULT_MODEL,
    use_openai: bool = False,
    execution_id: Optional[str] = None,
):
    """
    Factory function to create a properly configured QuantAgent.
    Args:
        mission: The mission to execute
        workspace_path: Path to the provisioned workspace
        master_context: MasterContext from preprocessing
        dependency_graph: Optional DependencyGraph for code analysis tools
        actor_matrix: Optional ActorMatrix for actor context filtering
        model: Model to use for inference
        use_openai: Whether to use OpenAI API directly
        execution_id: Optional execution ID for logging

    Returns:
        Configured QuantAgent ready for chat_with_tools()
    """
    from kai.agents.agent_types.quant_agent import QuantAgent

    # Create agent
    agent = QuantAgent(
        mission=mission,
        master_context=master_context,
        dependency_graph=dependency_graph,
        max_tool_turns=mission.max_turns,
        repo_path=workspace_path,
        model=model,
        use_openai=use_openai,
        execution_id=execution_id,
    )

    # Set workspace path for tools
    agent.workspace_path = workspace_path

    # Set up toolcalling prompt with invariant context
    if mission.invariant:
        actor_context = filter_actor_context(actor_matrix, mission.invariant)
        agent.set_toolcalling_prompt(
            invariant=mission.invariant,
            actor_context=actor_context,
        )

    return agent


def create_blackbox_agent(
    mission: Mission,
    workspace_path: str,
    master_context: MasterContext,
    dependency_graph: Optional[DependencyGraph] = None,
    actor_matrix: Optional[ActorMatrix] = None,
    model: str = settings.MAIN_DEFAULT_MODEL,
    use_openai: bool = False,
    execution_id: Optional[str] = None,
):
    """
    Factory function to create a properly configured BlackboxAgent.

    Note: BlackboxAgent uses a different setup flow via BlackboxProcess.
    This factory provides a simpler interface for Dispatcher integration.
    """
    from kai.agents.agent_types.blackbox_agent import BlackboxAgent
    from kai.schemas import CampaignBrief, CampaignScope
    from kai.utils.tool_adapters import get_supported_frameworks

    # Create a CampaignBrief from mission context
    framework = None
    if master_context and getattr(master_context, "frameworks", None):
        supported = set(get_supported_frameworks())
        for fw in master_context.frameworks or []:
            fw_lower = str(fw).lower()
            if fw_lower in supported:
                framework = fw_lower
                break

    campaign_brief = CampaignBrief(
        campaign_id=mission.campaign_id,
        agent_types=[mission.agent_type],
        framework=framework,
        scope=CampaignScope(),
        invariants=[mission.invariant] if mission.invariant else [],
        master_context=master_context,
    )

    agent = BlackboxAgent(
        campaign_brief=campaign_brief,
        dependency_graph=dependency_graph,
        repo_path=workspace_path,
        max_tool_turns=mission.max_turns,
        model=model,
        use_openai=use_openai,
        execution_id=execution_id,
    )

    return agent


# Registry of agent factories by type
AGENT_FACTORIES: Dict[MissionAgentType, Any] = {
    MissionAgentType.STATE: create_state_agent,
    MissionAgentType.QUANT: create_quant_agent,
    MissionAgentType.BLACKBOX: create_blackbox_agent,
    # MissionAgentType.GAMIFIED: create_gamified_agent,  # TODO
}


def get_agent_factory(agent_type: MissionAgentType):
    """
    Get the factory function for an agent type.

    Args:
        agent_type: The MissionAgentType

    Returns:
        Factory function or None if not supported
    """
    return AGENT_FACTORIES.get(agent_type)
