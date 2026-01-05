"""
VerifierProcess - Validates exploit findings from State/Quant agents.

Uses VerifierAgent to run the PoC test, analyze results, check for mock contracts,
evaluate economic feasibility, and produce a Verdict.
"""

import re
from pathlib import Path
from typing import Optional

from kai.agents.agent_types.verifier_agent import VerifierAgent
from kai.dispatcher.workspace import WorkspaceManager
from kai.processes.base import BaseProcess
from kai.schemas import (
    Verdict,
    VerifierProcessInput,
    VerifierProcessOutput,
)


class VerifierProcess(BaseProcess[VerifierProcessInput, VerifierProcessOutput]):
    """
    Process to run the VerifierAgent and validate exploit candidates.

    Takes an ExploitCandidate from State/Quant agents and produces
    a Verdict determining validity, severity, and reasoning.

    The verifier:
    1. Provisions a workspace
    2. Writes and compiles the PoC
    3. Runs the test
    4. Analyzes results and submits verdict
    """

    async def execute(self, input_data: VerifierProcessInput) -> VerifierProcessOutput:
        ctx = input_data.master_context
        repo_path = ctx.root_path
        candidate = input_data.exploit_candidate
        invariant = input_data.invariant

        self.logger.info(f"Starting verification for candidate: {candidate.mission_id}")

        # Provision workspace for verifier to run tests
        workspace_manager = WorkspaceManager(
            workspace_dir=str(self._project_root() / "kai_workspaces"),
            logger=self.logger,
        )
        workspace_id = f"verify_{candidate.mission_id}"
        workspace_path = workspace_manager.provision(
            workspace_id=workspace_id,
            master_path=repo_path,
            master_context=ctx,
        )

        # Create VerifierAgent
        agent = VerifierAgent(
            exploit_candidate=candidate,
            invariant=invariant,
            master_context=ctx,
            dependency_graph=input_data.dependency_graph,
            max_tool_turns=input_data.max_turns,
            repo_path=repo_path,
            model=input_data.model_name,
            use_openai=input_data.use_openai,
            execution_id=workspace_id,
        )

        # Set workspace path so tools can write/run tests
        agent.workspace_path = workspace_path

        # Set up the toolcalling prompt
        agent.set_toolcalling_prompt()

        prefix = "verifier"
        exception_msg = ""
        verdict: Optional[Verdict] = None

        try:
            # Run verification
            await agent.chat_with_tools("Begin verification.")

            # If no verdict was submitted, nudge the agent
            verdict = agent.get_verdict()
            if verdict is None:
                prefix = "verifier_retry"
                retry_prompt = (
                    "FORMAT REQUIREMENT: You must call submit_verdict({...}) "
                    "with your analysis. Call it now to finish."
                )
                await agent.chat_with_tools(retry_prompt)
                verdict = agent.get_verdict()

        except Exception as e:
            exception_msg = str(e)
            self.logger.error(f"Verification failed: {e}")
        finally:
            try:
                await agent.close()
            except Exception as e:
                raise e
            # Cleanup workspace
            workspace_manager.cleanup(workspace_id)

        # Conversation saving handled by dispatcher via state_manager

        # Determine success
        success = verdict is not None
        error_message = None
        if not success:
            error_message = exception_msg or "Verifier agent did not submit a verdict"

        # Log result
        if verdict:
            status = "VALID" if verdict.is_valid else "REJECTED"
            self.logger.info(
                f"Verification complete: {status} - {verdict.severity.value.upper()}"
            )
            if verdict.rejection_reason:
                self.logger.info(f"Rejection reason: {verdict.rejection_reason}")
        else:
            self.logger.warning(f"Verification incomplete for {candidate.mission_id}")

        return VerifierProcessOutput(
            verdict=verdict,
            success=success,
            error_message=error_message,
            estimated_cost=agent.estimated_cost,
            total_tokens=agent.total_tokens,
            agent_messages=agent.messages,
            agent_model=agent.model,
        )

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def _repo_slug(self, repo_path: str) -> str:
        name = Path(repo_path).name or "repo"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
        return safe_name
