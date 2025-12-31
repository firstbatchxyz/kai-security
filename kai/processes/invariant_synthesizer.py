import json
import hashlib
from typing import List, Optional, Any, Set

from kai.agents.agent_types.invariant_synthesizer_agent import InvariantSynthesizerAgent
from kai.processes.base import BaseProcess
from kai.schemas import (
    Invariant,
    InvariantType,
    Observation,
    InvariantSynthesizerInput,
    InvariantSynthesizerOutput,
)
from kai.utils.dependency.adapters import DomainAdapter, get_adapter
from kai.utils.dependency.analysis import FileSourceLoader, GraphQueryEngine
from kai.utils.dependency.models import EdgeKind, NodeKind


class InvariantSynthesizerProcess(
    BaseProcess[InvariantSynthesizerInput, InvariantSynthesizerOutput]
):
    """
    Process to synthesize grounded Invariants from Blackbox Observations.
    """

    async def execute(
        self, input_data: InvariantSynthesizerInput
    ) -> InvariantSynthesizerOutput:
        ctx = input_data.master_context
        graph = input_data.dependency_graph
        manifesto = input_data.protocol_manifesto

        if graph is None:
            return InvariantSynthesizerOutput(
                success=False,
                error_message="No dependency graph provided",
            )

        # Get adapter
        try:
            adapter: DomainAdapter = get_adapter(ctx.adapter)
        except ValueError as e:
            return InvariantSynthesizerOutput(
                success=False,
                error_message=str(e),
            )

        # Build query engine for grounding
        source_loader = FileSourceLoader(ctx.root_path)
        engine = GraphQueryEngine(
            graph=graph, adapter=adapter, source_loader=source_loader
        )

        final_invariants: List[Invariant] = []
        total_cost = 0.0
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        stats = {
            "seen": 0,
            "converted": 0,
            "no_invariant": 0,
            "unresolved_targets": 0,
            "llm_failed": 0,
        }

        for obs in input_data.observations:
            stats["seen"] += 1
            self.logger.info(
                f"Synthesizing invariant from observation: {obs.description[:100]}..."
            )

            # 1. Grounding Step (Required)
            target_ids = self._ground_observation(obs, engine)
            if not target_ids:
                self.logger.warning(
                    f"Observation could not resolve any target functions for '{obs.description[:50]}'"
                )
                # Do NOT drop: still run the agent on real observations.
                # Target IDs may remain empty; we will still persist the conversation + result.
                stats["unresolved_targets"] += 1

            # 2. LLM Step (Lightweight)
            agent = InvariantSynthesizerAgent(
                protocol_manifesto=manifesto,
                max_tool_turns=input_data.max_turns_per_observation,
                repo_path=ctx.root_path,
                model=input_data.model_name,
                use_openai=input_data.use_openai,
            )

            try:
                # Prepare prompt
                obs_dump = obs.model_dump()
                manifesto_summary = self._summarize_manifesto(manifesto)

                user_prompt = (
                    f"Convert the following Blackbox Observation into a tentative invariant rule.\n\n"
                    f"OBSERVATION:\n{json.dumps(obs_dump, indent=2)}\n\n"
                    f"PROTOCOL CONTEXT:\n{manifesto_summary}\n\n"
                    f"Determine if this observation reveals an invariant that should hold. "
                    f"If yes, call finalize_invariant with the draft. If no, call finalize_no_invariant."
                )

                await agent.chat_with_tools(user_prompt)

                # Check results
                invariant: Optional[Invariant] = None
                if agent._finalized_invariant_draft:
                    draft = agent._finalized_invariant_draft

                    # Create grounded invariant
                    inv_label = self._generate_label(obs, target_ids)
                    target_function_ids = sorted(list(target_ids))
                    target_var_ids = self._derive_target_var_ids(
                        target_function_ids, engine
                    )
                    target_file_ids = self._derive_target_file_ids(
                        obs, target_function_ids, graph
                    )

                    try:
                        inv_type = InvariantType(draft.get("type", "other").lower())
                    except ValueError:
                        inv_type = InvariantType.OTHER

                    invariant = Invariant(
                        label=inv_label,
                        type=inv_type,
                        rule=draft.get("rule", ""),
                        explanation=draft.get("explanation", ""),
                        target_function_ids=target_function_ids,
                        target_var_ids=target_var_ids,
                        target_file_ids=target_file_ids,
                        confidence=draft.get("confidence", 0.5),
                        source="observation_llm",
                    )
                    final_invariants.append(invariant)
                    stats["converted"] += 1
                else:
                    # "No invariant" is a valid outcome if the agent explicitly finalized it.
                    if agent._finalized_no_invariant_reason:
                        self.logger.info(
                            f"Agent finalized no invariant: {agent._finalized_no_invariant_reason}"
                        )
                        stats["no_invariant"] += 1
                    else:
                        # The agent did not comply with the required finalize tool contract.
                        self.logger.warning(
                            "Agent did not finalize an invariant or a no-invariant decision."
                        )
                        stats["llm_failed"] += 1

                # Accrue usage
                total_cost += agent.estimated_cost
                total_tokens["prompt_tokens"] += agent.total_tokens.get(
                    "prompt_tokens", 0
                )
                total_tokens["completion_tokens"] += agent.total_tokens.get(
                    "completion_tokens", 0
                )

                # Conversation/result saving handled by dispatcher via state_manager

            except Exception as e:
                self.logger.error(f"Agent synthesis failed for observation: {e}")
                stats["llm_failed"] += 1
            finally:
                await agent.close()

        return InvariantSynthesizerOutput(
            invariants=final_invariants,
            success=True,
            estimated_cost=total_cost,
            total_tokens=total_tokens,
            stats=stats,
        )

    def _ground_observation(
        self, obs: Observation, engine: GraphQueryEngine
    ) -> Set[str]:
        """
        Resolve Observation.affected_functions to DependencyGraph node IDs.
        """
        target_ids: Set[str] = set()
        g = engine.graph
        adapter = engine.adapter

        affected_files = [f for f in (obs.affected_files or []) if f]

        def _units_in_files() -> Set[str]:
            out: Set[str] = set()
            for f in affected_files:
                try:
                    for uid in g.units_in_file(f):
                        out.add(uid)
                except Exception:
                    continue
            return out

        for func_ref in obs.affected_functions:
            scope = None
            ref = func_ref

            if "." in func_ref:
                parts = func_ref.split(".")
                scope = parts[0]
                ref = parts[1]

            # 1) Resolve via engine (scope-aware when provided)
            try:
                results = engine.resolve(ref, scope=scope)
                if results:
                    # Pick the top one (engine.resolve returns ranked candidates)
                    target_ids.add(results[0].id)
                    continue
            except Exception as e:
                self.logger.debug(f"Failed to resolve {func_ref}: {e}")

            # 2) If no explicit scope, try file-scoped resolution to keep grounding relevant.
            # This is crucial for "constructor" observations (many exist globally).
            if scope is None and affected_files:
                try:
                    cands: Set[str] = set()
                    for uid in _units_in_files():
                        try:
                            n = g.node(uid)
                        except Exception:
                            continue

                        if ref == "constructor":
                            if n.kind == NodeKind.UNIT and n.meta.get("is_constructor"):
                                cands.add(uid)
                        else:
                            sig = (
                                (n.meta.get("signature") or "")
                                if n.kind == NodeKind.UNIT
                                else ""
                            )
                            if n.name == ref or (sig and sig.startswith(f"{ref}(")):
                                cands.add(uid)

                    if cands:
                        # For constructors, keep all file-scoped constructors (stable + informative).
                        if ref == "constructor":
                            target_ids |= cands
                        else:
                            ranked = sorted(
                                cands,
                                key=lambda uid: (
                                    0
                                    if adapter.is_public_entrypoint(g.node(uid))
                                    else 1,
                                    g.node(uid).name,
                                    uid,
                                ),
                            )
                            target_ids.add(ranked[0])
                        continue
                except Exception:
                    pass

        # 3) If still unresolved but we have affected files, anchor to entrypoints/constructors in those files.
        if not target_ids and affected_files:
            try:
                cands: list[str] = []
                for uid in sorted(_units_in_files()):
                    try:
                        n = g.node(uid)
                    except Exception:
                        continue
                    if n.kind != NodeKind.UNIT:
                        continue
                    if adapter.is_public_entrypoint(n) or n.meta.get("is_constructor"):
                        cands.append(uid)
                # Cap to keep invariants focused/deterministic.
                for uid in cands[:10]:
                    target_ids.add(uid)
            except Exception:
                pass

        return target_ids

    def _generate_label(self, obs: Observation, target_ids: Set[str]) -> str:
        """
        Generate a deterministic label based on description and targets.
        """
        sorted_targets = sorted(list(target_ids))
        raw = f"{obs.description}|{'|'.join(sorted_targets)}"
        h = hashlib.sha1(raw.encode()).hexdigest()[:10]
        return f"INV_OBS_{h}"

    def _derive_target_var_ids(
        self, target_function_ids: List[str], engine: GraphQueryEngine
    ) -> List[str]:
        """
        Deterministically derive variable targets from the dependency graph.

        Uses READS/WRITES edges out of the grounded target functions.
        """
        var_ids: Set[str] = set()
        for fid in target_function_ids:
            try:
                refs = engine.neighbors(
                    fid, [EdgeKind.READS, EdgeKind.WRITES], direction="out"
                )
                for r in refs:
                    if r.kind == NodeKind.VARIABLE:
                        var_ids.add(r.id)
            except Exception:
                # Best-effort enrichment; never fail the conversion pipeline.
                continue
        return sorted(var_ids)

    def _derive_target_file_ids(
        self, obs: Observation, target_function_ids: List[str], graph: Any
    ) -> List[str]:
        """
        Deterministically derive file targets.

        Note: InvariantProcess uses absolute paths (from unit node spans) as file IDs in vocab,
        so we mirror that convention here.
        """
        files: Set[str] = set()

        # 1) Prefer grounded file paths from graph node spans
        for fid in target_function_ids:
            try:
                node = getattr(graph, "_nodes", {}).get(fid)
                if (
                    node
                    and getattr(node, "span", None)
                    and getattr(node.span, "file", None)
                ):
                    files.add(node.span.file)
            except Exception:
                continue

        # 2) Also include observation-provided affected files (already absolute paths in practice)
        for f in obs.affected_files:
            if f:
                files.add(f)

        return sorted(files)

    def _summarize_manifesto(self, manifesto: Optional[Any]) -> str:
        """Minimal summary for LLM context."""
        if not manifesto:
            return "No protocol manifesto available."

        lines = []
        if getattr(manifesto, "name", None):
            lines.append(f"Name: {manifesto.name}")
        if getattr(manifesto, "purpose", None):
            lines.append(f"Purpose: {manifesto.purpose}")
        if getattr(manifesto, "domain", None):
            lines.append(f"Domain: {manifesto.domain}")

        return "\n".join(lines)
