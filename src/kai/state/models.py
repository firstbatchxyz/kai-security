"""Data models for state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    """Record of a single pipeline run."""

    run_id: str
    repo_path: str
    started_at: str  # ISO 8601
    status: str  # "running" | "completed" | "failed"
    root_model: str
    finished_at: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    usage_summary: dict[str, Any] | None = None
    execution_time: float | None = None
    total_exploits: int = 0
    total_fixes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "repo_path": self.repo_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "root_model": self.root_model,
            "config_snapshot": self.config_snapshot,
            "usage_summary": self.usage_summary,
            "execution_time": self.execution_time,
            "total_exploits": self.total_exploits,
            "total_fixes": self.total_fixes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Deserialize from a dict."""
        return cls(
            run_id=data["run_id"],
            repo_path=data["repo_path"],
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
            status=data.get("status", "running"),
            root_model=data.get("root_model", "unknown"),
            config_snapshot=data.get("config_snapshot", {}),
            usage_summary=data.get("usage_summary"),
            execution_time=data.get("execution_time"),
            total_exploits=data.get("total_exploits", 0),
            total_fixes=data.get("total_fixes", 0),
        )


@dataclass
class StatusUpdate:
    """A single iteration status update for progress tracking."""

    run_id: str
    iteration_num: int
    timestamp: str  # ISO 8601
    agent_name: str  # "exploit" for root, or sub-agent name
    has_spawn_calls: bool = False
    iteration_time: float | None = None
    spawn_agent: str | None = None
    spawn_kwargs: dict[str, Any] | None = None
    spawn_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "iteration_num": self.iteration_num,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "has_spawn_calls": self.has_spawn_calls,
            "iteration_time": self.iteration_time,
            "spawn_agent": self.spawn_agent,
            "spawn_kwargs": self.spawn_kwargs,
            "spawn_result": self.spawn_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatusUpdate:
        """Deserialize from a dict."""
        return cls(
            run_id=data["run_id"],
            iteration_num=data["iteration_num"],
            timestamp=data["timestamp"],
            agent_name=data["agent_name"],
            has_spawn_calls=data.get("has_spawn_calls", False),
            iteration_time=data.get("iteration_time"),
            spawn_agent=data.get("spawn_agent"),
            spawn_kwargs=data.get("spawn_kwargs"),
            spawn_result=data.get("spawn_result"),
        )


@dataclass
class ExploitRecord:
    """Record of a discovered exploit, progressively enriched."""

    run_id: str
    exploit_id: str
    timestamp: str
    source_agent: str  # "analyzer" | "verifier"
    status: str  # "candidate" | "verified" | "verified_and_fixed"
    hypothesis: str
    file: str
    function: str
    exploit_sketch: str = ""
    confirmed: bool | None = None
    poc_code: str | None = None
    test_output: str | None = None
    severity: str | None = None
    cvss_vector: str | None = None
    cvss_score: float | None = None
    patch: str | None = None
    test_results: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "exploit_id": self.exploit_id,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "status": self.status,
            "hypothesis": self.hypothesis,
            "file": self.file,
            "function": self.function,
            "exploit_sketch": self.exploit_sketch,
            "confirmed": self.confirmed,
            "poc_code": self.poc_code,
            "test_output": self.test_output,
            "severity": self.severity,
            "cvss_vector": self.cvss_vector,
            "cvss_score": self.cvss_score,
            "patch": self.patch,
            "test_results": self.test_results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExploitRecord:
        """Deserialize from a dict."""
        return cls(
            run_id=data["run_id"],
            exploit_id=data["exploit_id"],
            timestamp=data["timestamp"],
            source_agent=data["source_agent"],
            status=data.get("status", "candidate"),
            hypothesis=data["hypothesis"],
            file=data["file"],
            function=data["function"],
            exploit_sketch=data.get("exploit_sketch", ""),
            confirmed=data.get("confirmed"),
            poc_code=data.get("poc_code"),
            test_output=data.get("test_output"),
            severity=data.get("severity"),
            cvss_vector=data.get("cvss_vector"),
            cvss_score=data.get("cvss_score"),
            patch=data.get("patch"),
            test_results=data.get("test_results"),
        )


@dataclass
class FixRecord:
    """Record of a fix applied to an exploit."""

    run_id: str
    fix_id: str
    exploit_id: str
    timestamp: str
    hypothesis: str
    file: str
    function: str
    severity: str
    patch: str
    test_results: str
    cvss_vector: str = ""
    cvss_score: float | None = None
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "fix_id": self.fix_id,
            "exploit_id": self.exploit_id,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "file": self.file,
            "function": self.function,
            "severity": self.severity,
            "cvss_vector": self.cvss_vector,
            "cvss_score": self.cvss_score,
            "patch": self.patch,
            "test_results": self.test_results,
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixRecord:
        """Deserialize from a dict."""
        return cls(
            run_id=data["run_id"],
            fix_id=data["fix_id"],
            exploit_id=data["exploit_id"],
            timestamp=data["timestamp"],
            hypothesis=data["hypothesis"],
            file=data["file"],
            function=data["function"],
            severity=data["severity"],
            cvss_vector=data.get("cvss_vector", ""),
            cvss_score=data.get("cvss_score"),
            patch=data["patch"],
            test_results=data["test_results"],
            applied=data.get("applied", False),
        )
