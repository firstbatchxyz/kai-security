"""
Tests for the dependency graph module.

Uses the pre-built dependency_graph.json from bbp-public-assets for testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.utils.dependency import (
    DependencyGraph,
    EdgeKind,
    Node,
    NodeKind,
)
from kai.utils.dependency.models import SourceSpan


# Path to the cached dependency graph
GRAPH_JSON = Path(__file__).resolve().parent / "fixtures" / "dependency_graph.json"


@pytest.fixture(scope="module")
def graph() -> DependencyGraph:
    """Load the dependency graph from the cached JSON file."""
    if not GRAPH_JSON.exists():
        pytest.skip(f"Dependency graph not found at {GRAPH_JSON}")
    return DependencyGraph.from_json(GRAPH_JSON)


class TestGraphBasics:
    """Tests for basic graph operations."""

    def test_graph_loads(self, graph: DependencyGraph):
        """Graph should load with nodes and edges."""
        assert len(graph._nodes) > 0
        assert len(graph._edges) > 0

    def test_nodes_by_kind(self, graph: DependencyGraph):
        """Should be able to filter nodes by kind."""
        containers = graph.nodes(NodeKind.CONTAINER)
        units = graph.nodes(NodeKind.UNIT)
        variables = graph.nodes(NodeKind.VARIABLE)
        files = graph.nodes(NodeKind.FILE)

        assert len(containers) > 0, "Should have container nodes"
        assert len(units) > 0, "Should have unit nodes"
        assert len(variables) > 0, "Should have variable nodes"
        assert len(files) > 0, "Should have file nodes"

    def test_node_access(self, graph: DependencyGraph):
        """Should be able to access individual nodes."""
        container_ids = list(graph.nodes(NodeKind.CONTAINER))
        assert len(container_ids) > 0

        node = graph.node(container_ids[0])
        assert node.kind == NodeKind.CONTAINER
        assert node.name is not None
        assert node.id == container_ids[0]

    def test_find_containers(self, graph: DependencyGraph):
        """Should find containers by name."""
        # Look for a contract we know exists in bbp-public-assets
        results = graph.find_containers("StakedUSDeV2")
        assert len(results) > 0, "Should find StakedUSDeV2 container"

    def test_find_units(self, graph: DependencyGraph):
        """Should find units by name."""
        results = graph.find_units("withdraw")
        assert len(results) > 0, "Should find withdraw unit(s)"

    def test_node_span_present(self, graph: DependencyGraph):
        """Nodes should have span information."""
        container_ids = list(graph.nodes(NodeKind.CONTAINER))
        if container_ids:
            node = graph.node(container_ids[0])
            assert node.span is not None, "Container should have span"
            assert node.span.file is not None, "Span should have file"
            assert node.span.start_line >= 1, "Span should have valid start_line"


class TestPublicEntrypoints:
    """Tests for public entrypoint detection."""

    def test_public_entrypoints_exist(self, graph: DependencyGraph):
        """Should find public/external units."""
        entrypoints = graph.public_entrypoints()
        assert len(entrypoints) > 0, "Should have public entrypoints"

    def test_public_entrypoints_are_public(self, graph: DependencyGraph):
        """All entrypoints should have public/external visibility in meta."""
        entrypoints = graph.public_entrypoints()
        for ep in entrypoints[:20]:  # Sample first 20
            node = graph.node(ep)
            assert node.kind == NodeKind.UNIT
            visibility = node.meta.get("visibility", "")
            assert visibility in ("public", "external"), (
                f"Entrypoint {node.name} has visibility {visibility}"
            )

    def test_excludes_constructors(self, graph: DependencyGraph):
        """Constructors should not be in entrypoints."""
        entrypoints = graph.public_entrypoints()
        for ep in entrypoints:
            node = graph.node(ep)
            assert not node.meta.get("is_constructor", False), (
                f"Constructor {node.name} should not be an entrypoint"
            )


class TestDeriveRelatedFiles:
    """Tests for file relationship derivation."""

    def test_derive_related_files_minimal(self, graph: DependencyGraph):
        """MINIMAL mode should return only the target file."""
        target = "contracts/StakedUSDeV2.sol"
        related = graph.derive_related_files(target, depth=2, mode="MINIMAL")
        assert len(related) == 1
        assert target in related[0]

    def test_derive_related_files_real_source(self, graph: DependencyGraph):
        """REAL_SOURCE mode should return related files."""
        target = "contracts/StakedUSDeV2.sol"
        related = graph.derive_related_files(target, depth=2, mode="REAL_SOURCE")
        assert len(related) > 1, "Should find related files"
        # Target should be included
        assert any(target in f for f in related)

    def test_derive_related_files_broad(self, graph: DependencyGraph):
        """BROAD mode should return more files than REAL_SOURCE."""
        target = "contracts/StakedUSDeV2.sol"
        real_source = graph.derive_related_files(target, depth=2, mode="REAL_SOURCE")
        broad = graph.derive_related_files(target, depth=2, mode="BROAD")
        # Broad should be >= real_source (uses all edge types)
        assert len(broad) >= len(real_source)

    def test_excludes_test_files_by_default(self, graph: DependencyGraph):
        """Test files should be excluded by default."""
        target = "contracts/StakedUSDeV2.sol"
        related = graph.derive_related_files(target, depth=3, mode="REAL_SOURCE")
        for f in related:
            assert not graph._looks_like_test(f), f"Test file {f} should be excluded"


class TestBFS:
    """Tests for BFS traversal."""

    def test_bfs_from_unit(self, graph: DependencyGraph):
        """BFS should find connected nodes from a unit."""
        unit_ids = graph.find_units("withdraw")
        if not unit_ids:
            pytest.skip("No withdraw unit found")

        visited = graph.bfs(
            unit_ids[:1],
            max_hops=2,
            edge_kinds={EdgeKind.CALLS, EdgeKind.READS, EdgeKind.WRITES},
            direction="out",
        )
        assert len(visited) >= 1, "Should visit at least the start node"

    def test_bfs_respects_max_hops(self, graph: DependencyGraph):
        """BFS with lower max_hops should visit fewer nodes."""
        unit_ids = graph.find_units("withdraw")
        if not unit_ids:
            pytest.skip("No withdraw unit found")

        visited_1 = graph.bfs(unit_ids[:1], max_hops=1, direction="both")
        visited_3 = graph.bfs(unit_ids[:1], max_hops=3, direction="both")
        assert len(visited_1) <= len(visited_3)


class TestNeighbors:
    """Tests for neighbor queries."""

    def test_neighbors_out(self, graph: DependencyGraph):
        """Should find outgoing neighbors."""
        unit_ids = list(graph.nodes(NodeKind.UNIT))[:10]
        for uid in unit_ids:
            neighbors = list(
                graph.neighbors(uid, edge_kinds={EdgeKind.CALLS}, direction="out")
            )
            # Just verify it doesn't crash
            assert isinstance(neighbors, list)

    def test_neighbors_in(self, graph: DependencyGraph):
        """Should find incoming neighbors."""
        unit_ids = list(graph.nodes(NodeKind.UNIT))[:10]
        for uid in unit_ids:
            neighbors = list(
                graph.neighbors(uid, edge_kinds={EdgeKind.CALLS}, direction="in")
            )
            assert isinstance(neighbors, list)

    def test_neighbors_both(self, graph: DependencyGraph):
        """Should find both incoming and outgoing neighbors."""
        unit_ids = list(graph.nodes(NodeKind.UNIT))[:5]
        for uid in unit_ids:
            out_neighbors = set(
                graph.neighbors(uid, edge_kinds={EdgeKind.CALLS}, direction="out")
            )
            in_neighbors = set(
                graph.neighbors(uid, edge_kinds={EdgeKind.CALLS}, direction="in")
            )
            both_neighbors = set(
                graph.neighbors(uid, edge_kinds={EdgeKind.CALLS}, direction="both")
            )
            assert both_neighbors == out_neighbors | in_neighbors


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict(self, graph: DependencyGraph):
        """Graph should serialize to dict."""
        d = graph.to_dict()
        assert "root_dir" in d
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == len(graph._nodes)
        assert len(d["edges"]) == len(graph._edges)

    def test_roundtrip(self, graph: DependencyGraph, tmp_path):
        """Graph should survive JSON roundtrip."""
        json_path = tmp_path / "test_graph.json"
        graph.to_json(json_path)

        loaded = DependencyGraph.from_json(json_path)
        assert len(loaded._nodes) == len(graph._nodes)
        assert len(loaded._edges) == len(graph._edges)

    def test_node_structure_roundtrip(self, tmp_path):
        """Nodes with span and meta should survive roundtrip."""
        g = DependencyGraph(tmp_path)

        span = SourceSpan(
            file="Test.sol",
            start_line=10,
            end_line=25,
            start_col=1,
            end_col=None,
        )

        g.add_node(
            Node(
                id="container:Test",
                kind=NodeKind.CONTAINER,
                name="Test",
                span=span,
                parent_id=None,
                meta={"subkind": "contract", "abstract": False},
            )
        )

        g.add_node(
            Node(
                id="unit:Test:foo",
                kind=NodeKind.UNIT,
                name="foo",
                span=SourceSpan(file="Test.sol", start_line=15, end_line=20),
                parent_id="container:Test",
                meta={"visibility": "public", "signature": "foo()"},
            )
        )

        g.add_edge("container:Test", "unit:Test:foo", EdgeKind.DEFINES)

        # Save and reload
        json_path = tmp_path / "struct_graph.json"
        g.to_json(json_path)

        loaded = DependencyGraph.from_json(json_path)

        # Verify container
        container = loaded.node("container:Test")
        assert container.kind == NodeKind.CONTAINER
        assert container.name == "Test"
        assert container.span is not None
        assert container.span.file == "Test.sol"
        assert container.span.start_line == 10
        assert container.span.end_line == 25
        assert container.meta["subkind"] == "contract"

        # Verify unit
        unit = loaded.node("unit:Test:foo")
        assert unit.kind == NodeKind.UNIT
        assert unit.parent_id == "container:Test"
        assert unit.meta["visibility"] == "public"
        assert unit.meta["signature"] == "foo()"

        # Verify edge
        neighbors = list(
            loaded.neighbors(
                "container:Test", edge_kinds={EdgeKind.DEFINES}, direction="out"
            )
        )
        assert "unit:Test:foo" in neighbors


class TestManualGraphConstruction:
    """Tests for manually constructing graphs."""

    def test_create_minimal_graph(self, tmp_path):
        """Should be able to create a minimal graph manually."""
        g = DependencyGraph(tmp_path)

        # Add file node
        g.add_node(
            Node(
                id="file:Test.sol",
                kind=NodeKind.FILE,
                name="Test.sol",
                span=SourceSpan(file="Test.sol", start_line=1, end_line=1),
            )
        )

        # Add container node
        g.add_node(
            Node(
                id="container:Test",
                kind=NodeKind.CONTAINER,
                name="Test",
                span=SourceSpan(file="Test.sol", start_line=5, end_line=50),
                parent_id="file:Test.sol",
                meta={"subkind": "contract"},
            )
        )

        # Add unit node
        g.add_node(
            Node(
                id="unit:Test:transfer",
                kind=NodeKind.UNIT,
                name="transfer",
                span=SourceSpan(file="Test.sol", start_line=10, end_line=20),
                parent_id="container:Test",
                meta={
                    "visibility": "external",
                    "signature": "transfer(address,uint256)",
                },
            )
        )

        # Add variable node
        g.add_node(
            Node(
                id="var:Test:balances",
                kind=NodeKind.VARIABLE,
                name="balances",
                span=SourceSpan(file="Test.sol", start_line=6, end_line=6),
                parent_id="container:Test",
                meta={"type": "mapping(address => uint256)"},
            )
        )

        # Add edges
        g.add_edge("file:Test.sol", "container:Test", EdgeKind.DEFINES)
        g.add_edge("container:Test", "unit:Test:transfer", EdgeKind.DEFINES)
        g.add_edge("container:Test", "var:Test:balances", EdgeKind.DEFINES)
        g.add_edge("unit:Test:transfer", "var:Test:balances", EdgeKind.WRITES)

        # Verify structure
        assert len(g.nodes(NodeKind.FILE)) == 1
        assert len(g.nodes(NodeKind.CONTAINER)) == 1
        assert len(g.nodes(NodeKind.UNIT)) == 1
        assert len(g.nodes(NodeKind.VARIABLE)) == 1

        # Verify relationships
        containers = g.containers_in_file("Test.sol")
        assert "container:Test" in containers

        units = g.units_in_container("container:Test")
        assert "unit:Test:transfer" in units

        # Verify writes edge
        writes = list(
            g.neighbors(
                "unit:Test:transfer", edge_kinds={EdgeKind.WRITES}, direction="out"
            )
        )
        assert "var:Test:balances" in writes

    def test_interface_nodes(self, tmp_path):
        """Should be able to create interface (modifier) nodes."""
        g = DependencyGraph(tmp_path)

        g.add_node(
            Node(
                id="container:Ownable",
                kind=NodeKind.CONTAINER,
                name="Ownable",
                span=SourceSpan(file="Ownable.sol", start_line=1, end_line=50),
                meta={"subkind": "contract"},
            )
        )

        g.add_node(
            Node(
                id="interface:Ownable:onlyOwner",
                kind=NodeKind.INTERFACE,
                name="onlyOwner",
                span=SourceSpan(file="Ownable.sol", start_line=10, end_line=15),
                parent_id="container:Ownable",
                meta={"subkind": "modifier"},
            )
        )

        g.add_node(
            Node(
                id="unit:Ownable:transferOwnership",
                kind=NodeKind.UNIT,
                name="transferOwnership",
                span=SourceSpan(file="Ownable.sol", start_line=20, end_line=30),
                parent_id="container:Ownable",
                meta={
                    "visibility": "public",
                    "signature": "transferOwnership(address)",
                },
            )
        )

        # Unit ACCEPTS the interface (modifier)
        g.add_edge(
            "unit:Ownable:transferOwnership",
            "interface:Ownable:onlyOwner",
            EdgeKind.ACCEPTS,
        )
        g.add_edge("container:Ownable", "interface:Ownable:onlyOwner", EdgeKind.DEFINES)
        g.add_edge(
            "container:Ownable", "unit:Ownable:transferOwnership", EdgeKind.DEFINES
        )

        # Verify
        modifiers = list(
            g.neighbors(
                "unit:Ownable:transferOwnership",
                edge_kinds={EdgeKind.ACCEPTS},
                direction="out",
            )
        )
        assert "interface:Ownable:onlyOwner" in modifiers

        # Interface nodes should be findable via find_units (since they're indexed similarly)
        results = g.find_units("onlyOwner")
        assert "interface:Ownable:onlyOwner" in results

    def test_type_def_nodes(self, tmp_path):
        """Should be able to create type_def (struct/enum) nodes."""
        g = DependencyGraph(tmp_path)

        g.add_node(
            Node(
                id="container:Token",
                kind=NodeKind.CONTAINER,
                name="Token",
                span=SourceSpan(file="Token.sol", start_line=1, end_line=100),
                meta={"subkind": "contract"},
            )
        )

        g.add_node(
            Node(
                id="type_def:Token:UserInfo",
                kind=NodeKind.TYPE_DEF,
                name="UserInfo",
                span=SourceSpan(file="Token.sol", start_line=5, end_line=10),
                parent_id="container:Token",
                meta={"subkind": "struct", "fields": ["amount", "rewardDebt"]},
            )
        )

        g.add_node(
            Node(
                id="unit:Token:deposit",
                kind=NodeKind.UNIT,
                name="deposit",
                span=SourceSpan(file="Token.sol", start_line=20, end_line=40),
                parent_id="container:Token",
                meta={"visibility": "external", "signature": "deposit(uint256)"},
            )
        )

        # Unit uses the type
        g.add_edge("unit:Token:deposit", "type_def:Token:UserInfo", EdgeKind.USES_TYPE)

        # Verify
        used_types = list(
            g.neighbors(
                "unit:Token:deposit", edge_kinds={EdgeKind.USES_TYPE}, direction="out"
            )
        )
        assert "type_def:Token:UserInfo" in used_types


class TestEdgeKinds:
    """Tests for edge kind values."""

    def test_edge_kind_values(self):
        """EdgeKind enum should have expected values."""
        assert EdgeKind.DEFINES == "defines"
        assert EdgeKind.IMPORTS == "imports"
        assert EdgeKind.INHERITS == "inherits"
        assert EdgeKind.CALLS == "calls"
        assert EdgeKind.ACCEPTS == "accepts"
        assert EdgeKind.READS == "reads"
        assert EdgeKind.WRITES == "writes"
        assert EdgeKind.EMITS == "emits"
        assert EdgeKind.USES_TYPE == "uses_type"


class TestNodeKinds:
    """Tests for node kind values."""

    def test_node_kind_values(self):
        """NodeKind enum should have expected values."""
        assert NodeKind.FILE == "file"
        assert NodeKind.CONTAINER == "container"
        assert NodeKind.UNIT == "unit"
        assert NodeKind.INTERFACE == "interface"
        assert NodeKind.VARIABLE == "variable"
        assert NodeKind.TYPE_DEF == "type_def"
        assert NodeKind.EVENT == "event"
        assert NodeKind.EXTERNAL == "external"


class TestFileOperations:
    """Tests for file-related operations."""

    def test_file_node_lookup(self, graph: DependencyGraph):
        """Should be able to find file nodes by path."""
        # The fixture should have file nodes
        file_nodes = graph.nodes(NodeKind.FILE)
        if not file_nodes:
            pytest.skip("No file nodes in fixture")

        # Pick a file node and verify lookup
        fid = next(iter(file_nodes))
        node = graph.node(fid)
        if node.span:
            found_id = graph.file_node(node.span.file)
            assert found_id == fid

    def test_containers_in_file(self, graph: DependencyGraph):
        """Should find containers defined in a file."""
        containers = graph.containers_in_file("contracts/StakedUSDeV2.sol")
        assert len(containers) > 0, "Should find containers in StakedUSDeV2.sol"

    def test_units_in_file(self, graph: DependencyGraph):
        """Should find units defined in a file."""
        units = graph.units_in_file("contracts/StakedUSDeV2.sol")
        assert len(units) > 0, "Should find units in StakedUSDeV2.sol"
