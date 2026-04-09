"""Tests for FilteredGeffLoader."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from zarr.storage import MemoryStore

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import (
    FilteredGeffLoader,
    IndexedRXGraph,
    RustWorkXGraph,
    SQLGraph,
)
from tracksdata.graph._base_graph import BaseGraph


@pytest.fixture(
    params=[
        (IndexedRXGraph, {}),
        (RustWorkXGraph, {}),
        (SQLGraph, {"drivername": "sqlite", "database": ":memory:"}),
    ]
)
def graph_class_and_kwargs(request: pytest.FixtureRequest) -> tuple[type[BaseGraph], dict[str, Any]]:
    """Fixture that provides graph class types for FilteredGeffLoader testing."""
    return request.param


def _create_test_geff(store: MemoryStore) -> None:
    """Create a test GEFF file with multiple time points for filtering tests."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_node_attr_key("label", dtype=pl.Int64)

    graph.add_edge_attr_key("distance", dtype=pl.Float64)

    # Create nodes across multiple time points (t=0 to t=9)
    node_ids = []
    for t in range(10):
        for i in range(3):  # 3 nodes per time point
            node_id = graph.add_node(
                {
                    "t": t,
                    "x": float(t * 10 + i),
                    "y": float(t * 5 + i),
                    "label": i,
                }
            )
            node_ids.append(node_id)

    # Add edges between consecutive time points
    for t in range(9):
        for i in range(3):
            src_idx = t * 3 + i
            tgt_idx = (t + 1) * 3 + i
            src_node = node_ids[src_idx]
            tgt_node = node_ids[tgt_idx]

            # Calculate distance
            src_attrs = graph.nodes[src_node]
            tgt_attrs = graph.nodes[tgt_node]
            distance = np.sqrt((tgt_attrs["x"] - src_attrs["x"]) ** 2 + (tgt_attrs["y"] - src_attrs["y"]) ** 2)

            graph.add_edge(src_node, tgt_node, {"distance": float(distance)})

    # Save to geff
    graph.to_geff(geff_store=store)


def test_time_range_filtering() -> None:
    """Test loading subset by time range."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load with time filter
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") >= 5, NodeAttr("t") < 8],
    )
    graph, metadata = loader.load()

    # Verify only nodes with t in [5, 6, 7] are loaded
    times = graph.node_attrs(attr_keys=["t"])["t"].to_list()
    assert all(5 <= t < 8 for t in times)
    assert len(times) == 9  # 3 nodes per time point * 3 time points
    assert set(times) == {5, 6, 7}

    # Verify edges are also filtered
    assert graph.num_edges() == 6  # 3 edges between t=5->6, 3 edges between t=6->7


def test_multiple_filters_and_logic() -> None:
    """Test combining multiple filter conditions with AND logic."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load with multiple filters
    loader = FilteredGeffLoader(
        store,
        node_filters=[
            NodeAttr("t") >= 5,
            NodeAttr("x") > 50,
            NodeAttr("label") < 2,
        ],
    )
    graph, _ = loader.load()

    # Verify all conditions are satisfied
    df = graph.node_attrs(attr_keys=["t", "x", "label"])
    assert all(df["t"] >= 5)
    assert all(df["x"] > 50)
    assert all(df["label"] < 2)


def test_property_subset_loading() -> None:
    """Test loading only specified properties."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load only t and x properties
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") >= 5],
    )
    graph, metadata = loader.load(node_props=["t", "x"])

    # Verify only specified props are loaded
    node_keys = set(graph.node_attr_keys())
    assert "t" in node_keys
    assert "x" in node_keys
    assert "y" not in node_keys  # Not loaded
    assert "label" not in node_keys  # Not loaded


def test_empty_result() -> None:
    """Test handling case where filters match no nodes."""
    store = MemoryStore()
    _create_test_geff(store)

    # Filter that matches no nodes
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") > 1000],
    )
    graph, _ = loader.load()

    # Verify empty graph
    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0


def test_no_filters(graph_class_and_kwargs: tuple[type[BaseGraph], dict[str, Any]]) -> None:
    """Test loading without filters behaves like normal load."""
    store = MemoryStore()
    _create_test_geff(store)

    graph_class, graph_kwargs = graph_class_and_kwargs

    # Load without filters
    loader = FilteredGeffLoader(store)
    graph_filtered, _ = loader.load(graph_class=graph_class, **graph_kwargs)

    # Load normally for comparison
    graph_full, _ = graph_class.from_geff(store, **graph_kwargs)

    # Should have same number of nodes and edges
    assert graph_filtered.num_nodes() == graph_full.num_nodes()
    assert graph_filtered.num_edges() == graph_full.num_edges()


def test_equivalence_with_full_load() -> None:
    """Verify results match full load + filter + subgraph."""
    store = MemoryStore()
    _create_test_geff(store)

    # Method 1: Full load then filter
    graph_full, _ = IndexedRXGraph.from_geff(store)
    ref = graph_full.filter(NodeAttr("t") >= 5, NodeAttr("t") < 8).subgraph()

    # Method 2: Pre-filtered load
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") >= 5, NodeAttr("t") < 8],
    )
    graph_filtered, _ = loader.load()

    # Verify identical results
    assert graph_filtered.num_nodes() == ref.num_nodes()
    assert graph_filtered.num_edges() == ref.num_edges()

    # Compare node attributes including node_id
    ref_attrs = ref.node_attrs().sort(DEFAULT_ATTR_KEYS.NODE_ID)
    filtered_attrs = graph_filtered.node_attrs().sort(DEFAULT_ATTR_KEYS.NODE_ID)

    # edge ids are not guaranteed to be the exact same
    ref_edge_attrs = (
        ref.edge_attrs()
        .drop(DEFAULT_ATTR_KEYS.EDGE_ID)
        .sort(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET)
    )
    filtered_edge_attrs = (
        graph_filtered.edge_attrs()
        .drop(DEFAULT_ATTR_KEYS.EDGE_ID)
        .sort(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET)
    )

    assert ref_edge_attrs.equals(filtered_edge_attrs)
    assert ref_attrs.equals(filtered_attrs)


def test_all_nodes_match() -> None:
    """Test case where all nodes match the filter."""
    store = MemoryStore()
    _create_test_geff(store)

    # Filter that matches all nodes
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") >= 0],
    )
    graph, _ = loader.load()

    # Should have all 30 nodes (10 time points * 3 nodes each)
    assert graph.num_nodes() == 30
    assert graph.num_edges() == 27  # 9 time transitions * 3 nodes


def test_single_timepoint() -> None:
    """Test filtering to a single time point."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load only t=5
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") == 5],
    )
    graph, _ = loader.load()

    # Should have exactly 3 nodes
    assert graph.num_nodes() == 3
    times = graph.node_attrs(attr_keys=["t"])["t"].to_list()
    assert all(t == 5 for t in times)

    # No edges (no consecutive time points)
    assert graph.num_edges() == 0


def test_is_in_filter() -> None:
    """Test is_in filter for specific values."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load only t in [2, 5, 8]
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t").is_in([2, 5, 8])],
    )
    graph, _ = loader.load()

    # Should have 9 nodes (3 time points * 3 nodes each)
    assert graph.num_nodes() == 9
    times = set(graph.node_attrs(attr_keys=["t"])["t"].to_list())
    assert times == {2, 5, 8}


def test_with_tmp_path(tmp_path: Path) -> None:
    """Test with actual file path instead of memory store."""
    geff_path = tmp_path / "test.geff"

    # Create and save
    graph = RustWorkXGraph()
    graph.add_node_attr_key("x", dtype=pl.Float64)
    for t in range(5):
        graph.add_node({"t": t, "x": float(t)})

    graph.to_geff(geff_store=geff_path)

    # Load with filter
    loader = FilteredGeffLoader(
        geff_path,
        node_filters=[NodeAttr("t") >= 2],
    )
    graph_loaded, _ = loader.load()

    assert graph_loaded.num_nodes() == 3
    times = graph_loaded.node_attrs(attr_keys=["t"])["t"].to_list()
    assert set(times) == {2, 3, 4}


def test_graph_class_parameter(graph_class_and_kwargs: tuple[type[BaseGraph], dict[str, Any]]) -> None:
    """Test loading into different graph classes."""
    store = MemoryStore()
    _create_test_geff(store)

    graph_class, graph_kwargs = graph_class_and_kwargs

    # Load with specified graph class
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") < 3],
    )
    graph, _ = loader.load(graph_class=graph_class, **graph_kwargs)

    assert isinstance(graph, graph_class)
    assert graph.num_nodes() == 9


def test_reuse_loader_with_different_filters() -> None:
    """Test reusing a loader instance with different filters across multiple loads."""
    store = MemoryStore()
    _create_test_geff(store)

    loader = FilteredGeffLoader(store)

    # First load: t in [0, 3)
    loader.node_filters = [NodeAttr("t") >= 0, NodeAttr("t") < 3]
    graph1, _ = loader.load(node_props=["t", "x"])
    times1 = set(graph1.node_attrs(attr_keys=["t"])["t"].to_list())
    assert times1 == {0, 1, 2}
    assert graph1.num_nodes() == 9

    # Second load: t in [5, 8)
    loader.node_filters = [NodeAttr("t") >= 5, NodeAttr("t") < 8]
    graph2, _ = loader.load(node_props=["t", "x"])
    times2 = set(graph2.node_attrs(attr_keys=["t"])["t"].to_list())
    assert times2 == {5, 6, 7}
    assert graph2.num_nodes() == 9

    # Third load: different props
    loader.node_filters = [NodeAttr("t") == 4]
    graph3, _ = loader.load(node_props=["t", "y", "label"])
    assert graph3.num_nodes() == 3
    assert "y" in set(graph3.node_attr_keys())
    assert "label" in set(graph3.node_attr_keys())

    # Fourth load: no filters
    loader.node_filters = []
    graph4, _ = loader.load(node_props=["t"])
    assert graph4.num_nodes() == 30

    # Fifth load: with edge_props subset
    loader.node_filters = [NodeAttr("t") >= 3, NodeAttr("t") < 6]
    graph5, _ = loader.load(node_props=["t"], edge_props=["distance"])
    assert graph5.num_edges() == 6
    edge_keys = set(graph5.edge_attr_keys())
    assert "distance" in edge_keys


def test_edges_filtered_by_nodes() -> None:
    """Test that edges are filtered when both endpoints are in node set."""
    store = MemoryStore()
    _create_test_geff(store)

    # Load nodes with t in [2, 7) - should have edges between consecutive time points
    loader = FilteredGeffLoader(
        store,
        node_filters=[NodeAttr("t") >= 2, NodeAttr("t") < 7],
    )
    graph, _ = loader.load()

    # Verify node filter
    times = graph.node_attrs(attr_keys=["t"])["t"].to_list()
    assert all(2 <= t < 7 for t in times)
    assert len(times) == 15  # 5 time points * 3 nodes each

    # Verify edges - should only have edges between filtered nodes
    # Expected: 4 time transitions (2->3, 3->4, 4->5, 5->6) * 3 edges each = 12 edges
    assert graph.num_edges() == 12
