import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend


@pytest.fixture(params=[RustWorkXGraphBackend])
def graph_backend(request) -> BaseGraphBackend:
    """Fixture that provides all implementations of BaseGraphBackend."""
    return request.param()


def test_already_existing_keys(graph_backend: BaseGraphBackend) -> None:
    """Test that adding already existing keys raises an error."""
    graph_backend.add_node_feature_key("x", None)

    with pytest.raises(ValueError):
        graph_backend.add_node_feature_key("x", None)

    graph_backend.add_edge_feature_key("w", None)

    with pytest.raises(ValueError):
        graph_backend.add_edge_feature_key("w", None)

    with pytest.raises(ValueError):
        # missing x
        graph_backend.add_node(attributes={"t": 0})


def testing_empty_graph(graph_backend: BaseGraphBackend) -> None:
    """Test that the graph is empty."""
    assert graph_backend.num_nodes == 0
    assert graph_backend.num_edges == 0

    with pytest.raises(ValueError):
        graph_backend.node_features()

    # graph could be disconnected so this should not raise an error
    assert graph_backend.edge_features().is_empty()


def test_node_validation(graph_backend: BaseGraphBackend) -> None:
    """Test node validation."""
    # 't' key must exist by default
    graph_backend.add_node({"t": 1})

    with pytest.raises(ValueError):
        graph_backend.add_node({"t": 0, "x": 1.0})


def test_edge_validation(graph_backend: BaseGraphBackend) -> None:
    """Test edge validation."""
    with pytest.raises(ValueError):
        graph_backend.add_edge(0, 1, {"weight": 0.5})


def test_add_node(graph_backend: BaseGraphBackend) -> None:
    """Test adding nodes with various attributes."""

    for key in ["x", "y"]:
        graph_backend.add_node_feature_key(key, None)

    node_id = graph_backend.add_node({"t": 0, "x": 1.0, "y": 2.0})
    assert isinstance(node_id, int)

    # Check node features
    df = graph_backend.node_features(node_ids=[node_id])
    assert df["t"].to_list() == [0]
    assert df["x"].to_list() == [1.0]
    assert df["y"].to_list() == [2.0]


def test_add_edge(graph_backend: BaseGraphBackend) -> None:
    """Test adding edges with attributes."""
    # Add node feature key
    graph_backend.add_node_feature_key("x", None)

    # Add two nodes first
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})

    # Add edge feature key
    graph_backend.add_edge_feature_key("weight", 0.0)

    # Add edge
    edge_id = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})
    assert isinstance(edge_id, int)

    # Check edge features
    df = graph_backend.edge_features()
    assert df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list() == [node1]
    assert df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list() == [node2]
    assert df["weight"].to_list() == [0.5]

    # testing adding new add attribute
    graph_backend.add_edge_feature_key("new_feature", 0.0)
    edge_id = graph_backend.add_edge(
        node1, node2, attributes={"new_feature": 1.0, "weight": 0.1}
    )
    assert isinstance(edge_id, int)

    # testing default value was assigned correctly
    df = graph_backend.edge_features()
    assert df["new_feature"].to_list() == [0.0, 1.0]
    assert df["weight"].to_list() == [0.5, 0.1]


def test_node_ids(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving node IDs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    assert set(graph_backend.node_ids()) == {node1, node2}


def test_filter_nodes_by_attribute(graph_backend: BaseGraphBackend) -> None:
    """Test filtering nodes by attributes."""
    graph_backend.add_node_feature_key("label", None)

    node1 = graph_backend.add_node({"t": 0, "label": "A"})
    node2 = graph_backend.add_node({"t": 0, "label": "B"})
    node3 = graph_backend.add_node({"t": 1, "label": "A"})

    # Filter by time
    nodes = graph_backend.filter_nodes_by_attribute({"t": 0})
    assert set(nodes) == {node1, node2}

    # Filter by label
    nodes = graph_backend.filter_nodes_by_attribute({"label": "A"})
    assert set(nodes) == {node1, node3}

    # Filter by t and label
    nodes = graph_backend.filter_nodes_by_attribute({"t": 1, "label": "A"})
    assert set(nodes) == {node3}


def test_subgraph(graph_backend: BaseGraphBackend) -> None:
    """Test creating subgraphs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})
    graph_backend.add_node({"t": 2})

    subgraph = graph_backend.subgraph(node_ids=[node1, node2])
    assert isinstance(subgraph, BaseGraphBackend)
    # TODO: implement many more subgraph testing
    raise NotImplementedError


def test_time_points(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving time points."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 2})
    graph_backend.add_node({"t": 1})

    assert set(graph_backend.time_points()) == {0, 1, 2}


def test_node_features(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving node features."""
    graph_backend.add_node_feature_key("x", None)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})

    df = graph_backend.node_features(node_ids=[node1, node2], feature_keys=["x"])
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1.0, 2.0]


def test_edge_features(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    df = graph_backend.edge_features(feature_keys=["weight"])
    assert isinstance(df, pl.DataFrame)
    assert df["weight"].to_list() == [0.5]


def test_add_node_feature_key(graph_backend: BaseGraphBackend) -> None:
    """Test adding new node feature keys."""
    node = graph_backend.add_node({"t": 0})
    graph_backend.add_node_feature_key("new_feature", 42)

    df = graph_backend.node_features(node_ids=[node], feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_add_edge_feature_key(graph_backend: BaseGraphBackend) -> None:
    """Test adding new edge feature keys."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("new_feature", 42)
    edge_id = graph_backend.add_edge(node1, node2, attributes={"new_feature": 42})

    df = graph_backend.edge_features(node_ids=[edge_id], feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_update_node_features(graph_backend: BaseGraphBackend) -> None:
    """Test updating node features."""
    graph_backend.add_node_feature_key("x", None)

    node_1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node_2 = graph_backend.add_node({"t": 0, "x": 2.0})

    graph_backend.update_node_features(node_ids=[node_1], attributes={"x": 3.0})

    df = graph_backend.node_features(node_ids=[node_1, node_2], feature_keys="x")
    assert df["x"].to_list() == [3.0, 2.0]

    # inverted access on purpose
    graph_backend.update_node_features(
        node_ids=[node_2, node_1], attributes={"x": [5.0, 6.0]}
    )

    df = graph_backend.node_features(node_ids=[node_1, node_2], feature_keys="x")
    assert df["x"].to_list() == [6.0, 5.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_node_features(
            node_ids=[node_1, node_2], attributes={"x": [1.0]}
        )


def test_update_edge_features(graph_backend: BaseGraphBackend) -> None:
    """Test updating edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    edge_id = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    graph_backend.update_edge_features(edge_ids=[edge_id], attributes={"weight": 1.0})
    df = graph_backend.edge_features(node_ids=[node1, node2], feature_keys=["weight"])
    assert df["weight"].to_list() == [1.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_edge_features(
            edge_ids=[edge_id], attributes={"weight": [1.0, 2.0]}
        )


def test_num_edges(graph_backend: BaseGraphBackend) -> None:
    """Test counting edges."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    assert graph_backend.num_edges == 1


def test_num_nodes(graph_backend: BaseGraphBackend) -> None:
    """Test counting nodes."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 1})

    assert graph_backend.num_nodes == 2
