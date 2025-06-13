import numpy as np
import pytest

from tracksdata.array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes._mask import Mask

# NOTE: this could be generic test for all array backends
# when more slicing operations are implemented we could test as in:
#  - https://github.com/royerlab/ultrack/blob/main/ultrack/utils/_test/test_utils_array.py


def test_graph_array_view_init() -> None:
    """Test GraphArrayView initialization."""
    graph = RustWorkXGraph()

    # Add a attribute key
    graph.add_node_attribute_key("label", 0)

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="label", offset=0)

    assert array_view.graph is graph
    assert array_view.shape == (10, 100, 100)
    assert array_view._attribute_key == "label"
    assert array_view._offset == 0
    assert array_view.dtype == np.int32
    assert array_view.ndim == 3
    assert len(array_view) == 10


def test_graph_array_view_init_invalid_attribute_key() -> None:
    """Test GraphArrayView initialization with invalid attribute key."""
    graph = RustWorkXGraph()

    with pytest.raises(ValueError, match="Attribute key 'invalid_key' not found in graph"):
        GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="invalid_key")


def test_graph_array_view_getitem_empty_time() -> None:
    """Test __getitem__ with empty time point (no nodes)."""
    graph = RustWorkXGraph()
    graph.add_node_attribute_key("label", 0)

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="label")

    # Get data for time point 0 (no nodes)
    result = array_view[0]

    # Should return zeros with correct shape
    assert result.shape == (100, 100)
    assert np.all(result == 0)
    assert result.dtype == np.int32


def test_graph_array_view_getitem_with_nodes() -> None:
    """Test __getitem__ with nodes at time point."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attribute_key("label", 0)
    graph.add_node_attribute_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create a mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 12, 22]))  # y_min, x_min, y_max, x_max

    # Add a node with mask and label
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "label": 5, DEFAULT_ATTR_KEYS.MASK: mask})

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="label")

    # Get data for time point 0
    result = array_view[0]

    # Should have correct shape
    assert result.shape == (100, 100)

    # Check that the mask was painted with the label value
    # The mask should be painted at the bbox location
    assert result[10, 20] == 5  # Top-left of mask
    assert result[10, 21] == 5  # Top-right of mask
    assert result[11, 20] == 5  # Bottom-left of mask
    assert result[11, 21] == 0  # Bottom-right should be 0 (mask is False there)

    # Other areas should be 0
    assert result[0, 0] == 0
    assert result[50, 50] == 0


def test_graph_array_view_getitem_multiple_nodes() -> None:
    """Test __getitem__ with multiple nodes at same time point."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attribute_key("label", 0)
    graph.add_node_attribute_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create two masks at different locations
    mask1_data = np.array([[True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([10, 20, 11, 22]))

    mask2_data = np.array([[True]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([30, 40, 31, 41]))

    # Add nodes with different labels
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "label": 3, DEFAULT_ATTR_KEYS.MASK: mask1})

    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "label": 7, DEFAULT_ATTR_KEYS.MASK: mask2})

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="label")

    # Get data for time point 0
    result = array_view[0]

    # Check that both masks were painted with their respective labels
    assert result[10, 20] == 3
    assert result[10, 21] == 3
    assert result[30, 40] == 7

    # Other areas should be 0
    assert result[0, 0] == 0
    assert result[50, 50] == 0


def test_graph_array_view_getitem_boolean_dtype() -> None:
    """Test __getitem__ with boolean attribute values."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attribute_key("is_active", False)
    graph.add_node_attribute_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with boolean attribute
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "is_active": True, DEFAULT_ATTR_KEYS.MASK: mask})

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="is_active")

    # Get data for time point 0
    result = array_view[0]

    # Boolean values should be converted to uint8 for napari
    assert result.dtype == np.uint8
    assert result[10, 20] == 1  # True -> 1
    assert result[0, 0] == 0  # False -> 0


def test_graph_array_view_dtype_inference() -> None:
    """Test that dtype is properly inferred from data."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attribute_key("float_label", 0.0)
    graph.add_node_attribute_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with float attribute
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "float_label": 3.14, DEFAULT_ATTR_KEYS.MASK: mask})

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attribute_key="float_label")

    # Get data to trigger dtype inference
    _ = array_view[0]

    # Dtype should be updated based on the actual data
    assert array_view.dtype == np.float64
