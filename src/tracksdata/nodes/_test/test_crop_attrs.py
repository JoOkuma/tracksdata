import numpy as np
import pytest
from numpy.typing import NDArray

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import CropFuncAttrs, Mask


def test_crop_func_attrs_init_default() -> None:
    """Test CropFuncAttrs initialization with default parameters."""

    def dummy_func(mask: Mask, value: float) -> float:
        return value * 2.0

    operator = CropFuncAttrs(
        func=dummy_func,
        output_key="test_output",
    )

    assert operator.func == dummy_func
    assert operator.output_key == "test_output"
    assert operator.attr_keys == ()
    assert operator.show_progress is True


def test_crop_func_attrs_init_with_attr_keys() -> None:
    """Test CropFuncAttrs initialization with custom attr_keys."""

    def dummy_func(mask: Mask, value: float, multiplier: int) -> float:
        return value * multiplier

    operator = CropFuncAttrs(
        func=dummy_func,
        output_key="test_output",
        attr_keys=["multiplier"],
        show_progress=False,
    )

    assert operator.func == dummy_func
    assert operator.output_key == "test_output"
    assert operator.attr_keys == ["multiplier"]
    assert operator.show_progress is False


def test_crop_func_attrs_init_with_sequence_output_key() -> None:
    """Test CropFuncAttrs initialization with sequence output_key."""

    def dummy_func(mask: Mask, value: float) -> float:
        return value * 2.0

    operator = CropFuncAttrs(
        func=dummy_func,
        output_key=["test_output"],
    )

    assert operator.output_key == ["test_output"]


def test_crop_func_attrs_simple_function_no_frames() -> None:
    """Test applying a simple function without frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key("value", 0.0)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks and values
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1, "value": 10.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2, "value": 20.0})

    def double_value(mask: Mask, value: float) -> float:
        return value * 2.0

    # Create operator and add attributes
    operator = CropFuncAttrs(
        func=double_value,
        output_key="doubled_value",
        attr_keys=["value"],
        show_progress=False,
    )

    operator.add_node_attrs(graph)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "doubled_value" in nodes_df.columns

    # Check results
    doubled_values = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["doubled_value"], strict=False))
    assert doubled_values[node1] == 20.0
    assert doubled_values[node2] == 40.0


def test_crop_func_attrs_function_with_frames() -> None:
    """Test applying a function with frames."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2})

    # Create test frames
    frames = np.array(
        [
            np.array([[100, 200], [300, 400]]),  # Frame 0
        ]
    )

    def intensity_sum(mask: Mask, frame: NDArray) -> float:
        cropped = mask.crop(frame)
        return float(np.sum(cropped[mask.mask]))

    # Create operator and add attributes
    operator = CropFuncAttrs(
        func=intensity_sum,
        output_key="intensity_sum",
        show_progress=False,
    )

    operator.add_node_attrs(graph, t=0, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "intensity_sum" in nodes_df.columns

    # Check results
    intensity_sums = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["intensity_sum"], strict=False))

    # Expected: mask1 has 3 True pixels, mask2 has 1 True pixel
    # Frame values: [[100, 200], [300, 400]]
    # mask1 covers [0,0], [0,1], [1,0] -> 100 + 200 + 300 = 600
    # mask2 covers [0,0] -> 100
    assert intensity_sums[node1] == 600.0
    assert intensity_sums[node2] == 100.0


def test_crop_func_attrs_function_with_frames_and_attrs() -> None:
    """Test applying a function with frames and additional attributes."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key("multiplier", 1.0)

    # Create test masks
    mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Add nodes with masks and multipliers
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask1, "multiplier": 2.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask2, "multiplier": 3.0})

    # Create test frames
    frames = np.array(
        [
            np.array([[10, 20], [30, 40]]),  # Frame 0
        ]
    )

    def intensity_sum_times_multiplier(mask: Mask, frame: NDArray, multiplier: float) -> float:
        cropped = mask.crop(frame)
        return float(np.sum(cropped[mask.mask]) * multiplier)

    # Create operator and add attributes
    operator = CropFuncAttrs(
        func=intensity_sum_times_multiplier,
        output_key="weighted_intensity",
        attr_keys=["multiplier"],
        show_progress=False,
    )

    operator.add_node_attrs(graph, t=0, frames=frames)

    # Check that attributes were added
    nodes_df = graph.node_attrs()
    assert "weighted_intensity" in nodes_df.columns

    # Check results
    weighted_intensities = dict(zip(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID], nodes_df["weighted_intensity"], strict=False))

    # Expected:
    # mask1: sum = 10 + 20 + 30 = 60, multiplier = 2.0 -> 120.0
    # mask2: sum = 10, multiplier = 3.0 -> 30.0
    assert weighted_intensities[node1] == 120.0
    assert weighted_intensities[node2] == 30.0


def test_crop_func_attrs_function_returns_different_types() -> None:
    """Test that functions can return different types."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create test mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))

    # Add node
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask})

    def return_string(mask: Mask) -> str:
        return "test_string"

    def return_list(mask: Mask) -> list[int]:
        return [1, 2, 3]

    def return_dict(mask: Mask) -> dict[str, int]:
        return {"count": 3}

    def return_array(mask: Mask) -> NDArray:
        return np.asarray([1, 2, 3])

    # Test string return type
    operator_str = CropFuncAttrs(
        func=return_string,
        output_key="string_result",
        show_progress=False,
    )
    operator_str.add_node_attrs(graph)

    # Test list return type
    operator_list = CropFuncAttrs(
        func=return_list,
        output_key="list_result",
        show_progress=False,
    )
    operator_list.add_node_attrs(graph)

    # Test dict return type
    operator_dict = CropFuncAttrs(
        func=return_dict,
        output_key="dict_result",
        show_progress=False,
    )
    operator_dict.add_node_attrs(graph)

    # Test array return type
    operator_array = CropFuncAttrs(
        func=return_array,
        output_key="array_result",
        show_progress=False,
    )
    operator_array.add_node_attrs(graph)

    # Check results
    nodes_df = graph.node_attrs()
    assert nodes_df["string_result"][0] == "test_string"
    assert nodes_df["list_result"][0].to_list() == [1, 2, 3]
    assert nodes_df["dict_result"][0] == {"count": 3}
    np.testing.assert_array_equal(nodes_df["array_result"][0], np.asarray([1, 2, 3]))


def test_crop_func_attrs_error_handling_missing_attr_key() -> None:
    """Test error handling when required attr_key is missing."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    # Note: "value" is not registered

    # Create test mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([0, 0, 2, 2]))

    # Add node without the required attribute
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask})

    def use_value(mask: Mask, value: float) -> float:
        return value * 2.0

    # Create operator that requires "value" attribute
    operator = CropFuncAttrs(
        func=use_value,
        output_key="result",
        attr_keys=["value"],
        show_progress=False,
    )

    # Should raise an error when trying to access missing attribute
    with pytest.raises(KeyError):  # Specific exception type depends on graph backend
        operator.add_node_attrs(graph)


def test_crop_func_attrs_empty_graph() -> None:
    """Test behavior with an empty graph."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    def dummy_func(mask: Mask) -> float:
        return 1.0

    operator = CropFuncAttrs(
        func=dummy_func,
        output_key="result",
        show_progress=False,
    )

    # Should not raise an error, just do nothing
    operator.add_node_attrs(graph)

    # Check that no attributes were added
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 0
