import abc
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Literal, Optional, ParamSpec, TypeVar

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from skimage.util._map_array import ArrayMap

# NOTE:
# - maybe a single basegraph is better
# - nodes have a t, and space


class BaseReadOnlyGraph(abc.ABC):  # noqa: B024
    """
    Base class for viewing a graph.
    """


class BaseWritableGraph(BaseReadOnlyGraph):
    """
    Base class for writing to a graph.
    """

    # TODO


class BaseGraphBackend(abc.ABC):
    """
    Base class for a graph backend.
    """

    def __init__(self) -> None:
        self.set_parent(None, None)

    @property
    def parent(self) -> Optional["BaseGraphBackend"]:
        """
        The parent graph.
        """
        return self._parent

    def set_parent(
        self,
        parent: Optional["BaseGraphBackend"],
        node_ids: list[int] | None,
    ) -> None:
        """
        Set the parent graph and computes between the node IDs of
        the parent and child graphs.

        Parameters
        ----------
        parent : Optional[&quot;BaseGraphBackend&quot;]
            The parent graph.
        node_ids : Optional[list[int]]
            The node IDs of the subgraph.

        Raises
        ------
        ValueError
            If 'node_ids' is not provided when 'parent' is not None.
        """

        # TODO: do we need a edge inversion map?
        if parent is None:
            # resetting mapping
            self._parent = parent
            self._node_map = None
            self._node_inv_map = None
            return

        if node_ids is None or len(node_ids) == 0:
            raise ValueError("'node_ids' must be provided when setting graph 'parent'")

        node_ids = np.asarray(node_ids, dtype=int, copy=True)
        self._node_map = node_ids
        self._node_inv_map = ArrayMap(node_ids, np.arange(len(node_ids)))
        self._parent = parent

    def maybe_map_nodes(
        self,
        node_ids: ArrayLike,
        direction: Literal["child_to_root", "root_to_child"],
    ) -> np.ndarray:
        """
        Map the node IDs of the root graph to the child subgraph.

        For example:

        {root_graph} -> {intermediate_graph} -> {child_graph}

        Parameters
        ----------
        node_ids : ArrayLike
            The node IDs to map.
        direction : Literal["child_to_root", "root_to_child"]
            The direction of the mapping.

        Returns
        -------
        np.ndarray
            The mapped node IDs.
        """
        node_ids = np.asarray(node_ids, dtype=int)

        if self.parent is None:
            return node_ids

        if direction == "root_to_child":
            return self._node_inv_map[node_ids]
        elif direction == "child_to_root":
            return self._node_map[node_ids]
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @staticmethod
    def _validate_attributes(
        attributes: dict[str, Any],
        reference_keys: list[str],
        mode: str,
    ) -> None:
        """
        Validate the attributes of a node.

        Parameters
        ----------
        attributes : dict[str, Any]
            The attributes to validate.
        reference_keys : list[str]
            The keys to validate against.
        mode : str
            The mode to validate against, for example "node" or "edge".
        """
        for key in attributes.keys():
            if key not in reference_keys:
                raise ValueError(
                    f"{mode} feature key {key} not found in existing keys: "
                    f"'{reference_keys}'\nInitialize with "
                    "`graph.add_{mode}_feature_key(key, default_value)`"
                )

        for ref_key in reference_keys:
            if ref_key not in attributes.keys():
                raise ValueError(
                    f"Attribute '{ref_key}' not found in attributes: "
                    f"'{attributes.keys()}'\nAll '{reference_keys}' "
                    "attributes must be provided."
                )

    @abc.abstractmethod
    def add_node(
        self,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        attributes : Any
            The attributes of the node to be added, must have a "t" key.
            The keys of the attributes will be used as the attributes of the node.
            For example:
            >>> `graph.add_node(dict(t=0, label='A', intensity=100))`
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.

        TODO: should "t" be it's own parameter?

        Returns
        -------
        int
            The ID of the added node.
        """

    @abc.abstractmethod
    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.
        attributes : dict[str, Any]
            Additional attributes for the edge.
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.

        Returns
        -------
        int
            The ID of the added edge.
        """

    @abc.abstractmethod
    def node_ids(self) -> np.ndarray:
        """
        Get the IDs of all nodes in the graph.
        """

    @abc.abstractmethod
    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> np.ndarray:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        attributes : dict[str, Any]
            Attributes to filter by, for example:
            >>> `graph.filter_nodes_by_attribute(dict(t=0, label='A'))`

        Returns
        -------
        np.ndarray
            The IDs of the filtered nodes.
        """

    @abc.abstractmethod
    def subgraph(
        self,
        *,
        node_ids: Sequence[int],
    ) -> "BaseGraphBackend":
        """
        Create a subgraph from the graph from the given node IDs.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to include in the subgraph.

        Returns
        -------
        BaseGraphBackend
            A new graph with the specified nodes.
        """

    @abc.abstractmethod
    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """

    @abc.abstractmethod
    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        """
        Get the features of the nodes as a pandas DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the features for.
            If None, all nodes are used.
        feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.

        Returns
        -------
        pl.DataFrame
            A polars DataFrame with the features of the nodes.
        """

    @abc.abstractmethod
    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Get the features of the edges as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the subgraph to get the edge features for.
            If None, all edges of the graph are used.
        feature_keys : Sequence[str] | None
            The feature keys to get.
            If None, all features are used.
        """

    @property
    @abc.abstractmethod
    def node_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the nodes.
        """

    @property
    @abc.abstractmethod
    def edge_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the edges.
        """

    @abc.abstractmethod
    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing nodes will have the default value for the new feature key.
        """

    @abc.abstractmethod
    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing edges will have the default value for the new feature key.
        """

    @property
    @abc.abstractmethod
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """

    @property
    @abc.abstractmethod
    def num_nodes(self) -> int:
        """
        The number of nodes in the graph.
        """

    @abc.abstractmethod
    def update_node_features(
        self,
        *,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        """
        Update the features of the nodes.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to update.
        attributes : dict[str, Any]
            The attributes to update.
        """

    @abc.abstractmethod
    def update_edge_features(
        self,
        *,
        edge_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        """
        Update the features of the edges.

        Parameters
        ----------
        edge_ids : Sequence[int]
            The IDs of the edges to update.
        attributes : dict[str, Any]
            Attributes to be updated.
        """


P = ParamSpec("P")
R = TypeVar("R")


def remap_input_node_ids(node_param: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to remap node IDs from root to child before method execution.

    Parameters
    ----------
    node_param : str
        Name of the parameter containing node IDs to remap
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def _wrapper(self: BaseGraphBackend, **kwargs: P.kwargs) -> R:
            # Get the node IDs from args or kwargs
            node_ids = kwargs.get(node_param, None)
            if node_ids is not None:
                kwargs[node_param] = self.maybe_map_nodes(node_ids, "root_to_child")
            return func(self, **kwargs)

        return _wrapper

    return _decorator


def remap_output_node_ids(ids_columns: list[str] | None = None) -> Callable[P, R]:
    """
    Decorator to remap node IDs from child to root after method execution.
    Only works for methods that return a sequence of node IDs or a single node ID.

    Parameters
    ----------
    ids_columns : list[str] | None
        Optional list of columns to remap when returning a pl.DataFrame.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        The decorated function.
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def _wrapper(self: BaseGraphBackend, *args: P.args, **kwargs: P.kwargs) -> R:
            result = func(self, *args, **kwargs)
            if isinstance(result, pl.DataFrame):
                if ids_columns is None:
                    raise ValueError(
                        "'node_params' must be provided when returning a pl.DataFrame"
                    )
                for node_param in ids_columns:
                    values = self.maybe_map_nodes(
                        result[node_param].to_numpy(), "child_to_root"
                    )
                    result = result.with_columns(
                        pl.Series(name=node_param, values=values)
                    )
                return result
            elif isinstance(result, list | np.ndarray | Sequence):
                return self.maybe_map_nodes(result, "child_to_root")
            elif isinstance(result, int | np.integer):
                return int(self.maybe_map_nodes([result], "child_to_root")[0])
            raise ValueError(f"Invalid return type: {type(result)}")

        return _wrapper

    return _decorator
