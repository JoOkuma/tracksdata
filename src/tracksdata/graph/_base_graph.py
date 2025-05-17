import abc
from typing import Any

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

    @abc.abstractmethod
    def add_node(
        self,
        *,
        t: int,
        **kwargs: Any,
    ) -> int:
        """
        Add a node to the graph.

        Parameters
        ----------
        t : int
            The time of the node.
        kwargs : Any
            Additional attributes for the node.

        TODO: make add additional attributes
            - x and y
            - z=0
            - mask=None
            - bbox=None
        thoughts?

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
        **kwargs: Any,
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.
        kwargs : Any
            Additional attributes for the edge.

        Returns
        -------
        int
            The ID of the added edge.
        """

    @abc.abstractmethod
    def filter_nodes_by_attribute(
        self,
        **kwargs: Any,
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        kwargs : Any
            Attributes to filter by.

        Returns
        -------
        BaseGraphBackend
            A new graph with the filtered nodes.
        """

    @abc.abstractmethod
    def subgraph(
        self,
        *,
        node_ids: list[int] | None = None,
        **filter_kwargs: Any,
    ) -> "BaseReadOnlyGraph":
        """
        Create a subgraph from the graph from the given node IDs or by filtering
        the nodes by attributes -- both cannot be used at the same time.

        Parameters
        ----------
        node_ids : List[int] | None
            If provided, the IDs of the nodes to include in the subgraph.
        filter_kwargs : Any
            Attributes to filter by the nodes of the original graph.

        Returns
        -------
        BaseReadOnlyGraph
            A new graph with the specified nodes.
        """
