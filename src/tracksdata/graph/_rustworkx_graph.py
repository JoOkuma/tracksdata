from typing import Any

import rustworkx as rx

from tracksdata.graph._base_graph import BaseGraphBackend, BaseReadOnlyGraph

# TODO:
# - use a better name for the default graph backend


class RustWorkXReadOnlyGraph(BaseReadOnlyGraph):
    def __init__(
        self,
        graph: rx.PyDiGraph | None = None,
    ) -> None:
        """
        TODO
        """
        if graph is None:
            self._graph = graph
        else:
            self._graph = rx.PyDiGraph()


class RustWorkXGraphBackend(BaseGraphBackend):
    def __init__(self) -> None:
        """
        TODO
        """
        self._graph = rx.PyDiGraph()
        self._time_to_nodes: dict[int, list[int]] = {}

    def add_node(
        self,
        *,
        t: int,
        **kwargs: Any,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        t : int
            The time at which to add the node.
        **kwargs : Any
            The attributes of the node to be added.
            The keys of the kwargs will be used as the attributes of the node.
            For example:
            >>> `graph.add_node(t=0, label='A', intensity=100)`
        """
        # avoiding copying kwargs on purpose, it could be a problem in the future
        kwargs["t"] = t
        node_id = self._graph.add_node(kwargs)
        self._time_to_nodes.setdefault(t, []).append(node_id)
        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        **kwargs: Any,
    ) -> int:
        # TODO doc
        edge_id = self._graph.add_edge(source_id, target_id, **kwargs)
        return edge_id

    def filter_nodes_by_attribute(
        self,
        **kwargs: Any,
    ) -> list[int]:
        # TODO doc
        def _filter_func(node_id: int) -> bool:
            for key, value in kwargs.items():
                try:
                    if self._graph[node_id][key] != value:
                        return False
                except KeyError:
                    return False
            return True

        return list(self._graph.filter_nodes(_filter_func))

    def subgraph(
        self,
        *,
        node_ids: list[int] | None = None,
        **filter_kwargs,
    ) -> RustWorkXReadOnlyGraph:
        # TODO doc
        if node_ids is not None:
            subgraph = self._graph.subgraph(node_ids)
        else:
            subgraph = self._graph.subgraph(
                filter_nodes=filter_kwargs,
            )
        return RustWorkXReadOnlyGraph(graph=subgraph)
