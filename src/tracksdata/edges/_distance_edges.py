from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._base_edges import BaseEdgesOperator
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG


class DistanceEdges(BaseEdgesOperator):
    """
    Operator that adds edges to a graph based on the distance between nodes.

    Parameters
    ----------
    distance_threshold : float
        The distance threshold for adding edges.
    n_neighbors : int
        The maximum number of neighbors to consider for adding edges.
        This in respect from the current to the previous frame.
        That means, a node in frame t will have edges to the closest
        n_neighbors nodes in frame t-1.
    attr_keys : Sequence[str] | None
        The attribute keys to use for the distance calculation.
        When None, "z", "y", "x" are used.
    show_progress : bool
        Whether to print progress of the edges addition.
    """

    def __init__(
        self,
        distance_threshold: float,
        n_neighbors: int,
        output_key: str = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        attr_keys: Sequence[str] | None = None,
        show_progress: bool = True,
    ):
        super().__init__(output_key=output_key, show_progress=show_progress)
        self.distance_threshold = distance_threshold
        self.n_neighbors = n_neighbors
        self.output_key = output_key
        self.attr_keys = attr_keys

    def _add_edges_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
    ) -> None:
        """
        Add edges to a graph based on the distance between nodes.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add edges to.
        t : int
            The time point to add edges for.
        """
        if self.output_key not in graph.edge_attr_keys:
            # negative value to indicate that the edge is not valid
            graph.add_edge_attr_key(self.output_key, -99999.0)

        if self.attr_keys is None:
            if "z" in graph.node_attr_keys:
                attr_keys = ["z", "y", "x"]
            else:
                attr_keys = ["y", "x"]
        else:
            attr_keys = self.attr_keys

        prev_node_ids = graph.filter_nodes_by_attrs(NodeAttr(DEFAULT_ATTR_KEYS.T) == t - 1)
        cur_node_ids = graph.filter_nodes_by_attrs(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if len(prev_node_ids) == 0:
            LOG.warning(
                "No nodes found for time point %d",
                t - 1,
            )
            return

        if len(cur_node_ids) == 0:
            LOG.warning(
                "No nodes found for time point %d",
                t,
            )
            return

        prev_attrs = graph.node_attrs(node_ids=prev_node_ids, attr_keys=attr_keys)
        cur_attrs = graph.node_attrs(node_ids=cur_node_ids, attr_keys=attr_keys)

        prev_kdtree = KDTree(prev_attrs.to_numpy())

        distances, prev_neigh_ids = prev_kdtree.query(
            cur_attrs.to_numpy(),
            k=self.n_neighbors,
            distance_upper_bound=self.distance_threshold,
        )
        is_valid = ~np.isinf(distances)

        prev_node_ids = np.asarray(prev_node_ids)
        # kdtree return from 0 to n-1
        # converting back to arbitrary indexing
        prev_neigh_ids[is_valid] = prev_node_ids[prev_neigh_ids[is_valid]]

        edges_data = []
        for cur_id, neigh_ids, neigh_dist, neigh_valid in zip(
            cur_node_ids, prev_neigh_ids, distances, is_valid, strict=True
        ):
            for neigh_id, dist in zip(neigh_ids[neigh_valid].tolist(), neigh_dist[neigh_valid].tolist(), strict=True):
                edges_data.append(
                    {
                        "source_id": neigh_id,
                        "target_id": cur_id,
                        self.output_key: dist,
                    }
                )

        if len(edges_data) > 0:
            graph.bulk_add_edges(edges_data)
        else:
            LOG.warning("No valid edges found for the pair of time point (%d, %d)", t, t - 1)


class ExtendDistanceEdges(DistanceEdges):
    """
    Operator that extends the neighbors of a node to different time points
    until it meets the `n_neighbors` condition or the maximum number of time points.
    """

    def __init__(
        self,
        distance_threshold: float,
        n_neighbors: int,
        max_time_points: int,
        output_key: str = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        show_progress: bool = True,
    ):
        super().__init__(
            distance_threshold=distance_threshold,
            n_neighbors=n_neighbors,
            output_key=output_key,
            show_progress=show_progress,
        )
        self.max_time_points = max_time_points
        self._kdtrees = {}

    def add_edges(self, graph: BaseGraph, *, t: int | None = None) -> None:
        """
        Add missing edges to a graph based and their current number of neighbors.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add edges to.
        t : int | None
            The time point to add edges for.
            If None, all time points are considered.
        """
        super().add_edges(graph, t=t)
        self._kdtrees.clear()

    def _add_edges_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
    ) -> None:
        """
        Add edges to a graph based on the distance between nodes.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add edges to.
        t : int
            The time point to add edges for.
        """
