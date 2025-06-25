from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator


class RandomNodes(BaseNodesOperator):
    def __init__(
        self,
        n_time_points: int,
        n_nodes_per_tp: tuple[int, int],
        n_dim: Literal[2, 3] = 3,
        random_state: int = 0,
        show_progress: bool = False,
    ):
        super().__init__(show_progress=show_progress)
        if isinstance(n_nodes_per_tp, int):
            raise ValueError("`n_nodes_per_tp` must be a tuple of two integers")

        self.n_time_points = n_time_points
        self.n_nodes = n_nodes_per_tp

        if n_dim == 2:
            self.spatial_cols = ["x", "y"]
        elif n_dim == 3:
            self.spatial_cols = ["x", "y", "z"]
        else:
            raise ValueError(f"Invalid number of dimensions: {n_dim}")

        self.rng = np.random.default_rng(random_state)

    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Override the base add_nodes method to handle n_time_points parameter.

        When t=None, iterates over range(n_time_points) instead of graph.time_points().
        When t is specified, uses the base implementation.
        """
        if t is None:
            for t in tqdm(range(self.n_time_points), disable=not self.show_progress, desc="Adding nodes"):
                self._add_nodes_per_time(
                    graph,
                    t=t,
                    **kwargs,
                )
        else:
            self._add_nodes_per_time(
                graph,
                t=t,
                **kwargs,
            )

    def _add_nodes_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
        **kwargs: Any,
    ) -> None:
        """
        Add nodes for a specific time point.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        t : int
            The time point to add nodes for.
        **kwargs : Any
            Additional keyword arguments to pass to add_node.
        """
        # Register each spatial column individually
        for col in self.spatial_cols:
            if col not in graph.node_attr_keys:
                graph.add_node_attr_key(col, -999_999_999.0)

        n_nodes_at_t = self.rng.integers(
            self.n_nodes[0],
            self.n_nodes[1],
        )

        coords = self.rng.uniform(
            low=0,
            high=1,
            size=(n_nodes_at_t, len(self.spatial_cols)),
        )

        graph.bulk_add_nodes(
            [{"t": t, **dict(zip(self.spatial_cols, c.tolist(), strict=True)), **kwargs} for c in coords]
        )
