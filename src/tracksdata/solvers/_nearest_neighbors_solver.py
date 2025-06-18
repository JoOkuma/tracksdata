import numpy as np
from numba import njit, typed

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.expr import EdgeAttr, ExprInput, NodeAttr
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._graph_view import GraphView
from tracksdata.solvers._base_solver import BaseSolver


@njit
def _constrained_nearest_neighbors(
    sorted_source: np.ndarray,
    sorted_target: np.ndarray,
    solution: np.ndarray,
    max_children: int,
) -> None:
    children_counter = typed.Dict.empty(
        key_type=np.int64,
        value_type=np.int64,
    )
    seen_targets = set()  # Track targets that already have a parent
    size = len(sorted_source)

    for i in range(size):
        source_id = sorted_source[i]
        target_id = sorted_target[i]

        # Check if target already has a parent (one parent constraint)
        if target_id in seen_targets:
            continue

        # Check if source already has max_children (max children constraint)
        source_children_count = children_counter.get(source_id, np.int64(0))
        if source_children_count >= max_children:
            continue

        # Accept this edge
        seen_targets.add(target_id)
        children_counter[source_id] = source_children_count + 1

        solution[i] = True


class NearestNeighborsSolver(BaseSolver):
    """
    Solver tracking problem with nearest neighbor ordering of edges.
    Each node can have only one parent and up to `max_children` child.

    Parameters
    ----------
    max_children : int
        The maximum number of children a node can have.
    edge_weight : str | EdgeAttr
        Key to get the edge weight from the graph or an expression to evaluate
        composing edge attributes of the graph.

        For example:
        >>> `edge_weight=-EdgeAttr("iou")`
        will use the negative IoU as edge weight.

        >>> `edge_weight=EdgeAttr("iou").log() * EdgeAttr("weight")`
        will use the log of IoU times the default weight as edge weight.

    output_key : str
        The key to store the solution in the graph.
    reset : bool
        Whether to reset the solution values in the whole graph before solving.
    """

    def __init__(
        self,
        max_children: int = 2,
        edge_weight: str | ExprInput = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
        reset: bool = True,
        return_solution: bool = True,
    ):
        super().__init__(
            output_key=output_key,
            reset=reset,
            return_solution=return_solution,
        )
        self.max_children = max_children
        self.edge_weight_expr = EdgeAttr(edge_weight)

    def solve(
        self,
        graph: BaseGraph,
    ) -> GraphView | None:
        """
        Solve the tracking problem with nearest neighbor ordering of edges.
        Each node can have only one parent and up to `max_children` child.

        Parameters
        ----------
        graph : BaseGraph
            The graph to solve.

        Returns
        -------
        GraphView | None
            The graph view of the solution if `return_solution` is True, otherwise None.
        """
        # get edges and sort them by weight
        edges_df = graph.edge_attrs(attr_keys=self.edge_weight_expr.columns)

        if len(edges_df) == 0:
            raise ValueError("No edges found in the graph, there is nothing to solve.")

        weights = self.edge_weight_expr.evaluate(edges_df).to_numpy()
        sorted_indices = np.argsort(weights)

        sorted_source = edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy()[sorted_indices].astype(np.int64)
        sorted_target = edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy()[sorted_indices].astype(np.int64)
        sorted_solution = np.zeros(len(sorted_source), dtype=bool)

        _constrained_nearest_neighbors(
            sorted_source,
            sorted_target,
            sorted_solution,
            self.max_children,
        )
        del sorted_source, sorted_target

        inverted_indices = np.empty_like(sorted_indices)
        inverted_indices[sorted_indices] = np.arange(len(sorted_indices))
        solution = sorted_solution[inverted_indices]
        del sorted_solution, inverted_indices, sorted_indices

        solution_edges_df = edges_df.filter(solution)

        if self.output_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.output_key, False)
        elif self.reset:
            graph.update_edge_attrs(attrs={self.output_key: False})

        graph.update_edge_attrs(
            edge_ids=solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
            attrs={self.output_key: True},
        )

        node_ids = np.unique(
            np.concatenate(
                [
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
                ]
            )
        )

        graph.add_node_attr_key(self.output_key, False)
        graph.update_node_attrs(
            node_ids=node_ids,
            attrs={self.output_key: True},
        )

        if self.return_solution:
            return graph.subgraph(
                NodeAttr(self.output_key) == True,
                EdgeAttr(self.output_key) == True,
            )
