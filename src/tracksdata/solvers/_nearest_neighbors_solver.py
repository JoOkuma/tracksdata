import numba as nb
import numpy as np
from numba import typed

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.expr import AttrExpr
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.solvers._base_solver import BaseSolver


@nb.njit
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
    seen = set()
    size = len(sorted_source)

    for i in range(size):
        source_id = sorted_source[i]
        target_id = sorted_target[i]

        if source_id in seen:
            continue

        target_count = children_counter.get(target_id, 0)
        if target_count >= max_children:
            continue

        seen.add(source_id)
        children_counter[target_id] = target_count + 1

        solution[i] = True


class NearestNeighborsSolver(BaseSolver):
    """
    Solver tracking problem with nearest neighbor ordering of edges.
    Each node can have only one parent and up to `max_children` child.

    Parameters
    ----------
    max_children : int
        The maximum number of children a node can have.
    edge_weight : str | AttrExpr
        Key to get the edge weight from the graph or an expression to evaluate
        composing edge attributes of the graph.

        For example:
        >>> `edge_weight=-AttrExpr("iou")`
        will use the negative IoU as edge weight.

        >>> `edge_weight=AttrExpr("iou").log() * AttrExpr("weight")`
        will use the log of IoU times the default weight as edge weight.

    output_key : str
        The key to store the solution in the graph.
    """

    def __init__(
        self,
        max_children: int = 2,
        edge_weight: str | AttrExpr = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
    ):
        self.max_children = max_children
        self.solution_key = output_key

        if not isinstance(edge_weight, AttrExpr):
            self.edge_weight_expr = AttrExpr(edge_weight)
        else:
            self.edge_weight_expr = edge_weight

    def solve(
        self,
        graph: BaseGraphBackend,
    ) -> None:
        """
        Solve the tracking problem with nearest neighbor ordering of edges.
        Each node can have only one parent and up to `max_children` child.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to solve.
        """
        # get edges and sort them by weight
        edges_df = graph.edge_features(
            feature_keys=self.edge_weight_expr.column_names()
        )
        weights = self.edge_weight_expr.evaluate(edges_df).to_numpy()
        sorted_indices = np.argsort(weights)

        sorted_source = edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy()[
            sorted_indices
        ]
        sorted_target = edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy()[
            sorted_indices
        ]
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

        graph.add_edge_feature_key(self.solution_key, False)
        graph.update_edge_features(
            edge_ids=solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
            attributes={self.solution_key: True},
        )

        node_ids = np.unique(
            np.concatenate(
                [
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
                ]
            )
        )

        graph.add_node_feature_key(self.solution_key, False)
        graph.update_node_features(
            node_ids=node_ids,
            attributes={self.solution_key: True},
        )
