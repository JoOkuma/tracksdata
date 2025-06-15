import numpy as np
from numba import njit, typed

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.expr import AttrExpr, ExprInput
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.solvers._base_solver import BaseSolver


@njit
def _constrained_nearest_neighbors(
    sorted_source: np.ndarray,
    sorted_target: np.ndarray,
    solution: np.ndarray,
    max_children: int,
) -> None:
    """
    Optimized constraint satisfaction for nearest neighbor tracking.

    Numba-compiled function that efficiently applies tracking constraints by
    processing edges in weight-sorted order (best first) and greedily accepting
    edges that don't violate constraints. Ensures optimal solution for the
    nearest neighbor tracking problem in O(n) time.

    Parameters
    ----------
    sorted_source : np.ndarray
        Array of source node IDs, sorted by edge weights (best first).
    sorted_target : np.ndarray
        Array of target node IDs, corresponding to sorted_source.
    solution : np.ndarray
        Output boolean array to store the solution. Modified in-place.
        True indicates the edge is selected in the solution.
    max_children : int
        Maximum number of children each source node can have.
    """
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
    Solver for tracking problems using nearest neighbor edge selection.

    Implements a greedy nearest neighbor approach to solve tracking problems by
    selecting the best edges while enforcing constraints on parent-child relationships.
    Works by sorting all edges by weight, greedily selecting edges starting from the
    best weights, and enforcing constraints (each node can have at most one parent
    and max_children children). Runs in O(n log n) time due to sorting, where n is
    the number of edges.

    Parameters
    ----------
    max_children : int, default 2
        Maximum number of children (successors) each node can have.
        This constrains cell division events in biological tracking.
    edge_weight : str | AttrExpr, default DEFAULT_ATTR_KEYS.EDGE_WEIGHT
        Edge attribute key or expression to use as edge weights for sorting.
        Lower weights are preferred (treated as better matches).
        Can be a string key or AttrExpr for complex expressions.
    output_key : str, default DEFAULT_ATTR_KEYS.SOLUTION
        Attribute key to store the solution boolean values in nodes and edges.

    Attributes
    ----------
    max_children : int
        Maximum number of children per node.
    solution_key : str
        Key used to store solution results.
    edge_weight_expr : AttrExpr
        Expression used to compute edge weights.
    output_key : str
        The key to store the solution in the graph.
    reset : bool
        Whether to reset the solution values in the whole graph before solving.

    Examples
    --------
    Basic usage with default settings:

    >>> from tracksdata.solvers import NearestNeighborsSolver
    >>> solver = NearestNeighborsSolver()
    >>> solver.solve(graph)

    Customize maximum children for cell division tracking:

    >>> solver = NearestNeighborsSolver(max_children=3)
    >>> solver.solve(graph)

    Use custom edge weight expression:

    >>> from tracksdata.expr import AttrExpr
    >>> solver = NearestNeighborsSolver(
    ...     edge_weight=-AttrExpr("iou"),  # Higher IoU is better
    ...     max_children=2,
    ... )

    Combine multiple edge attributes:

    >>> weight_expr = AttrExpr("distance") + 0.5 * AttrExpr("color_diff")
    >>> solver = NearestNeighborsSolver(edge_weight=weight_expr)
    """

    def __init__(
        self,
        max_children: int = 2,
        edge_weight: str | ExprInput = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
        reset: bool = True,
    ):
        self.max_children = max_children
        self.solution_key = output_key
        self.edge_weight_expr = AttrExpr(edge_weight)
        self.reset = reset

    def solve(
        self,
        graph: BaseGraph,
    ) -> None:
        """
        Solve the tracking problem using nearest neighbor edge selection.

        Applies the nearest neighbor algorithm to find the optimal set of edges
        that form valid tracking paths while respecting parent-child relationship
        constraints. Automatically extends the graph schema to include the solution
        key if it doesn't already exist.

        Parameters
        ----------
        graph : BaseGraph
            The graph containing nodes and edges to solve. The graph will be
            modified in-place to add solution attributes to nodes and edges.

        Examples
        --------
        >>> solver = NearestNeighborsSolver(max_children=2)
        >>> solver.solve(graph)

        Access solution edges:

        >>> solution_edges = graph.edge_attrs().filter(pl.col("solution") == True)
        """
        # get edges and sort them by weight
        edges_df = graph.edge_attrs(attr_keys=self.edge_weight_expr.columns)
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

        if self.solution_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.solution_key, False)
        elif self.reset:
            graph.update_edge_attrs(attrs={self.solution_key: False})

        graph.update_edge_attrs(
            edge_ids=solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
            attrs={self.solution_key: True},
        )

        node_ids = np.unique(
            np.concatenate(
                [
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
                    solution_edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
                ]
            )
        )

        graph.add_node_attr_key(self.solution_key, False)
        graph.update_node_attrs(
            node_ids=node_ids,
            attrs={self.solution_key: True},
        )
