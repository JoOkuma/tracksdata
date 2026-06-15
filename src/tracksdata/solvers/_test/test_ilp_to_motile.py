import polars as pl
import pytest

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.solvers import ILPSolver

motile = pytest.importorskip("motile")

from motile.variables import EdgeSelected, NodeSelected  # noqa: E402


def _selected(graph: RustWorkXGraph, solver: ILPSolver) -> tuple[set[int], set[tuple[int, int]]]:
    """Solve with motile and return selected node ids / (source, target) edges."""
    motile_solver = solver.to_motile_solver(graph)
    solution = motile_solver.solve()

    node_indicators = motile_solver.get_variables(NodeSelected)
    edge_indicators = motile_solver.get_variables(EdgeSelected)

    nodes = {n for n, idx in node_indicators.items() if solution[idx] > 0.5}
    edges = {e for e, idx in edge_indicators.items() if solution[idx] > 0.5}
    return nodes, edges


def _build_chain() -> tuple[RustWorkXGraph, list[int], list[int]]:
    graph = RustWorkXGraph()
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    n0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0})
    n1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0})
    n2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, "x": 2.0})

    e0 = graph.add_edge(n0, n1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})
    e1 = graph.add_edge(n1, n2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})
    return graph, [n0, n1, n2], [e0, e1]


def test_to_motile_solver_returns_solver() -> None:
    graph, _, _ = _build_chain()
    solver = ILPSolver(edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST)
    motile_solver = solver.to_motile_solver(graph)
    assert isinstance(motile_solver, motile.Solver)


def test_to_motile_solver_selects_full_chain() -> None:
    graph, node_ids, _ = _build_chain()
    solver = ILPSolver(edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST)

    nodes, edges = _selected(graph, solver)

    assert nodes == set(node_ids)
    assert edges == {(node_ids[0], node_ids[1]), (node_ids[1], node_ids[2])}


def test_to_motile_solver_matches_ilp_solution() -> None:
    """The motile solver should select the same edges as the native ILP solver."""
    graph, _, _ = _build_chain()
    solver = ILPSolver(edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST, appearance_weight=0.5)

    _, motile_edges = _selected(graph, solver)

    solution_graph = solver.solve(graph)
    ilp_edges = {
        (row[DEFAULT_ATTR_KEYS.EDGE_SOURCE], row[DEFAULT_ATTR_KEYS.EDGE_TARGET])
        for row in solution_graph.edge_attrs().iter_rows(named=True)
    }

    assert motile_edges == ilp_edges


def test_to_motile_solver_division() -> None:
    """A node with two children is allowed (division) at low division cost."""
    graph = RustWorkXGraph()
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    n0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0})
    n1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1})
    n2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1})
    graph.add_edge(n0, n1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})
    graph.add_edge(n0, n2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})

    solver = ILPSolver(edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST, division_weight=0.0)
    nodes, edges = _selected(graph, solver)

    assert nodes == {n0, n1, n2}
    assert edges == {(n0, n1), (n0, n2)}


def test_to_motile_solver_merge_pins_max_parents() -> None:
    graph, _, _ = _build_chain()
    solver = ILPSolver(edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST, merge_weight=1.0)
    motile_solver = solver.to_motile_solver(graph)
    # smoke test: a merge-enabled solver builds and solves
    motile_solver.solve()


def test_to_motile_solver_rejects_indicator_pins() -> None:
    graph, _, _ = _build_chain()
    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
        appearance_weight=(NodeAttr(DEFAULT_ATTR_KEYS.T) > 0) * float("inf"),
    )
    # appearance pins map to indicator variables motile cannot pin
    with pytest.raises(NotImplementedError):
        solver.to_motile_solver(graph)
