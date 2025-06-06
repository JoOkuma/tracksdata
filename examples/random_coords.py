from profilehooks import profile

from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.expr import AttrExpr
from tracksdata.graph._sql_graph import SQLGraph
from tracksdata.nodes._random import RandomNodes
from tracksdata.solvers._nearest_neighbors_solver import NearestNeighborsSolver


@profile(immediate=True, sort="cumulative")
def main() -> None:
    n_nodes = 200
    nodes_operator = RandomNodes(
        n_time_points=50,
        n_nodes_per_tp=(int(n_nodes * 0.95), int(n_nodes * 1.05)),
        n_dim=3,
        show_progress=False,
    )
    dist_operator = DistanceEdges(distance_threshold=30.0, n_neighbors=5, show_progress=False)

    solver = NearestNeighborsSolver(
        edge_weight=AttrExpr("weight"),
        max_children=2,
    )

    # graph = RustWorkXGraph()
    graph = SQLGraph(drivername="sqlite", database=":memory:")
    nodes_operator.add_nodes(graph)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")

    solver.solve(graph)


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start} seconds")
