
```python
def track(
    graph: BaseGraphBackend,
    nodes_init: BaseNodesOperator,
    edges_init: BaseEdgesOperator,
    solver: BaseSolver,
    solution_key: str | None,
) -> BaseGraphBackend:

    nodes_init.init_nodes(graph, ...)
    edges_init.init_edges(graph, ...)
    solver.solve(graph, solution_key)

    return graph

nodes_with_overlap = NodesWithOverlapOperator()
# custom usecase
nodes_with_overlap.init_nodes(graph, ...)
edges_init.init_edges(graph, ...)

apply_network_edge_prediction(graph, ...)

solver.solve(graph, solution_key)


class TracksCurationWidget:
    def __init__(graph: BaseWritableGraph):
        pass


def plot_tracks(graph: BaseReadOnlyGraph):
    pass

```
