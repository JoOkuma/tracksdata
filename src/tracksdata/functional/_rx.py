import numpy as np
import polars as pl
import rustworkx as rx
from numba import njit, typed, types
from numpy.typing import NDArray

from tracksdata.utils._logging import LOG

NO_PARENT = -1


@njit
def _fast_path_transverse(
    node: int,
    track_id: int,
    queue: list[tuple[int, int]],
    idx_to_node: NDArray[np.int64],
    indptr: NDArray[np.int64],
    child_idx: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Transverse a path in the forest directed graph and add path (track) split into queue.

    Parameters
    ----------
    node : int
        Source path node.
    track_id : int
        Reference track id for path split.
    queue : list[tuple[int, int]]
        Source nodes and path (track) id reference queue.
    idx_to_node : np.ndarray
        Mapping of node indices to their respective node id.
    indptr : np.ndarray
        Indices of the children for each parent.
    child_idx : np.ndarray
        Indices of the children for each parent.

    Returns
    -------
    list[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    for _ in range(len(idx_to_node)):  # better than while True, to avoid infinite loop
        path.append(idx_to_node[node])

        start = indptr[node]
        end = indptr[node + 1]
        n_children = end - start

        if n_children == 0:
            # end of track
            break

        elif n_children == 1:
            node = child_idx[start]
        elif n_children == 2:
            for child in child_idx[start:end]:
                queue.append((child, track_id))
            break
        else:
            raise ValueError(f"Node {node} ({idx_to_node[node]}) has {n_children} children, expected 0 or 1")

    return path


@njit
def _fast_dag_transverse(
    idx_to_node: NDArray[np.int64],
    indptr: NDArray[np.int64],
    child_idx: NDArray[np.int64],
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    """
    Transverse the tracks DAG creating a distinct id to each path.

    Parameters
    ----------
    idx_to_node : np.ndarray
        Mapping of node indices to their respective node id.
    indptr : np.ndarray
        Indices of the children for each parent.
    child_idx : np.ndarray
        Indices of the children for each parent.

    Returns
    -------
    tuple[list[list[int]], list[int], list[int], list[int]]
        Sequence of paths, their respective track_id, parent_track_id and length.
    """
    track_id = 1
    paths = []
    track_ids = []  # equivalent to arange
    parent_track_ids = []
    lengths = []

    for root in child_idx[indptr[0] : indptr[1]]:  # roots are the first nodes in the graph
        queue = [(root, NO_PARENT)]

        while queue:
            node, parent_track_id = queue.pop()
            path = _fast_path_transverse(node, track_id, queue, idx_to_node, indptr, child_idx)
            paths.append(path)
            track_ids.append(track_id)
            parent_track_ids.append(parent_track_id)
            lengths.append(len(path))
            track_id += 1

    return paths, track_ids, parent_track_ids, lengths


@njit
def _numba_dag(node_ids: np.ndarray, parent_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a dict DAG of track lineages

    Parameters
    ----------
    node_ids : np.ndarray
        Nodes indices.
    parent_ids : np.ndarray
        Parent indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - idx_to_node: Mapping of node indices to their respective node id.
        - indptr: Indices of the children for each parent.
        - child_idx: Indices of the children for each parent.
    """
    dag = {}

    idx_to_node = np.zeros(len(node_ids) + 1, dtype=np.int64)
    node_to_idx = {}

    for parent in parent_ids:
        dag[parent] = typed.List.empty_list(types.int64)

    idx_to_node[0] = NO_PARENT

    n_edges = 0
    for i, node in enumerate(node_ids):
        idx_to_node[i + 1] = node
        node_to_idx[node] = i + 1
        dag[parent_ids[i]].append(node)
        n_edges += 1

    indptr = np.zeros(len(node_ids) + 2, dtype=np.int64)
    child_idx = np.zeros(n_edges, dtype=np.int64)
    current_edge = 0

    for i, node_id in enumerate(idx_to_node):
        indptr[i] = current_edge
        if node_id in dag:
            for child in dag[node_id]:
                child_idx[current_edge] = node_to_idx[child]
                current_edge += 1

    indptr[-1] = current_edge

    return idx_to_node, indptr, child_idx


def _rx_graph_to_dict_dag(graph: rx.PyDiGraph) -> dict[int, list[int]]:
    """Creates the DAG of track lineages

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of nodes.

    Returns
    -------
    dict[int, list[int]]
        DAG where parent maps to their children (parent -> children)
    """
    # target are the children
    # source are the parents
    node_indices = np.asarray(graph.node_indices(), dtype=np.int64)
    graph_df = pl.DataFrame({"target": node_indices})
    edge_list = pl.from_numpy(
        np.asarray(graph.edge_list(), dtype=np.int64),
        schema=["source", "target"],
    )
    try:
        graph_df = (
            graph_df.join(edge_list, on="target", how="left", validate="1:1")
            .with_columns(pl.col("source").fill_null(NO_PARENT))
            .select(pl.col("target"), pl.col("source"))
            .to_numpy(order="fortran")
            .T
        )
    except pl.exceptions.ComputeError as e:
        if "join keys did not fulfill 1:1" in str(e):
            raise RuntimeError("Invalid graph structure, found node with multiple parents") from e
        else:
            raise e

    # above we convert to numpy representation and then create numba dict
    # inside a njit function, otherwise it's very slow
    forest = _numba_dag(graph_df[0], graph_df[1])

    return forest


def graph_track_ids(
    graph: rx.PyDiGraph,
) -> tuple[np.ndarray, np.ndarray, rx.PyDiGraph]:
    """
    Assigns an unique `track_id` to each simple path in the graph and
    their respective parent -> child relationships.

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of tracks.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, rx.PyDiGraph]
        - node_ids: Sequence of node ids.
        - track_ids: The respective track_id for each node.
        - tracks_graph: Graph indicating the parent -> child relationships.
    """
    if graph.num_nodes() == 0:
        raise ValueError("Graph is empty")

    LOG.info(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    # was it better (faster) when using a numpy array for the digraph as in ultrack?
    idx_to_node, indptr, child_idx = _rx_graph_to_dict_dag(graph)

    paths, track_ids, parent_track_ids, lengths = _fast_dag_transverse(idx_to_node, indptr, child_idx)

    n_tracks = len(track_ids)

    tracks_graph = rx.PyDiGraph(node_count_hint=n_tracks, edge_count_hint=n_tracks)
    tracks_graph.add_nodes_from([None] * (n_tracks + 1))
    tracks_graph.add_edges_from_no_data(
        [(p, c) for p, c in zip(parent_track_ids, track_ids, strict=True) if p != NO_PARENT]
    )

    paths = np.concatenate(paths)
    nodes_track_ids = np.repeat(track_ids, lengths)

    return paths, nodes_track_ids, tracks_graph
