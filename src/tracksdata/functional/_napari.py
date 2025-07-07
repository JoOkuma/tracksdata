import polars as pl

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph


def to_napari_format(
    graph: BaseGraph,
    shape: tuple[int, ...],
    solution_key: str | None = DEFAULT_ATTR_KEYS.SOLUTION,
    output_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
) -> (
    tuple[
        GraphArrayView,
        pl.DataFrame,
        dict[int, int],
    ]
    | tuple[
        pl.DataFrame,
        dict[int, int],
    ]
):
    """
    Convert the subgraph of solution nodes to a napari-ready format.

    This includes:
    - a labels layer with the solution nodes IF masks attribute exists.
    - a tracks layer with the solution tracks
    - a graph with the parent-child relationships for the solution tracks

    IMPORTANT: This function will reset the track ids if they already exist.

    Parameters
    ----------
    graph : BaseGraph
        The graph to convert.
    shape : tuple[int, ...]
        The shape of the labels layer.
    solution_key : str | None, optional
        The key of the solution attribute.
        If None, it assumes the graph is already a directed acyclic graph.
    output_track_id_key : str, optional
        The key of the output track id attribute.

    Returns
    -------
    tuple[GraphArrayView, pl.DataFrame, dict[int, int]] | tuple[pl.DataFrame, dict[int, int]]
        - array_view: The array view of the solution graph if 'masks' are present, otherwise None.
        - tracks_data: The tracks data as a polars DataFrame.
        - dict_graph: A dictionary of parent -> child relationships.
    """
    if solution_key is not None:
        solution_graph = graph.subgraph(NodeAttr(solution_key) == True, EdgeAttr(solution_key) == True)
    else:
        solution_graph = graph

    tracks_graph = solution_graph.assign_track_ids(output_track_id_key)
    dict_graph = {child: parent for parent, child in tracks_graph.edge_list()}

    spatial_cols = ["z", "y", "x"][-len(shape) + 1 :]

    tracks_data = solution_graph.node_attrs(
        attr_keys=[output_track_id_key, DEFAULT_ATTR_KEYS.T, *spatial_cols],
    )
    # sorting columns
    tracks_data = tracks_data.select([output_track_id_key, DEFAULT_ATTR_KEYS.T, *spatial_cols])

    if DEFAULT_ATTR_KEYS.MASK not in solution_graph.node_attr_keys:
        return tracks_data, dict_graph

    array_view = GraphArrayView(
        solution_graph,
        shape,
        attr_key=output_track_id_key,
    )

    return array_view, tracks_data, dict_graph
