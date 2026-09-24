"""Memory-efficient GEFF loader with pre-filtering support."""

import copy
import operator
from typing import Any

import geff
import numpy as np
import polars as pl
from geff.core_io._base_read import GeffReader
from geff_spec import GeffMetadata
from numpy.typing import NDArray
from zarr.storage import StoreLike

from tracksdata.attrs import AttrComparison
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import IndexedRXGraph


class FilteredGeffLoader:
    """
    Load GEFF files with pre-filtering to reduce I/O and improve runtime.

    Filters GEFF data before loading into memory by:
    1. Reading only filter attribute zarr arrays to identify matching nodes
    2. Creating boolean masks for nodes and edges
    3. Loading only the filtered subset using integer-indexed zarr reads

    This is significantly more I/O and runtime efficient than loading the entire
    graph and then filtering, especially for large datasets.

    Parameters
    ----------
    geff_store : StoreLike
        Path or zarr store to the GEFF file
    node_filters : list[AttrComparison] | None
        Filter conditions for nodes (e.g., [NodeAttr("t") >= 10, NodeAttr("t") < 20]).
        Multiple filters are combined with AND logic. Edges are automatically included
        only when BOTH endpoints (source and target) are in the filtered node set.
    validate : bool
        Whether to validate GEFF structure on initialization

    Examples
    --------
    Load only nodes with t >= 10 and t < 20:

    >>> from tracksdata.graph import FilteredGeffLoader
    >>> from tracksdata.attrs import NodeAttr
    >>> loader = FilteredGeffLoader("data.geff", node_filters=[NodeAttr("t") >= 10, NodeAttr("t") < 20])
    >>> graph, metadata = loader.load()

    Load specific properties only:

    >>> loader = FilteredGeffLoader("data.geff", node_filters=[NodeAttr("t").is_in([5, 10, 15])])
    >>> graph, metadata = loader.load(node_props=["t", "x", "y"], edge_props=["distance"])
    """

    def __init__(
        self,
        geff_store: StoreLike,
        node_filters: list[AttrComparison] | None = None,
        validate: bool = True,
    ):
        """Initialize the filtered GEFF loader."""
        self.geff_store = geff_store
        self.validate = validate
        self.node_filters = node_filters or []

        # Create and cache a GeffReader (pay validation cost once)
        self._reader = GeffReader(geff_store, validate=validate)
        self.metadata = self._reader.metadata

        # Cache all zarr property references (lightweight, just zarr array refs)
        self._reader.read_node_props(None)
        self._reader.read_edge_props(None)

        # Cache node IDs and edge IDs in memory (small arrays, always needed)
        self._node_ids: NDArray = np.asarray(self._reader.nodes[:])
        self._edge_ids: NDArray = np.asarray(self._reader.edges[:])

    def _get_required_node_props(self) -> set[str]:
        """Extract property names needed for node filtering."""
        return {comp.column for comp in self.node_filters}

    def _create_node_mask(self) -> NDArray[np.bool_] | None:
        """
        Create boolean mask by evaluating filters on zarr arrays.

        Strategy:
        1. Identify which node properties are needed for filtering
        2. Load ONLY those properties from zarr (not entire graph)
        3. Apply filter conditions using polars to create mask
        4. Return boolean array of length num_nodes

        Returns
        -------
        NDArray[np.bool_] | None
            Boolean mask for nodes, or None if no filters
        """
        if not self.node_filters:
            return None

        # Read minimal properties needed for filtering using cached zarr refs
        filter_prop_names = self._get_required_node_props()

        temp_data = {}
        for prop in filter_prop_names:
            zarr_prop = self._reader.node_props[prop]
            prop_metadata = self._reader.metadata.node_props_metadata[prop]
            in_memory = self._reader._load_prop_to_memory(zarr_prop, None, prop_metadata)
            temp_data[prop] = in_memory["values"]

        df = pl.DataFrame(temp_data)

        # Apply filter conditions
        mask_expressions = []
        for attr_comp in self.node_filters:
            col_expr = pl.col(str(attr_comp.column))
            mask_expressions.append(attr_comp.op(col_expr, attr_comp.other))

        mask_expr = pl.reduce(operator.and_, mask_expressions)
        mask = df.select(mask_expr).to_series().to_numpy()

        return mask

    @staticmethod
    def _load_zarr_by_indices(
        zarr_arr: Any,
        indices: NDArray[np.intp] | None,
        dtype: type | None = None,
    ) -> NDArray:
        """Load zarr array subset using integer indices (faster than bool masks for sparse selections)."""
        if indices is None:
            data = zarr_arr[:]
        elif len(indices) == 0:
            shape = (0, *zarr_arr.shape[1:])
            return np.empty(shape, dtype=dtype or zarr_arr.dtype)
        else:
            data = zarr_arr.oindex[indices]
        return np.asarray(data, dtype=dtype) if dtype else np.asarray(data)

    def _build_filtered(
        self,
        node_mask: NDArray[np.bool_] | None,
        node_props_names: list[str],
        edge_props_names: list[str],
    ) -> dict[str, Any]:
        """
        Build in-memory GEFF dict using integer-indexed zarr reads.

        Uses integer indices instead of boolean masks for zarr access,
        which is significantly faster for sparse selections.
        """
        reader = self._reader

        if node_mask is not None:
            node_indices = np.where(node_mask)[0]
            nodes = self._node_ids[node_mask]
        else:
            node_indices = None
            nodes = self._node_ids

        # Load node props with integer indices (fast zarr access)
        node_props: dict[str, dict] = {}
        for name in node_props_names:
            zarr_prop = reader.node_props[name]
            prop_metadata = reader.metadata.node_props_metadata[name]
            if prop_metadata.varlength:
                # varlength needs special deserialization via _load_prop_to_memory
                node_props[name] = reader._load_prop_to_memory(zarr_prop, node_mask, prop_metadata)
            else:
                values = self._load_zarr_by_indices(zarr_prop["values"], node_indices)
                missing = None
                if "missing" in zarr_prop:
                    missing = self._load_zarr_by_indices(zarr_prop["missing"], node_indices, dtype=bool)
                node_props[name] = {"values": values, "missing": missing}

        # Filter edges using cached edge IDs
        edges = self._edge_ids
        if node_mask is not None and len(edges) > 0 and len(nodes) > 0:
            # Lookup array for fast edge filtering
            max_id = int(max(edges.max(), nodes.max())) + 1
            lookup = np.zeros(max_id, dtype=bool)
            lookup[nodes] = True
            edge_mask = lookup[edges[:, 0]] & lookup[edges[:, 1]]
            edge_indices = np.where(edge_mask)[0]
            edges = edges[edge_mask]
        elif node_mask is not None:
            # No nodes or no edges: empty result
            edge_mask = None
            edge_indices = np.array([], dtype=np.intp)
            edges = np.empty((0, 2), dtype=self._edge_ids.dtype)
        else:
            edge_mask = None
            edge_indices = None

        # Load edge props with integer indices
        edge_props: dict[str, dict] = {}
        for name in edge_props_names:
            zarr_prop = reader.edge_props[name]
            prop_metadata = reader.metadata.edge_props_metadata[name]
            if prop_metadata.varlength:
                edge_props[name] = reader._load_prop_to_memory(zarr_prop, edge_mask, prop_metadata)
            else:
                values = self._load_zarr_by_indices(zarr_prop["values"], edge_indices)
                missing = None
                if "missing" in zarr_prop:
                    missing = self._load_zarr_by_indices(zarr_prop["missing"], edge_indices, dtype=bool)
                edge_props[name] = {"values": values, "missing": missing}

        # Clean metadata: remove properties not loaded
        output_metadata = copy.deepcopy(reader.metadata)
        node_props_set = set(node_props_names)
        edge_props_set = set(edge_props_names)
        for prop in list(output_metadata.node_props_metadata.keys()):
            if prop not in node_props_set:
                del output_metadata.node_props_metadata[prop]
        for prop in list(output_metadata.edge_props_metadata.keys()):
            if prop not in edge_props_set:
                del output_metadata.edge_props_metadata[prop]

        return {
            "metadata": output_metadata,
            "node_ids": nodes,
            "node_props": node_props,
            "edge_ids": edges,
            "edge_props": edge_props,
        }

    def load(
        self,
        graph_class: type[BaseGraph] = IndexedRXGraph,
        node_props: list[str] | None = None,
        edge_props: list[str] | None = None,
        node_attr_key_map: dict[str, str] | None = None,
        edge_attr_key_map: dict[str, str] | None = None,
        **graph_kwargs: Any,
    ) -> tuple[BaseGraph, GeffMetadata]:
        """
        Load filtered GEFF data into a graph.

        Parameters
        ----------
        graph_class : type[BaseGraph]
            Graph class to instantiate (default: IndexedRXGraph)
        node_props : list[str] | None
            Subset of node properties to load (None = all)
        edge_props : list[str] | None
            Subset of edge properties to load (None = all)
        node_attr_key_map : dict[str, str] | None
            Rename node attributes during loading
        edge_attr_key_map : dict[str, str] | None
            Rename edge attributes during loading
        **graph_kwargs
            Additional arguments passed to graph constructor

        Returns
        -------
        tuple[BaseGraph, GeffMetadata]
            Loaded graph and metadata
        """
        # Create node mask BEFORE loading data
        node_mask = self._create_node_mask()

        # Resolve property names (preserve order for deterministic column ordering)
        node_props_names = list(node_props) if node_props is not None else list(self._reader.node_prop_names)
        edge_props_names = list(edge_props) if edge_props is not None else list(self._reader.edge_prop_names)

        # Build filtered data using fast integer-indexed reads
        in_memory_geff = self._build_filtered(node_mask, node_props_names, edge_props_names)

        # Construct graph using geff backend
        rx_graph = geff.construct(
            in_memory_geff["metadata"],
            in_memory_geff["node_ids"],
            in_memory_geff["edge_ids"],
            in_memory_geff["node_props"],
            in_memory_geff["edge_props"],
            backend="rustworkx",
        )

        # Apply attribute key mapping if provided
        if node_attr_key_map is not None:
            for src_k, dst_k in node_attr_key_map.items():
                in_memory_geff["metadata"].node_props_metadata[dst_k] = in_memory_geff[
                    "metadata"
                ].node_props_metadata.pop(src_k)
                for node_attr in rx_graph.nodes():
                    node_attr[dst_k] = node_attr.pop(src_k)

        if edge_attr_key_map is not None:
            for src_k, dst_k in edge_attr_key_map.items():
                in_memory_geff["metadata"].edge_props_metadata[dst_k] = in_memory_geff[
                    "metadata"
                ].edge_props_metadata.pop(src_k)
                for edge_attr in rx_graph.edges():
                    edge_attr[dst_k] = edge_attr.pop(src_k)

        # Wrap in requested graph class
        node_id_map = rx_graph.attrs.get("to_rx_id_map", {})

        if graph_class == IndexedRXGraph:
            graph = IndexedRXGraph(rx_graph=rx_graph, node_id_map=node_id_map)
        else:
            # Create IndexedRXGraph first, then convert to target class
            indexed_graph = IndexedRXGraph(rx_graph=rx_graph, node_id_map=node_id_map)
            graph = graph_class.from_other(indexed_graph, **graph_kwargs)

        return graph, in_memory_geff["metadata"]
