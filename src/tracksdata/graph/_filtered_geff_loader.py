"""Memory-efficient GEFF loader with pre-filtering support."""

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
    3. Loading only the filtered subset via GeffReader.build()

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
    >>> loader = FilteredGeffLoader(
    ...     "data.geff",
    ...     node_filters=[NodeAttr("t") >= 10, NodeAttr("t") < 20]
    ... )
    >>> graph, metadata = loader.load()

    Load specific properties only:

    >>> loader = FilteredGeffLoader(
    ...     "data.geff",
    ...     node_filters=[NodeAttr("t").is_in([5, 10, 15])]
    ... )
    >>> graph, metadata = loader.load(
    ...     node_props=["t", "x", "y"],
    ...     edge_props=["distance"]
    ... )
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

        # Read metadata to know available properties
        self.metadata = GeffMetadata.read(geff_store)

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

        # Read minimal properties needed for filtering
        filter_prop_names = self._get_required_node_props()
        reader = GeffReader(self.geff_store, validate=self.validate)
        reader.read_node_props(filter_prop_names)

        # Build DataFrame from zarr arrays (load to memory for filtering)
        temp_data = {}
        for prop in filter_prop_names:
            zarr_prop = reader.node_props[prop]
            prop_metadata = reader.metadata.node_props_metadata[prop]
            in_memory = reader._load_prop_to_memory(zarr_prop, None, prop_metadata)
            temp_data[prop] = in_memory["values"]

        df = pl.DataFrame(temp_data)

        # Apply filter conditions - use pl.col for expressions, not df[...]
        # polars_reduce_attr_comps needs Expr, not Series
        mask_expressions = []
        for attr_comp in self.node_filters:
            col_expr = pl.col(str(attr_comp.column))
            mask_expressions.append(attr_comp.op(col_expr, attr_comp.other))

        # Combine with AND logic
        mask_expr = pl.reduce(operator.and_, mask_expressions)
        mask = df.select(mask_expr).to_series().to_numpy()

        return mask

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

        # Use GeffReader with node mask
        # Note: GeffReader.build() automatically filters edges where both endpoints
        # are in the filtered node set
        reader = GeffReader(self.geff_store, validate=self.validate)
        reader.read_node_props(node_props)
        reader.read_edge_props(edge_props)

        # Build loads ONLY filtered data (nodes + incident edges)
        in_memory_geff = reader.build(node_mask=node_mask)

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
