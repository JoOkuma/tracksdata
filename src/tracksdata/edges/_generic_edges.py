from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._base_edge_attrs import BaseEdgeAttrsOperator
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG


class GenericNodeFunctionEdgeAttrs(BaseEdgeAttrsOperator):
    """
    Add weights to the edges of the graph based on the output of a function.

    When provided multiple attribute keys, the function should take a dict
    with the keys as values for each node.

    When provided a single attribute key, the function should take the value
    for each node.

    For example, if the function is ``func(source_attr, target_attr)``,
    and the attribute keys are ``["a", "b"]``, then the function should be
    ``func({"a": 1, "b": 2}, {"a": 3, "b": 4})``.

    For a single attribute key "a", the function should take a single value
    for each node, as ``func(1, 3)``.

    Parameters
    ----------
    func : Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any]
        The function to apply to the source and target attributes.
    attr_keys : Sequence[str] | str
        The keys of the attributes to pass to the function.
    output_key : str
        The key to store the output of the function.
    show_progress : bool
        Whether to show a progress bar.
    """

    def __init__(
        self,
        func: Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any],
        attr_keys: Sequence[str] | str,
        output_key: str,
        show_progress: bool = True,
    ) -> None:
        super().__init__(output_key=output_key, show_progress=show_progress)
        self.attr_keys = attr_keys
        self.func = func

    def _add_edge_attrs_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
    ) -> None:
        """
        Add weights to the edges of the graph based on the output of a function
        for a specific time point.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add weights to.
        t : int
            The time point to add weights for.
        """
        source_ids = graph.filter_nodes_by_attrs({DEFAULT_ATTR_KEYS.T: t})
        edges_df = graph.edge_attrs(node_ids=source_ids, include_targets=True)

        if len(edges_df) == 0:
            LOG.warning(f"No edges found for time point {t} to sucessors")
            return

        source_df = graph.node_attrs(
            node_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
            attr_keys=self.attr_keys,
        )
        target_df = graph.node_attrs(
            node_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
            attr_keys=self.attr_keys,
        )

        weights = np.zeros(len(edges_df), dtype=np.float32)

        if isinstance(self.attr_keys, str):
            # faster than creating a dict
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.attr_keys],
                    target_df[self.attr_keys],
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)
        else:
            # a bit more expensive to create a dict but more flexible
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.attr_keys].iter_rows(named=True),
                    target_df[self.attr_keys].iter_rows(named=True),
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)

        if self.output_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.output_key, -99999.0)

        graph.update_edge_attrs(
            edge_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
            attrs={self.output_key: weights},
        )
