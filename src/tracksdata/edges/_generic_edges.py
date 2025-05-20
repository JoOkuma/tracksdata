from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.utils._logging import LOG
from tracksdata.utils._processing import maybe_show_progress


class GenericFunctionEdgesOperator:
    """
    Add weights to the edges of the graph based on the output of a function.

    When provided multiple feature keys, the function should take a dict
    with the keys as values for each node.

    When provided a single feature key, the function should take the value
    for each node.

    For example, if the function is ``func(source_attr, target_attr)``,
    and the feature keys are ``["a", "b"]``, then the function should be
    ``func({"a": 1, "b": 2}, {"a": 3, "b": 4})``.

    For a single feature key "a", the function should take a single value
    for each node, as ``func(1, 3)``.

    Parameters
    ----------
    func : Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any]
        The function to apply to the source and target attributes.
    feature_keys : Sequence[str] | str
        The keys of the attributes to pass to the function.
    output_key : str
        The key to store the output of the function.
    show_progress : bool
        Whether to show a progress bar.
    """

    def __init__(
        self,
        func: Callable[[dict[str, Any] | Any, dict[str, Any] | Any], Any],
        feature_keys: Sequence[str] | str,
        output_key: str,
        show_progress: bool = True,
    ) -> None:
        self.feature_keys = feature_keys

        self.func = func
        self.show_progress = show_progress
        self.output_key = output_key

    def add_weights(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
    ) -> None:
        """
        Add weights to the edges of the graph based on the output of a function.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to add weights to.
        t : int | None
            The time point to add weights to.
        """
        if t is None:
            for t in maybe_show_progress(
                graph.time_points(),
                desc="Adding weights to edges",
                show_progress=self.show_progress,
            ):
                self.add_weights(graph, t=t)
            return

        source_ids = graph.filter_nodes_by_attribute({DEFAULT_ATTR_KEYS.T: t})
        edges_df = graph.edge_features(node_ids=source_ids)

        if len(edges_df) == 0:
            LOG.warning(f"No edges found for time point {t} to sucessors")
            return

        source_df = graph.node_features(
            node_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
            feature_keys=self.feature_keys,
        )
        target_df = graph.node_features(
            node_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
            feature_keys=self.feature_keys,
        )

        weights = np.zeros(len(edges_df), dtype=np.float32)

        if isinstance(self.feature_keys, str):
            # faster than creating a dict
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.feature_keys],
                    target_df[self.feature_keys],
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)
        else:
            # a bit more expensive to create a dict but more flexible
            for i, (source_attr, target_attr) in enumerate(
                zip(
                    source_df[self.feature_keys].iter_rows(named=True),
                    target_df[self.feature_keys].iter_rows(named=True),
                    strict=True,
                )
            ):
                weights[i] = self.func(source_attr, target_attr)

        if self.output_key not in graph.edge_features_keys:
            graph.add_edge_feature_key(self.output_key, -1.0)

        graph.update_edge_features(
            edge_ids=edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
            attributes={self.output_key: weights},
        )
