from collections.abc import Callable
from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike
from skimage.measure._regionprops import RegionProperties, regionprops

from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.utils._processing import maybe_show_progress


class RegionPropsOperator(BaseNodesOperator):
    """Operator that adds nodes to a graph using scikit-image's regionprops."""

    def __init__(
        self,
        cache: bool = True,
        extra_properties: list[str | Callable[[RegionProperties], Any]] | None = None,
        spacing: tuple[float, float] | None = None,
        show_progress: bool = True,
    ):
        super().__init__()
        self._cache = cache
        self._extra_properties = extra_properties or []
        self._spacing = spacing
        self._show_progress = show_progress

    @override
    # pylint: disable=arguments-differ
    def add_nodes(
        self,
        graph: BaseGraphBackend,
        *,
        labels: ArrayLike,
        t: int | None = None,
        intensity_image: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the nodes in the provided graph using skimage's region properties.
        If t is None, the labels are considered to be a timelapse where axis=0 is time.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to initialize the nodes in.
        labels : ArrayLike
            The labels of the nodes to be initialized.
        t : int | None
            The time at which to initialize the nodes.
            If None, labels are considered to be a timelapse where axis=0 is time.
        intensity_image : ArrayLike | None
            The intensity image to use for the region properties.
            If None, the intensity image is not used.
        """
        if t is None:
            for t in maybe_show_progress(
                range(labels.shape[0]),
                desc="Processing time points",
                show_progress=self._show_progress,
            ):
                if intensity_image is not None:
                    self.add_nodes(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                        intensity_image=intensity_image[t],
                    )
                else:
                    self.add_nodes(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                    )
            return

        if labels.ndim == 2:
            axis_names = ["y", "x"]
        elif labels.ndim == 3:
            axis_names = ["z", "y", "x"]
        else:
            raise ValueError(
                f"`labels` must be 2D or 3D, got {labels.ndim} dimensions."
            )

        labels = np.asarray(labels)

        for obj in maybe_show_progress(
            list(
                regionprops(
                    labels,
                    intensity_image=intensity_image,
                    spacing=self._spacing,
                )
            ),
            show_progress=self._show_progress,
            desc=f"Processing regions of time {t}",
        ):
            attributes = dict(zip(axis_names, obj.centroid, strict=False))

            for prop in self._extra_properties:
                if callable(prop):
                    attributes[prop.__name__] = prop(obj)
                else:
                    attributes[prop] = getattr(obj, prop)

            graph.add_node(
                t=t,
                mask=obj.image,
                bbox=obj.bbox,
                **attributes,
            )
