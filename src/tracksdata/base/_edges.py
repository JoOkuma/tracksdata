import abc
from collections.abc import Sequence

from tracksdata._graph import BaseGraphBackend

DEFAULT_EDGE_WEIGHT_KEY = "weight"


class BaseEdgesOperator(abc.ABC):
    """
    Base class indicating methods required to insert edges into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    @abc.abstractmethod
    def __call__(
        self,
        graph: BaseGraphBackend,
        t: int | None = None,
        weight_key: Sequence[str] = (DEFAULT_EDGE_WEIGHT_KEY,),
    ) -> None:
        r"""
        Initialize the edges from nodes of given `time` to nodes in neighboring
        times (`time` + `\delta time`)

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to initialize the edges in.
        t: int
            The time of the nodes to initialize the edges from.
        weight_key: str
            The key to add the edge weights to.
        """
