import abc
from typing import Any

from tracksdata.graph._base_graph import BaseGraphBackend


class BaseNodesOperator(abc.ABC):
    """
    Base class indicating methods required to insert nodes into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    @abc.abstractmethod
    def add_nodes(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the nodes for a given time.
        """
