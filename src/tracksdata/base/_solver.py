import abc

from tracksdata._graph import BaseGraphBackend

DEFAULT_SOLUTION_KEY = "solution"


class BaseSolver(abc.ABC):
    """
    Base class for a solver tracking problem solver.
    It should return a subset of nodes and edges that are part of the solution.
    This class will interact with a `BaseGraphBackend` to do so.
    """

    @abc.abstractmethod
    def __call__(
        self, graph: BaseGraphBackend, solution_key: str = DEFAULT_SOLUTION_KEY
    ) -> None:
        """
        Solve the tracking problem and add the result to the graph with
        key `solution_key`.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to solve the tracking problem on.
        solution_key: str
            The edge and node attributes key to add the solution to.
        """
