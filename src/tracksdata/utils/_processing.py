from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def maybe_show_progress(
    iterator: Iterable[T],
    show_progress: bool,
    **tqdm_kwargs: Any,
) -> tqdm | Iterable[T]:
    """
    Wraps an iterable with a progress bar if show_progress is True.

    Parameters
    ----------
    iterator : Iterable[T]
        The iterable to wrap.
    show_progress : bool
        Whether to show the progress bar.
    tqdm_kwargs : Any
        Additional arguments to pass to tqdm.
        See tqdm documentation for more details.
        https://tqdm.github.io/docs/tqdm/

    Returns
    -------
    tqdm | Iterable[T]
        The wrapped iterable if show_progress is True, otherwise the original iterable.
    """

    if show_progress:
        iterator = tqdm(iterator, **tqdm_kwargs)

    return iterator
