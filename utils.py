"""Shared utility functions for the information integration project."""
from typing import Callable, Optional


def make_status_printer(
    status_callback: Optional[Callable[[str], None]]
) -> Callable[[str], None]:
    """Create a callable for emitting status messages.

    Parameters
    ----------
    status_callback : callable, optional
        Callback that receives status messages. If ``None`` messages are printed
        to standard output.

    Returns
    -------
    Callable[[str], None]
        Function accepting a message string.
    """

    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    return _status
