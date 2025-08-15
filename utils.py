"""Shared utility functions for the information integration project."""
from typing import Callable, Optional


def make_status_printer(
    status_callback: Optional[Callable[[str], None]]
) -> Callable[[str], None]:
    """Return a status printing function.

    The returned function will forward messages to ``status_callback`` if it is
    provided. Otherwise, messages are printed to standard output.
    """

    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    return _status
