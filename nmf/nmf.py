"""Main library file for nmf."""

import logging

log = logging.getLogger(__name__)


def is_positive(x: int) -> bool:
    """
    Check that an integer is positive.

    Parameters
    ----------
    x : int
        The integer to check.

    Returns
    -------
    result: bool
        True if integer >= 0.

    """
    log.info("Checking integer {}".format(x))
    result = (x > 0)
    return result
