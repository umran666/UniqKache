"""Logging helpers.

UniqKache deliberately does **not** ship a global "silence everything" switch.
Diagnostics that indicate a correctness risk are emitted at ``WARNING`` or
above, so that a benchmark run cannot quietly discard a problem.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT_NAME = "uniqkache"
_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def _configure_root() -> None:
    """Attach a single stderr handler to the package root logger, once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)

    level_name = os.environ.get("UNIQKACHE_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    # Library code must not hijack the application's root logger.
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``uniqkache`` root.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.
    """
    _configure_root()
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


__all__ = ["get_logger"]
