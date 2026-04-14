"""Compatibility shim for legacy imports.

New code should import from ``utils.placecell_core``.
"""

from utils.placecell_core import *  # noqa: F401,F403
from utils.placecell_core import _compute_moving_epochs, _compute_quiet_epochs  # noqa: F401
