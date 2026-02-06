"""
Local utilities.

Keep this package import-light so scripts like `utils/DS_motion_correction.py` can
import `utils.minivi_io` in minimal Python environments (e.g., older cluster
Python) without pulling in optional dependencies.
"""

__all__ = []
