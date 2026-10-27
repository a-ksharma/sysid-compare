"""Pytest test session configuration.

Sets MPLCONFIGDIR to a temporary directory to avoid matplotlib permission
delays in restricted environments.
"""

from __future__ import annotations

import os
import tempfile

mpl_temp_dir = os.path.join(tempfile.gettempdir(), "sysid_mpl")
os.makedirs(mpl_temp_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_temp_dir)
