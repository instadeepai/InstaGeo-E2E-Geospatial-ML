"""InstaGeo: A Geospatial Data Analysis Library.

This package provides tools for processing and analyzing geospatial data,
with components for data handling (instageo.data), modeling (instageo.model),
and applications (instageo.apps).
"""

from __future__ import annotations

import pathlib

# Package information
__version__ = "0.0.1"

# Package paths
INSTAGEO_PATH = pathlib.Path(__file__).parent.resolve()
"""Path to InstaGeo package."""

INSTAGEO_APPS_PATH = INSTAGEO_PATH / "apps"
"""Path to the InstaGeo apps sub-package."""

INSTAGEO_DATA_PATH = INSTAGEO_PATH / "data"
"""Path to the InstaGeo data sub-package."""

INSTAGEO_MODEL_PATH = INSTAGEO_PATH / "model"
"""Path to the InstaGeo model sub-package."""

INSTAGEO_PARENT_PATH = INSTAGEO_PATH.parent
"""Path to the parent directory containing InstaGeo."""

INSTAGEO_TEST_PATH = INSTAGEO_PARENT_PATH / "tests"
"""Path to InstaGeo test directory."""
