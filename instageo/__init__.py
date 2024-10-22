# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

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
