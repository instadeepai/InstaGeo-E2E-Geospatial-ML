"""Centralized flag definitions for InstaGeo."""

from absl import flags

from instageo.data.data_pipeline import MASK_DECODING_POS

# Define all shared flags here
flags.DEFINE_integer("chip_size", 256, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_float(
    "spatial_resolution",
    0.0002694945852358564,
    "Spatial Resolution in the specified CRS",
)
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "num_steps",
    3,
    """Number of temporal steps. When `is_time_series_task` is set to True, an attempt
    will be made to retrieve `num_steps` chips prior to the observation date.
    Otherwise, the value of `num_steps` will default to 1 and an attempt will be made to retrieve
    the chip corresponding to the observation date.
    """,
    lower_bound=1,
)
flags.DEFINE_integer(
    "temporal_step",
    30,
    """Temporal step size. When dealing with a time series task, an attempt will be made to
    fetch the data up to `temporal_step` days away from the date of observation. A tolerance might
    be applied when fetching the data for the different time steps.""",
)
flags.DEFINE_integer(
    "temporal_tolerance", 5, "Tolerance used when searching for the closest tile"
)
flags.DEFINE_integer(
    "temporal_tolerance_minutes", 0, """Additional tolerance in minutes"""
)
flags.DEFINE_enum("data_source", "HLS", ["HLS", "S2", "S1"], "Data source to use.")
flags.DEFINE_integer(
    "cloud_coverage",
    10,
    "Percentage of cloud cover to use. Accepted values are between 0.0001 and 100.",
    lower_bound=0,
    upper_bound=100,
)
flags.DEFINE_integer(
    "window_size",
    0,
    """Size of the window defined around the observation pixel. For instance, a value of 1 means
    that the label of the observation will be assigned to a 3x3 pixels window centered around the
    pixel of observation. The values are assigned within the bounds of a specific chip, i.e the
    window will be clipped to the extents of the chip in case it falls outside of the chip. A
    non-zero value for this parameter typically means that the observation covers more ground or
    pixels and can, in some cases account for low geolocation precision. Keep the default value
    if only interested in the pixel in which the observation falls.""",
    lower_bound=0,
)
flags.DEFINE_list(
    "mask_types",
    [],
    "List of different types of masking to apply",
)
flags.register_validator(
    "mask_types",
    lambda val_list: all(v in MASK_DECODING_POS["HLS"].keys() for v in val_list),
    message=f"Valid values are {list(MASK_DECODING_POS['HLS'].keys())}",
)
flags.DEFINE_enum(
    "masking_strategy",
    "each",
    ["each", "any"],
    """Method to use when applying masking:
    - "each" for timestep-wise masking.
    - "any" to exclude pixels if the mask is present for at least one timestep.
    """,
)
flags.DEFINE_bool(
    "daytime_only", False, "Whether to select only daytime satellite observations."
)

flags.DEFINE_enum(
    "task_type",
    "seg",
    ["seg", "reg"],
    """Type of the task for which the chips are being generated. This will
    impact the data type used to save the labels and thus the storage used.
    - "seg" for segmentation tasks. The labels raster file will be saved as int16.
    - "reg" for regression tasks. The labels raster file will be saved as float32.
    """,
)
FLAGS = flags.FLAGS
