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

"""InstaGeo Chip Creator Module."""

import ast
import json
import logging as std_logging
import os
from typing import Any, Callable, Dict, List, Tuple

import geopandas as gpd
import pandas as pd
from absl import app, flags, logging
from dotenv import load_dotenv
from pystac_client import Client
from shapely.geometry import Point

from instageo.data import hls_utils, s1_utils, s2_utils
from instageo.data.data_pipeline import get_tiles
from instageo.data.flags import FLAGS
from instageo.data.hls_utils import HLSPointsPipeline, add_hls_stac_items
from instageo.data.s1_utils import S1PointsPipeline, add_s1_stac_items
from instageo.data.s2_utils import S2PointsPipeline, add_s2_stac_items
from instageo.data.stac_utils import create_records_with_items

load_dotenv(os.path.expanduser("~/.credentials"))
logging.set_verbosity(logging.INFO)
std_logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame file.")

flags.DEFINE_integer("min_count", 100, "Minimum observation counts per tile", lower_bound=1)
flags.DEFINE_boolean(
    "shift_to_month_start",
    True,
    "Indicates whether or not to shift the observation date to the beginning of the month",
)
flags.DEFINE_boolean(
    "is_time_series_task",
    True,
    """Indicates whether or not the current task is a time series one. The data will be then
    retrieved before the date of observation""",
)
flags.DEFINE_enum(
    "processing_method",
    "cog",
    ["cog", "download", "download-only"],
    """Method to use to process the tiles:
    - "cog" corresponds to creating the chips by utilizing the Cloud Optimized GeoTIFFs.
    - "download" corresponds to downloading entire tiles to be used for creating
    the chips.
    - "download-only" corresponds to a simple download of the tiles without further
    processing.""",
)
flags.DEFINE_enum(
    "data_format",
    "csv",
    ["csv", "parquet"],
    """Format of the original file containing the observations. The data must contain
    columns named 'date', 'x', 'y', 'label'.
    In case of a Parquet file, the partitions must be done by the 'year' and 'mgrs_tile_id'
    columns.
    """,
)
flags.DEFINE_string(
    "filters",
    None,
    """List of filters to use. Filters must be provided as tuples following the
    structure ('col_to_filter_on' ? 'operator' ? value). Applies only in case of
    Parquet files.
    - The columns on which to filter and the operators should be provided as strings.
    - The operators allowed are ['==', '=', '>', '>=', '<', '<=', '!=', 'in', 'not in']
    Example: "('year' ? '==' ? 2016); ('mgrs_tile_id' ? '!=' ? 'BAN')"
    "('year' ? 'in' ? [2016, 2020]); ('mgrs_tile_id' ? 'not in' ? ['13SCS', 'BAN'])"
    """,
)


def parse_tuple_list(flag_value: str) -> List[Tuple]:
    """Converts a string into a list of tuples.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        (List[Tuple]): List of tuples extracted from the original string.
    """
    try:
        return [tuple(item.strip("()").split("?")) for item in flag_value.split(";")]
    except Exception as e:
        raise ValueError(f"Error parsing string {flag_value} to extract filters list: {e}")


def parse_filters(flag_value: str) -> List[Tuple[str, str, Any]]:
    """Converts a list of tuples into valid filters.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        (List[Tuple[str, str, Any]]): A parsed version of the list of tuples to be used as filters.
    """
    try:
        filters = parse_tuple_list(flag_value)
        parsed_filters = []
        ops = ["==", "=", ">", ">=", "<", "<=", "!=", "in", "not in"]
        for filter in filters:
            col, op, val = filter
            try:
                col = ast.literal_eval(col)
                op = ast.literal_eval(op)
                val = ast.literal_eval(val)
            except Exception as e:
                raise flags.ValidationError(f"Could not properly parse filter {filter}: {e}")
            if not isinstance(col, str):
                raise flags.ValidationError("Provide the filter column as a string")
            if op not in ops:
                raise flags.ValidationError(f"Operators must be one of {ops}")
            parsed_filters.append((col, op, val))
    except Exception as e:
        raise flags.ValidationError(
            f"Filters must be provided as tuple ('col_to_filter_on' ? 'operator' ? value): {e}"
        )
    return parsed_filters


# TODO: Use flags.mark_flags_as_required
def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


# TODO: Improve this workflow by using a more generic approach
# The pipeline classes and the add_stac_items_func are similar for all data sources

# Data source configuration
DATA_SOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "HLS": {
        "add_stac_items_func": add_hls_stac_items,
        "pipeline_class": HLSPointsPipeline,
        "granules_field": "hls_granules",
        "items_field": "hls_items",
        "client_func": lambda: Client.open(hls_utils.API.URL),
        "extra_params": ["temporal_tolerance_minutes", "cloud_coverage", "daytime_only"],
    },
    "S2": {
        "add_stac_items_func": add_s2_stac_items,
        "pipeline_class": S2PointsPipeline,
        "granules_field": "s2_granules",
        "items_field": "s2_items",
        "client_func": lambda: Client.open(s2_utils.API.URL),
        "extra_params": ["temporal_tolerance_minutes", "cloud_coverage", "daytime_only"],
    },
    "S1": {
        "add_stac_items_func": add_s1_stac_items,
        "pipeline_class": S1PointsPipeline,
        "granules_field": "s1_granules",
        "items_field": "s1_items",
        "client_func": lambda: Client.open(s1_utils.API.URL),
        "extra_params": ["temporal_tolerance_minutes"],
    },
}


def process_data_source(
    data_source: str,
    sub_data: pd.DataFrame,
    add_stac_items_func: Callable,
    pipeline_class: type,
    granules_field: str,
    items_field: str,
    client_func: Callable,
    **kwargs,
) -> None:
    """Generic function to process any data source (HLS, S2, S1).

    Args:
        data_source: Name of the data source (e.g., "HLS", "S2", "S1")
        sub_data: Input data DataFrame
        add_stac_items_func: Function to add STAC items (e.g., add_hls_stac_items)
        pipeline_class: Pipeline class to instantiate (e.g., HLSPointsPipeline)
        granules_field: Field name for granules (e.g., "hls_granules")
        items_field: Field name for items (e.g., "hls_items")
        client_func: Function to get STAC client
        **kwargs: Additional arguments for add_stac_items_func
    """
    dataset_file = os.path.join(FLAGS.output_directory, f"{data_source.lower()}_dataset.json")
    records_file = os.path.join(FLAGS.output_directory, "filtered_obsv_records.gpkg")

    if not (os.path.exists(dataset_file) and os.path.exists(records_file)):
        logging.info(f"Creating {data_source} dataset JSON.")
        logging.info(f"Retrieving {data_source} tile ID for each observation.")
        os.makedirs(os.path.join(FLAGS.output_directory), exist_ok=True)

        # Convert to GeoPandas DataFrame
        geometry = [Point(xy) for xy in zip(sub_data["x"], sub_data["y"])]
        sub_data = gpd.GeoDataFrame(sub_data, geometry=geometry, crs=f"EPSG:{FLAGS.src_crs}")
        sub_data["geometry_4326"] = sub_data["geometry"].to_crs("EPSG:4326")

        client = client_func()
        sub_data_with_items = add_stac_items_func(client, sub_data, **kwargs)

        (
            filtered_obsv_records,
            dataset,
        ) = create_records_with_items(sub_data_with_items, granules_field, items_field)

        with open(dataset_file, "w") as json_file:
            json.dump(dataset, json_file, indent=4)
        filtered_obsv_records.to_file(records_file, driver="GPKG")
    else:
        logging.info(f"{data_source} dataset JSON already created")
        with open(dataset_file) as json_file:
            dataset = json.load(json_file)
        filtered_obsv_records = gpd.read_file(records_file)

    logging.info("Creating Chips and Segmentation Maps")

    pipeline = pipeline_class(
        output_directory=FLAGS.output_directory,
        chip_size=FLAGS.chip_size,
        mask_types=getattr(FLAGS, "mask_types", []),
        masking_strategy=FLAGS.masking_strategy,
        src_crs=FLAGS.src_crs,
        spatial_resolution=getattr(FLAGS, "spatial_resolution", None),
        window_size=FLAGS.window_size,
        task_type=FLAGS.task_type,
    )

    # Run pipeline
    pipeline.run(dataset, filtered_obsv_records)


def main(argv: Any) -> None:
    """CSV/Parquet Chip Creator.

    Given a file containing geo-located point observations and labels, the Chip
    Creator creates small chip from larger tiles which is suitable for training
    segmentation models.
    """
    del argv
    if FLAGS.data_format == "parquet":
        try:
            filters = None
            if FLAGS.filters:
                filters = parse_filters(FLAGS.filters)
            data = pd.read_parquet(
                FLAGS.dataframe_path,
                engine="pyarrow",
                filters=filters,
            )
        except Exception as e:
            raise ValueError(f"Provide valid path and filters: {e}")
    else:
        data = pd.read_csv(FLAGS.dataframe_path)

    # Convert date column to datetime
    data["date"] = pd.to_datetime(data["date"])

    # If time column exists, combine it with date
    # Time format expected is HH:MM:SS
    if "time" in data.columns:
        data["date"] = data["date"] + pd.to_timedelta(data["time"])

    # Shift to month start if requested
    if FLAGS.shift_to_month_start:
        data["date"] = data["date"] - pd.offsets.MonthBegin(1)

    data["input_features_date"] = (
        data["date"] - pd.DateOffset(days=FLAGS.temporal_step)
        if FLAGS.is_time_series_task
        else data["date"]
    )
    FLAGS.num_steps = 1 if not FLAGS.is_time_series_task else FLAGS.num_steps
    sub_data = get_tiles(data, src_crs=FLAGS.src_crs, min_count=FLAGS.min_count)

    # Process data source using configuration registry
    if FLAGS.data_source not in DATA_SOURCE_CONFIG:
        raise ValueError(
            f"Error: data_source value '{FLAGS.data_source}' is not correct. "
            f"Please enter one of: {', '.join(DATA_SOURCE_CONFIG.keys())}."
        )

    config = DATA_SOURCE_CONFIG[FLAGS.data_source]
    logging.info(f"Using {FLAGS.data_source} pipeline")

    # Build extra parameters from FLAGS
    extra_params = {param: getattr(FLAGS, param) for param in config["extra_params"]}

    process_data_source(
        data_source=FLAGS.data_source,
        sub_data=sub_data,
        add_stac_items_func=config["add_stac_items_func"],
        pipeline_class=config["pipeline_class"],
        granules_field=config["granules_field"],
        items_field=config["items_field"],
        client_func=config["client_func"],
        num_steps=FLAGS.num_steps,
        temporal_step=FLAGS.temporal_step,
        temporal_tolerance=FLAGS.temporal_tolerance,
        **extra_params,
    )


if __name__ == "__main__":
    app.run(main)
