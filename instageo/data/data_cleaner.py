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

"""Chips and segmentation maps cleaner module."""

import os
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from absl import app, flags, logging
from mgrs import MGRS
from pyproj import Transformer

# Define flags
flags.DEFINE_string(
    "chips_dataset_csv",
    None,
    "Path to input CSV file containing Input and Label columns.",
)
flags.mark_flag_as_required("chips_dataset_csv")
flags.DEFINE_string(
    "output_chips_dataset_csv",
    None,
    "Path to save the cleaned CSV file.",
)
flags.mark_flag_as_required("output_chips_dataset_csv")
flags.DEFINE_bool(
    "drop_chips",
    False,
    "Whether to drop chips based on no_data threshold.",
)
flags.DEFINE_enum(
    "drop_chips_strategy",
    "any",
    ["any", "all"],
    """Strategy to use when dropping chips. If `any`, a chip is dropped
    if any band has no_data pixels above the threshold. If `all`, a chip
    is dropped if all bands have no_data pixels above the threshold.""",
)

flags.DEFINE_float(
    "no_data_threshold",
    0.5,
    "Threshold of no_data pixels to drop a chip.",
    lower_bound=0.0,
    upper_bound=1.0,
)

flags.DEFINE_integer(
    "no_data_value",
    -9999,
    "Value representing no_data in chips.",
)
flags.DEFINE_bool(
    "clean_seg_maps",
    False,
    "Whether to clean the segmentation maps.",
)
flags.DEFINE_enum(
    "cleaning_method",
    "buffer",
    ["buffer", "limit"],
    """Method to apply to segmentation map cleaning.
        - buffer: Buffer observation pixels by a given window size.
        - limit: Limit segmentation map to observation pixels.""",
)
flags.DEFINE_string(
    "observation_points_csv",
    None,
    """Path to CSV file containing observation points. Must have
    columns: x, y, mgrs_tile_id, date.""",
)

flags.DEFINE_string(
    "seg_map_output_dir",
    None,
    """Path to save the cleaned segmentation map. If None,
    saves in the same directory as the original file which will be replaced.""",
)

flags.DEFINE_integer(
    "ignore_index",
    -1,
    "Value to set for pixels that are not observation points.",
)

flags.DEFINE_integer(
    "window_size",
    1,
    """Size of window around observation points.
    A value of 1 creates a 3x3 window, 2 creates a 5x5 window, etc.""",
    lower_bound=1,
)


FLAGS = flags.FLAGS


def should_drop_chip(
    chip_fname: str,
    no_data_threshold: float,
    no_data_value: int,
    drop_chips_strategy: str,
) -> bool:
    """Check if a chip should be dropped based on no_data threshold.

    Args:
        chip_fname (str): Path to chip file.
        no_data_threshold (float): Threshold of no_data pixels to drop a chip.
        no_data_value (int): Value representing no_data in chips.
        drop_chips_strategy (str): Strategy to use when dropping chips.

    Returns:
        bool: True if the chip should be dropped, False otherwise.
    """
    chip = rasterio.open(chip_fname).read()
    if drop_chips_strategy == "any":
        no_data_mask = np.any(chip == no_data_value, axis=0)
    elif drop_chips_strategy == "all":
        no_data_mask = np.all(chip == no_data_value, axis=0)
    no_data_ratio = np.mean(no_data_mask)
    return no_data_ratio > no_data_threshold


def buffer_observation_pixels(
    seg_map_fname: str,
    chip_fname: str,
    window_size: int,
    no_data_value: int,
    ignore_index: int,
    seg_map_output_dir: str | None,
) -> str | None:
    """Buffer observation pixels by a given window size.

    Args:
        seg_map_fname (str): Path to segmentation map file.
        chip_fname (str): Path to chip file.
        window_size (int): Size of window around observation pixels.
        no_data_value (int): Value representing no_data in chips.
        ignore_index (int): Value to set for pixels that are not observation points.
        seg_map_output_dir (str | None): Path to save the cleaned segmentation map. If None,
            saves in the same directory as the original file which will be replaced.

    Returns:
        str | None: Path to the cleaned segmentation map.
    """
    chip = rasterio.open(chip_fname).read()
    with rasterio.open(seg_map_fname) as src:
        seg_map = src.read()
        # Get coordinates of valid pixels in seg_map
        rows, cols = np.where(seg_map[0] != ignore_index)

        # Create window coordinates for all valid pixels
        offsets = np.arange(-window_size, window_size + 1)
        offset_rows, offset_cols = np.meshgrid(offsets, offsets)
        window_rows = np.clip(
            rows[:, np.newaxis, np.newaxis] + offset_rows, 0, seg_map.shape[1] - 1
        )
        window_cols = np.clip(
            cols[:, np.newaxis, np.newaxis] + offset_cols, 0, seg_map.shape[2] - 1
        )

        # Get values to propagate and set window pixels to corresponding values
        values = seg_map[0, rows, cols]
        seg_map[0, window_rows.ravel(), window_cols.ravel()] = np.repeat(
            values, (2 * window_size + 1) ** 2
        )

        # Mask corresponding pixels in segmentation map that are not valid in the image
        mask = np.all(chip == no_data_value, axis=0)
        seg_map[0, mask] = ignore_index

        # Determine output path
        if seg_map_output_dir is None:
            seg_map_output_dir = os.path.dirname(seg_map_fname)
        else:
            os.makedirs(seg_map_output_dir, exist_ok=True)

        output_path = os.path.join(seg_map_output_dir, os.path.basename(seg_map_fname))
        with rasterio.open(output_path, "w", **src.profile) as dst:
            dst.write(seg_map)
        return output_path


def limit_seg_map_to_observation_pixels(
    seg_map_fname: str,
    observation_points: pd.DataFrame,
    ignore_index: int,
    seg_map_output_dir: str | None,
) -> str | None:
    """Limit segmentation map to observation pixels.

    Args:
        seg_map_fname (str): Path to segmentation map file.
        observation_points (pd.DataFrame): DataFrame containing observation points.
        ignore_index (int): Value to set for pixels that are not observation points.
        seg_map_output_dir (str | None): Path to save the cleaned segmentation map. If None,
            saves in the same directory as the original file which will be replaced.

    Returns:
        str | None: Path to the cleaned segmentation map.
    """
    # Extract date and MGRS tile ID from filename
    # Example: seg_map_20200101_L30_T13SCS_2020001T173930_9_11.tif
    filename = os.path.basename(seg_map_fname)
    parts = filename.split("_")
    if len(parts) >= 4:
        chip_date = parts[2]
        mgrs_tile = parts[4][1:]

    # Create a mask where only relevant observation points are kept
    with rasterio.open(seg_map_fname) as src:
        seg_map = src.read()
        crs = src.crs

        # Filter observation points for this specific tile and date
        relevant_points = observation_points[
            (observation_points["mgrs_tile_id"] == mgrs_tile)
            & (observation_points["date"].str.replace("-", "") == chip_date)
        ]

        if not relevant_points.empty:
            # Create a mask where only relevant observation points are kept
            mask = np.zeros(seg_map[0].shape, dtype=bool)
            transformer = Transformer.from_crs(4326, crs, always_xy=True)
            # Transform coordinates
            x_coords, y_coords = transformer.transform(
                relevant_points["x"].values, relevant_points["y"].values
            )
            for x, y in zip(x_coords, y_coords):
                # Convert x,y coordinates to row,col indices
                row_idx, col_idx = src.index(x, y)
                if 0 <= row_idx < seg_map.shape[1] and 0 <= col_idx < seg_map.shape[2]:
                    mask[row_idx, col_idx] = True

            # Set all non-observation points to ignore_index
            seg_map = np.where(mask, seg_map, ignore_index)

            # Save the filtered segmentation map
            if seg_map_output_dir is None:
                seg_map_output_dir = os.path.dirname(seg_map_fname)
            else:
                os.makedirs(seg_map_output_dir, exist_ok=True)
            output_path = os.path.join(seg_map_output_dir, filename)
            with rasterio.open(output_path, "w", **src.profile) as dst:
                dst.write(seg_map)
            return output_path
        else:
            logging.warning(f"No relevant points found for {seg_map_fname}")
            return None


def clean_data(
    chips_dataset_csv: str,
    output_chips_dataset_csv: str,
    drop_chips: bool = False,
    drop_chips_strategy: str = "any",
    no_data_threshold: float = 0.5,
    no_data_value: int = -9999,
    clean_seg_maps: bool = False,
    observation_points_csv: str | None = None,
    cleaning_method: str = "buffer",
    ignore_index: int = -1,
    window_size: int = 1,
    seg_map_output_dir: str | None = None,
) -> None:
    """Clean and process data from CSV file.

    Args:
        chips_dataset_csv (str): Path to input CSV file containing Input and Label columns.
        output_chips_dataset_csv (str): Path to save the cleaned CSV file.
        drop_chips (bool): Whether to drop chips based on no_data threshold.
        drop_chips_strategy (str): Strategy to use when dropping chips. If `any`,
            a chip is dropped if any band has no_data pixels above the threshold.
            If `all`, a chip is dropped if all bands have no_data pixels above the threshold.
        no_data_threshold (float): Threshold of no_data pixels to drop a chip.
        no_data_value (int): Value representing no_data in chips.
        clean_seg_maps (bool): Whether to clean the segmentation maps.
        observation_points_csv (str | None): Path to CSV file containing observation points.
            Must have columns: x, y, mgrs_tile_id, date.
        cleaning_method (str): Method to apply to segmentation map cleaning.
        ignore_index (int): Value to set for pixels that are not observation points.
        window_size (int): Size of window around observation points. A value of 1 creates a
            3x3 window, 2 creates a 5x5 window, etc.
        seg_map_output_dir (str): Path to save the cleaned segmentation map.

    Returns:
        None
    """
    # Read input CSV
    try:
        df = pd.read_csv(chips_dataset_csv)
        num_rows = len(df)
        if not all(col in df.columns for col in ["Input", "Label"]):
            raise ValueError("CSV must contain 'Input' and 'Label' columns")

        if drop_chips:
            df = df[
                ~df["Input"].apply(
                    should_drop_chip,
                    args=(no_data_threshold, no_data_value, drop_chips_strategy),
                )
            ]

        if clean_seg_maps:
            if cleaning_method == "buffer":
                df["Label"] = df.apply(
                    lambda row: buffer_observation_pixels(
                        row["Label"],
                        row["Input"],
                        window_size,
                        no_data_value,
                        ignore_index,
                        seg_map_output_dir,
                    ),
                    axis=1,
                )
            elif cleaning_method == "limit":
                # Read observation points if provided
                observation_points = None
                if observation_points_csv:
                    observation_points = pd.read_csv(observation_points_csv)
                    if not all(col in observation_points.columns for col in ["x", "y", "date"]):
                        raise ValueError(
                            "Observation points CSV must contain 'x', 'y', and 'date' columns"
                        )
                    # Extract MGRS tile ID from x and y coordinates if not provided
                    if "mgrs_tile_id" not in observation_points.columns:
                        mgrs_object = MGRS()
                        observation_points["mgrs_tile_id"] = observation_points.apply(
                            lambda row: mgrs_object.toMGRS(row["y"], row["x"], MGRSPrecision=0),
                            axis=1,
                        )
                    df["Label"] = df["Label"].apply(
                        limit_seg_map_to_observation_pixels,
                        args=(observation_points, ignore_index, seg_map_output_dir),
                    )
                    # Filter out rows where Label is None
                    df = df[df["Label"].notna()]
                else:
                    raise ValueError(
                        "Observation points CSV is required for `limit` cleaning method"
                    )
            else:
                raise ValueError(f"Invalid cleaning method: {cleaning_method}")

        # Save cleaned CSV
        df.to_csv(output_chips_dataset_csv, index=False)
        logging.info(
            f"Cleaned data saved to {output_chips_dataset_csv}.\n"
            f"Dropped {num_rows - len(df)} rows."
        )
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")


def main(argv: Any) -> None:
    """Command line interface for data cleaning."""
    del argv  # Unused

    clean_data(
        chips_dataset_csv=FLAGS.chips_dataset_csv,
        output_chips_dataset_csv=FLAGS.output_chips_dataset_csv,
        drop_chips=FLAGS.drop_chips,
        drop_chips_strategy=FLAGS.drop_chips_strategy,
        no_data_threshold=FLAGS.no_data_threshold,
        no_data_value=FLAGS.no_data_value,
        clean_seg_maps=FLAGS.clean_seg_maps,
        observation_points_csv=FLAGS.observation_points_csv,
        cleaning_method=FLAGS.cleaning_method,
        ignore_index=FLAGS.ignore_index,
        window_size=FLAGS.window_size,
        seg_map_output_dir=FLAGS.seg_map_output_dir,
    )


if __name__ == "__main__":
    app.run(main)
