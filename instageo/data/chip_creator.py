import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from absl import app, flags
from rasterio.transform import Affine

from instageo.data.geo_utils import open_mf_tiff_dataset, read_csv_gdf

FLAGS = flags.FLAGS

flags.DEFINE_string("hls_tile_path", None, "Path to the HLS tile.")
flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame CSV file.")
flags.DEFINE_integer("chip_size", 224, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_integer("dst_crs", 32613, "CRS of the geo-coordinates in `hls_tile_path`")
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "no_data_value", -1, "Value to use for no data areas in the segmentation maps."
)


def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["hls_tile_path", "dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


def lonlat_to_pixel(
    longitude: float, latitude: float, transform: Affine
) -> Tuple[int, int]:
    """Convert longitude and latitude to pixel coordinates.

    Args:
        longitude (float): Longitude value.
        latitude (float): Latitude value.
        transform (Affine): Affine transformation to convert geographic coordinates to
            pixel coordinates.

    Returns:
        Tuple[int, int]: The pixel coordinates as a tuple (row, column).
    """
    pixel_coords = rasterio.transform.rowcol(transform, longitude, latitude)
    return pixel_coords


def create_segmentation_map(
    chip: Any,
    df: pd.DataFrame,
    transform: Affine,
    chip_offset: Dict[str, int],
    no_data_value: int,
) -> np.ndarray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        transform (Affine): Affine transformation for the dataset.
        chip_offset (Dict[str, int]): The x and y offsets of the chip.
        no_data_value (int): Value to be used for pixels with no data.

    Returns:
        np.ndarray: The created segmentation map as a NumPy array.
    """
    # Initialize the segmentation map with no_data_value
    seg_map = np.full((chip.dims["y"], chip.dims["x"]), no_data_value, dtype=int)

    # TODO create a subset from df so that we dont iterate over everything
    for _, row in df.iterrows():
        # Convert geolocation to pixel coordinates in the full image
        x_pixel, y_pixel = lonlat_to_pixel(
            row["geometry"].x, row["geometry"].y, transform
        )

        # Adjust for the current chip's offset
        x_pixel -= chip_offset["x"]
        y_pixel -= chip_offset["y"]

        # Check if the point is within the current chip
        if 0 <= x_pixel < chip.dims["x"] and 0 <= y_pixel < chip.dims["y"]:
            seg_map[y_pixel, x_pixel] = row["label"]

    return seg_map


def create_and_save_chips_with_seg_maps(
    hls_tile_path: Dict[str, str],
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
) -> int:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a HLS tile and save them to
    an output directory.

    Args:
        hls_tile_path (Dict): A dict mapping band names to HLS tile filepath.
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.

    Returns:
        int: The number of chips created and saved.
    """
    # Load the HLS tile file as an xarray Dataset
    ds = open_mf_tiff_dataset(hls_tile_path)

    # Calculate the number of chips in each dimension
    n_chips_x = ds.dims["x"] // chip_size
    n_chips_y = ds.dims["y"] // chip_size

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    chip_count = 0
    for x in range(n_chips_x):
        for y in range(n_chips_y):
            # Extract the chip
            chip = ds.isel(
                x=slice(x * chip_size, (x + 1) * chip_size),
                y=slice(y * chip_size, (y + 1) * chip_size),
            )

            # Calculate chip offset
            chip_offset = {"x": x * chip_size, "y": y * chip_size}

            # Create a segmentation map for the chip
            seg_map = create_segmentation_map(
                chip, df, ds.attrs["transform"], chip_offset, no_data_value
            )

            # Save the chip and segmentation map
            chip_filename = os.path.join(output_directory, f"chip_{chip_count}.tif")
            seg_map_filename = os.path.join(
                output_directory, f"seg_map_{chip_count}.tif"
            )

            chip.rio.to_raster(chip_filename)
            with rasterio.open(
                seg_map_filename,
                "w",
                driver="GTiff",
                height=seg_map.shape[0],
                width=seg_map.shape[1],
                count=1,
                dtype=str(seg_map.dtype),
                crs=ds.attrs["crs"],
                transform=ds.attrs["transform"],
            ) as dst:
                dst.write(seg_map, 1)

            chip_count += 1

    return chip_count


def main(argv: Any) -> None:
    """Chip Creator Entry Point"""
    del argv
    check_required_flags()

    hls_tile_path = FLAGS.hls_tile_path
    dataframe_path = FLAGS.dataframe_path
    chip_size = FLAGS.chip_size
    output_directory = FLAGS.output_directory
    no_data_value = FLAGS.no_data_value
    src_crs = FLAGS.src_crs
    dst_crs = FLAGS.dst_crs

    # Load DataFrame
    gdf = read_csv_gdf(dataframe_path, src_crs=src_crs, dst_crs=dst_crs)

    # Create and Save Chips
    num_chips = create_and_save_chips_with_seg_maps(
        hls_tile_path, gdf, chip_size, output_directory, no_data_value
    )
    print(f"Created and saved {num_chips} chips.")


if __name__ == "__main__":
    app.run(main)
