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

import bisect
import json
import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import xarray as xr
from absl import app, flags, logging
from shapely.geometry import Point
from tqdm import tqdm

import instageo.data.hls_utils as hls_utils
from instageo.data.geo_utils import open_mf_tiff_dataset

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame CSV file.")
flags.DEFINE_integer("chip_size", 224, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "no_data_value", -1, "Value to use for no data areas in the segmentation maps."
)
flags.DEFINE_integer(
    "min_count", 100, "Minimum observation counts per tile", lower_bound=1
)
flags.DEFINE_boolean(
    "shift_to_month_start",
    False,
    "Indicates whether or not to shift the observation date to the beginning of the month",
)
flags.DEFINE_boolean(
    "is_time_series_task",
    True,
    """Indicates whether or not the current task is a time series one. The data will be then 
    retrieved before the date of observation""",
)
flags.DEFINE_integer(
    "num_steps",
    3,
    """Number of temporal steps. When `is_time_series_task` is set to True, an attempt 
    will be made to retrieve `num_steps` HLS chips prior to the observation date. 
    Otherwise, the value of `num_steps` will default to 1 and an attempt will be made to retrieve 
    the HLS chip corresponding to the observation date.
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
flags.DEFINE_boolean(
    "download_only", False, "Downloads HLS dataset without creating chips."
)
flags.DEFINE_boolean("mask_cloud", False, "Perform Cloud Masking")
flags.DEFINE_float(
    "point_to_pixel_coverage",
    0.1,
    """Buffer radius to use around the observations to assign corresponding label values to 
    each touching pixel. A higher value typically means that the observation will cover more
    ground/pixels. Keep the default value/ or use a small value if only interested in the pixel 
    in which the observation falls.""",
)


def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


def create_segmentation_map(
    chip: Any, df: pd.DataFrame, no_data_value: int, point_to_pixel_coverage: float
) -> np.ndarray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        no_data_value (int): Value to be used for pixels with no data.
        point_to_pixel_coverage (float): Radius of the buffer to use around the observation
        to assign labels to pixels.

    Returns:
        np.ndarray: The created segmentation map as a NumPy array.
    """
    seg_map = chip.isel(band=0).assign(
        {
            "band_data": (
                ("y", "x"),
                no_data_value * np.ones((chip.sizes["x"], chip.sizes["y"])),
            )
        }
    )
    df = df[
        (chip["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= chip["x"].max().item())
        & (chip["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= chip["y"].max().item())
    ]
    mask = rasterio.features.rasterize(
        [*zip(df.geometry.buffer(point_to_pixel_coverage), df.label)],
        out_shape=seg_map.band_data.shape,
        fill=np.nan,
        transform=seg_map.rio.transform(),
        dtype=seg_map.band_data.dtype,
        all_touched=True,  # to assign the observation label to all the pixels touched
    )
    mask = xr.DataArray(mask, dims=seg_map.dims, coords=seg_map.coords)
    seg_map = seg_map.where(np.isnan(mask), mask)
    return seg_map.band_data.squeeze()


def get_chip_coords(
    df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int
) -> list[tuple[int, int]]:
    """Get Chip Coordinates.

    Given a list of x,y coordinates tuples of a point and an xarray dataarray, this
    function returns the corresponding x,y indices of the grid where each point will fall
    when the DataArray is gridded such that each grid has size `chip_size`
    indices where it will fall.

    Args:
        gdf (gpd.GeoDataFrame): GeoPandas dataframe containing the point.
        tile (xr.DataArray): Tile DataArray.
        chip_size (int): Size of each chip.

    Returns:
        List of chip indices.
    """
    coords = []
    for _, row in df.iterrows():
        x = bisect.bisect_left(tile["x"].values, row["geometry"].x)
        y = bisect.bisect_left(tile["y"].values[::-1], row["geometry"].y)
        y = tile.sizes["y"] - y - 1
        x = int(x // chip_size)
        y = int(y // chip_size)
        coords.append((x, y))
    return coords


def create_and_save_chips_with_seg_maps(
    hls_tile_dict: dict[str, dict[str, str]],
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
    point_to_pixel_coverage: float
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a HLS tile and save them to
    an output directory.

    Args:
        hls_tile_dict (Dict): A dict mapping band names to HLS tile filepath.
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_cloud (bool): Perform cloud masking if True.
        point_to_pixel_coverage (float): Radius of the buffer to use around the observation
        to assign labels to pixels.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    ds, crs = open_mf_tiff_dataset(hls_tile_dict, mask_cloud)
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=src_crs, inplace=True)
    df = df.to_crs(crs=crs)

    df = df[
        (ds["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= ds["x"].max().item())
        & (ds["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= ds["y"].max().item())
    ]
    os.makedirs(output_directory, exist_ok=True)
    tile_name_splits = hls_tile_dict["tiles"]["B02_0"].split(".")
    tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
    date_id = df.iloc[0]["date"].strftime("%Y%m%d")
    chips = []
    seg_maps: list[str | None] = []
    n_chips_x = ds.sizes["x"] // chip_size
    n_chips_y = ds.sizes["y"] // chip_size
    chip_coords = list(set(get_chip_coords(df, ds, chip_size)))
    for x, y in chip_coords:
        if (x >= n_chips_x) or (y >= n_chips_y):
            continue
        chip_id = f"{date_id}_{tile_id}_{x}_{y}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"

        chip_filename = os.path.join(output_directory, "chips", chip_name)
        seg_map_filename = os.path.join(output_directory, "seg_maps", seg_map_name)
        if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
            continue

        chip = ds.isel(
            x=slice(x * chip_size, (x + 1) * chip_size),
            y=slice(y * chip_size, (y + 1) * chip_size),
        )
        if chip.count().values == 0:
            continue
        seg_map = create_segmentation_map(
            chip, df, no_data_value, point_to_pixel_coverage
        )
        if seg_map.where(seg_map != no_data_value).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chip = chip.fillna(no_data_value)
        chips.append(chip_name)
        chip.band_data.rio.to_raster(chip_filename)
    return chips, seg_maps


def create_hls_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, dict[str, str]]], set[str]]:
    """Creates HLS Dataset.

    A HLS dataset is a list of dictionary mapping band names to corresponding GeoTiff
    filepath. It is required for creating chips.

    Args:
        data_with_tiles (pd.DataFrame): A dataframe containing observations that fall
            within a dense tile. It also has `hls_tiles` column that contains a temporal
            series of HLS granules.
        outdir (str): Output directory where tiles could be downloaded to.

    Returns:
        A tuple containing HLS dataset and a list of tiles that needs to be downloaded.
    """
    data_with_tiles = data_with_tiles.drop_duplicates(subset=["hls_tiles"])
    data_with_tiles = data_with_tiles[
        data_with_tiles["hls_tiles"].apply(
            lambda granule_lst: all("HLS" in str(item) for item in granule_lst)
        )
    ]
    assert not data_with_tiles.empty, "No observation record with valid HLS tiles"
    hls_dataset = {}
    granules_to_download = []
    s30_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    l30_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
    for hls_tiles, obsv_date in zip(
        data_with_tiles["hls_tiles"], data_with_tiles["date"]
    ):
        band_id, band_path = [], []
        mask_id, mask_path = [], []
        for idx, tile in enumerate(hls_tiles):
            tile = tile.strip(".")
            if "HLS.S30" in tile:
                for band in s30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )
            else:
                for band in l30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "hls_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )

        hls_dataset[f'{obsv_date.strftime("%Y-%m-%d")}_{tile.split(".")[2]}'] = {
            "tiles": {k: v for k, v in zip(band_id, band_path)},
            "fmasks": {k: v for k, v in zip(mask_id, mask_path)},
        }
    return hls_dataset, set(granules_to_download)


def main(argv: Any) -> None:
    """CSV Chip Creator.

    Given a csv file containing geo-located point observations and labels, the Chip
    Creator creates small chips from large HLS tiles which is suitable for training
    segmentation models.
    """
    del argv
    data = pd.read_csv(FLAGS.dataframe_path)
    data["date"] = (
        pd.to_datetime(data["date"]) - pd.offsets.MonthBegin(1)
        if FLAGS.shift_to_month_start
        else pd.to_datetime(data["date"])
    )
    data["input_features_date"] = (
        data["date"] - pd.DateOffset(days=FLAGS.temporal_step)
        if FLAGS.is_time_series_task
        else data["date"]
    )
    FLAGS.num_steps = 1 if not FLAGS.is_time_series_task else FLAGS.num_steps
    sub_data = hls_utils.get_hls_tiles(data, min_count=FLAGS.min_count)

    if not (
        os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
        and os.path.exists(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    ):
        logging.info("Creating HLS dataset JSON.")
        logging.info("Retrieving HLS tile ID for each observation.")
        sub_data_with_tiles = hls_utils.add_hls_granules(
            sub_data,
            num_steps=FLAGS.num_steps,
            temporal_step=FLAGS.temporal_step,
            temporal_tolerance=FLAGS.temporal_tolerance,
        )
        logging.info("Retrieving HLS tiles that will be downloaded.")
        hls_dataset, granules_to_download = create_hls_dataset(
            sub_data_with_tiles, outdir=FLAGS.output_directory
        )
        with open(
            os.path.join(FLAGS.output_directory, "hls_dataset.json"), "w"
        ) as json_file:
            json.dump(hls_dataset, json_file, indent=4)
        pd.DataFrame({"tiles": list(granules_to_download)}).to_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    else:
        logging.info("HLS dataset JSON already created")
        with open(
            os.path.join(FLAGS.output_directory, "hls_dataset.json")
        ) as json_file:
            hls_dataset = json.load(json_file)
        granules_to_download = pd.read_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )["tiles"].tolist()
    os.makedirs(os.path.join(FLAGS.output_directory, "hls_tiles"), exist_ok=True)
    logging.info("Downloading HLS Tiles")
    hls_utils.parallel_download(
        granules_to_download,
        outdir=os.path.join(FLAGS.output_directory, "hls_tiles"),
    )
    if FLAGS.download_only:
        return
    logging.info("Creating Chips and Segmentation Maps")
    all_chips = []
    all_seg_maps = []
    os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
    for key, hls_tile_dict in tqdm(hls_dataset.items(), desc="Processing HLS Dataset"):
        obsv_date_str, tile_id = key.split("_")
        obsv_data = sub_data[
            (sub_data["date"] == pd.to_datetime(obsv_date_str))
            & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
        ]
        try:
            chips, seg_maps = create_and_save_chips_with_seg_maps(
                hls_tile_dict,
                obsv_data,
                chip_size=FLAGS.chip_size,
                output_directory=FLAGS.output_directory,
                no_data_value=FLAGS.no_data_value,
                src_crs=FLAGS.src_crs,
                mask_cloud=FLAGS.mask_cloud,
                point_to_pixel_coverage=FLAGS.point_to_pixel_coverage
            )
            all_chips.extend(chips)
            all_seg_maps.extend(seg_maps)
        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Error {e} when reading dataset containing: {hls_tile_dict}")
        except IndexError as e:
            logging.error(f"Error {e} when processing {key}")
    logging.info("Saving dataframe of chips and segmentation maps.")
    pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
        os.path.join(FLAGS.output_directory, "hls_chips_dataset.csv")
    )


if __name__ == "__main__":
    app.run(main)
