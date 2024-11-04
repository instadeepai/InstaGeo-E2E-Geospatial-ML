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

"""HLS pipeline Module."""

import os

import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS
from shapely.geometry import Point

from instageo.data.geo_utils import create_segmentation_map, get_chip_coords


def open_mf_tiff_dataset(
    band_files: dict[str, dict[str, str]], mask_cloud: bool, water_mask: bool
) -> tuple[xr.Dataset, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.

    Returns:
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files and its CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
    )
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = xr.open_mfdataset(
        mask_paths,
        concat_dim="band",
        combine="nested",
    )
    if water_mask:
        mask_water = decode_fmask_value(mask_dataset, 5)
        mask_water = mask_water.band_data.values.any(axis=0).astype(int)
        bands_dataset = bands_dataset.where(mask_water == 0)
    if mask_cloud:
        cloud_mask = decode_fmask_value(mask_dataset, 1)
        cloud_mask = cloud_mask.band_data.values.any(axis=0).astype(int)
        bands_dataset = bands_dataset.where(cloud_mask == 0)
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, crs


def decode_fmask_value(value: xr.Dataset, position: int) -> xr.Dataset:
    """Decodes HLS v2.0 Fmask.

    The decoding strategy is described in Appendix A of the user manual
    (https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

    Arguments:
        value: Input xarray Dataset created from Fmask.tif
        position: Bit position to decode.

    Returns:
        Xarray dataset containing decoded bits.
    """
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


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


def create_and_save_chips_with_seg_maps_hls(
    hls_tile_dict: dict[str, dict[str, str]],
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
    water_mask: bool,
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
        water_mask (bool): Perform water masking if True.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    ds, crs = open_mf_tiff_dataset(hls_tile_dict, mask_cloud, water_mask)
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
        seg_map = create_segmentation_map(chip, df, no_data_value)
        if seg_map.where(seg_map != -1).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chip = chip.fillna(no_data_value)
        chips.append(chip_name)
        chip.band_data.rio.to_raster(chip_filename)
    return chips, seg_maps
