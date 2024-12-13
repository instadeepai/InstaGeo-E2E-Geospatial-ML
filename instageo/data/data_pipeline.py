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

"""InstaGeo Data pipeline Module."""

from itertools import chain
from typing import Any

import dask
import dask.delayed
import rasterio
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Point


def create_and_save_chips_with_seg_maps(
    mf_reader: Callable,
    tile_dict: dict[str, dict[str, str]],
    tile_id: str,
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
    water_mask: bool,
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a satellite image tile, read in as Xarray
    dataset, and save them to an output directory.

    Args:
        mf_reader (callable[dict[str, dict[str, str]], bool, bool]): A multi-file reader that
            accepts a dictionary of satellite image tile paths and reads it into an Xarray dataset.
            Optionally performs water and cloud masking based on the boolean flags passed.
        tile_dict (Dict): A dict mapping band names to HLS tile filepath.
        tile_id (str): MGRS ID of the tiles in `tile_dict`.
        df (pd.DataFrame): DataFrame containing the labelled data for creating segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_cloud (bool): Perform cloud masking if True.
        water_mask (bool): Perform water masking if True.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    ds, crs = mf_reader(tile_dict, mask_cloud, water_mask)
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

# Block sizes for the internal tiling of HLS COGs
BLOCKSIZE_X = 256
BLOCKSIZE_Y = 256

# Masks decoding positions
MASK_DECODING_POS = {"cloud": 1, "water": 5}


def open_mf_tiff_dataset(
    band_files: dict[str, Any], load_masks: bool
) -> tuple[xr.Dataset, xr.Dataset | None, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        load_masks (bool): Whether or not to load the masks files.

    Returns:
        (xr.Dataset, xr.Dataset | None, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files, (optionally) the masks, and the CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
        mask_and_scale=False,  # Scaling will be applied manually
    )
    bands_dataset.band_data.attrs["scale_factor"] = 1
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = (
        xr.open_mfdataset(
            mask_paths,
            concat_dim="band",
            combine="nested",
        )
        if load_masks
        else None
    )
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, mask_dataset, crs


@dask.delayed
def load_cog(url: str) -> xr.DataArray:
    """Load a COG file as an xarray DataArray.

    Args:
        url (str): COG url.

    Returns:
        xr.DataArray: An array exposing the data loaded from the COG
    """
    return rxr.open_rasterio(
        url,
        chunks=dict(band=1, x=BLOCKSIZE_X, y=BLOCKSIZE_Y),
        lock=False,
        mask_and_scale=False,  # Scaling will be applied manually
    )


def open_hls_cogs(
    bands_infos: dict[str, Any], load_masks: bool
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Open multiple COGs as an xarray DataArray.

    Args:
        bands_infos (dict[str, Any]): A dictionary containing data links for
        all bands and for all timesteps of interest.
        load_masks (bool): Whether or not to load the masks COGs.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray Dataset combining
        data from all the COGs bands, (optionally) the COGs masks and the CRS used
    """
    cogs_urls = bands_infos["data_links"]
    # For each timestep, this will contain a list of links for the different bands
    # with the masks being at the last position

    bands_links = list(chain.from_iterable(urls[:-1] for urls in cogs_urls))
    masks_links = [urls[-1] for urls in cogs_urls]

    all_timesteps_bands = xr.concat(
        dask.compute(*[load_cog(link) for link in bands_links]), dim="band"
    )
    all_timesteps_bands.attrs["scale_factor"] = 1

    # only read masks if necessary
    all_timesteps_masks = (
        xr.concat(dask.compute(*[load_cog(link) for link in masks_links]), dim="band")
        if load_masks
        else None
    )
    return (
        all_timesteps_bands,
        all_timesteps_masks,
        all_timesteps_bands.spatial_ref.crs_wkt,
    )


def decode_fmask_value(
    value: xr.Dataset | xr.DataArray, position: int
) -> xr.Dataset | xr.DataArray:
    """Decodes HLS v2.0 Fmask.

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
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


def apply_mask(
    chip: xr.DataArray,
    mask: xr.DataArray,
    no_data_value: int,
    masking_strategy: str = "each",
    mask_types: list[str] = list(MASK_DECODING_POS.keys()),
) -> xr.DataArray:
    """Apply masking to a chip.

    Args:
        chip (xr.DataArray): Chip array containing the pixels to be masked out.
        mask (xr.DataArray): Array containing the masks.
        no_data_value (int): Value to be used for masked pixels.
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)
        mask_types (list[str]): Mask types to apply.

    Returns:
        xr.DataArray: An array representing a chip with the relevant pixels being masked.
    """
    for mask_type in mask_types:
        pos = MASK_DECODING_POS.get(mask_type, None)
        if pos:
            decoded_mask = decode_fmask_value(mask, pos)
            if masking_strategy == "each":
                # repeat across timesteps so that, each mask is applied to its
                # corresponding timestep
                decoded_mask = decoded_mask.values.repeat(
                    chip.shape[0] // mask.shape[0], axis=0
                )
            elif masking_strategy == "any":
                # collapse the mask to exclude a pixel if its corresponding mask value
                # for at least one timestep is 1
                decoded_mask = decoded_mask.values.any(axis=0)
            chip = chip.where(decoded_mask == 0, other=no_data_value)
    return chip
