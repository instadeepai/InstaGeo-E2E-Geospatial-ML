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

"""Geo Utils Module."""

from itertools import chain
from typing import Any

import dask
import dask.delayed
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS

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
        (xr.Dataset, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files, (optionally) the masks and its CRS
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


def apply_mask(
    chip: xr.DataArray,
    mask: xr.DataArray,
    no_data_value: int,
    masking_strategy: str = "each",
    mask_types: list[str] = ["cloud", "water"],
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
            mask = decode_fmask_value(mask, pos)
            if masking_strategy == "each":
                # repeat across timesteps so that, each mask is applied to its
                # corresponding timestep
                mask = mask.values.repeat(chip.shape[0] // mask.shape[0], axis=0)
            elif masking_strategy == "any":
                # collapse the mask to exclude a pixel if its corresponding mask value
                # for at least one timestep is 1
                mask = mask.values.any(axis=0)
            chip = chip.where(mask == 0, other=no_data_value)
    return chip
