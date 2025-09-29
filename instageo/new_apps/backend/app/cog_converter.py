"""Simple COG converter using GDAL commands for merging TIF chips."""

import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class COGConverter:
    """Simple service for merging TIF chips into Cloud Optimized GeoTIFFs using GDAL."""

    def __init__(self) -> None:
        """Initialize COG converter."""
        pass

    def merge_task_files_to_cog(
        self, data_path: str, chip_size: int = 256, compute_seg_stats: bool = False
    ) -> Dict[str, Any]:
        """Merge TIF chips and prediction results from a directory into COGs using GDAL in parallel.

        Args:
            data_path: Directory containing files to merge into COG.
            chip_size: Size of chips (used as block size).
            compute_seg_stats: Whether to compute segmentation stats.

        Returns:
            Tuple of (merged_chips_cog_path, merged_predictions_cog_path)
        """
        start_time = datetime.now()
        try:
            data_path_obj = Path(data_path)
            output_path = data_path_obj / "cogs"
            output_path.mkdir(parents=True, exist_ok=True)

            chips_dir = data_path_obj / "chips"
            predictions_dir = data_path_obj / "predictions"

            chip_files = list(chips_dir.glob("*.tif"))
            prediction_files = list(predictions_dir.glob("*.tif"))

            if not chip_files:
                raise ValueError(f"No chips found in {chips_dir}")
            if not prediction_files:
                raise ValueError(f"No prediction files found in {predictions_dir}")

            logger.info(f"Starting parallel merge of {len(chip_files)} chips")

            # Run chips and predictions merging in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                chips_future = executor.submit(
                    self.merge_files_to_cog,
                    chip_files,
                    str(output_path / "chips_merged.tif"),
                    chip_size,
                    True,  # Enable band selection for chips (keep only B, G, R)
                )
                predictions_future = executor.submit(
                    self.merge_files_to_cog,
                    prediction_files,
                    str(output_path / "predictions_merged.tif"),
                    chip_size,
                    False,  # No band selection for predictions
                )

                # Wait for both to complete and get results
                merged_chips_cog_path = chips_future.result()
                merged_predictions_cog_path = predictions_future.result()

            # Compute segmentation stats if needed
            seg_stats = None
            if compute_seg_stats:
                seg_stats = self.compute_seg_stats(merged_predictions_cog_path)

            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            logger.info(f"Merging to COGs completed successfully in {processing_duration} s")

            return {
                "chips_merged_cog_path": merged_chips_cog_path,
                "predictions_merged_cog_path": merged_predictions_cog_path,
                "processing_duration": f"{processing_duration:.1f}",
                "segmentation_stats": seg_stats,
            }

        except Exception as e:
            logger.error(f"Failed to merge files to COG: {str(e)}")
            raise

    def merge_files_to_cog(
        self,
        tif_files: List[Path],
        output_path: str,
        chip_size: int = 256,
        select_bands: bool = False,
    ) -> str:
        """Merge TIF files from a list of files into a single COG using GDAL.

        Args:
            tif_files: List of TIF file paths to merge.
            output_path: Path for output COG file.
            chip_size: Size of chips (used as block size).
            select_bands: If True, keep only first 3 bands (B, G, R) for RGB data.

        Returns:
            Path to the created COG file.
        """
        try:
            if not tif_files:
                raise ValueError("No TIF files provided for merging")

            logger.info(f"Merging {len(tif_files)} TIF files using GDAL")

            # Step 1: Merge files using gdal_merge.py to temporary file
            temp_merged = tempfile.mktemp(suffix=".tif")

            merge_cmd = [
                "gdal_merge.py",
                "-o",
                temp_merged,
                *[str(f) for f in tif_files],
            ]

            logger.info(f"Step 1 - Merging: gdal_merge.py with {len(tif_files)} files")

            merge_result = subprocess.run(merge_cmd, capture_output=True, text=True, check=True)

            if merge_result.stderr:
                logger.warning(f"GDAL merge warnings: {merge_result.stderr}")

            # Step 2: Convert merged file to COG using gdal_translate
            cog_cmd = [
                "gdal_translate",
                temp_merged,
                str(output_path),
                "-of",
                "COG",
            ]

            # Add band selection for RGB data (chips only)
            if select_bands:
                cog_cmd.extend(
                    ["-b", "1", "-b", "2", "-b", "3"]
                )  # Keep only first 3 bands (B, G, R)
                logger.info("Selecting only first 3 bands (B, G, R) for RGB data")

            cog_cmd.extend(
                [
                    "-co",
                    f"BLOCKSIZE={chip_size}",
                    "-co",
                    "COMPRESS=LZW",
                    "-co",
                    "BIGTIFF=IF_SAFER",
                    "-co",
                    "OVERVIEW_RESAMPLING=BILINEAR",
                    "-co",
                    "OVERVIEW_COUNT=6",
                    "-co",
                    "NUM_THREADS=ALL_CPUS",
                ]
            )

            logger.info(f"Step 2 - Converting to COG: {output_path}")

            cog_result = subprocess.run(cog_cmd, capture_output=True, text=True, check=True)

            if cog_result.stderr:
                logger.warning(f"GDAL COG conversion warnings: {cog_result.stderr}")
            logger.info(f"Successfully created COG with overviews: {output_path}")
            return str(output_path)

        except subprocess.CalledProcessError as e:
            error_msg = f"GDAL command failed: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Failed to merge files to COG: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            Path(temp_merged).unlink(missing_ok=True)

    def compute_seg_stats(self, pred_cog_path: str) -> Dict[str, Any]:
        """Compute  pixels stats from a COG file.

        Args:
            pred_cog_path: Path to the COG file to compute stats from.

        Returns:
            Dictionary containing the segmentation stats.
        """
        try:
            logger.info(f"Computing segmentation stats for {pred_cog_path}")
            pred_cog = Path(pred_cog_path)
            with rasterio.open(pred_cog) as src:
                band = src.read(1, masked=True)
                valid_vals = band.compressed().astype(np.int8)  # Ensure np.int8 for class indices
                total_valid = int(valid_vals.size)
                class_counts: Dict[str, int] = {}
                if total_valid > 0:
                    max_label = int(valid_vals.max()) if valid_vals.size else 0
                    counts_arr = np.bincount(valid_vals, minlength=max_label + 1)
                    class_counts = {str(i): int(c) for i, c in enumerate(counts_arr) if c > 0}
                segmentation_stats = {
                    "valid_pixels": total_valid,
                    "class_counts": class_counts,
                    "unique_values": len(class_counts),
                }
            return segmentation_stats
        except Exception as seg_e:
            logger.warning(f"Segmentation stats computation failed: {seg_e}")
            raise
