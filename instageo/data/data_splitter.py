"""Module for splitting and visualizing geospatial datasets.

This module provides functionality to split geospatial datasets into train, validation,
and test sets while maintaining geographical coherence. It includes visualization tools
for analyzing the distribution of splits across space and time.
"""

import os
import random
import re
from typing import Any, Dict, List, Set, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import mgrs
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags, logging
from haversine import haversine
from matplotlib.patches import Patch
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

# Visualization constants
# These values may need to be tuned for different datasets
VIZ_COLORS = {
    "Train": "#4e79a7",
    "Validation": "#59a14f",
    "Test": "#e15759",
    "Data": "#4e79a7",  # Using same color as Train for consistency
}
VIZ_CMAPS = {
    "Train": "Blues",
    "Validation": "Greens",
    "Test": "Reds",
    "Data": "Blues",  # Using same colormap as Train for consistency
}

VIZ_STYLE: Any = {
    "facecolor": "#2b2b2b",
    "grid_color": "#666666",
    "grid_alpha": 0.3,
    "grid_linestyle": "--",
    "title_color": "white",
    "label_color": "white",
    "legend_facecolor": "#3a3a3a",
    "legend_edgecolor": "#666666",
    "point_alpha": 0.7,
    "point_edgecolor": "white",
    "point_linewidth": 0.1,
    "legend_linewidth": 0.3,
    "gridsize": 40,  # approximately 1000km by 1000km to get a more
    # global density estimate when kde is used
    "bw_adjust": 0.4,  # low value = tighter contours
    "cut": 0.8,  # cut value for kde
    "levels": 5,  # number of levels for kde
    "thresh": 0.4,  # threshold for kde
}

mgrs_object = mgrs.MGRS()
FLAGS = flags.FLAGS

# Global coordinate cache
mgrs_coord_cache: Dict[str, Tuple[float, float]] = {}

# Define flags
flags.DEFINE_integer(
    "random_state", 42, "Random seed for reproducibility", lower_bound=0
)

flags.DEFINE_float(
    "test_ratio",
    0.20,
    "Ratio of data to use for test set",
    lower_bound=0.0,
    upper_bound=1.0,
)

flags.DEFINE_float(
    "val_ratio",
    0.20,
    "Ratio of data to use for validation set",
    lower_bound=0.0,
    upper_bound=1.0,
)

flags.DEFINE_boolean("visualize", True, "Whether to generate visualizations")

flags.DEFINE_boolean("include_val", True, "Whether to include validation split")

flags.DEFINE_boolean("include_test", True, "Whether to include test split")

flags.DEFINE_boolean(
    "allow_group_overlap",
    True,
    """Whether to allow data groups to be split across sets. If False,
    maintains sets integrity but might not achieve exact proportions.""",
)

flags.DEFINE_float(
    "distance_threshold",
    400.0,
    "Maximum distance in kilometers to consider MGRS tiles as close",
    lower_bound=0.0,
)

flags.DEFINE_string("input_file", "", "Path to input CSV file")

flags.DEFINE_string("output_dir", "", "Base directory for output files")

flags.DEFINE_integer("n_clusters", 20, "Number of clusters to create")

flags.DEFINE_bool("use_kmeans", True, "Whether to use KMeans clustering")


# ===== Basic Utility Functions =====
def extract_mgrs_tile(file_path: str) -> str | None:
    """Extract MGRS tile from file path.

    Args:
        file_path: Path to the input file

    Returns:
        MGRS tile code or None if not found
    """
    filename = os.path.basename(file_path)
    # Look for a 4 or 5-character MGRS tile
    match = re.search(r"(\d{1,2}[a-zA-Z]{3})", filename)
    if match:
        # Return MGRS tile in uppercase
        mgrs = match.group(1).upper()
        return mgrs
    return None


def extract_year(file_path: str) -> int | None:
    """Extract a 4-digit year from the file path.

    This function is generalized to work with filenames that may contain
    a year anywhere in the name.

    Args:
        file_path: Path to the input file

    Returns:
        Year as integer or None if not found
    """
    filename = os.path.basename(file_path)
    # Search for all 4-digit sequences [1900-2099]
    matches = re.findall(r"(19[0-9]{2}|20[0-9]{2})", filename)
    if matches:
        return int(matches[0])  # Return the first valid match
    return None


# ===== MGRS Tile Operations =====
def find_connected_tiles(
    tile: str, remaining_tiles: Set[str], distance_threshold: float
) -> Set[str]:
    """Recursively find all tiles connected to the given tile within the distance threshold.

    Args:
        tile: MGRS tile code to find connections for
        remaining_tiles: Set of MGRS tiles to check for connections
        distance_threshold: Maximum distance in kilometers to consider tiles as connected

    Returns:
        Set of connected MGRS tiles including the input tile
    """
    connected = {tile}

    current_coords = mgrs_coord_cache.get(tile)
    if not current_coords:
        return connected

    current_lat, current_lon = current_coords

    # Quick approximation: 1 degree of latitude ≈ 111km, 1 degree of longitude varies with latitude
    lat_threshold = distance_threshold / 111.0  # Convert km to degrees
    lon_threshold = lat_threshold / abs(
        np.cos(np.radians(current_lat))
    )  # Adjust for latitude

    # First pass: Quick filter using lat/lon bounds to reduce number of candidates
    candidates = []
    for other_tile in remaining_tiles:
        other_coords = mgrs_coord_cache.get(other_tile)
        if not other_coords:
            continue

        other_lat, other_lon = other_coords
        # Quick check if tile is within rough bounds
        if (
            abs(other_lat - current_lat) <= lat_threshold
            and abs(other_lon - current_lon) <= lon_threshold
        ):
            candidates.append((other_tile, other_lat, other_lon))

    # Second pass: Calculate exact distances only for candidates
    tile_distances = []
    for other_tile, other_lat, other_lon in candidates:
        distance = haversine((current_lat, current_lon), (other_lat, other_lon))
        if distance <= distance_threshold:
            tile_distances.append((other_tile, distance))

    # Sort by distance
    tiles_to_check = [t for t, _ in sorted(tile_distances, key=lambda x: x[1])]

    for other_tile in tiles_to_check:
        if other_tile in remaining_tiles:
            remaining_tiles.remove(other_tile)
            connected.update(
                find_connected_tiles(other_tile, remaining_tiles, distance_threshold)
            )
    return connected


def group_close_mgrs_tiles(
    mgrs_tiles: List[str], distance_threshold: float = 400.0
) -> List[Set[str]]:
    """Group MGRS tiles that are geographically close together.

    Args:
        mgrs_tiles: List of MGRS tile codes to group
        distance_threshold: Maximum distance in kilometers to consider tiles as close

    Returns:
        List of sets, where each set contains MGRS tiles that are close to each other
    """
    groups = []
    remaining_tiles = set(mgrs_tiles)

    while remaining_tiles:
        # Start with any tile from remaining tiles
        current_tile = next(iter(remaining_tiles))
        remaining_tiles.remove(current_tile)
        group = find_connected_tiles(current_tile, remaining_tiles, distance_threshold)
        groups.append(group)

    return groups


# ===== Visualization Functions =====
def visualize_splits_locations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    output_dir: str = "visualizations",
) -> None:
    """Visualize the geographical distribution of dataset splits as a PNG file.

    Args:
        train_df: Training dataset
        val_df: Validation dataset (optional)
        test_df: Test dataset (optional)
        output_dir: Directory to save the visualization
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(14, 10), facecolor=VIZ_STYLE["facecolor"])
        ax = plt.axes(projection=ccrs.PlateCarree(), facecolor=VIZ_STYLE["facecolor"])

        # Add world map features with lower zorder
        ax.add_feature(cfeature.LAND, facecolor="#3a3a3a", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#1a1a1a", zorder=0)
        ax.add_feature(
            cfeature.COASTLINE,
            edgecolor=VIZ_STYLE["grid_color"],
            linewidth=0.5,
            zorder=0,
        )
        ax.add_feature(
            cfeature.BORDERS,
            linestyle="-",
            edgecolor=VIZ_STYLE["grid_color"],
            linewidth=0.5,
            zorder=0,
        )

        ax.set_global()

        has_data = False
        legend_handles = []

        # Determine if we should show "Data" instead of "Train"
        show_data = val_df is None and test_df is None

        # Plot each dataset
        for df, color, label in [
            (
                train_df,
                VIZ_COLORS["Data" if show_data else "Train"],
                "Data" if show_data else "Train",
            ),
            (val_df, VIZ_COLORS["Validation"], "Validation"),
            (test_df, VIZ_COLORS["Test"], "Test"),
        ]:
            if df is not None and not df.empty:
                # Get coordinates for all MGRS tiles at once
                coords = df["mgrs_tile"].map(mgrs_coord_cache)
                valid_coords = coords[coords.notna()]

                if not valid_coords.empty:
                    # Create points from coordinates
                    points = [Point(lon, lat) for lat, lon in valid_coords]
                    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

                    # Add points to plot
                    gdf.plot(
                        ax=ax,
                        color=color,
                        alpha=VIZ_STYLE["point_alpha"],
                        edgecolor=VIZ_STYLE["point_edgecolor"],
                        linewidth=VIZ_STYLE["point_linewidth"],
                        label=label,
                        zorder=2,
                        markersize=8,
                    )

                    # Create KDE plot with colorbar
                    gridsize = VIZ_STYLE["gridsize"]
                    bw_adjust = VIZ_STYLE["bw_adjust"]
                    levels = VIZ_STYLE["levels"]
                    thresh = VIZ_STYLE["thresh"]
                    cut = VIZ_STYLE["cut"]
                    kde = sns.kdeplot(
                        x=gdf.geometry.x,
                        y=gdf.geometry.y,
                        cmap=VIZ_CMAPS[label],
                        fill=True,
                        alpha=0.7,
                        bw_adjust=bw_adjust,
                        levels=levels,
                        thresh=thresh,
                        cut=cut,
                        ax=ax,
                        cbar=True,
                        cbar_kws={
                            "label": f"{label} Count",
                            "orientation": "horizontal",
                            "pad": -0.2,
                            "shrink": 0.2,
                            "aspect": 30,
                            "location": "bottom",
                            "anchor": (0.02, 0.5),
                            "panchor": (0.02, -0.2),
                        },
                        gridsize=gridsize,
                    )

                    # Style the colorbar
                    cbar = kde.collections[-1].colorbar
                    cbar.ax.set_facecolor(VIZ_STYLE["legend_facecolor"])
                    cbar.ax.tick_params(colors=VIZ_STYLE["label_color"])

                    # Calculate grid area for count conversion
                    x_min, x_max = (
                        gdf.geometry.x.min() - bw_adjust * cut,
                        gdf.geometry.x.max() + bw_adjust * cut,
                    )
                    y_min, y_max = (
                        gdf.geometry.y.min() - bw_adjust * cut,
                        gdf.geometry.y.max() + bw_adjust * cut,
                    )
                    # Calculate grid cell area in square degrees
                    grid_area = ((x_max - x_min) * (y_max - y_min)) / (
                        (gridsize - 1) ** 2
                    )

                    # Set all ticks with proper count conversion
                    ticks = cbar.get_ticks()
                    # Convert density to counts: density * grid_area * number_of_points
                    counts = [int(tick * grid_area * len(gdf)) for tick in ticks]
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{count}" for count in counts])
                    cbar.ax.set_xlabel("")
                    cbar.ax.set_ylabel("")

                    # Add legend handle for this dataset
                    patch = Patch(
                        facecolor=color,
                        edgecolor=VIZ_STYLE["point_edgecolor"],
                        linewidth=VIZ_STYLE["legend_linewidth"],
                        alpha=VIZ_STYLE["point_alpha"],
                        label=label,
                    )
                    legend_handles.append(patch)
                    has_data = True

        if has_data:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                frameon=True,
                facecolor=VIZ_STYLE["legend_facecolor"],
                edgecolor=VIZ_STYLE["legend_edgecolor"],
                labelcolor=VIZ_STYLE["label_color"],
                fontsize=10,
                bbox_to_anchor=(0.01, 0.5),
            )

            ax.set_title(
                "Distribution of data across the world"
                if show_data
                else "Distribution of splits across the world",
                color=VIZ_STYLE["title_color"],
                fontsize=14,
                pad=20,
            )
            ax.text(
                0.01,
                0.25 if show_data else 0.35,
                "Density based estimated counts\n(1000km x 1000km)",
                transform=ax.transAxes,
                color=VIZ_STYLE["label_color"],
                fontsize=8,
            )
            # Set gridlines approximately every 1000km (roughly 9 degrees at equator)
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                xlocs=np.arange(-180, 180, 9),
                ylocs=np.arange(-90, 90, 9),
                linewidth=0.5,
                color=VIZ_STYLE["grid_color"],
                alpha=VIZ_STYLE["grid_alpha"],
                linestyle=VIZ_STYLE["grid_linestyle"],
            )
            gl.top_labels = False
            gl.right_labels = False

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "splits_map.png"),
                dpi=300,
                bbox_inches="tight",
                facecolor=VIZ_STYLE["facecolor"],
            )
            plt.close()

    except Exception as e:
        logging.error(f"Error creating PNG export: {str(e)}")


def visualize_splits_years(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    output_dir: str = "visualizations",
) -> None:
    """Visualize the year-based distribution of the dataset splits.

    Args:
        train_df: Training dataset
        val_df: Validation dataset (optional)
        test_df: Test dataset (optional)
        output_dir: Directory to save the visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    years = []
    sets = []

    # Determine if we should show "Data" instead of "Train"
    show_data = val_df is None and test_df is None

    if test_df is not None:
        test_years = test_df["Input"].apply(extract_year)
        years.extend(test_years)
        sets.extend(["Test"] * len(test_years))

    if val_df is not None:
        val_years = val_df["Input"].apply(extract_year)
        years.extend(val_years)
        sets.extend(["Validation"] * len(val_years))

    train_years = train_df["Input"].apply(extract_year)
    years.extend(train_years)
    sets.extend(["Data" if show_data else "Train"] * len(train_years))

    viz_df = pd.DataFrame({"Year": years, "Set": sets})

    plt.figure(figsize=(12, 6), facecolor=VIZ_STYLE["facecolor"])
    ax = plt.axes(facecolor=VIZ_STYLE["facecolor"])

    sns.histplot(
        data=viz_df,
        x="Year",
        hue="Set",
        multiple="stack",
        discrete=True,
        shrink=0.75,
        palette=VIZ_COLORS,
        ax=ax,
    )

    unique_years = sorted(viz_df["Year"].unique())
    ax.set_xticks(unique_years)
    ax.set_xticklabels(unique_years, color=VIZ_STYLE["label_color"])

    ax.set_facecolor(VIZ_STYLE["facecolor"])
    ax.grid(
        True,
        color=VIZ_STYLE["grid_color"],
        alpha=VIZ_STYLE["grid_alpha"],
        linestyle=VIZ_STYLE["grid_linestyle"],
    )

    ax.set_title(
        "Distribution of data over the years"
        if show_data
        else "Distribution of splits over the years",
        color=VIZ_STYLE["title_color"],
        pad=20,
    )
    ax.set_xlabel("Year", color=VIZ_STYLE["label_color"])
    ax.set_ylabel("Count", color=VIZ_STYLE["label_color"])

    # Create legend handles manually
    legend_handles = []
    for label, color in VIZ_COLORS.items():
        if label in viz_df["Set"].unique():
            patch = Patch(
                facecolor=color,
                edgecolor=VIZ_STYLE["point_edgecolor"],
                linewidth=VIZ_STYLE["legend_linewidth"],
                alpha=VIZ_STYLE["point_alpha"],
                label=label,
            )
            legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor=VIZ_STYLE["legend_facecolor"],
        edgecolor=VIZ_STYLE["legend_edgecolor"],
        labelcolor=VIZ_STYLE["label_color"],
        fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_color(VIZ_STYLE["grid_color"])

    plt.xticks(rotation=45, color=VIZ_STYLE["label_color"])
    plt.yticks(color=VIZ_STYLE["label_color"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "splits_distribution_by_year.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=VIZ_STYLE["facecolor"],
    )
    plt.close()


# ===== Main Dataset Splitting Functions =====
def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    base_output_dir: str = "dataset_splits",
    save_viz: bool = True,
) -> None:
    """Save dataset splits to CSV files and generate visualizations.

    Args:
        train_df: Training dataset
        val_df: Validation dataset (optional)
        test_df: Test dataset (optional)
        base_output_dir: Directory to save the splits and visualizations
        save_viz: Whether to generate visualizations
    """
    os.makedirs(base_output_dir, exist_ok=True)
    splits_dir = os.path.join(base_output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    # Save train set
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)

    # Save validation set if it exists
    if val_df is not None:
        val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)

    # Save test set if it exists
    if test_df is not None:
        test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # Create visualizations if requested
    if save_viz:
        viz_dir = os.path.join(splits_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        visualize_splits_locations(train_df, val_df, test_df, viz_dir)
        visualize_splits_years(train_df, val_df, test_df, viz_dir)


def _try_mgrs_groups(
    df: pd.DataFrame, distance_threshold: float
) -> List[Set[str]] | None:
    """Try to create groups based on MGRS tiles proximity.

    Args:
        df: Input DataFrame
        distance_threshold: Maximum distance in kilometers to consider MGRS tiles as close

    Returns:
        List of sets of MGRS tiles that are close to each other, or None if not enough groups
    """
    valid_mgrs = df["mgrs_tile"].unique()
    if len(valid_mgrs) < 2:
        logging.info("Not enough MGRS tiles for grouping")
        return None

    mgrs_groups = group_close_mgrs_tiles(valid_mgrs, distance_threshold)
    logging.info(f"Created {len(mgrs_groups)} MGRS groups")
    return mgrs_groups


def _try_year_groups(df: pd.DataFrame) -> List[Set[str]] | None:
    """Try to create groups based on years.

    Args:
        df: Input DataFrame

    Returns:
        List of sets containing years, or None if not enough years
    """
    years = sorted(df["year"].unique(), reverse=True)
    groups = [{year} for year in years]
    logging.info(f"Created {len(groups)} year groups")
    return groups


def _try_random_split(
    df: pd.DataFrame,
    random_state: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    include_test: bool = True,
    include_val: bool = True,
) -> Tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None]:
    """Create random splits using sklearn's train_test_split.

    Args:
        df: Input DataFrame
        random_state: Random seed for reproducibility
        test_ratio: Ratio of data to use for test set
        val_ratio: Ratio of data to use for validation set
        include_test: Whether to include test split
        include_val: Whether to include validation split

    Returns:
        Tuple of (test_df, train_df, val_df) where each can be None if not included
    """
    try:
        test_df = None
        val_df = None
        train_df = df.copy()

        if include_test:
            # Split into train+val and test
            train_val_df, test_df = train_test_split(
                df, test_size=test_ratio, random_state=random_state
            )
            train_df = train_val_df

        if include_val:
            # Split remaining data into train and val
            # Adjust val_ratio to account for the first split if test was included
            adjusted_val_ratio = (
                val_ratio / (1 - test_ratio) if include_test else val_ratio
            )
            train_df, val_df = train_test_split(
                train_df, test_size=adjusted_val_ratio, random_state=random_state
            )

        logging.info(
            f"Created random splits: test={len(test_df) if test_df is not None else 0}, "
            f"train={len(train_df)}, val={len(val_df) if val_df is not None else 0}"
        )
        return test_df, train_df, val_df
    except Exception as e:
        logging.error(f"Error in random splitting: {str(e)}")
        return None, df, None


def _split_data(
    df: pd.DataFrame,
    groups: List[Set[str]],
    test_ratio: float,
    val_ratio: float,
    include_test: bool,
    include_val: bool,
    allow_group_overlap: bool = True,
) -> Tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None]:
    """Helper function to split data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        groups: List of sets containing group IDs
        test_ratio: Ratio of data to use for test set
        val_ratio: Ratio of data to use for validation set
        include_test: Whether to include test split
        include_val: Whether to include validation split
        allow_group_overlap: Whether to allow groups to be split across sets

    Returns:
        Tuple of (test_df, train_df, val_df) where each can be None if not included
    """
    # Assign group IDs to each record
    df["group_id"] = -1  #
    for i, group in enumerate(groups):
        mask = df["mask"].isin(group)
        df.loc[mask, "group_id"] = i

    # Calculate target sizes
    target_test_size = int(len(df) * test_ratio) if include_test else 0
    target_val_size = int(len(df) * val_ratio) if include_val else 0
    target_train_size = len(df) - target_test_size - target_val_size

    # Calculate average year for each group (relevant for mgrs based groups)
    group_years = []

    for i in range(len(groups)):
        group_data = df[df["group_id"] == i]
        if len(group_data) > 0:
            avg_year = group_data["year"].mean()
            group_years.append((i, avg_year, len(group_data)))

    # Initialize sets
    test_df = None
    val_df = None
    train_df = None

    if include_test:
        # Get test set - always take most recent groups
        # Sort groups by average year
        group_years.sort(key=lambda x: x[1], reverse=True)
        test_records = []
        current_size = 0
        i = 0
        while current_size <= target_test_size and i < len(group_years):
            current_group = df[df["group_id"] == group_years[i][0]]
            current_size += len(current_group)
            test_records.extend(current_group.index)
            i += 1

        test_records = test_records[:target_test_size]
        test_df = df.loc[test_records].copy()
        logging.info(f"Test set size: {len(test_df)} (target: {target_test_size})")

    # Get remaining data for train/val
    remaining_df = df[~df.index.isin(test_df.index)] if test_df is not None else df

    # Shuffle groups for train/val split
    test_groups = set(test_df["group_id"].unique()) if test_df is not None else set()
    remaining_groups = (
        group_years
        if allow_group_overlap
        else [g for g in group_years if g[0] not in test_groups]
    )
    random.shuffle(remaining_groups)

    if include_val:
        # Get validation set
        val_records = []
        current_size = 0
        i = 0
        while current_size < target_val_size and i < len(remaining_groups):
            current_group = remaining_df[
                remaining_df["group_id"] == remaining_groups[i][0]
            ]
            current_size += len(current_group)
            val_records.extend(current_group.index)
            i += 1

        val_records = val_records[:target_val_size]
        val_df = remaining_df.loc[val_records].copy()
        logging.info(f"Validation set size: {len(val_df)} (target: {target_val_size})")

    # Get training set
    train_df = (
        remaining_df[~remaining_df.index.isin(val_df.index)]
        if val_df is not None
        else remaining_df
    )
    val_groups = set(val_df["group_id"].unique()) if val_df is not None else set()
    remaining_groups = (
        remaining_groups
        if allow_group_overlap
        else [g for g in remaining_groups if g[0] not in val_groups]
    )
    if not allow_group_overlap:
        train_df = train_df[train_df.group_id.isin([g[0] for g in remaining_groups])]
    logging.info(f"Train set size: {len(train_df)} (target: {target_train_size})")

    # Clean up temporary column
    for df_split in [test_df, train_df, val_df]:
        if df_split is not None:
            df_split.drop(columns=["mask", "group_id"], inplace=True)

    return test_df, train_df, val_df


def find_closest_clusters(
    centroids: np.ndarray, available_clusters: Set[int]
) -> Tuple[int, int] | None:
    """Function to find closest clusters.

    Args:
        centroids: np.ndarray of cluster centroids
        available_clusters: Set of available clusters

    Returns:
        Tuple of (int, int) of the closest pair of clusters, or None if no closest pair is found
    """
    min_dist = float("inf")
    closest_pair = None

    for i in available_clusters:
        for j in available_clusters:
            if i < j:  # Avoid comparing same cluster and duplicates
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (i, j)

    return closest_pair


def merge_clusters(
    cluster1: int, cluster2: int, df_with_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Function to merge two clusters.

    Args:
        cluster1: int of the first cluster
        cluster2: int of the second cluster
        df_with_clusters: pd.DataFrame with clusters

    Returns:
        pd.DataFrame with clusters
    """
    # Assign all points from cluster2 to cluster1
    df_with_clusters.loc[df_with_clusters["cluster"] == cluster2, "cluster"] = cluster1
    return df_with_clusters


def _try_kmeans_groups(df: pd.DataFrame, n_clusters: int) -> None:
    """Try to create groups based on KMeans clustering.

    Args:
        df: Input DataFrame
        n_clusters: Number of clusters to create
    """
    df = df.copy()
    df[["lat", "lon"]] = df["mgrs_tile"].apply(
        lambda x: pd.Series(mgrs_coord_cache.get(x))
    )
    std = StandardScaler()
    df[["lat", "lon"]] = std.fit_transform(df[["lat", "lon"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["lat", "lon"]])

    # Calculate target sizes
    total_size = len(df)
    target_test_size = int(total_size * FLAGS.test_ratio)
    target_val_size = int(total_size * FLAGS.val_ratio)

    # Get cluster centroids
    cluster_centroids = kmeans.cluster_centers_

    # Initialize splits
    test_clusters = set()
    val_clusters = set()
    train_clusters = set()

    # Start with all clusters available
    available_clusters = set(range(n_clusters))
    current_test_size = 0
    current_val_size = 0

    # Merge clusters for test set
    while current_test_size < target_test_size and len(available_clusters) > 1:
        # Find closest pair of clusters
        closest_pair = find_closest_clusters(cluster_centroids, available_clusters)
        if closest_pair is None:
            break

        cluster1, cluster2 = closest_pair

        # Merge the clusters
        df = merge_clusters(cluster1, cluster2, df)

        # Update available clusters
        available_clusters.remove(cluster2)
        test_clusters.add(cluster1)

        # Update current size
        current_test_size = len(df[df["cluster"].isin(test_clusters)])

    # Get remaining data for validation set
    remaining_df = df[~df["cluster"].isin(test_clusters)]
    available_clusters = available_clusters - test_clusters

    # Merge clusters for validation set
    while current_val_size < target_val_size and len(available_clusters) > 1:
        # Find closest pair of clusters
        closest_pair = find_closest_clusters(cluster_centroids, available_clusters)
        if closest_pair is None:
            break

        cluster1, cluster2 = closest_pair

        # Merge the clusters
        remaining_df = merge_clusters(cluster1, cluster2, remaining_df)

        # Update available clusters
        available_clusters.remove(cluster2)
        val_clusters.add(cluster1)

        # Update current size
        current_val_size = len(remaining_df[remaining_df["cluster"].isin(val_clusters)])

    # Get training set (everything else)
    train_clusters = available_clusters - val_clusters

    # Create the splits
    test_df = df[df["cluster"].isin(test_clusters)].copy()
    val_df = remaining_df[remaining_df["cluster"].isin(val_clusters)].copy()
    train_df = remaining_df[remaining_df["cluster"].isin(train_clusters)].copy()

    # Clean up temporary columns
    for df_split in [test_df, val_df, train_df]:
        if df_split is not None:
            df_split.drop(columns=["cluster", "lat", "lon"], inplace=True)

    logging.info(
        f"KMeans splits created: test={len(test_df)}, val={len(val_df)}, train={len(train_df)}"
    )

    # Save the splits
    save_splits(train_df, val_df, test_df, FLAGS.output_dir, FLAGS.visualize)

    return None  # Return None since we've already saved the splits


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_viz: bool = True,
    include_val: bool = True,
    include_test: bool = True,
    distance_threshold: float = 400.0,
    output_dir: str = "dataset_splits",
    allow_group_overlap: bool = True,
) -> None:
    """Split dataset into train, validation, and test sets.

    First tries to split based on KMeans clustering, then attempts to split based
    on MGRS tiles,
    keeping geographically close
    tiles together and ensuring test set has more recent years. If not enough
    groups are available, falls back to splitting by years only. If year-based
    splitting is not possible, falls back to random splitting.

    Args:
        df: Input DataFrame
        val_ratio: Ratio of data to use for validation set
        test_ratio: Ratio of data to use for test set
        random_state: Random seed for reproducibility
        save_viz: Whether to generate visualizations
        include_val: Whether to include validation split
        include_test: Whether to include test split
        distance_threshold: Maximum distance in kilometers to consider MGRS tiles as close
        output_dir: Base directory for output files
        allow_group_overlap: Whether to allow groups to be split across sets
    """
    random.seed(random_state)
    np.random.seed(random_state)
    pd.set_option("mode.chained_assignment", None)

    # Try KMeans clustering first if enabled
    if FLAGS.use_kmeans:
        logging.info("Using KMeans clustering strategy")
        _try_kmeans_groups(df, n_clusters=FLAGS.n_clusters)
        return

    # Try MGRS tile grouping first
    mgrs_groups = _try_mgrs_groups(df, distance_threshold)
    if mgrs_groups and len(mgrs_groups) >= 2:
        logging.info("Using MGRS tile grouping strategy")
        df["mask"] = df["mgrs_tile"]
        test_df, train_df, val_df = _split_data(
            df,
            mgrs_groups,
            test_ratio,
            val_ratio,
            include_test,
            include_val,
            allow_group_overlap,
        )
        save_splits(train_df, val_df, test_df, output_dir, save_viz)
        return

    # Try year-based splitting next
    year_groups = _try_year_groups(df)
    if year_groups and len(year_groups) >= 2:
        logging.info("Using year-based splitting strategy")
        df["mask"] = df["year"]
        test_df, train_df, val_df = _split_data(
            df,
            year_groups,
            test_ratio,
            val_ratio,
            include_test,
            include_val,
            allow_group_overlap,
        )
        save_splits(train_df, val_df, test_df, output_dir, save_viz)
        return

    # Fall back to random splitting
    logging.info("Using random splitting strategy")
    test_df, train_df, val_df = _try_random_split(
        df, random_state, test_ratio, val_ratio, include_test, include_val
    )
    if test_df is not None or val_df is not None or train_df is not None:
        save_splits(train_df, val_df, test_df, output_dir, save_viz)
        return

    raise ValueError("Failed to create valid splits with any strategy")


def main(argv: Any) -> None:
    """Split dataset into train, validation, and test sets.

    Args:
        argv: Command line arguments
    """
    del argv

    df = pd.read_csv(FLAGS.input_file)

    # Extract MGRS tiles and populate coordinate cache
    df["mgrs_tile"] = df["Input"].apply(extract_mgrs_tile)
    df["year"] = df["Input"].apply(extract_year)

    # Filter out records with invalid MGRS tiles or years
    valid_mask = df["mgrs_tile"].notna() & df["year"].notna()
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        logging.warning(
            f"Filtering out {invalid_count} records with invalid MGRS tiles or years"
        )
        df = df[valid_mask].copy()

    # Populate cache with all unique MGRS tiles
    unique_mgrs = df["mgrs_tile"].unique()
    for mgrs_code in unique_mgrs:
        try:
            mgrs_coord_cache[mgrs_code] = mgrs_object.toLatLon(f"{mgrs_code}")
        except (ValueError, IndexError):
            continue

    split_dataset(
        df,
        test_ratio=FLAGS.test_ratio,
        val_ratio=FLAGS.val_ratio,
        save_viz=FLAGS.visualize,
        include_val=FLAGS.include_val,
        include_test=FLAGS.include_test,
        distance_threshold=FLAGS.distance_threshold,
        random_state=FLAGS.random_state,
        output_dir=FLAGS.output_dir,
        allow_group_overlap=FLAGS.allow_group_overlap,
    )


if __name__ == "__main__":
    app.run(main)
