"""Minimal TiTiler service to visualize tasks.

This service exposes endpoints to visualize tasks in the frontend.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from titiler.core.factory import TilerFactory

from instageo.new_apps.backend.app.auth import is_task_owner, verify_access_token
from instageo.new_apps.backend.app.db import get_db

logger = logging.getLogger(__name__)


class InstaGeoTilerService:
    """Minimal TiTiler service to visualize tasks."""

    def __init__(self, base_directory: str = "/app/instageo-data"):
        """Initialize TiTiler service.

        Args:
            base_directory: Base directory containing task files.
        """
        self.base_directory = Path(base_directory)

        # Create TiTiler factory with minimal configuration
        self.tiler = TilerFactory()

        # Custom router for task-based endpoints
        self.router = APIRouter(
            prefix="/api/visualize",
            tags=["visualization"],
            dependencies=[Depends(verify_access_token)],
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up routes."""

        @self.router.get("/{task_id}")
        @is_task_owner
        async def visualize_task(
            task_id: str,
            db: Session = Depends(get_db),
            claims: Dict[str, Any] = Depends(verify_access_token),
        ) -> Dict[str, Any]:
            """Get visualization data for a task."""
            try:
                # Try to get COG files for this task
                self._get_chips_cog_file(task_id)
                self._get_predictions_cog_file(task_id)

                # Return safe URLs with task_id instead of file paths
                # The middleware will map task_id to actual file paths
                return {
                    "task_id": task_id,
                    "satellite": {
                        "tiles_url": (
                            f"/api/titiler/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x.png"
                            f"?url={task_id}_chips"
                        ),
                        "tilejson_url": (
                            f"/api/titiler/WebMercatorQuad/tilejson.json" f"?url={task_id}_chips"
                        ),
                        "preview_url": f"/api/titiler/preview.png?url={task_id}_chips",
                        "stats_url": f"/api/titiler/statistics?url={task_id}_chips",
                    },
                    "prediction": {
                        "tiles_url": (
                            f"/api/titiler/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x.png"
                            f"?url={task_id}_predictions"
                        ),
                        "tilejson_url": (
                            f"/api/titiler/WebMercatorQuad/tilejson.json"
                            f"?url={task_id}_predictions"
                        ),
                        "preview_url": f"/api/titiler/preview.png?url={task_id}_predictions",
                        "stats_url": f"/api/titiler/statistics?url={task_id}_predictions",
                    },
                    "status": "ready",
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get visualization for task {task_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _get_chips_cog_file(self, task_id: str) -> Path:
        """Get satellite chips COG file path for task_id."""
        task_cog_dir = self.base_directory / task_id / "data" / "cogs"
        cog_file = task_cog_dir / "chips_merged.tif"

        if not cog_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Satellite chips COG file not found for task {task_id}",
            )

        return cog_file

    def _get_predictions_cog_file(self, task_id: str) -> Path:
        """Get predictions COG file path for task_id."""
        task_cog_dir = self.base_directory / task_id / "data" / "cogs"
        cog_file = task_cog_dir / "predictions_merged.tif"

        if not cog_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Predictions COG file not found for task {task_id}",
            )

        return cog_file

    def get_router(self) -> APIRouter:
        """Get the custom router."""
        return self.router

    def get_tiler_router(self) -> APIRouter:
        """Get the TiTiler router."""
        return self.tiler.router
