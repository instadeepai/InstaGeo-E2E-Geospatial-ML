import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    Box,
    IconButton,
    Alert,
    CircularProgress,
    Card,
    CardContent,
    CardMedia,
    Chip,
    Tooltip
} from '@mui/material';
import {
    Close as CloseIcon,
    Map as MapIcon,
    Download as DownloadIcon
} from '@mui/icons-material';
import { generateTiTilerColormap } from '../utils/segmentationColors';
import { logger } from '../utils/logger';
import apiService from '../services/apiService';
import { useAuth0 } from '@auth0/auth0-react';
import { prefixTitilerUrl } from '../config';

const VisualizationDialog = ({ open, onClose, task, onAddToMap, onCloseTasksMonitor }) => {
    const { getAccessTokenSilently } = useAuth0();
    const [satelliteImageLoading, setSatelliteImageLoading] = useState(true);
    const [satelliteImageError, setSatelliteImageError] = useState(false);
    const [predictionImageLoading, setPredictionImageLoading] = useState(true);
    const [predictionImageError, setPredictionImageError] = useState(false);
    const [predictionStats, setPredictionStats] = useState(null);
    const [statsLoading, setStatsLoading] = useState(false);

    const satellite_params = '&expression=b3;b2;b1&rescale=0,3000';
    // prediction_params will be dynamic based on stats


    // Fetch prediction statistics when component mounts or task changes
    useEffect(() => {
        const fetchPredictionStats = async () => {
            // Ensure visualization data is available before attempting to fetch
            if (!task?.titiler_data) {
                setPredictionStats(null);
                return;
            }
            // Check if this is a segmentation task
            const isSegmentation = task.model_type === 'seg';

            const titiler_data = task?.titiler_data;

            setStatsLoading(true);
            try {

                if (isSegmentation) {

                    const segStats = task?.stages?.visualization_preparation?.result?.segmentation_stats;
                    const classCounts = (segStats && segStats.class_counts) ? segStats.class_counts : {};
                    const uniqueCount = (segStats && typeof segStats.unique_values === 'number') ? segStats.unique_values : 0;
                    const totalValidPixels = (segStats && typeof segStats.valid_pixels === 'number') ? segStats.valid_pixels : 0;

                    // Calculate proportions for each class
                    const classProportions = {};
                    if (classCounts && Object.keys(classCounts).length > 0 && totalValidPixels) {
                        Object.entries(classCounts).forEach(([idx, count]) => {
                            classProportions[Number(idx)] = ((Number(count) / totalValidPixels) * 100).toFixed(1);
                        });
                    }
                    setPredictionStats({
                        type: "seg",
                        class_indices: Object.keys(task.classes_mapping).map(Number).sort((a, b) => a - b), // All possible classes
                        unique_values: uniqueCount, // Store count for display
                        classes_mapping: task.classes_mapping,
                        class_proportions: classProportions, // Add proportions data
                        valid_pixels: totalValidPixels, // Total valid pixels
                    });
                } else {
                    // For regression tasks
                    if (!titiler_data?.prediction?.stats_url) {
                        setPredictionStats(null);
                        return;
                    }
                    const response = await apiService.getTitilerData(titiler_data.prediction.stats_url, getAccessTokenSilently);
                    const stats = await response;
                    setPredictionStats({
                        type: "reg",
                        min: stats.b1?.min || 0,
                        max: stats.b1?.max || 1,
                        histogram: [
                            stats.b1?.histogram?.[0] || [], // counts
                            stats.b1?.histogram?.[1] || []  // bin edges
                        ],
                        valid_pixels: stats.b1?.valid_pixels
                    });
                }
            } catch (error) {
                logger.warn('Failed to fetch prediction statistics:', error);
                if (isSegmentation) {
                    // Fallback to classes_mapping
                    const classIndices = Object.keys(task.classes_mapping).map(Number).sort((a, b) => a - b);
                    const emptyProportions = {};
                    classIndices.forEach(index => {
                        emptyProportions[index] = '0.0';
                    });
                    setPredictionStats({
                        type: "seg",
                        class_indices: classIndices,
                        unique_values: classIndices.length,
                        classes_mapping: task.classes_mapping,
                        class_proportions: emptyProportions
                    });
                } else {
                    setPredictionStats({ type: "reg", min: 0, max: 1 });
                }
            } finally {
                setStatsLoading(false);
            }
        };

        fetchPredictionStats();
    }, [task, getAccessTokenSilently]);

    const getDynamicPredictionParams = () => {
        if (!predictionStats) return null;

        if (predictionStats.type === "seg") {
            // For segmentation, generate custom colormap based on class indices
            const classIndices = predictionStats.class_indices || [];
            const colormapJson = generateTiTilerColormap(classIndices);
            return `&expression=b1&colormap=${encodeURIComponent(colormapJson)}`;
        } else {
            // For regression, use rescaling with actual min/max values
            return `&expression=b1&rescale=${predictionStats.min},${predictionStats.max}&colormap_name=viridis`;
        }
    };

    if (!task?.titiler_data) {
        return null;
    }

    const { titiler_data } = task;

    const handleAddToMap = async () => {
        if (onAddToMap && titiler_data.satellite && titiler_data.prediction) {
            try {
                // Fetch TileJSON for both satellite and prediction to get bounds and zoom levels
                const [satelliteTileJsonResponse, predictionTileJsonResponse] = await Promise.all([
                    apiService.getTitilerData(titiler_data.satellite.tilejson_url, getAccessTokenSilently),
                    apiService.getTitilerData(titiler_data.prediction.tilejson_url, getAccessTokenSilently)
                ]);

                const satelliteTileJson = await satelliteTileJsonResponse;
                const predictionTileJson = await predictionTileJsonResponse;

                // Use satellite bounds as primary (they should be the same for same task)
                // But fall back to prediction bounds if satellite is missing
                const bounds = satelliteTileJson.bounds || predictionTileJson.bounds;

                // Convert bounds to Leaflet format [[south, west], [north, east]]
                const leafletBounds = [
                    [bounds[1], bounds[0]], // [south, west]
                    [bounds[3], bounds[2]]  // [north, east]
                ];

                // Get dynamic zoom levels (use the more restrictive range)
                const minZoom = Math.max(satelliteTileJson.minzoom || 0, predictionTileJson.minzoom || 0);
                const maxZoom = Math.min(satelliteTileJson.maxzoom || 18, predictionTileJson.maxzoom || 18);

                // Add both satellite and prediction layers in one call
                const satelliteRgbTilesUrl = prefixTitilerUrl(titiler_data.satellite.tiles_url) + satellite_params;
                const predictionTilesUrl = prefixTitilerUrl(titiler_data.prediction.tiles_url) + getDynamicPredictionParams();

                onAddToMap({
                    taskId: task.task_id,
                    satelliteTilesUrl: satelliteRgbTilesUrl,
                    predictionTilesUrl: predictionTilesUrl,
                    satellitePreviewUrl: getSatellitePreviewUrl(),
                    predictionPreviewUrl: getPredictionPreviewUrl(),
                    taskName: `Task ${task.task_id}`,
                    name: `Task ${task.task_id}`,
                    bounds: leafletBounds,
                    minZoom: minZoom,
                    maxZoom: maxZoom,
                    predictionStats: predictionStats,
                    modelType: task.model_type,
                    modelShortName: task.model_short_name,
                    modelName: task.model_name,
                    classMapping: task.classes_mapping
                });

                onClose(); // Close dialog after adding to map
                if (onCloseTasksMonitor) {
                    onCloseTasksMonitor(); // Close tasks monitor after adding to map
                }
            } catch (error) {
                logger.error('Error fetching TileJSON:', error);
                // Fallback without bounds and zoom constraints
                const satelliteRgbTilesUrl = prefixTitilerUrl(titiler_data.satellite.tiles_url) + satellite_params;
                const predictionTilesUrl = prefixTitilerUrl(titiler_data.prediction.tiles_url) + getDynamicPredictionParams();

                onAddToMap({
                    taskId: task.task_id,
                    satelliteTilesUrl: satelliteRgbTilesUrl,
                    predictionTilesUrl: predictionTilesUrl,
                    satellitePreviewUrl: getSatellitePreviewUrl(),
                    predictionPreviewUrl: getPredictionPreviewUrl(),
                    taskName: `Task ${task.task_id}`,
                    name: `Task ${task.task_id}`,
                    predictionStats: predictionStats,
                    modelType: task.model_type,
                    modelShortName: task.model_short_name,
                    modelName: task.model_name,
                    classMapping: task.classes_mapping
                });

                onClose();
                if (onCloseTasksMonitor) {
                    onCloseTasksMonitor();
                }
            }
        }
    };

    const getSatellitePreviewUrl = () => {
        if (!titiler_data?.satellite?.preview_url) return '';
        return prefixTitilerUrl(titiler_data.satellite.preview_url + satellite_params);
    };

    const getPredictionPreviewUrl = () => {
        if (!titiler_data?.prediction?.preview_url) return '';
        const params = getDynamicPredictionParams();
        if (!params) return '';
        return prefixTitilerUrl(titiler_data.prediction.preview_url + params);
    };

    const handleSatelliteImageLoad = () => {
        setSatelliteImageLoading(false);
        setSatelliteImageError(false);
    };

    const handleSatelliteImageError = () => {
        setSatelliteImageLoading(false);
        setSatelliteImageError(true);
    };

    const handlePredictionImageLoad = () => {
        setPredictionImageLoading(false);
        setPredictionImageError(false);
    };

    const handlePredictionImageError = () => {
        setPredictionImageLoading(false);
        setPredictionImageError(true);
    };

    const handleDownloadPreview = (url, filename) => {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderBottom: '1px solid',
                borderColor: 'divider',
                py: 2
            }}>
                <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Visualization Preview
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <Typography variant="body2" color="text.secondary">
                            Task ID:
                        </Typography>
                        <Chip
                            label={task.task_id}
                            size="small"
                            variant="outlined"
                            sx={{
                                fontFamily: 'monospace',
                                fontSize: '0.75rem',
                                fontWeight: 'normal'
                            }}
                        />
                    </Box>
                </Box>
                <IconButton onClick={onClose} size="small">
                    <CloseIcon />
                </IconButton>
            </DialogTitle>

            <DialogContent>
                <Box sx={{ mt: 1, display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
                    {/* Satellite Preview */}
                    <Box>
                        <Card>
                            <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                    <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                                        Satellite Data
                                    </Typography>
                                    <Chip label="RGB Composite" size="small" color="primary" />
                                </Box>
                                <Box sx={{
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    minHeight: '200px',
                                    position: 'relative'
                                }}>
                                    {satelliteImageLoading && (
                                        <CircularProgress />
                                    )}

                                    {satelliteImageError && (
                                        <Alert severity="error" size="small">
                                            Failed to load satellite data
                                        </Alert>
                                    )}

                                    {titiler_data.satellite?.preview_url && (
                                        <CardMedia
                                            component="img"
                                            src={getSatellitePreviewUrl()}
                                            alt="Satellite Data Preview"
                                            sx={{
                                                maxWidth: '100%',
                                                maxHeight: '300px',
                                                display: satelliteImageError ? 'none' : 'block',
                                                borderRadius: '4px',
                                                objectFit: 'contain'
                                            }}
                                            onLoad={handleSatelliteImageLoad}
                                            onError={handleSatelliteImageError}
                                        />
                                    )}
                                </Box>
                                {/* Action buttons */}
                                <Box display="flex" justifyContent="center" gap={1} mt={1}>
                                    <Tooltip title="Download preview">
                                        <span>
                                        <IconButton
                                            size="small"
                                            onClick={() => handleDownloadPreview(getSatellitePreviewUrl(), 'satellite_preview.png')}
                                            disabled={satelliteImageError || satelliteImageLoading}
                                        >
                                            <DownloadIcon />
                                        </IconButton>
                                        </span>
                                    </Tooltip>
                                </Box>
                            </CardContent>
                        </Card>
                    </Box>

                    {/* Prediction Preview */}
                    <Box>
                        <Card>
                            <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                    <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                                        Model Prediction
                                    </Typography>
                                    <Chip
                                        label={task.model_short_name}
                                        size="small"
                                        color="secondary"
                                    />
                                </Box>
                                <Box sx={{
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    minHeight: '200px',
                                    position: 'relative'
                                }}>
                                    {statsLoading && (
                                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                                            <CircularProgress size={24} />
                                            <Typography variant="caption" color="text.secondary">
                                                Loading prediction stats...
                                            </Typography>
                                        </Box>
                                    )}

                                    {predictionImageLoading && !statsLoading && (
                                        <CircularProgress />
                                    )}

                                    {predictionImageError && !statsLoading && (
                                        <Alert severity="error" size="small">
                                            Failed to load prediction data
                                        </Alert>
                                    )}

                                    {titiler_data.prediction?.preview_url && !statsLoading && predictionStats && (
                                        <CardMedia
                                            component="img"
                                            src={getPredictionPreviewUrl()}
                                            alt="Model Prediction Preview"
                                            sx={{
                                                maxWidth: '100%',
                                                maxHeight: '300px',
                                                display: predictionImageError ? 'none' : 'block',
                                                borderRadius: '4px',
                                                objectFit: 'contain'
                                            }}
                                            onLoad={handlePredictionImageLoad}
                                            onError={handlePredictionImageError}
                                        />
                                    )}
                                </Box>
                                {/* Action buttons */}
                                <Box display="flex" justifyContent="center" gap={1} mt={1}>
                                    <Tooltip title="Download preview">
                                        <span>
                                        <IconButton
                                            size="small"
                                            onClick={() => handleDownloadPreview(getPredictionPreviewUrl(), 'prediction_preview.png')}
                                            disabled={predictionImageError || predictionImageLoading}
                                        >
                                            <DownloadIcon />
                                        </IconButton>
                                        </span>
                                    </Tooltip>
                                </Box>
                            </CardContent>
                        </Card>
                    </Box>
                </Box>
            </DialogContent>

            <DialogActions sx={{ p: 3 }}>
                <Button onClick={onClose} variant="outlined">
                    Close
                </Button>
                <Button
                    onClick={handleAddToMap}
                    variant="contained"
                    startIcon={<MapIcon />}
                    size="large"
                    disabled={statsLoading || !predictionStats}
                >
                    Add to Map
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default VisualizationDialog;
