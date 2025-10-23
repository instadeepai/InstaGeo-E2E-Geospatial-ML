import React, { useState, useEffect, useRef } from 'react';
import { useMap } from 'react-leaflet';
import { useAuth0 } from '@auth0/auth0-react';
import L from 'leaflet';
import {
    Box,
    Paper,
    Typography,
    IconButton,
    Collapse,
    Slider,
    Divider,
    Tooltip
} from '@mui/material';
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Layers as LayersIcon,
    Delete as DeleteIcon,
    ZoomIn as ZoomInIcon,
    Visibility as VisibilityIcon,
    VisibilityOff as VisibilityOffIcon,
    PictureAsPdf as PictureAsPdfIcon,
} from '@mui/icons-material';
import { createPortal } from 'react-dom';
import { generateSegmentationColors } from '../utils/segmentationColors';
import { generateTaskPdf } from '../utils/pdfReport';
import { logger } from '../utils/logger';

const TaskLayersControl = ({ taskLayers = [], onTaskLayerChange }) => {
    const map = useMap();
    const controlRef = useRef(null);
    const { getAccessTokenSilently } = useAuth0();
    const controlInstanceRef = useRef(null);
    const [expanded, setExpanded] = useState(true);
    const [taskCollapsed, setTaskCollapsed] = useState({}); // Track collapsed state per task

    logger.log('TaskLayersControl render:', { taskLayersCount: taskLayers.length });

    // Create layer control container (always present)
    useEffect(() => {
        if (!map || controlInstanceRef.current) return;

        logger.log('Creating unified task layers control container');

        // Create control container
        const TaskLayersControlClass = L.Control.extend({
            onAdd: function () {
                const div = L.DomUtil.create('div', 'task-layers-control');
                div.style.background = 'transparent';
                div.style.padding = '0';
                div.style.border = 'none';

                // Prevent map interactions on control
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);

                controlRef.current = div;
                logger.log('Task layers control div created');
                return div;
            }
        });

        const control = new TaskLayersControlClass({ position: 'bottomleft' });
        control.addTo(map);
        controlInstanceRef.current = control;
        logger.log('Task layers control added to map');

        return () => {
            logger.log('Removing task layers control');
            if (controlInstanceRef.current) {
                map.removeControl(controlInstanceRef.current);
                controlInstanceRef.current = null;
            }
        };
    }, [map]);

    // Handle task layer visibility changes
    const handleTaskLayerVisibilityChange = (taskLayerId, layerType, visible) => {
        if (onTaskLayerChange) {
            onTaskLayerChange(taskLayerId, layerType, 'visibility', visible);
        }
    };

    // Handle task layer opacity changes
    const handleTaskLayerOpacityChange = (taskLayerId, layerType, opacity) => {
        if (onTaskLayerChange) {
            onTaskLayerChange(taskLayerId, layerType, 'opacity', opacity);
        }
    };

    // Handle task collapse toggle
    const toggleTaskCollapse = (taskId) => {
        setTaskCollapsed(prev => ({
            ...prev,
            [taskId]: !prev[taskId]
        }));
    };

    // Handle individual layer removal
    const handleRemoveLayer = (taskLayerId) => {
        if (onTaskLayerChange) {
            onTaskLayerChange(taskLayerId, null, 'remove', null);
        }
    };

    // Handle remove all layers
    const handleRemoveAll = () => {
        if (onTaskLayerChange) {
            taskLayers.forEach(taskLayer => {
                onTaskLayerChange(taskLayer.id, null, 'remove', null);
            });
        }
    };

    // Handle zoom to task layers
    const handleZoomToTask = (taskLayer) => {
        if (map && taskLayer.bounds && Array.isArray(taskLayer.bounds) && taskLayer.bounds.length === 2) {
            try {
                // Calculate center of the bounds
                const bounds = taskLayer.bounds;
                const centerLat = (bounds[0][0] + bounds[1][0]) / 2;
                const centerLng = (bounds[0][1] + bounds[1][1]) / 2;
                const center = [centerLat, centerLng];

                // Set zoom level first
                let targetZoom;
                if (taskLayer.maxZoom && taskLayer.maxZoom > 0) {
                    targetZoom = taskLayer.maxZoom;
                } else {
                    // Fallback: calculate appropriate zoom based on bounds size
                    const latDiff = Math.abs(bounds[1][0] - bounds[0][0]);
                    const lngDiff = Math.abs(bounds[1][1] - bounds[0][1]);
                    const maxDiff = Math.max(latDiff, lngDiff);

                    // Calculate zoom level based on bounds size
                    if (maxDiff > 1) targetZoom = 8;
                    else if (maxDiff > 0.5) targetZoom = 10;
                    else if (maxDiff > 0.1) targetZoom = 12;
                    else if (maxDiff > 0.05) targetZoom = 14;
                    else targetZoom = 16;
                }

                // Center the map and set zoom
                map.setView(center, targetZoom);

                logger.log('Centered map on task bounds:', center, 'zoom:', targetZoom);
            } catch (error) {
                logger.error('Error zooming to task bounds:', error);
            }
        } else {
            logger.warn('No valid bounds available for task:', taskLayer.taskName);
        }
    };

    // Render control content
    const renderControl = () => {
        logger.log('renderControl called:', {
            hasControlRef: !!controlRef.current,
            taskLayersCount: taskLayers.length
        });

        if (!controlRef.current || taskLayers.length === 0) return null;

        return createPortal(
            <Paper
                elevation={3}
                sx={{
                    minWidth: expanded ? 280 : 'auto',
                    maxWidth: 320,
                    backgroundColor: 'background.paper',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid',
                    borderColor: 'divider',
                    overflowX: 'hidden' // Prevent horizontal scrolling
                }}
            >
                {/* Header */}
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 1.5,
                        pb: expanded ? 1 : 1.5,
                        backgroundColor: 'primary.main',
                        color: 'white'
                    }}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LayersIcon fontSize="small" />
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            Task Layers ({taskLayers.length})
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {taskLayers.length > 0 && (
                            <Tooltip title="Remove All Tasks Layers">
                                <IconButton
                                    size="small"
                                    onClick={handleRemoveAll}
                                    sx={{
                                        color: 'white',
                                        '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
                                    }}
                                >
                                    <DeleteIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        )}
                        <Tooltip title={expanded ? "Collapse Panel" : "Expand Panel"}>
                    <IconButton
                        size="small"
                        onClick={() => setExpanded(!expanded)}
                        sx={{
                            color: 'white',
                            '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
                        }}
                    >
                        {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                    </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

                {/* Collapsible Content */}
                <Collapse in={expanded}>
                    <Box
                        sx={{
                            p: 2,
                            pt: 1.5,
                            maxHeight: '400px', // Limit height to prevent panel from growing too large
                            overflowY: 'auto', // Add vertical scrolling when needed
                            overflowX: 'hidden', // Prevent horizontal scrolling
                            '&::-webkit-scrollbar': {
                                width: '0px',
                                height: '0px',
                            },
                            '&::-webkit-scrollbar-track': {
                                backgroundColor: 'transparent',
                            },
                            '&::-webkit-scrollbar-thumb': {
                                backgroundColor: 'transparent',
                            },
                        }}
                    >
                        {taskLayers.slice().reverse().map((taskLayer, index) => (
                            <Box key={taskLayer.id} mb={index < taskLayers.length - 1 ? 3 : 0}>
                                {/* Task Header */}
                                <Box sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    mb: 1.5,
                                    borderBottom: '1px solid',
                                    borderColor: 'divider',
                                    pb: 0.5
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Tooltip title={taskCollapsed[taskLayer.id] ? "Expand Task" : "Collapse Task"}>
                                            <IconButton
                                                size="small"
                                                onClick={() => toggleTaskCollapse(taskLayer.id)}
                                                sx={{
                                                    color: 'primary.main',
                                                    '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' }
                                                }}
                                            >
                                                {taskCollapsed[taskLayer.id] ? <ExpandMoreIcon fontSize="small" /> : <ExpandLessIcon fontSize="small" />}
                                            </IconButton>
                                        </Tooltip>
                                <Typography
                                    variant="subtitle2"
                                    sx={{
                                        fontWeight: 600,
                                                color: 'primary.main'
                                            }}
                                        >
                                            {taskLayer.taskName}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Tooltip title={`${(taskLayer.satelliteVisible !== false && taskLayer.predictionVisible !== false) ? 'Hide' : 'Show'} All Task Layers`}>
                                            <IconButton
                                                size="small"
                                                onClick={() => {
                                                    const newVisibility = !(taskLayer.satelliteVisible !== false && taskLayer.predictionVisible !== false);
                                                    handleTaskLayerVisibilityChange(taskLayer.id, 'satellite', newVisibility);
                                                    handleTaskLayerVisibilityChange(taskLayer.id, 'prediction', newVisibility);
                                                }}
                                                sx={{
                                                    color: (taskLayer.satelliteVisible !== false && taskLayer.predictionVisible !== false) ? 'primary.main' : 'text.disabled',
                                                    '&:hover': { backgroundColor: 'rgba(25, 118, 210, 0.04)' }
                                                }}
                                            >
                                                {(taskLayer.satelliteVisible !== false && taskLayer.predictionVisible !== false) ?
                                                    <VisibilityIcon fontSize="small" /> :
                                                    <VisibilityOffIcon fontSize="small" />
                                                }
                                            </IconButton>
                                        </Tooltip>
                                        <Tooltip title="Zoom to Task Layers">
                                            <IconButton
                                                size="small"
                                                onClick={() => handleZoomToTask(taskLayer)}
                                                sx={{
                                        color: 'primary.main',
                                                    '&:hover': { backgroundColor: 'rgba(25, 118, 210, 0.04)' }
                                                }}
                                            >
                                                <ZoomInIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                        <Tooltip title="Remove Task Layers">
                                            <IconButton
                                                size="small"
                                                onClick={() => handleRemoveLayer(taskLayer.id)}
                                                sx={{
                                                    color: 'error.main',
                                                    '&:hover': { backgroundColor: 'rgba(211, 47, 47, 0.04)' }
                                                }}
                                            >
                                                <DeleteIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                        <Tooltip title="Generate Report">
                                            <IconButton
                                                size="small"
                                                onClick={() => generateTaskPdf(taskLayer, getAccessTokenSilently)}
                                                sx={{
                                                    color: 'info.main',
                                                    '&:hover': { backgroundColor: 'rgba(33, 150, 243, 0.04)' }
                                                }}
                                            >
                                                <PictureAsPdfIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                </Box>

                                {/* Layer Controls - Collapsible */}
                                <Collapse in={!taskCollapsed[taskLayer.id]}>
                                    <Box>
                                        {/* Prediction Layer - shown first */}
                                        {taskLayer.predictionTilesUrl && (
                                    <Box mb={2}>
                                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                                    <Tooltip title={`${taskLayer.predictionVisible !== false ? 'Hide' : 'Show'} Model Prediction`}>
                                                        <IconButton
                                                    size="small"
                                                            onClick={() => handleTaskLayerVisibilityChange(taskLayer.id, 'prediction', !(taskLayer.predictionVisible !== false))}
                                                            sx={{
                                                                color: taskLayer.predictionVisible !== false ? 'primary.main' : 'text.disabled',
                                                                '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' }
                                                            }}
                                                        >
                                                            {taskLayer.predictionVisible !== false ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
                                                        </IconButton>
                                                    </Tooltip>
                                                    <Box
                                                        width={24}
                                                        height={24}
                                                        borderRadius="4px"
                                                        border="1px solid"
                                                        borderColor="divider"
                                                        overflow="hidden"
                                                        sx={{
                                                            backgroundImage: `url(${taskLayer.predictionPreviewUrl || ''})`,
                                                            backgroundSize: 'cover',
                                                            backgroundPosition: 'center',
                                                            backgroundColor: 'secondary.main', // Fallback color
                                                            opacity: taskLayer.predictionVisible !== false ? 1 : 0.5
                                                        }}
                                                    />
                                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                        Model Prediction
                                                    </Typography>
                                                </Box>
                                        <Box ml={0.5}>
                                            <Typography variant="caption" color="text.secondary" gutterBottom>
                                                        Opacity: {Math.round((taskLayer.predictionOpacity || 0.8) * 100)}%
                                            </Typography>
                                            <Slider
                                                        value={taskLayer.predictionOpacity || 0.8}
                                                        onChange={(e, value) => handleTaskLayerOpacityChange(taskLayer.id, 'prediction', value)}
                                                min={0}
                                                max={1}
                                                step={0.1}
                                                size="small"
                                                sx={{ mt: 0.5 }}
                                            />
                                        </Box>
                                                {/* Regression Scale Bar */}
                                                {taskLayer.predictionStats?.type === 'reg' && (
                                                    <Box mt={1} ml={0.5} width="100%">
                                                        <Typography variant="caption" color="text.secondary" gutterBottom>
                                                            Values Scale ({taskLayer.predictionStats?.min?.toFixed(5) || '0.00000'} - {taskLayer.predictionStats?.max?.toFixed(5) || '1.00000'}):
                                                        </Typography>
                                                        <Box
                                                            display="flex"
                                                            alignItems="center"
                                                            mt={0.5}
                                                            sx={{
                                                                background: 'linear-gradient(to right, #440154, #482777, #3f4a8a, #31678e, #26838f, #1f9d8a, #6cce5a, #b6de2b, #fee825)',
                                                                height: '16px',
                                                                borderRadius: '8px',
                                                                border: '1px solid',
                                                                borderColor: 'divider',
                                                                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                                                                width: '100%'
                                                            }}
                                                        />
                                                        <Box display="flex" justifyContent="space-between" mt={0.5}>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {taskLayer.predictionStats?.min?.toFixed(3) || '0.000'}
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {taskLayer.predictionStats ? ((taskLayer.predictionStats.min + taskLayer.predictionStats.max) / 2).toFixed(3) : '0.500'}
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {taskLayer.predictionStats?.max?.toFixed(3) || '1.000'}
                                                            </Typography>
                                                        </Box>
                                                    </Box>
                                                )}
                                                {taskLayer.predictionStats?.type === 'seg' && (
                                                    <Box mt={1} ml={0.5}>
                                                        <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                                            Classes ({taskLayer.predictionStats.unique_values || taskLayer.predictionStats.class_indices?.length} present out of {taskLayer.predictionStats.class_indices?.length} total)
                                                        </Typography>
                                                        <Box mt={0.5}>
                                                            {(() => {
                                                                const classColors = generateSegmentationColors(taskLayer.predictionStats.class_indices);
                                                                return taskLayer.predictionStats.class_indices.map((idx) => {
                                                                    const color = classColors[idx];
                                                                    const className = taskLayer.predictionStats.classes_mapping?.[idx] || `Class ${idx}`;
                                                                    return (
                                                                        <Box key={idx} display="flex" alignItems="center" gap={1} mb={0.5}>
                                                                            <Box width={12} height={12} borderRadius="2px" border="1px solid" borderColor="divider" sx={{ backgroundColor: color }} />
                                                                            <Typography variant="caption">{className}</Typography>
                                                                        </Box>
                                                                    );
                                                                });
                                                            })()}
                                                        </Box>
                                                    </Box>
                                                )}
                                    </Box>
                                )}

                                        {/* Satellite Layer */}
                                        {taskLayer.satelliteTilesUrl && (
                                    <Box>
                                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                                    <Tooltip title={`${taskLayer.satelliteVisible !== false ? 'Hide' : 'Show'} Satellite Data`}>
                                                        <IconButton
                                                    size="small"
                                                            onClick={() => handleTaskLayerVisibilityChange(taskLayer.id, 'satellite', !(taskLayer.satelliteVisible !== false))}
                                                            sx={{
                                                                color: taskLayer.satelliteVisible !== false ? 'primary.main' : 'text.disabled',
                                                                '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' }
                                                            }}
                                                        >
                                                            {taskLayer.satelliteVisible !== false ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
                                                        </IconButton>
                                                    </Tooltip>
                                                    <Box
                                                        width={24}
                                                        height={24}
                                                        borderRadius="4px"
                                                        border="1px solid"
                                                        borderColor="divider"
                                                        overflow="hidden"
                                                        sx={{
                                                            backgroundImage: `url(${taskLayer.satellitePreviewUrl || ''})`,
                                                            backgroundSize: 'cover',
                                                            backgroundPosition: 'center',
                                                            backgroundColor: 'primary.main', // Fallback color
                                                            opacity: taskLayer.satelliteVisible !== false ? 1 : 0.5
                                                        }}
                                                    />
                                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                        Satellite Data
                                                    </Typography>
                                                </Box>
                                        <Box ml={0.5}>
                                            <Typography variant="caption" color="text.secondary" gutterBottom>
                                                        Opacity: {Math.round((taskLayer.satelliteOpacity || 0.8) * 100)}%
                                            </Typography>
                                            <Slider
                                                        value={taskLayer.satelliteOpacity || 0.8}
                                                        onChange={(e, value) => handleTaskLayerOpacityChange(taskLayer.id, 'satellite', value)}
                                                min={0}
                                                max={1}
                                                step={0.1}
                                                size="small"
                                                sx={{ mt: 0.5 }}
                                            />
                                        </Box>
                                    </Box>
                                )}
                                    </Box>
                                </Collapse>

                                {/* Divider between tasks */}
                                {index < taskLayers.length - 1 && (
                                    <Divider sx={{ mt: 2 }} />
                                )}
                            </Box>
                        ))}
                    </Box>
                </Collapse>
            </Paper>,
            controlRef.current
        );
    };

    return renderControl();
};

export default TaskLayersControl;
