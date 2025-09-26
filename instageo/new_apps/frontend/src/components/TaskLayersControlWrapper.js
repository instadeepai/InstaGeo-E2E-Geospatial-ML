import { useEffect, useRef, useState } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import {
    Box,
    Paper,
    FormControlLabel,
    Switch,
    Typography,
    IconButton,
    Collapse,
    Slider,
    Divider
} from '@mui/material';
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Layers as LayersIcon
} from '@mui/icons-material';
import { createPortal } from 'react-dom';
import { logger } from '../utils/logger';

const TaskLayersControlWrapper = ({ taskLayers = [], onTaskLayerChange }) => {
    const map = useMap();
    const controlRef = useRef(null);
    const [expanded, setExpanded] = useState(true);

    logger.log('TaskLayersControlWrapper render:', { taskLayersCount: taskLayers.length });

    // Create layer control
    useEffect(() => {
        if (!map) return;

        logger.log('Creating unified layer control');

        // Create control container
        const TaskLayersControlWrapperClass = L.Control.extend({
            onAdd: function() {
                const div = L.DomUtil.create('div', 'unified-layer-control');
                div.style.background = 'transparent';
                div.style.padding = '0';
                div.style.border = 'none';

                // Prevent map interactions on control
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);

                controlRef.current = div;
                logger.log('Unified control div created');
                return div;
            }
        });

        const control = new TaskLayersControlWrapperClass({ position: 'bottomleft' });
        control.addTo(map);
        logger.log('Unified control added to map');

        return () => {
            logger.log('Removing unified control');
            if (control) {
                map.removeControl(control);
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
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid rgba(0, 0, 0, 0.1)'
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
                            {expanded ? `Layers (${taskLayers.length})` : 'Layers'}
                        </Typography>
                    </Box>
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
                </Box>

                {/* Collapsible Content */}
                <Collapse in={expanded}>
                    <Box sx={{ p: 2, pt: 1.5 }}>
                        {taskLayers.map((taskLayer, index) => (
                            <Box key={taskLayer.id} mb={index < taskLayers.length - 1 ? 3 : 0}>
                                {/* Task Header */}
                                <Typography
                                    variant="subtitle2"
                                    sx={{
                                        fontWeight: 600,
                                        mb: 1.5,
                                        color: 'primary.main',
                                        borderBottom: '1px solid',
                                        borderColor: 'divider',
                                        pb: 0.5
                                    }}
                                >
                                    {taskLayer.taskName}
                                </Typography>

                                {/* Satellite Layer */}
                                {taskLayer.satelliteTilesUrl && (
                                    <Box mb={2}>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    checked={taskLayer.satelliteVisible !== false}
                                                    onChange={(e) => handleTaskLayerVisibilityChange(taskLayer.id, 'satellite', e.target.checked)}
                                                    size="small"
                                                    color="primary"
                                                />
                                            }
                                            label={
                                                <Box display="flex" alignItems="center" gap={1}>
                                                    <Box
                                                        width={14}
                                                        height={14}
                                                        bgcolor="primary.main"
                                                        borderRadius="2px"
                                                        border="1px solid rgba(0,0,0,0.2)"
                                                    />
                                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                        Satellite Data
                                                    </Typography>
                                                </Box>
                                            }
                                            sx={{ mb: 1, ml: 0 }}
                                        />
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
                                                disabled={taskLayer.satelliteVisible === false}
                                                sx={{ mt: 0.5 }}
                                            />
                                        </Box>
                                    </Box>
                                )}

                                {/* Prediction Layer */}
                                {taskLayer.predictionTilesUrl && (
                                    <Box>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    checked={taskLayer.predictionVisible !== false}
                                                    onChange={(e) => handleTaskLayerVisibilityChange(taskLayer.id, 'prediction', e.target.checked)}
                                                    size="small"
                                                    color="secondary"
                                                />
                                            }
                                            label={
                                                <Box display="flex" alignItems="center" gap={1}>
                                                    <Box
                                                        width={14}
                                                        height={14}
                                                        bgcolor="secondary.main"
                                                        borderRadius="2px"
                                                        border="1px solid rgba(0,0,0,0.2)"
                                                    />
                                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                        Model Prediction
                                                    </Typography>
                                                </Box>
                                            }
                                            sx={{ mb: 1, ml: 0 }}
                                        />
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
                                                disabled={taskLayer.predictionVisible === false}
                                                sx={{ mt: 0.5 }}
                                            />
                                        </Box>
                                    </Box>
                                )}

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

export default TaskLayersControlWrapper;
