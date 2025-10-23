import React, { useState, useEffect, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import MyLocationIcon from '@mui/icons-material/MyLocation';
import L from 'leaflet';
import { IconButton, Tooltip, ThemeProvider, CssBaseline, Snackbar, Alert } from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ListAltIcon from '@mui/icons-material/ListAlt';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import { useAuth0 } from '@auth0/auth0-react';
import 'leaflet-draw/dist/leaflet.draw.css';
import { lightTheme, darkTheme } from './theme';
import MapComponent from './components/MapComponent';
import TaskLayers from './components/TaskLayers';
import ControlPanel from './components/ControlPanel';
import BoundingBoxInfo from './components/BoundingBoxInfo';
import TaskResultPopup from './components/TaskResultPopup';
import TasksMonitor from './components/TasksMonitor';
import { CONFIG } from './config';
import { APP_THEMES, BASE_MAP_CONFIG, DARK_MODE_MAP_FILTER } from './constants';
import TaskLayersControl from './components/TaskLayersControl';
import { logger } from './utils/logger';
import apiService from './services/apiService';

// Configure default Leaflet marker icons
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const App = () => {
    // Auth0 hook
    const { getAccessTokenSilently } = useAuth0();

    const [controlPanelOpen, setControlPanelOpen] = useState(false);
    const [hasBoundingBox, setHasBoundingBox] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [totalArea, setTotalArea] = useState(0);
    const [showInfo, setShowInfo] = useState(false);
    const [taskResult, setTaskResult] = useState(null);
    const [taskError, setTaskError] = useState(null);
    const [showTaskPopup, setShowTaskPopup] = useState(false);
    const [showTasksMonitor, setShowTasksMonitor] = useState(false);
    const [taskLayers, setTaskLayers] = useState([]);
    const [appTheme, setAppTheme] = useState(APP_THEMES.DARK);
    const [snackbarOpen, setSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState('');
    const featureGroupRef = React.useRef(null);
    const statusPollingRef = useRef(null);

    // Status polling effect
    useEffect(() => {
        if (taskResult?.task_id && taskResult.status !== 'completed' && taskResult.status !== 'failed') {
            const pollStatus = async () => {
                try {
                    const updatedResult = await apiService.getTaskStatus(taskResult.task_id, getAccessTokenSilently);
                    setTaskResult(updatedResult);

                        // Stop polling if task is completed or failed
                        if (updatedResult.status === 'completed' || updatedResult.status === 'failed') {
                            if (statusPollingRef.current) {
                                clearInterval(statusPollingRef.current);
                                statusPollingRef.current = null;
                            }
                        }
                    } catch (error) {
                        logger.error('Error polling task status:', error);
                    }
                };

            // Poll every 15 seconds
            statusPollingRef.current = setInterval(pollStatus, 15000);

            // Cleanup on unmount or when taskResult changes
            return () => {
                if (statusPollingRef.current) {
                    clearInterval(statusPollingRef.current);
                    statusPollingRef.current = null;
                }
            };
        }
    }, [taskResult?.task_id, taskResult?.status, getAccessTokenSilently]);

    const handleAddTaskLayer = (taskLayerData) => {
        logger.log('handleAddTaskLayer called with:', taskLayerData);

        // Add new task layer to the list
        const newTaskLayer = {
            ...taskLayerData,
            id: Date.now(), // Simple ID for React key
            visible: true,
            opacity: 0.8,
            satelliteVisible: false,
            predictionVisible: true,
            satelliteOpacity: 0.8,
            predictionOpacity: 0.8
        };

        logger.log('Creating task layer:', newTaskLayer);

        setTaskLayers(prev => {
            const filtered = prev.filter(taskLayer => taskLayer.taskId !== taskLayerData.taskId);
            const newTaskLayers = [...filtered, newTaskLayer];
            logger.log('Updated task layers:', newTaskLayers);
            return newTaskLayers;
        });
    };

    const handleTaskLayerChange = (taskLayerId, layerType, changeType, value) => {
        logger.log('handleTaskLayerChange:', { taskLayerId, layerType, changeType, value });

        setTaskLayers(prev => {
            if (changeType === 'remove') {
                // Remove the task layer completely
                return prev.filter(taskLayer => taskLayer.id !== taskLayerId);
            }

            return prev.map(taskLayer => {
                if (taskLayer.id === taskLayerId) {
                    const updated = { ...taskLayer };

                    if (changeType === 'visibility') {
                        if (layerType === 'satellite') {
                            updated.satelliteVisible = value;
                        } else if (layerType === 'prediction') {
                            updated.predictionVisible = value;
                        }
                    } else if (changeType === 'opacity') {
                        if (layerType === 'satellite') {
                            updated.satelliteOpacity = value;
                        } else if (layerType === 'prediction') {
                            updated.predictionOpacity = value;
                        }
                    }

                    return updated;
                }
                return taskLayer;
            });
        });
    };

    const handleDrawCreated = (layer) => {
        setHasBoundingBox(true);
        setShowInfo(true);
    };

    const handleDrawEdited = (layers) => {
        setHasBoundingBox(layers.getLayers().length > 0);
        setShowInfo(true);
    };

    const handleDrawDeleted = () => {
        const hasLayers = featureGroupRef.current && featureGroupRef.current.getLayers().length > 0;
        setHasBoundingBox(hasLayers);
        setShowInfo(hasLayers);
    };

    const showSnackbar = useCallback((message) => {
        setSnackbarMessage(message);
        setSnackbarOpen(true);
    }, []);

    const handleSnackbarClose = () => {
        setSnackbarOpen(false);
    };

    const handleAreaChange = (area) => {
        setTotalArea(area);
        // Show info for valid areas (when clicking on existing bounding box)
        // Invalid areas are handled by Snackbar in MapComponent
        if (area >= CONFIG.MIN_AREA_KM2 && area <= CONFIG.MAX_AREA_KM2) {
            setShowInfo(true);
        }
    };

    const handleRunModel = async (modelParams) => {
        if (!hasBoundingBox || !featureGroupRef.current) return;

        setIsProcessing(true);
        setTaskResult(null);
        setTaskError(null);

        try {
            const boundingBoxes = featureGroupRef.current.getLayers().map(layer => {
                const bounds = layer.getBounds();
                // return [29.95, -2.05, 30.15, -1.85];
                return [
                    bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()
                ];
            });

            const payload = {
                bboxes: boundingBoxes,
                ...(modelParams),
            };
            logger.log('Payload being sent:', payload);

            const result = await apiService.runModel(payload, getAccessTokenSilently);
            logger.log('Task result:', result);

            // Set the task result and show popup
            setTaskResult(result);
            setShowTaskPopup(true);

        } catch (error) {
            logger.error('Error running model:', error);
            setTaskError({
                message: error.message || 'Failed to submit task. Please try again.'
            });
            setShowTaskPopup(true);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleCloseTaskPopup = () => {
        setShowTaskPopup(false);
        setTaskResult(null);
        setTaskError(null);

        // Stop polling when popup is closed
        if (statusPollingRef.current) {
            clearInterval(statusPollingRef.current);
            statusPollingRef.current = null;
        }
    };

    const toggleAppTheme = () => {
        setAppTheme(prevTheme => {
            const newTheme = prevTheme === APP_THEMES.LIGHT ? APP_THEMES.DARK : APP_THEMES.LIGHT;
            logger.log('Toggling app theme from', prevTheme, 'to', newTheme);
            return newTheme;
        });
    };

    const currentTheme = appTheme === APP_THEMES.DARK ? darkTheme : lightTheme;

    return (
        <ThemeProvider theme={currentTheme}>
            <CssBaseline />
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
                <MapContainer
                    center={[0, 0]}
                    zoom={3}
                    minZoom={3}
                    maxBounds={[[-90, -180], [90, 180]]}
                    maxBoundsViscosity={1.0}
                    style={{ width: '100%', height: '100%' }}
                >
                    <TileLayer
                        url={BASE_MAP_CONFIG.url}
                        attribution={BASE_MAP_CONFIG.attribution}
                    />
                    <BaseMapThemeController appTheme={appTheme} />

                    {/* Locate (GPS) button */}
                    <LocateControl />

                    {/* Task Layers */}
                    {taskLayers.map(taskLayer => (
                        <TaskLayers
                            key={taskLayer.id}
                            satelliteTilesUrl={taskLayer.satelliteTilesUrl}
                            predictionTilesUrl={taskLayer.predictionTilesUrl}
                            satelliteVisible={taskLayer.satelliteVisible}
                            predictionVisible={taskLayer.predictionVisible}
                            satelliteOpacity={taskLayer.satelliteOpacity}
                            predictionOpacity={taskLayer.predictionOpacity}
                            visible={taskLayer.visible}
                            bounds={taskLayer.bounds}
                            minZoom={taskLayer.minZoom}
                            maxZoom={taskLayer.maxZoom}
                            taskName={taskLayer.taskName}
                        />
                    ))}

                    {/* Task Layers Control */}
                    <TaskLayersControl
                        taskLayers={taskLayers}
                        onTaskLayerChange={handleTaskLayerChange}
                    />

                    <MapComponent
                        onDrawCreated={handleDrawCreated}
                        onDrawEdited={handleDrawEdited}
                        onDrawDeleted={handleDrawDeleted}
                        onAreaChange={handleAreaChange}
                        onShowMessage={showSnackbar}
                        featureGroupRef={featureGroupRef}
                    />
                    {showInfo && (
                        <BoundingBoxInfo
                            onClose={() => setShowInfo(false)}
                            featureGroup={featureGroupRef.current}
                            totalArea={totalArea}
                        />
                    )}
                </MapContainer>
            </div>

            <div style={{ position: 'absolute', top: 20, right: 20, zIndex: 2 }}>
                <Tooltip title="Open Control Panel">
                    <IconButton
                        onClick={() => setControlPanelOpen(true)}
                        sx={{
                            backgroundColor: 'white',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                            '&:hover': {
                                backgroundColor: '#f5f5f5'
                            },
                            '& .MuiSvgIcon-root': {
                                color: '#1E88E5'
                            }
                        }}
                    >
                        <AnalyticsIcon />
                    </IconButton>
                </Tooltip>
                <Tooltip title="View Task History">
                    <IconButton
                        onClick={() => setShowTasksMonitor(true)}
                        sx={{
                            backgroundColor: 'white',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                            marginLeft: 1,
                            '&:hover': {
                                backgroundColor: '#f5f5f5'
                            },
                            '& .MuiSvgIcon-root': {
                                color: '#4CAF50'
                            }
                        }}
                    >
                        <ListAltIcon />
                    </IconButton>
                </Tooltip>
                <Tooltip title={`Switch to ${appTheme === APP_THEMES.LIGHT ? 'Dark' : 'Light'} Mode`}>
                    <IconButton
                        onClick={toggleAppTheme}
                        sx={{
                            backgroundColor: 'white',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                            marginLeft: 1,
                            '&:hover': {
                                backgroundColor: '#f5f5f5'
                            },
                            '& .MuiSvgIcon-root': {
                                color: '#2196F3'
                            }
                        }}
                    >
                        {appTheme === APP_THEMES.LIGHT ? <DarkModeIcon /> : <LightModeIcon />}
                    </IconButton>
                </Tooltip>
            </div>

            <ControlPanel
                open={controlPanelOpen}
                onClose={() => setControlPanelOpen(false)}
                hasBoundingBox={hasBoundingBox}
                onRunModel={handleRunModel}
                isProcessing={isProcessing}
                appTheme={appTheme}
            />

            <TaskResultPopup
                open={showTaskPopup}
                onClose={handleCloseTaskPopup}
                result={taskResult}
                error={taskError}
                onOpenTasksMonitor={() => {
                    setShowTaskPopup(false);
                    setShowTasksMonitor(true);
                }}
            />

            <TasksMonitor
                open={showTasksMonitor}
                onClose={() => setShowTasksMonitor(false)}
                onAddTaskLayer={handleAddTaskLayer}
            />

            {/* Snackbar for area validation messages */}
            <Snackbar
                open={snackbarOpen}
                autoHideDuration={4000}
                onClose={handleSnackbarClose}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                <Alert
                    onClose={handleSnackbarClose}
                    severity="warning"
                    sx={{
                        width: '100%',
                        backgroundColor: 'background.paper',
                        color: 'text.primary',
                        '& .MuiAlert-icon': {
                            color: 'warning.main'
                        }
                    }}
                >
                    {snackbarMessage}
                </Alert>
            </Snackbar>
        </ThemeProvider>
    );
};

function BaseMapThemeController({ appTheme }) {
    const map = useMap();

    React.useEffect(() => {
        const tilePane = map.getPane('tilePane');
        if (!tilePane) return;

        // Find the first tile layer (base map)
        const firstTileLayer = tilePane.querySelector('div');
        if (!firstTileLayer) return;

        if (appTheme === APP_THEMES.DARK) {
            // Apply dark theme filter to first tile layer only
            firstTileLayer.style.filter = DARK_MODE_MAP_FILTER;
            logger.log('Applied dark theme filter to base map tiles');
        } else {
            // Remove filter for light theme
            firstTileLayer.style.filter = '';
            logger.log('Removed filter from base map tiles');
        }
    }, [map, appTheme]);

    return null;
}

function LocateControl() {
    const map = useMap();
    const [locating, setLocating] = React.useState(false);
    const markerRef = React.useRef(null);

    const handleLocate = () => {
        if (!navigator.geolocation || locating) return;
        setLocating(true);
        navigator.geolocation.getCurrentPosition(
            ({ coords }) => {
                const { latitude, longitude, accuracy } = coords;
                const latlng = [latitude, longitude];
                if (markerRef.current) markerRef.current.remove();
                markerRef.current = L.marker(latlng).addTo(map);
                const currentZoom = map.getZoom();
                const targetZoom = currentZoom < 10 ? (accuracy && accuracy > 2000 ? 10 : 12) : currentZoom;
                map.flyTo(latlng, targetZoom, { duration: 1 });
                setLocating(false);
            },
            () => setLocating(false),
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    };

    return (
        <Tooltip title="Fly to my location">
            <IconButton
                aria-label="Locate me"
                onClick={handleLocate}
                sx={{
                    position: 'absolute',
                    left: '50%',
                    bottom: 20,
                    transform: 'translateX(-50%)',
                    zIndex: 1000,
                    backgroundColor: 'background.paper',
                    color: 'text.primary',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    '&:hover': {
                        backgroundColor: 'action.hover',
                        color: 'text.primary'
                    }
                }}
                size="small"
            >
                <MyLocationIcon fontSize="small" color={locating ? 'disabled' : 'inherit'} />
            </IconButton>
        </Tooltip>
    );
}

export default App;
