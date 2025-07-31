import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import { IconButton, Tooltip } from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ListAltIcon from '@mui/icons-material/ListAlt';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import MapComponent from './components/MapComponent';
import ControlPanel from './components/ControlPanel';
import BoundingBoxInfo from './components/BoundingBoxInfo';
import TaskResultPopup from './components/TaskResultPopup';
import TasksMonitor from './components/TasksMonitor';
import { DEFAULT_PARAMS } from './constants';
import { INSTAGEO_BACKEND_API_ENDPOINTS } from './config';

const App = () => {
    const [controlPanelOpen, setControlPanelOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState('aod');
    const [modelParams, setModelParams] = useState(DEFAULT_PARAMS);
    const [hasBoundingBox, setHasBoundingBox] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [totalArea, setTotalArea] = useState(0);
    const [showInfo, setShowInfo] = useState(true);
    const [taskResult, setTaskResult] = useState(null);
    const [taskError, setTaskError] = useState(null);
    const [showTaskPopup, setShowTaskPopup] = useState(false);
    const [showTasksMonitor, setShowTasksMonitor] = useState(false);
    const featureGroupRef = React.useRef(null);
    const statusPollingRef = useRef(null);

    // Status polling effect
    useEffect(() => {
        if (taskResult?.task_id && taskResult.status !== 'completed' && taskResult.status !== 'failed') {
            const pollStatus = async () => {
                try {
                    const response = await fetch(INSTAGEO_BACKEND_API_ENDPOINTS.TASK_STATUS(taskResult.task_id));
                    if (response.ok) {
                        const updatedResult = await response.json();
                        setTaskResult(updatedResult);

                        // Stop polling if task is completed or failed
                        if (updatedResult.status === 'completed' || updatedResult.status === 'failed') {
                            if (statusPollingRef.current) {
                                clearInterval(statusPollingRef.current);
                                statusPollingRef.current = null;
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error polling task status:', error);
                }
            };

            // Poll every 5 seconds
            statusPollingRef.current = setInterval(pollStatus, 5000);

            // Cleanup on unmount or when taskResult changes
            return () => {
                if (statusPollingRef.current) {
                    clearInterval(statusPollingRef.current);
                    statusPollingRef.current = null;
                }
            };
        }
    }, [taskResult?.task_id, taskResult?.status]);

    const handleModelChange = (model) => {
        setSelectedModel(model);
    };

    const handleParamsChange = (params) => {
        setModelParams(params);
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

    const handleAreaChange = (area) => {
        setTotalArea(area);
        // Show info if area is invalid, regardless of previous state
        if (area < 1 || area > 100000) {
            setShowInfo(true);
        }
    };

    const handleRunModel = async () => {
        if (!hasBoundingBox || !featureGroupRef.current) return;

        setIsProcessing(true);
        setTaskResult(null);
        setTaskError(null);

        try {
            // Get all bounding box coordinates from the feature group
            const boundingBoxes = featureGroupRef.current.getLayers().map(layer => {
                const bounds = layer.getBounds();
                return [
                    bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()
                ];
            });

            const payload = {
                    bboxes: boundingBoxes,
                    model_type: selectedModel,
                    ...modelParams
            };
            console.log('Payload being sent:', payload);
            const response = await fetch(INSTAGEO_BACKEND_API_ENDPOINTS.RUN_MODEL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('Task result:', result);

            // Set the task result and show popup
            setTaskResult(result);
            setShowTaskPopup(true);

        } catch (error) {
            console.error('Error running model:', error);
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

    return (
        <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
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
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    />
                    <MapComponent
                        onDrawCreated={handleDrawCreated}
                        onDrawEdited={handleDrawEdited}
                        onDrawDeleted={handleDrawDeleted}
                        onAreaChange={handleAreaChange}
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
            </div>

            <ControlPanel
                open={controlPanelOpen}
                onClose={() => setControlPanelOpen(false)}
                onModelChange={handleModelChange}
                onParamsChange={handleParamsChange}
                hasBoundingBox={hasBoundingBox}
                onRunModel={handleRunModel}
                isProcessing={isProcessing}
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
            />
        </div>
    );
};

export default App;
