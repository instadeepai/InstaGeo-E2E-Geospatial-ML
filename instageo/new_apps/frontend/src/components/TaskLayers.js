import { useEffect, useRef } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import { logger } from '../utils/logger';

const TaskLayers = ({
    satelliteTilesUrl,
    predictionTilesUrl,
    satelliteVisible = false,
    predictionVisible = true,
    satelliteOpacity = 0.8,
    predictionOpacity = 0.8,
    visible = true,
    bounds = null,
    minZoom = 0,
    maxZoom = 22,
    taskName = "Layer"
}) => {
    const map = useMap();
    const satelliteLayerRef = useRef(null);
    const predictionLayerRef = useRef(null);

    logger.log('TaskLayers render:', { taskName, satelliteVisible, predictionVisible });

    // Handle bounds fitting when component mounts or bounds change
    useEffect(() => {
        if (map && bounds && Array.isArray(bounds) && bounds.length === 2) {
            try {
                map.fitBounds(bounds);
                logger.log('Map fitted to bounds:', bounds);
            } catch (error) {
                logger.error('Error fitting bounds:', error);
            }
        }
    }, [map, bounds]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (satelliteLayerRef.current && map && map.hasLayer(satelliteLayerRef.current)) {
                map.removeLayer(satelliteLayerRef.current);
            }
            if (predictionLayerRef.current && map && map.hasLayer(predictionLayerRef.current)) {
                map.removeLayer(predictionLayerRef.current);
            }
        };
    }, [map]);

    // Handle satellite layer visibility
    useEffect(() => {
        if (visible && satelliteVisible && satelliteTilesUrl) {
            // Create layer if it doesn't exist or is invalid
            if (!satelliteLayerRef.current || !map.hasLayer(satelliteLayerRef.current)) {
                try {
                    // Remove existing layer if it exists but is not on map
                    if (satelliteLayerRef.current && map.hasLayer(satelliteLayerRef.current)) {
                        map.removeLayer(satelliteLayerRef.current);
                    }

                    // Create new layer
                    satelliteLayerRef.current = L.tileLayer(satelliteTilesUrl, {
                        opacity: satelliteOpacity,
                        zIndex: 1000,
                        bounds,
                        maxNativeZoom: maxZoom,
                        maxZoom: 22,
                        errorTileUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
                    });

                    satelliteLayerRef.current.addTo(map);
                    logger.log('Satellite layer created and added to map');
                } catch (error) {
                    logger.error('Error creating/adding satellite layer to map:', error);
                }
            }
        } else {
            // Remove layer if it exists on map
            if (satelliteLayerRef.current && map.hasLayer(satelliteLayerRef.current)) {
                map.removeLayer(satelliteLayerRef.current);
                logger.log('Satellite layer removed from map');
            }
        }
    }, [visible, satelliteVisible, satelliteTilesUrl, satelliteOpacity, bounds, maxZoom, map]);

    // Handle prediction layer visibility
    useEffect(() => {
        if (visible && predictionVisible && predictionTilesUrl) {
            // Create layer if it doesn't exist or is invalid
            if (!predictionLayerRef.current || !map.hasLayer(predictionLayerRef.current)) {
                try {
                    // Remove existing layer if it exists but is not on map
                    if (predictionLayerRef.current && map.hasLayer(predictionLayerRef.current)) {
                        map.removeLayer(predictionLayerRef.current);
                    }

                    // Create new layer
                    predictionLayerRef.current = L.tileLayer(predictionTilesUrl, {
                        opacity: predictionOpacity,
                        zIndex: 1001,
                        bounds,
                        maxNativeZoom: maxZoom,
                        maxZoom: 22,
                        errorTileUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
                    });

                    predictionLayerRef.current.addTo(map);
                    logger.log('Prediction layer created and added to map');
                } catch (error) {
                    logger.error('Error creating/adding prediction layer to map:', error);
                }
            }
        } else {
            // Remove layer if it exists on map
            if (predictionLayerRef.current && map.hasLayer(predictionLayerRef.current)) {
                map.removeLayer(predictionLayerRef.current);
                logger.log('Prediction layer removed from map');
            }
        }
    }, [visible, predictionVisible, predictionTilesUrl, predictionOpacity, bounds, maxZoom, map]);

    // Handle satellite opacity changes
    useEffect(() => {
        if (satelliteLayerRef.current) {
            satelliteLayerRef.current.setOpacity(satelliteOpacity);
        }
    }, [satelliteOpacity]);

    // Handle prediction opacity changes
    useEffect(() => {
        if (predictionLayerRef.current) {
            predictionLayerRef.current.setOpacity(predictionOpacity);
        }
    }, [predictionOpacity]);

    return null;
};

export default TaskLayers;
