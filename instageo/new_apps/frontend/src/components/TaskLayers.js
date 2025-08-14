import { useEffect, useRef } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';

const TaskLayers = ({
    satelliteTilesUrl,
    predictionTilesUrl,
    satelliteVisible = true,
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

    console.log('TaskLayers render:', { taskName, satelliteVisible, predictionVisible });

    // Create layers
    useEffect(() => {
        if (!map || (!satelliteTilesUrl && !predictionTilesUrl)) return;

        console.log('TaskLayers: Creating layers', {
            satelliteTilesUrl,
            predictionTilesUrl,
            bounds,
            minZoom,
            maxZoom
        });

        // Create satellite layer
        if (satelliteTilesUrl && !satelliteLayerRef.current) {
            satelliteLayerRef.current = L.tileLayer(satelliteTilesUrl, {
                opacity: satelliteOpacity,
                zIndex: 1000,
                bounds,
                // Allow tile stretching/scaling for better UX outside optimal zoom range
                maxNativeZoom: maxZoom, // Use native zoom for best quality when available
                maxZoom: 22, // Allow zooming beyond native data for closer inspection
                // Error tile url is a transparent image to avoid showing a white square when a tile is not available
                errorTileUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
            });
            console.log('Satellite layer created with flexible zoom');
        }

        // Create prediction layer
        if (predictionTilesUrl && !predictionLayerRef.current) {
            predictionLayerRef.current = L.tileLayer(predictionTilesUrl, {
                opacity: predictionOpacity,
                zIndex: 1001, // Prediction layer on top
                bounds,
                // Allow tile stretching/scaling for better UX outside optimal zoom range
                maxNativeZoom: maxZoom, // Use native zoom for best quality when available
                maxZoom: 22, // Allow zooming beyond native data for closer inspection
                // Error tile url is a transparent image to avoid showing a white square when a tile is not available
                errorTileUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
            });
            console.log('Prediction layer created with flexible zoom');
        }

        // Fit bounds if provided
        if (bounds && Array.isArray(bounds) && bounds.length === 2) {
            try {
                map.fitBounds(bounds);
                console.log('Map fitted to bounds:', bounds);
            } catch (error) {
                console.error('Error fitting bounds:', error);
            }
        }

        // Cleanup
        return () => {
            if (satelliteLayerRef.current && map.hasLayer(satelliteLayerRef.current)) {
                map.removeLayer(satelliteLayerRef.current);
            }
            if (predictionLayerRef.current && map.hasLayer(predictionLayerRef.current)) {
                map.removeLayer(predictionLayerRef.current);
            }
        };
    }, [map, satelliteTilesUrl, predictionTilesUrl, bounds, minZoom, maxZoom]);

    // Handle satellite layer visibility
    useEffect(() => {
        if (satelliteLayerRef.current) {
            if (visible && satelliteVisible) {
                if (!map.hasLayer(satelliteLayerRef.current)) {
                    satelliteLayerRef.current.addTo(map);
                }
            } else {
                if (map.hasLayer(satelliteLayerRef.current)) {
                    map.removeLayer(satelliteLayerRef.current);
                }
            }
        }
    }, [visible, satelliteVisible, map]);

    // Handle prediction layer visibility
    useEffect(() => {
        if (predictionLayerRef.current) {
            if (visible && predictionVisible) {
                if (!map.hasLayer(predictionLayerRef.current)) {
                    predictionLayerRef.current.addTo(map);
                }
            } else {
                if (map.hasLayer(predictionLayerRef.current)) {
                    map.removeLayer(predictionLayerRef.current);
                }
            }
        }
    }, [visible, predictionVisible, map]);

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
