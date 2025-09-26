import { useEffect, useRef, useCallback, useState } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-draw';
import { CONFIG } from '../config';

const MapComponent = ({ onDrawCreated, onDrawEdited, onDrawDeleted, onAreaChange, onShowMessage, featureGroupRef }) => {
    const map = useMap();
    const drawControlRef = useRef(null);
    const [drawingEnabled, setDrawingEnabled] = useState(true);

    const calculateArea = (bounds) => {
        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();

        // Calculate width and height using Leaflet's distance calculation
        const width = L.latLng(sw.lat, sw.lng).distanceTo(L.latLng(sw.lat, ne.lng));
        const height = L.latLng(sw.lat, sw.lng).distanceTo(L.latLng(ne.lat, sw.lng));

        // Convert to square kilometers (distance is in meters)
        return (width * height) / 1000000;
    };

    const calculateTotalArea = useCallback(() => {
        if (!featureGroupRef.current) return 0;
        let totalArea = 0;
        featureGroupRef.current.eachLayer((layer) => {
            totalArea += calculateArea(layer.getBounds());
        });
        return totalArea;
    }, [featureGroupRef]);

    const initializeDrawControl = useCallback(() => {
        if (!map || !featureGroupRef.current) return;

        // Remove existing control if it exists
        if (drawControlRef.current) {
            map.removeControl(drawControlRef.current);
        }

        // Create new draw control
        drawControlRef.current = new L.Control.Draw({
            position: 'topleft',
            draw: {
                polygon: false,
                circle: false,
                circlemarker: false,
                marker: false,
                polyline: false,
                rectangle: drawingEnabled ? {
                    shapeOptions: {
                        color: '#1E88E5',
                        fillColor: '#1E88E5',
                        fillOpacity: 0.2,
                        weight: 2
                    },
                    showArea: false,
                    metric: true,
                    repeatMode: false
                } : false
            },
            edit: {
                featureGroup: featureGroupRef.current,
                remove: true,
                edit: {
                    selectedPathOptions: {
                        maintainColor: true,
                        dashArray: '10, 10'
                    }
                }
            }
        });
        map.addControl(drawControlRef.current);
    }, [map, featureGroupRef, drawingEnabled]);

    useEffect(() => {
        if (!map) return;

        // Create feature group if it doesn't exist
        if (!featureGroupRef.current) {
            featureGroupRef.current = new L.FeatureGroup();
        }

        // Initialize draw control
        initializeDrawControl();

        const addClickListenersToExistingBoxes = () => {
            if (featureGroupRef.current) {
                featureGroupRef.current.eachLayer((layer) => {
                    // Remove existing right-click listener to avoid duplicates
                    layer.off('contextmenu');
                    // Add right-click listener to show bbox info
                    layer.on('contextmenu', (e) => {
                        // Prevent default context menu
                        e.originalEvent.preventDefault();
                        const area = calculateArea(layer.getBounds());
                        onAreaChange(area);
                    });
                });
            }
        };

        // Add click listeners to existing boxes
        addClickListenersToExistingBoxes();

        const handleDrawCreated = (e) => {
            const layer = e.layer;
            const bounds = layer.getBounds();

            // Clear any existing bounding boxes first
            featureGroupRef.current.eachLayer((existingLayer) => {
                featureGroupRef.current.removeLayer(existingLayer);
                map.removeLayer(existingLayer);
            });

            // Get the area from the drawn rectangle
            const area = calculateArea(bounds);

            // Validate single box size
            if (area >= CONFIG.MIN_AREA_KM2 && area <= CONFIG.MAX_AREA_KM2) {
                featureGroupRef.current.addLayer(layer);
                map.addLayer(layer);
                onDrawCreated(layer);
                onAreaChange(area);
                // Disable adding new drawing after successful creation
                // Editing and deleting existing drawings are still allowed
                setDrawingEnabled(false);

            } else {
                onAreaChange(area);

                // Show warning message
                const message = area < CONFIG.MIN_AREA_KM2
                    ? `Area too small (${area.toFixed(4)} km²)! Minimum required: ${CONFIG.MIN_AREA_KM2} km². Your bounding box will not be considered.`
                    : `Area too large (${area.toFixed(4)} km²)! Maximum allowed: ${CONFIG.MAX_AREA_KM2} km². Your bounding box will not be considered.`;

                if (onShowMessage) {
                    onShowMessage(message);
                }

                // After 3 seconds, revert to showing 0
                setTimeout(() => {
                    onAreaChange(0);
                }, 3000);
            }
        };

        const handleDrawEdited = (e) => {
            const layers = e.layers;
            const totalArea = calculateTotalArea();

            // Check if single box area is valid
            if (totalArea >= CONFIG.MIN_AREA_KM2 && totalArea <= CONFIG.MAX_AREA_KM2) {
                onDrawEdited(layers);
                onAreaChange(totalArea);
            } else {
                onAreaChange(totalArea);

                // Show warning message
                const message = totalArea < CONFIG.MIN_AREA_KM2
                    ? `Area too small (${totalArea.toFixed(4)} km²)! Minimum required: ${CONFIG.MIN_AREA_KM2} km². Your bounding box will not be considered.`
                    : `Area too large (${totalArea.toFixed(4)} km²)! Maximum allowed: ${CONFIG.MAX_AREA_KM2} km². Your bounding box will not be considered.`;

                if (onShowMessage) {
                    onShowMessage(message);
                }

                setTimeout(() => {
                    layers.eachLayer((layer) => {
                        featureGroupRef.current.removeLayer(layer);
                        map.removeLayer(layer);
                    });
                    onDrawDeleted();
                    onAreaChange(0);
                    // Re-enable drawing after removing invalid layer
                    setDrawingEnabled(true);
                }, 3000);
            }
        };

        const handleDrawDeleted = (e) => {
            let hasLayers = false;
            e.layers.eachLayer((layer) => {
                hasLayers = true;
                featureGroupRef.current.removeLayer(layer);
                map.removeLayer(layer);
            });

            // Only proceed if layers were actually deleted
            if (hasLayers) {
                onDrawDeleted();
                onAreaChange(0);
                // Re-enable drawing after deletion
                setDrawingEnabled(true);
            }
        };

        // Add event listeners
        map.on('draw:created', handleDrawCreated);
        map.on('draw:edited', handleDrawEdited);
        map.on('draw:deleted', handleDrawDeleted);

        // Cleanup function
        return () => {
            map.off('draw:created', handleDrawCreated);
            map.off('draw:edited', handleDrawEdited);
            map.off('draw:deleted', handleDrawDeleted);
            if (drawControlRef.current) {
                map.removeControl(drawControlRef.current);
            }
            if (featureGroupRef.current) {
                featureGroupRef.current.remove();
            }
        };
    }, [map, onDrawCreated, onDrawEdited, onDrawDeleted, onAreaChange, onShowMessage, initializeDrawControl, featureGroupRef, calculateTotalArea]);

    return null;
};

export default MapComponent;
