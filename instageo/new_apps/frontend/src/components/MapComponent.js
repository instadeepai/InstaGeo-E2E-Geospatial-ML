import { useEffect, useRef, useCallback } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-draw';

const MapComponent = ({ onDrawCreated, onDrawEdited, onDrawDeleted, onAreaChange, featureGroupRef }) => {
    const map = useMap();
    const drawControlRef = useRef(null);

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
                rectangle: {
                    shapeOptions: {
                        color: '#1E88E5',
                        fillColor: '#1E88E5',
                        fillOpacity: 0.2,
                        weight: 2
                    },
                    showArea: false,
                    metric: true,
                    repeatMode: false
                }
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
    }, [map, featureGroupRef]);

    useEffect(() => {
        if (!map) return;

        // Create feature group if it doesn't exist
        if (!featureGroupRef.current) {
            featureGroupRef.current = new L.FeatureGroup();
        }

        // Initialize draw control
        initializeDrawControl();

        const handleDrawCreated = (e) => {
            const layer = e.layer;
            const bounds = layer.getBounds();

            // Get the area from the drawn rectangle
            const area = calculateArea(bounds);

            // Calculate area including existing layers
            let totalArea = calculateTotalArea() + area;

            // Validate total box size
            if (totalArea >= 1 && totalArea <= 100000) {
                featureGroupRef.current.addLayer(layer);
                map.addLayer(layer);
                onDrawCreated(layer);
                onAreaChange(totalArea);
            } else {
                // Show the attempted total area briefly
                onAreaChange(totalArea);
                // After 2 seconds, revert to showing the current valid total
                setTimeout(() => {
                    const currentTotalArea = totalArea - area;
                    onAreaChange(currentTotalArea);
                }, 2000);
            }
        };

        const handleDrawEdited = (e) => {
            const layers = e.layers;
            const totalArea = calculateTotalArea();

            // Check if total area is valid
            if (totalArea >= 1 && totalArea <= 100000) {
                onDrawEdited(layers);
                onAreaChange(totalArea);
            } else {
                onAreaChange(totalArea);
                setTimeout(() => {
                    layers.eachLayer((layer) => {
                        featureGroupRef.current.removeLayer(layer);
                        map.removeLayer(layer);
                    });
                    const currentTotalArea = calculateTotalArea();
                    onAreaChange(currentTotalArea);
                    onDrawDeleted();
                }, 2000);

            }
        };

        const handleDrawDeleted = (e) => {
            e.layers.eachLayer((layer) => {
                featureGroupRef.current.removeLayer(layer);
                map.removeLayer(layer);
            });
            onDrawDeleted();
            const currentTotalArea = calculateTotalArea();
            if (currentTotalArea < 1) {
                onAreaChange(currentTotalArea);
                setTimeout(() => {
                    onAreaChange(0);
                    onDrawDeleted();
                    // Remove all layers from map and feature group
                    featureGroupRef.current.eachLayer((layer) => {
                        map.removeLayer(layer);
                    });
                    featureGroupRef.current.clearLayers();

                }, 2000);

            } else {
                onAreaChange(currentTotalArea);
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
    }, [map, onDrawCreated, onDrawEdited, onDrawDeleted, onAreaChange, initializeDrawControl, featureGroupRef, calculateTotalArea]);

    return null;
};

export default MapComponent;
