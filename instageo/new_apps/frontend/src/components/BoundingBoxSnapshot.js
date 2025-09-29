import React, { useEffect, useRef } from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import L from 'leaflet';
import { BASE_MAP_CONFIG, DARK_MODE_MAP_FILTER, APP_THEMES } from '../constants';

const BoundingBoxSnapshot = ({ bbox, taskId }) => {
    const theme = useTheme();
    const mapRef = useRef(null);
    const mapInstanceRef = useRef(null);

    useEffect(() => {
        if (!bbox || bbox.length !== 4) return;

        // Clean up existing map
        if (mapInstanceRef.current) {
            mapInstanceRef.current.remove();
            mapInstanceRef.current = null;
        }

        // Create new map
        if (mapRef.current) {
            const map = L.map(mapRef.current, {
                center: [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
                zoom: 6,
                zoomControl: false,
                attributionControl: false,
                dragging: false,
                touchZoom: false,
                doubleClickZoom: false,
                scrollWheelZoom: false,
                boxZoom: false,
                keyboard: false,
                tapHold: false,
                fadeAnimation: false,
                markerZoomAnimation: false,
                zoomAnimation: false,
            });

            // Add tile layer
            L.tileLayer(BASE_MAP_CONFIG.url, {
                attribution: BASE_MAP_CONFIG.attribution
            }).addTo(map);

            // Apply dark mode filter
            if (theme.palette.mode === APP_THEMES.DARK) {
                const tilePane = map.getPanes().tilePane;
                if (tilePane) {
                    tilePane.style.filter = DARK_MODE_MAP_FILTER;
                }
            }

            // Add bounding box rectangle
            const rectangle = L.rectangle([
                [bbox[1], bbox[0]], // [south, west]
                [bbox[3], bbox[2]]  // [north, east]
            ], {
                color: '#1E88E5',
                fillColor: '#1E88E5',
                fillOpacity: 0.2,
                weight: 2
            }).addTo(map);

            // Fit map to bounding box
            map.fitBounds(rectangle.getBounds(), { padding: [20, 20] });

            mapInstanceRef.current = map;
        }

        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, [bbox, taskId, theme.palette.mode]);

    if (!bbox || bbox.length !== 4) {
        return (
            <Box sx={{ width: 120, height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="caption" color="textSecondary">
                    No bbox data
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{
            width: '100%',
            height: '100%',
            overflow: 'hidden'
        }}>
            <div
                ref={mapRef}
                style={{ width: '100%', height: '100%' }}
            />
        </Box>
    );
};

export default BoundingBoxSnapshot;
