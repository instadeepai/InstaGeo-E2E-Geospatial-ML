import React from 'react';
import { Paper, Typography, Box, Alert, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const BoundingBoxInfo = ({ onClose, featureGroup, totalArea }) => {
  if (!featureGroup) return null;

  const formatCoordinate = (coord) => coord.toFixed(6);

  // Calculate bounds of all boxes
  let north = -90;
  let south = 90;
  let east = -180;
  let west = 180;

  featureGroup.eachLayer((layer) => {
    const bounds = layer.getBounds();
    north = Math.max(north, bounds.getNorth());
    south = Math.min(south, bounds.getSouth());
    east = Math.max(east, bounds.getEast());
    west = Math.min(west, bounds.getWest());
  });

  const isOutsideBounds = totalArea < 1 || totalArea > 100000;

  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        position: 'absolute',
        bottom: 20,
        left: 20,
        zIndex: 1000,
        minWidth: 300,
        backgroundColor: 'rgba(255, 255, 255, 0.95)'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
        <Typography variant="h6" component="h2" gutterBottom>
          Areas of Interest
        </Typography>
        <IconButton
          size="small"
          onClick={onClose}
          sx={{
            color: 'text.secondary',
            '&:hover': {
              color: 'text.primary'
            }
          }}
        >
          <CloseIcon />
        </IconButton>
      </Box>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Bounds: [{formatCoordinate(west)}, {formatCoordinate(south)}, {formatCoordinate(east)}, {formatCoordinate(north)}]
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Total Area: {totalArea.toFixed(4)} km²
      </Typography>
      {isOutsideBounds && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          {totalArea < 1
            ? 'Total area is too small (minimum 1 km²)'
            : 'Total area is too large (maximum 100,000 km²)'}
        </Alert>
      )}
    </Paper>
  );
};

export default BoundingBoxInfo;
