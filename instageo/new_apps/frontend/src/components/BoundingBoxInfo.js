import React from 'react';
import { Paper, Typography, Box, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const BoundingBoxInfo = ({ onClose, featureGroup, totalArea }) => {
  if (!featureGroup) return null;

  const formatCoordinate = (coord) => coord.toFixed(6);

  // Get bounds of the single bounding box
  let bounds = null;
  featureGroup.eachLayer((layer) => {
    bounds = layer.getBounds();
  });

  if (!bounds) return null;

  const north = bounds.getNorth();
  const south = bounds.getSouth();
  const east = bounds.getEast();
  const west = bounds.getWest();


  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        position: 'absolute',
        bottom: 20,
        left: 20,
        zIndex: 1000,
        minWidth: 300
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
        <Typography variant="h6" component="h2" gutterBottom>
          Area of Interest
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
        Area: {totalArea.toFixed(4)} km²
      </Typography>
    </Paper>
  );
};

export default BoundingBoxInfo;
