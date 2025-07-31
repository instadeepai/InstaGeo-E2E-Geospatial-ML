import React, { useState } from 'react';
import { Drawer, Box, Typography, Select, MenuItem, FormControl, InputLabel, Slider, Button, TextField, CircularProgress } from '@mui/material';
import { DEFAULT_PARAMS, MODEL_CONFIGS } from '../constants';

const ControlPanel = ({ open, onClose, onModelChange, onParamsChange, hasBoundingBox, onRunModel, isProcessing }) => {
  const [model, setModel] = useState('aod');
  const [params, setParams] = useState(DEFAULT_PARAMS);

  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setModel(newModel);
    const newParams = MODEL_CONFIGS[newModel];
    setParams(newParams);
    onModelChange(newModel);
    onParamsChange(newParams);
  };

  const handleParamChange = (param, value) => {
    const newParams = { ...params, [param]: value };
    setParams(newParams);
    onParamsChange(newParams);
  };

  const handleDateChange = (event) => {
    handleParamChange('date', event.target.value);
  };

  const handleSatelliteSourceChange = (event) => {
    handleParamChange('data_source', event.target.value);
  };

  const renderSlider = (label, param, min, max, step = 1) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography gutterBottom>{label}</Typography>
        <Typography variant="caption" color="text.secondary">
          {min} - {max}
        </Typography>
      </Box>
      <Slider
        value={params[param]}
        onChange={(_, value) => handleParamChange(param, value)}
        min={min}
        max={max}
        step={step}
        valueLabelDisplay="auto"
      />
    </Box>
  );

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: 350,
          padding: 2,
          backgroundColor: 'white',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          zIndex: 1200
        }
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <img
            src="https://raw.githubusercontent.com/instadeepai/InstaGeo-E2E-Geospatial-ML/0910341cef9a858137f4dffc1467de7b3240ec0f/assets/logo.png"
            alt="InstaGeo Logo"
            style={{
              maxWidth: '200px',
              height: 'auto',
              display: 'block'
            }}
            onError={(e) => {
              console.error('Error loading logo:', e);
              e.target.style.display = 'none';
            }}
          />
        </Box>

        <Typography variant="h6" sx={{ mb: 2, color: '#1E88E5', textAlign: 'center' }}>
          Parameters
        </Typography>

        {!hasBoundingBox && (
          <Box sx={{ mb: 3, p: 2, bgcolor: '#fff3e0', borderRadius: 1 }}>
            <Typography color="warning.main">
              Please draw a bounding box on the map first
            </Typography>
          </Box>
        )}

        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Model</InputLabel>
          <Select
            value={model}
            label="Model"
            onChange={handleModelChange}
            disabled={!hasBoundingBox}
          >
            <MenuItem value="aod">Aerosol Optical Depth Estimation</MenuItem>
            <MenuItem value="locust">Locust Breeding Ground Prediction</MenuItem>
          </Select>
        </FormControl>

        <TextField
          label="Date"
          type="date"
          value={params.date}
          onChange={handleDateChange}
          fullWidth
          sx={{ mb: 3 }}
          InputLabelProps={{
            shrink: true,
          }}
          inputProps={{
            max: new Date().toISOString().split('T')[0]
          }}
        />

        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Satellite Source</InputLabel>
          <Select
            value={params.data_source}
            label="Satellite Source"
            onChange={handleSatelliteSourceChange}
          >
            <MenuItem value="HLS">HLS (Harmonized Landsat Sentinel-2)</MenuItem>
            <MenuItem value="S2">Sentinel-2 (S2)</MenuItem>
            <MenuItem value="S1">Sentinel-1 (S1)</MenuItem>
          </Select>
        </FormControl>

        {renderSlider('Temporal Tolerance (days)', 'temporal_tolerance', 1, 30)}

        {model === 'aod' ? (
          <>
            <Box sx={{ mb: 2, backgroundColor: '#f5f5f5', p: 2, borderRadius: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography gutterBottom color="text.secondary">
                  Number of Steps
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  1 - 10
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Fixed value: 1
              </Typography>
            </Box>
            <Box sx={{ mb: 2, backgroundColor: '#f5f5f5', p: 2, borderRadius: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography gutterBottom color="text.secondary">
                  Temporal Step (days)
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  1 - 90
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Fixed value: 0
              </Typography>
            </Box>
          </>
        ) : (
          <>
            {renderSlider('Number of Steps', 'num_steps', 1, 10)}
            {renderSlider('Temporal Step (days)', 'temporal_step', 1, 90)}
          </>
        )}

        {renderSlider('Maximum Cloud Cover', 'cloud_coverage', 0, 100)}

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={onRunModel}
          disabled={!hasBoundingBox || isProcessing}
          sx={{
            mt: 2,
            mb: 3,
            py: 1.5,
            position: 'relative',
            '&.Mui-disabled': {
              backgroundColor: isProcessing ? 'primary.main' : undefined,
              opacity: isProcessing ? 0.8 : undefined,
            }
          }}
        >
          {isProcessing ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
              <CircularProgress
                size={20}
                sx={{
                  color: 'white',
                  animation: 'pulse 1.5s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 0.6 },
                    '50%': { opacity: 1 },
                    '100%': { opacity: 0.6 },
                  }
                }}
              />
              <span>Processing...</span>
            </Box>
          ) : (
            'Run Model'
          )}
        </Button>
      </Box>
    </Drawer>
  );
};

export default ControlPanel;
