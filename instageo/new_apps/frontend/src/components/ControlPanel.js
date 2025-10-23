import React, { useEffect, useMemo, useState } from 'react';
import { Drawer, Box, Typography, Select, MenuItem, FormControl, InputLabel, Slider, Button, TextField, CircularProgress, Tooltip, IconButton, Collapse, Divider, Paper, Chip, InputAdornment } from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { useAuth0 } from '@auth0/auth0-react';
import dayjs from 'dayjs';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import RefreshIcon from '@mui/icons-material/Refresh';
import { DEFAULT_TASK_PARAMS, PARAMS_HELP, LOGO_PATHS } from '../constants';
import { logger } from '../utils/logger';
import { fetchModelsWithTTL, clearModelsCache } from '../utils/modelsCache';
import ProfileMenu from './ProfileMenu';

const ControlPanel = ({ open, onClose, hasBoundingBox, onRunModel, isProcessing, appTheme }) => {
  const { getAccessTokenSilently } = useAuth0();

  const [params, setParams] = useState(DEFAULT_TASK_PARAMS);
  const [models, setModels] = useState([]);
  const [selectedModelKey, setSelectedModelKey] = useState('');
  const [selectedSize, setSelectedSize] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  const [helpOpen, setHelpOpen] = useState({
    model_description: false,
    chip_size: false,
    num_steps: false,
    data_source: false,
    temporal_step: false,
    temporal_tolerance: false,
    cloud_coverage: false,
  });



  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoadingModels(true);
      try {
        const data = await fetchModelsWithTTL(getAccessTokenSilently);
        if (!mounted) return;
        setModels(data || []);
      } catch (e) {
        logger.warn('Failed to load models:', e);
        if (mounted) setModels([]);
      } finally {
        if (mounted) setLoadingModels(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, [getAccessTokenSilently]);

  const handleReloadModels = async () => {
    try {
      clearModelsCache();
    } catch {}
    setLoadingModels(true);
    try {
      const data = await fetchModelsWithTTL(getAccessTokenSilently);
      setModels(data || []);
    } catch (e) {
      logger.warn('Failed to reload models:', e);
      setModels([]);
    } finally {
      setLoadingModels(false);
    }
  };

  const resetFormToDefaults = () => {
    setParams(DEFAULT_TASK_PARAMS);
    setSelectedModelKey('');
    setSelectedSize('');
  };

  const modelsByKey = useMemo(() => {
    const map = {};
    if (models && Array.isArray(models)) {
      for (const m of models) {
        if (!map[m.model_key]) map[m.model_key] = [];
        map[m.model_key].push(m);
      }
    }
    return map;
  }, [models]);

  const modelKeys = useMemo(() => Object.keys(modelsByKey).sort(), [modelsByKey]);
  const sizesForSelected = useMemo(() => (selectedModelKey ? (modelsByKey[selectedModelKey] || []).map(m => m.model_size) : []), [modelsByKey, selectedModelKey]);

  const handleParamChange = (param, value) => {
    const newParams = { ...params, [param]: value };
    setParams(newParams);
  };



  const renderSlider = (label, param, min, max, step = 1, infoKey = null) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography gutterBottom>
            {label}
          </Typography>
          {infoKey && (
            <Tooltip title="More info">
              <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, [infoKey]: !prev[infoKey] }))}>
                <InfoOutlinedIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
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
      {infoKey && (
        <Collapse in={helpOpen[infoKey] === true}>
          <Typography variant="caption" color="text.secondary">
            {PARAMS_HELP[infoKey]}
          </Typography>
        </Collapse>
      )}
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
          zIndex: 1200
        }
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <img
            src={appTheme === 'dark' ? LOGO_PATHS.DARK_BG : LOGO_PATHS.DEFAULT}
            alt="InstaGeo Logo"
            style={{
              maxWidth: '200px',
              height: 'auto',
              display: 'block'
            }}
            onError={(e) => {
              logger.error('Error loading logo:', e);
              e.target.style.display = 'none';
            }}
          />
        </Box>

        <ProfileMenu appTheme={appTheme} />

        <Typography variant="h6" sx={{ mb: 2, color: '#1E88E5', textAlign: 'center' }}>
          Parameters
        </Typography>

        {!hasBoundingBox && (
          <Box sx={{
            mb: 3,
            p: 2,
            borderRadius: 1,
            border: '1px solid',
            borderColor: 'warning.main'
          }}>
            <Typography color="warning.main">
              Please draw a bounding box on the map first to select your area of interest.
            </Typography>
          </Box>
        )}

        {/* Dynamic model list from /api/models (cached 24h) */}
        <TextField
          select
          fullWidth
          label="Available Models"
          value={selectedModelKey}
          onChange={(e) => setSelectedModelKey(e.target.value)}
          disabled={!hasBoundingBox || loadingModels || modelKeys.length === 0}
          sx={{
            mb: 2,
            position: 'relative',
            '& .MuiSelect-icon': { zIndex: 3 },
            '& .MuiInputBase-input': { pr: 7 }
          }}
          slotProps={{
            input: {
              endAdornment: (
                <InputAdornment position="end" sx={{ position: 'absolute', right: 30, zIndex: 2 }}>
                <Tooltip title="Reload models">
                  <IconButton size="small" onClick={handleReloadModels} disabled={!hasBoundingBox || loadingModels} edge="start">
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                </InputAdornment>
              )
            }
          }}
        >
          {modelKeys.map((k) => {
            const name = (modelsByKey[k]?.[0]?.model_name) || k;
            return (
              <MenuItem key={k} value={k}>{name}</MenuItem>
            );
          })}
        </TextField>

        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Model Size</InputLabel>
          <Select
            value={selectedSize}
            label="Model Size"
            onChange={(e) => setSelectedSize(e.target.value)}
            disabled={!selectedModelKey}
          >
            {sizesForSelected.map((s) => (
              <MenuItem key={s} value={s}>{s}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Read-only model metadata */}
        {selectedModelKey && selectedSize && (() => {
          const info = (modelsByKey[selectedModelKey] || []).find(m => m.model_size === selectedSize);
          if (!info) return null;
          return (
            <Paper elevation={0} sx={{
              mb: 3,
              p: 2,
              border: '1px solid',
              borderColor: 'primary.light',
              borderLeft: '4px solid',
              borderLeftColor: 'primary.main'
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1" sx={{
                  color: 'text.primary',
                  fontSize: '0.875rem'
                }}>
                  Model Info
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title="About this model">
                    <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, model_description: !prev.model_description }))}>
                      <InfoOutlinedIcon fontSize="inherit" />
                    </IconButton>
                  </Tooltip>
                  <Chip size="small" color="primary" variant="outlined" label={`${info.model_type} • ${info.model_short_name} • ${selectedSize}`} />
                </Box>
              </Box>
              {info.model_description && (
                <Collapse in={helpOpen.model_description}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
                    {info.model_description}
                  </Typography>
                </Collapse>
              )}
              <Divider sx={{ mb: 1.5 }} />

              {/* Number of Parameters */}
              <Box sx={{ mb: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="subtitle2">
                    Number of Parameters
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {typeof info.num_params === 'number' ? `${Number(info.num_params.toFixed(2))} M` : info.num_params}
                  </Typography>
                </Box>
              </Box>

              {/* Chip Size */}
              <Box sx={{ mb: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      Chip Size
                    </Typography>
                    <Tooltip title="More info">
                      <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, chip_size: !prev.chip_size }))}>
                        <InfoOutlinedIcon fontSize="inherit" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {info.chip_size} x {info.chip_size} pixels
                  </Typography>
                </Box>
                <Collapse in={helpOpen.chip_size}>
                  <Typography variant="caption" color="text.secondary">
                    {PARAMS_HELP.chip_size}
                  </Typography>
                </Collapse>
              </Box>

              {/* Num Steps */}
              <Box sx={{ mb: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      Number of Steps
                    </Typography>
                    <Tooltip title="More info">
                      <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, num_steps: !prev.num_steps }))}>
                        <InfoOutlinedIcon fontSize="inherit" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {info.num_steps}
                  </Typography>
                </Box>
                <Collapse in={helpOpen.num_steps}>
                  <Typography variant="caption" color="text.secondary">
                    {PARAMS_HELP.num_steps}
                  </Typography>
                </Collapse>
              </Box>

              {/* Temporal Step */}
              <Box sx={{ mb: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      Temporal Step
                    </Typography>
                    <Tooltip title="More info">
                      <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, temporal_step: !prev.temporal_step }))}>
                        <InfoOutlinedIcon fontSize="inherit" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {info.temporal_step}
                  </Typography>
                </Box>
                <Collapse in={helpOpen.temporal_step}>
                  <Typography variant="caption" color="text.secondary">
                    {PARAMS_HELP.temporal_step}
                  </Typography>
                </Collapse>
              </Box>

              {/* Data Source */}
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      Data Source
                    </Typography>
                    <Tooltip title="More info">
                      <IconButton size="small" onClick={() => setHelpOpen(prev => ({ ...prev, data_source: !prev.data_source }))}>
                        <InfoOutlinedIcon fontSize="inherit" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {info.data_source}
                  </Typography>
                </Box>
                <Collapse in={helpOpen.data_source}>
                  <Typography variant="caption" color="text.secondary">
                    {PARAMS_HELP.data_source}
                  </Typography>
                </Collapse>
              </Box>
            </Paper>
          );
        })()}

        {/* Date Picker */}
        <LocalizationProvider dateAdapter={AdapterDayjs}>
          <DatePicker
            label="Date"
            value={dayjs(params.date)}
            onChange={(newValue) => {
              if (newValue) {
                handleParamChange('date', newValue.format('YYYY-MM-DD'));
              }
            }}
            minDate={dayjs('2016-01-01')}
            maxDate={dayjs()}
            slotProps={{
              textField: {
                fullWidth: true,
                sx: { mb: 3 },
                onKeyDown: (e) => {
                  e.preventDefault();
                  e.stopPropagation();
                },
                onKeyDownCapture: (e) => {
                  e.preventDefault();
                  e.stopPropagation();
                },
                inputProps: {readOnly: true}
              }
            }}
          />
        </LocalizationProvider>

        {renderSlider('Temporal Tolerance (days)', 'temporal_tolerance', 1, 30, 1, 'temporal_tolerance')}


        {renderSlider('Maximum Cloud Cover', 'cloud_coverage', 0, 100, 1, 'cloud_coverage')}

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={() => {
            // Inject selected model info into params for submission
            const info = (modelsByKey[selectedModelKey] || []).find(m => m.model_size === selectedSize);
            if (info) {
              const merged = {
                ...params,
                model_key: info.model_key,
                model_size: info.model_size,
              };
              logger.log('Merged params:', merged);
              onRunModel(merged);
              resetFormToDefaults();
              return;
            }
          }}
          disabled={!hasBoundingBox || isProcessing || !selectedModelKey || !selectedSize}
          sx={{
            mt: 2,
            mb: 3,
            py: 1.5,
            position: 'relative'
          }}
        >
          {isProcessing ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
              <CircularProgress
                size={20}
                sx={{
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
