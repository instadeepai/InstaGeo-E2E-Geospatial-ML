import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    Button,
    Typography,
    Box,
    Card,
    CardContent,
    Chip,
    IconButton,
    Alert,
    CircularProgress,
    LinearProgress,
    TextField,
    MenuItem,
    Pagination,
    Link,
    Tooltip as MuiTooltip,
    useTheme
} from '@mui/material';
import {
    Close as CloseIcon,
    Refresh as RefreshIcon,
    CheckCircle as CheckCircleIcon,
    Error as ErrorIcon,
    PlayArrow as PlayArrowIcon,
    Schedule as ScheduleIcon,
    Pause as PauseIcon,
    Visibility as VisibilityIcon,
    FilterList as FilterListIcon,
    Info as InfoIcon,
    Login as LoginIcon
} from '@mui/icons-material';
import { useAuth0 } from '@auth0/auth0-react';
import VisualizationDialog from './VisualizationDialog';
import BoundingBoxSnapshot from './BoundingBoxSnapshot';
import { logger } from '../utils/logger';
import { fetchModelsWithTTL } from '../utils/modelsCache';
import apiService from '../services/apiService';
import { isAuth0Configured } from '../auth0-config';
import { isAuthenticationError } from '../utils/authErrors';

const TasksMonitor = ({ open, onClose, onAddTaskLayer }) => {
    const { getAccessTokenSilently, loginWithRedirect } = useAuth0();
    const auth0Enabled = isAuth0Configured();
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';

    // Theme-aware styling using MUI theme colors
    const primaryColor = theme.palette.primary.main;
    const primaryDark = theme.palette.primary.dark;

    const themeStyles = {
        primaryColor: primaryColor,
        buttonBorder: primaryColor,
        buttonHover: isDark ? 'rgba(33, 150, 243, 0.08)' : 'rgba(33, 150, 243, 0.04)',
        buttonHoverBorder: isDark ? primaryDark : primaryColor,
    };

    const [tasks, setTasks] = useState([]);
    const [filteredTasks, setFilteredTasks] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedTask, setExpandedTask] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [modelFilter, setModelFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [visualizationDialogOpen, setVisualizationDialogOpen] = useState(false);
    const [selectedTaskForVisualization, setSelectedTaskForVisualization] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const tasksPerPage = 5;
    const pollingInterval = useRef(null);

    const fetchTasks = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await apiService.getAllTasks(getAccessTokenSilently);
            setTasks(data);
        } catch (err) {
            logger.error('Error fetching tasks:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [getAccessTokenSilently]);

    // Filter tasks based on search term, status filter, and model filter
    useEffect(() => {
        let filtered = tasks;

        // Filter by search term (task ID)
        if (searchTerm) {
            filtered = filtered.filter(task =>
                task.task_id.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        // Filter by status
        if (statusFilter !== 'all') {
            filtered = filtered.filter(task => task.status === statusFilter);
        }

        // Filter by model
        if (modelFilter !== 'all') {
            filtered = filtered.filter(task => task.model_name === modelFilter);
        }

        setFilteredTasks(filtered); // Update filteredTasks state
    }, [tasks, searchTerm, statusFilter, modelFilter]);

    // Reset to first page when filters change (but not when tasks data updates)
    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, statusFilter, modelFilter]);

    // Calculate pagination
    const totalPages = Math.ceil(filteredTasks.length / tasksPerPage); // Use filteredTasks.length
    const startIndex = (currentPage - 1) * tasksPerPage;
    const endIndex = startIndex + tasksPerPage;
    const currentTasks = filteredTasks.slice(startIndex, endIndex); // Use filteredTasks directly

    useEffect(() => {
        if (open) {
            fetchTasks();
            // Start polling every 60 seconds
            pollingInterval.current = setInterval(fetchTasks, 60000);
        } else {
            // Clear polling when dialog is closed
            if (pollingInterval.current) {
                clearInterval(pollingInterval.current);
                pollingInterval.current = null;
            }
        }

        return () => {
            if (pollingInterval.current) {
                clearInterval(pollingInterval.current);
            }
        };
    }, [open, fetchTasks]);

    const getStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'success';
            case 'failed':
                return 'error';
            case 'data_processing':
            case 'model_prediction':
            case 'visualization_preparation':
                return 'warning';
            default:
                return 'default';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <CheckCircleIcon fontSize="small" />;
            case 'failed':
                return <ErrorIcon fontSize="small" />;
            case 'data_processing':
            case 'model_prediction':
            case 'visualization_preparation':
                return <PlayArrowIcon fontSize="small" />;
            default:
                return <ScheduleIcon fontSize="small" />;
        }
    };

    const getStageStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'success';
            case 'failed':
                return 'error';
            case 'running':
                return 'warning';
            case 'pending':
                return 'default';
            default:
                return 'default';
        }
    };

    const getStageDisplayName = (stageName) => {
        switch (stageName) {
            case 'data_processing':
                return 'Data Processing';
            case 'model_prediction':
                return 'Model Prediction';
            case 'visualization_preparation':
                return 'Visualization Preparation';
            default:
                return stageName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    };

    const getStageStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <CheckCircleIcon fontSize="small" />;
            case 'failed':
                return <ErrorIcon fontSize="small" />;
            case 'running':
                return <PlayArrowIcon fontSize="small" />;
            case 'pending':
                return <PauseIcon fontSize="small" />;
            default:
                return <ScheduleIcon fontSize="small" />;
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    const formatDateOnly = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    const getTaskProgress = (task) => {
        const stages = task.stages;
        const totalStages = 3; // data_processing, model_prediction, and visualization_preparation
        let completedStages = 0;

        if (stages.data_processing?.status === 'completed') completedStages++;
        if (stages.model_prediction?.status === 'completed') completedStages++;
        if (stages.visualization_preparation?.status === 'completed') completedStages++;

        return (completedStages / totalStages) * 100;
    };

    const handleRefresh = () => {
        fetchTasks();
    };

    const handleTaskExpand = (taskId) => {
        setExpandedTask(expandedTask === taskId ? null : taskId);
    };

    const handleVisualize = async (task) => {
        try {
            // Call the TiTiler service API to get visualization data
            const titilerData = await apiService.visualizeTask(task.task_id, getAccessTokenSilently);

            if (!titilerData || (titilerData && !titilerData.prediction && !titilerData.satellite)) {
                throw new Error('Visualization data is not available yet for this task');
            }

            // Set the visualization data and open dialog
            setSelectedTaskForVisualization({
                ...task,
                titiler_data: titilerData
            });
            setVisualizationDialogOpen(true);

        } catch (error) {
            logger.error('Error getting visualization data:', error);
            setError(`Failed to load visualization data: ${error.message}`);
        }
    };

    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const handleStatusFilterChange = (event) => {
        setStatusFilter(event.target.value);
    };

    const handleModelFilterChange = (event) => {
        setModelFilter(event.target.value);
    };

    const clearFilters = () => {
        setSearchTerm('');
        setStatusFilter('all');
        setModelFilter('all');
    };

    const handlePageChange = (event, newPage) => {
        setCurrentPage(newPage);
    };

    // Fetch models from cache
    const fetchModels = useCallback(async () => {
        try {
            const models = await fetchModelsWithTTL(getAccessTokenSilently);
            setAvailableModels(models || []);
        } catch (error) {
            logger.warn('Failed to fetch models for filter:', error);
            setAvailableModels([]);
        }
    }, [getAccessTokenSilently]);

    // Get unique model names from available models
    const getUniqueModelNames = () => {
        return availableModels
            .map(model => model.model_name)
            .filter((name, index, self) => name && self.indexOf(name) === index)
            .sort();
    };

    // Fetch models when component mounts
    useEffect(() => {
        if (open) {
            fetchModels();
        }
    }, [open, fetchModels]);

    return (
        <>
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{
                style: { maxHeight: '90vh' }
            }}
        >
            <DialogTitle>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h6">Tasks Monitor</Typography>
                    <Box>
                        <IconButton
                            onClick={handleRefresh}
                            disabled={loading}
                            sx={{
                                '&:hover': {
                                    backgroundColor: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'rgba(255, 255, 255, 0.08)'
                                            : 'rgba(0, 0, 0, 0.04)'
                                }
                            }}
                        >
                            <RefreshIcon />
                        </IconButton>
                        <IconButton
                            onClick={onClose}
                            sx={{
                                '&:hover': {
                                    backgroundColor: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'rgba(255, 255, 255, 0.08)'
                                            : 'rgba(0, 0, 0, 0.04)'
                                }
                            }}
                        >
                            <CloseIcon />
                        </IconButton>
                    </Box>
                </Box>
            </DialogTitle>
            <DialogContent sx={{ pt: 2 }}>
                {/* Search and Filter Controls */}
                <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start', pt: 1 }}>
                    <TextField
                        label="Search by Task ID"
                        variant="outlined"
                        size="small"
                        value={searchTerm}
                        onChange={handleSearchChange}
                        InputProps={{
                            startAdornment: <FilterListIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                        }}
                        sx={{ minWidth: 250 }}
                    />

                    <TextField
                        label="Status Filter"
                        variant="outlined"
                        size="small"
                        value={statusFilter}
                        onChange={handleStatusFilterChange}
                        InputProps={{
                            startAdornment: <FilterListIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                        }}
                        sx={{ minWidth: 150 }}
                        select
                    >
                        <MenuItem value="all">All Statuses</MenuItem>
                        <MenuItem value="data_processing">Data Processing</MenuItem>
                        <MenuItem value="model_prediction">Model Prediction</MenuItem>
                        <MenuItem value="visualization_preparation">Visualization Preparation</MenuItem>
                        <MenuItem value="completed">Completed</MenuItem>
                        <MenuItem value="failed">Failed</MenuItem>
                    </TextField>

                    <TextField
                        label="Model Filter"
                        variant="outlined"
                        size="small"
                        value={modelFilter}
                        onChange={handleModelFilterChange}
                        InputProps={{
                            startAdornment: <FilterListIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                        }}
                        sx={{ minWidth: 200 }}
                        select
                    >
                        <MenuItem value="all">All Models</MenuItem>
                        {getUniqueModelNames().map((modelName) => (
                            <MenuItem key={modelName} value={modelName}>
                                {modelName}
                            </MenuItem>
                        ))}
                    </TextField>

                    <Button
                        variant="outlined"
                        size="small"
                        onClick={clearFilters}
                        disabled={searchTerm === '' && statusFilter === 'all' && modelFilter === 'all'}
                    >
                        Clear Filters
                    </Button>

                    <Typography variant="body2" color="textSecondary" sx={{ alignSelf: 'center' }}>
                        Showing {filteredTasks.length} of {tasks.length} tasks
                    </Typography>
                </Box>

                {loading && tasks.length === 0 && (
                    <Box display="flex" justifyContent="center" p={3}>
                        <CircularProgress />
                    </Box>
                )}

                {error && (() => {
                    const isAuthError = isAuthenticationError(error);

                    const handleSignIn = () => {
                        if (auth0Enabled) {
                            loginWithRedirect({
                                appState: {
                                    returnTo: window.location.pathname,
                                },
                            });
                        }
                    };

                    return (
                        <Alert
                            severity="error"
                            sx={{ mb: 2 }}
                            action={
                                isAuthError && auth0Enabled ? (
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        onClick={handleSignIn}
                                        startIcon={<LoginIcon />}
                                        sx={{
                                            borderColor: themeStyles.buttonBorder,
                                            color: themeStyles.primaryColor,
                                            '&:hover': {
                                                borderColor: themeStyles.buttonHoverBorder,
                                                backgroundColor: themeStyles.buttonHover
                                            }
                                        }}
                                    >
                                        Sign In
                                    </Button>
                                ) : null
                            }
                        >
                            <Typography variant="body2" component="div">
                                {error}
                            </Typography>
                        </Alert>
                    );
                })()}

                {filteredTasks.length === 0 && !loading && !error && (
                    <Box textAlign="center" p={3}>
                        <Typography variant="body1" color="textSecondary">
                            {tasks.length === 0
                                ? 'No tasks found. Create a new task to see it here.'
                                : 'No tasks match the current filters.'
                            }
                        </Typography>
                    </Box>
                )}

                {currentTasks.map((task) => (
                    <Card key={task.task_id} sx={{
                        mb: 2,
                        border: 'none',
                        boxShadow: (theme) =>
                            theme.palette.mode === 'dark'
                                ? '0 4px 12px rgba(0, 0, 0, 0.4)'
                                : '0 2px 8px rgba(0, 0, 0, 0.15)',
                        '&:hover': {
                            boxShadow: (theme) =>
                                theme.palette.mode === 'dark'
                                    ? '0 8px 24px rgba(0, 0, 0, 0.5)'
                                    : '0 4px 16px rgba(0, 0, 0, 0.25)'
                        }
                    }}>
                        <CardContent sx={{ p: 0 }}>
                            {/* Header with Area Snapshot and Basic Info */}
                            <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                p: 2,
                                borderBottom: 1,
                                borderColor: 'divider',
                                bgcolor: 'grey.50',
                                background: (theme) =>
                                    theme.palette.mode === 'dark'
                                        ? 'linear-gradient(135deg, #718096 0%, #718096 100%)'
                                        : 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)'
                            }}>
                                {/* Area Snapshot - Left Side */}
                                <Box sx={{
                                    width: 120,
                                    height: 120,
                                    mr: 2,
                                    borderRadius: 1,
                                    overflow: 'hidden',
                                    border: 1,
                                    borderColor: 'divider',
                                    bgcolor: 'background.paper',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}>
                                    <BoundingBoxSnapshot
                                        bbox={task.bboxes?.[0] || null}
                                        taskId={task.task_id}
                                    />
                                </Box>

                                {/* Task Info - Right Side */}
                                <Box sx={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Box>
                                        <Typography variant="h6" sx={{
                                            mb: 1,
                                            fontFamily: 'monospace',
                                            fontWeight: 400,
                                            fontSize: '1rem',
                                            letterSpacing: '0.5px',
                                            color: (theme) =>
                                                theme.palette.mode === 'dark'
                                                    ? 'white'
                                                    : 'text.primary',
                                            background: (theme) =>
                                                theme.palette.mode === 'dark'
                                                    ? 'none'
                                                    : 'linear-gradient(45deg, #1976d2, #42a5f5)',
                                            backgroundClip: (theme) =>
                                                theme.palette.mode === 'dark'
                                                    ? 'unset'
                                                    : 'text',
                                            WebkitBackgroundClip: (theme) =>
                                                theme.palette.mode === 'dark'
                                                    ? 'unset'
                                                    : 'text',
                                            WebkitTextFillColor: (theme) =>
                                                theme.palette.mode === 'dark'
                                                    ? 'white'
                                                    : 'transparent'
                                        }}>
                                            {task.task_id}
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                                <Typography variant="body2" color="textPrimary" sx={{ fontWeight: 500, width: 60, textAlign: 'right' }}>
                                                    Status:
                                                </Typography>
                                                <Chip
                                                    icon={getStatusIcon(task.status)}
                                                    label={task.status.replace('_', ' ')}
                                                    color={getStatusColor(task.status)}
                                                    size="small"
                                                />
                                            </Box>

                                            {/* Model Details as Chip */}
                                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                                <Typography variant="body2" color="textPrimary" sx={{ fontWeight: 500, width: 60, textAlign: 'right' }}>
                                                    Model:
                                                </Typography>
                                                <Chip
                                                    label={`${task.model_short_name}${task.model_size ? ` • ${task.model_size}` : ''}`}
                                                    variant="outlined"
                                                    size="small"
                                                    sx={{
                                                        fontFamily: 'monospace',
                                                        fontSize: '0.75rem',
                                                        backgroundColor: 'white',
                                                        '& .MuiChip-label': {
                                                            color: 'black'
                                                        }
                                                    }}
                                                />
                                            </Box>

                                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                                <Typography variant="body2" color="textPrimary" sx={{ fontWeight: 500, width: 60, textAlign: 'right' }}>
                                                    Created:
                                                </Typography>
                                                <Chip
                                                    label={formatDate(task.created_at)}
                                                    variant="outlined"
                                                    size="small"
                                                    sx={{
                                                        fontFamily: 'monospace',
                                                        fontSize: '0.75rem',
                                                        backgroundColor: 'white',
                                                        '& .MuiChip-label': {
                                                            color: 'black'
                                                        }
                                                    }}
                                                />
                                            </Box>
                                        </Box>
                                    </Box>

                                    {/* Action Buttons */}
                                    <Box display="flex" gap={1}>
                                        <Button
                                            size="small"
                                            onClick={() => handleTaskExpand(task.task_id)}
                                            sx={{ color: (theme) => theme.palette.mode === 'dark' ? 'white' : undefined }}
                                        >
                                            {expandedTask === task.task_id ? 'Hide' : 'Details'}
                                        </Button>
                                        {task.status === 'completed' && (
                                            <Button
                                                size="small"
                                                variant="contained"
                                                startIcon={<VisibilityIcon />}
                                                onClick={() => handleVisualize(task)}
                                                sx={{
                                                    minWidth: 'auto',
                                                    backgroundColor: (theme) =>
                                                        theme.palette.mode === 'dark'
                                                            ? theme.palette.primary.main
                                                            : undefined,
                                                    '&:hover': {
                                                        backgroundColor: (theme) =>
                                                            theme.palette.mode === 'dark'
                                                                ? theme.palette.primary.dark
                                                                : undefined,
                                                        boxShadow: (theme) =>
                                                            theme.palette.mode === 'dark'
                                                                ? '0 4px 8px rgba(0, 0, 0, 0.3)'
                                                                : undefined
                                                    }
                                                }}
                                            >
                                                Visualize
                                            </Button>
                                        )}
                                    </Box>
                                </Box>
                            </Box>

                        {/* Progress Bar */}
                        <Box sx={{ mt: 1, px: 2, py: 0.125 }}>
                            <Typography variant="subtitle2" color="textSecondary" sx={{ mb: 0.25, fontSize: '0.75rem' }}>
                                Progress
                            </Typography>
                            <LinearProgress
                                variant="determinate"
                                value={getTaskProgress(task)}
                                sx={{ height: 4, borderRadius: 2, mb: 0.25 }}
                            />
                            <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.7rem' }}>
                                {Math.round(getTaskProgress(task))}% Complete
                            </Typography>
                        </Box>

                        {/* Expanded Details */}
                        {expandedTask === task.task_id && (
                            <Box sx={{ mt: 3, px: 2, py: 2 }}>
                                <Typography variant="h6" sx={{
                                    mb: 1,
                                    fontFamily: 'monospace',
                                    fontWeight: 400,
                                    fontSize: '1rem',
                                    letterSpacing: '0.5px',
                                    color: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'white'
                                            : 'text.primary',
                                    background: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'none'
                                            : 'linear-gradient(45deg, #1976d2, #42a5f5)',
                                    backgroundClip: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'unset'
                                            : 'text',
                                    WebkitBackgroundClip: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'unset'
                                            : 'text',
                                    WebkitTextFillColor: (theme) =>
                                        theme.palette.mode === 'dark'
                                            ? 'white'
                                            : 'transparent'
                                }}>
                                    Stage Details
                                </Typography>
                                <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)' } }}>
                                    {Object.entries(task.stages).map(([stageName, stageData]) => (
                                        <Box key={stageName}>
                                            <Card variant="outlined" sx={{ height: '100%' }}>
                                                <CardContent sx={{ p: 2 }}>
                                                    <Box display="flex" alignItems="center" mb={1}>
                                                        <Chip
                                                            icon={getStageStatusIcon(stageData.status)}
                                                            label={stageData.status}
                                                            color={getStageStatusColor(stageData.status)}
                                                            size="small"
                                                            sx={{ mr: 1 }}
                                                        />
                                                        <Typography variant="subtitle2">
                                                            {getStageDisplayName(stageName)}
                                                        </Typography>
                                                    </Box>

                                                        {stageData.started_at && (
                                                            <Typography variant="caption" display="block">
                                                                Initiated: {formatDate(stageData.started_at)}
                                                            </Typography>
                                                        )}

                                                        {stageData.completed_at && (
                                                            <Typography variant="caption" display="block">
                                                                Completed: {formatDate(stageData.completed_at)}
                                                            </Typography>
                                                        )}

                                                        {stageData.error && (
                                                            <Alert severity="error" sx={{ mt: 1 }}>
                                                                {stageData.error}
                                                            </Alert>
                                                        )}

                                                        {stageData.result && (
                                                            <Box sx={{ mt: 1 }}>
                                                                {stageName === 'data_processing' && stageData.result.chips_created && (
                                                                    <>
                                                                        <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                                                                            ✅ Chips Created: {stageData.result.chips_created}
                                                                        </Typography>

                                                                        {/* Additional data extraction metrics */}
                                                                        {stageData.result.chip_size && (
                                                                            <Typography variant="caption" display="block">
                                                                                📐 Chip Size: {stageData.result.chip_size} x {stageData.result.chip_size} pixels
                                                                            </Typography>
                                                                        )}

                                                                        {stageData.result.processing_duration && (
                                                                            <Typography variant="caption" display="block">
                                                                                ⏱️ Extraction Duration: {stageData.result.processing_duration} s
                                                                            </Typography>
                                                                        )}


                                                                        {stageData.result.data_source && stageData.result.data_source !== 'unknown' && (
                                                                            <Typography variant="caption" display="block">
                                                                                🛰️ Source: {stageData.result.data_source}
                                                                            </Typography>
                                                                        )}

                                                                        {stageData.result.target_date && stageData.result.target_date !== 'unknown' && stageData.result.temporal_tolerance && (
                                                                            <Typography variant="caption" display="block">
                                                                                📅 Target Date: {formatDateOnly(stageData.result.target_date)} (±  📅 {stageData.result.temporal_tolerance} days)
                                                                            </Typography>
                                                                        )}

                                                                        {stageData.result.bboxes_processed && (
                                                                            <Typography variant="caption" display="block">
                                                                                📍 Bounding Boxes Processed: {stageData.result.bboxes_processed}
                                                                            </Typography>
                                                                        )}

                                                                    </>
                                                                )}

                                                                {stageName === 'model_prediction' && (
                                                                    <>
                                                                        {stageData.result["model/GFLOPs"] !== undefined && (
                                                                            <Typography variant="caption" display="block">
                                                                                🧮 GFLOPs: {Number(stageData.result["model/GFLOPs"]).toFixed(4)}
                                                                            </Typography>
                                                                        )}
                                                                        {stageData.result.CO2_emissions !== undefined && (
                                                                            <Box display="flex" alignItems="center" gap={0.5}>
                                                                                <Typography variant="caption" display="block">
                                                                                    🌿 CO2 emissions: {Number(stageData.result.CO2_emissions).toFixed(6)} (g CO₂)
                                                                                </Typography>
                                                                                <MuiTooltip title={
                                                                                    <Typography variant="caption">
                                                                                        CO₂ emissions calculated using CodeCarbon - an open-source tool for tracking compute-based carbon emissions.
                                                                                        <br/>
                                                                                        <Link
                                                                                            href="http://codecarbon.io"
                                                                                            target="_blank"
                                                                                            rel="noopener noreferrer"
                                                                                            sx={{ color: 'inherit', textDecoration: 'underline' }}
                                                                                        >
                                                                                            Learn more
                                                                                        </Link>
                                                                                    </Typography>
                                                                                }>
                                                                                    <InfoIcon sx={{ fontSize: 16, color: 'text.secondary', cursor: 'help' }} />
                                                                                </MuiTooltip>
                                                                            </Box>
                                                                        )}
                                                                        {stageData.result.energy_consumed !== undefined && (
                                                                            <Typography variant="caption" display="block">
                                                                                ⚡ Energy consumed: {Number(stageData.result.energy_consumed).toFixed(10)} (kWh)
                                                                            </Typography>
                                                                        )}
                                                                        {stageData.result.inference_time !== undefined && (
                                                                            <Typography variant="caption" display="block">
                                                                                ⏱️ Inference time: {Number(stageData.result.inference_time).toFixed(4)} s
                                                                            </Typography>
                                                                        )}
                                                                    </>
                                                                )}

                                                                {stageName === 'visualization_preparation' && stageData.result && (
                                                                    <>
                                                                        <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                                                                            ✅ Visualization Data Prepared
                                                                        </Typography>

                                                                        {stageData.result.processing_duration && (
                                                                            <Typography variant="caption" display="block">
                                                                                ⏱️ Processing Duration: {stageData.result.processing_duration} s
                                                                            </Typography>
                                                                        )}

                                                                    </>
                                                                )}

                                                                {/* Fallback for other results */}
                                                                {!(stageName === 'data_processing' && (stageData.result.chips_created || stageData.result.processing_date)) &&
                                                                 !(stageName === 'model_prediction' && (stageData.result["model/GFLOPs"] !== undefined || stageData.result.CO2_emissions !== undefined || stageData.result.energy_consumed !== undefined || stageData.result.inference_time !== undefined)) &&
                                                                 !(stageName === 'visualization_preparation' && stageData.result.processing_duration) && (

                                                                    <Typography variant="caption" display="block">
                                                                        Result: {JSON.stringify(stageData.result).slice(0, 100)}...
                                                                    </Typography>
                                                                )}
                                                            </Box>
                                                        )}
                                                    </CardContent>
                                                </Card>
                                            </Box>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </CardContent>
                    </Card>
                ))}

                {/* Pagination */}
                {filteredTasks.length > tasksPerPage && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3, mb: 1 }}>
                        <Pagination
                            count={totalPages}
                            page={currentPage}
                            onChange={handlePageChange}
                            color="primary"
                            showFirstButton
                            showLastButton
                        />
                    </Box>
                )}

                {/* Page Info */}
                {filteredTasks.length > 0 && (
                    <Box sx={{ textAlign: 'center', mb: 1 }}>
                        <Typography variant="body2" color="textSecondary">
                            Page {currentPage} of {totalPages} •
                            Showing tasks {startIndex + 1}-{Math.min(endIndex, filteredTasks.length)} of {filteredTasks.length}
                        </Typography>
                    </Box>
                )}
            </DialogContent>
        </Dialog>

        {/* Visualization Data Dialog */}
        <VisualizationDialog
            open={visualizationDialogOpen}
            onClose={() => setVisualizationDialogOpen(false)}
            task={selectedTaskForVisualization}
            onAddToMap={onAddTaskLayer}
            onCloseTasksMonitor={onClose}
        />
    </>
    );
};

export default TasksMonitor;
