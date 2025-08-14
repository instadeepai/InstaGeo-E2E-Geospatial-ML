import React, { useState, useEffect, useRef } from 'react';
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
    Pagination
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
    FilterList as FilterListIcon
} from '@mui/icons-material';
import VisualizationDialog from './VisualizationDialog';
import { INSTAGEO_BACKEND_API_ENDPOINTS } from '../config';

const TasksMonitor = ({ open, onClose, onAddTaskLayer }) => {
    const [tasks, setTasks] = useState([]);
    const [filteredTasks, setFilteredTasks] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedTask, setExpandedTask] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [visualizationDialogOpen, setVisualizationDialogOpen] = useState(false);
    const [selectedTaskForVisualization, setSelectedTaskForVisualization] = useState(null);
    const tasksPerPage = 5;
    const pollingInterval = useRef(null);

    // Filter states
    // const [searchTerm, setSearchTerm] = useState('');
    // const [statusFilter, setStatusFilter] = useState('all');

    // Pagination states
    // const [currentPage, setCurrentPage] = useState(1);
    // const tasksPerPage = 5;

    const fetchTasks = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetch(INSTAGEO_BACKEND_API_ENDPOINTS.GET_ALL_TASKS);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            setTasks(data);
        } catch (err) {
            console.error('Error fetching tasks:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Filter tasks based on search term and status filter
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

        setFilteredTasks(filtered); // Update filteredTasks state
    }, [tasks, searchTerm, statusFilter]);

    // Reset to first page when filters change (but not when tasks data updates)
    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, statusFilter]);

    // Calculate pagination
    const totalPages = Math.ceil(filteredTasks.length / tasksPerPage); // Use filteredTasks.length
    const startIndex = (currentPage - 1) * tasksPerPage;
    const endIndex = startIndex + tasksPerPage;
    const currentTasks = filteredTasks.slice(startIndex, endIndex); // Use filteredTasks directly

    useEffect(() => {
        if (open) {
            fetchTasks();
            // Start polling every 30 seconds
            pollingInterval.current = setInterval(fetchTasks, 30000);
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
    }, [open]);

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
            const response = await fetch(INSTAGEO_BACKEND_API_ENDPOINTS.VISUALIZE(task.task_id));

            if (!response.ok) {
                throw new Error(`Failed to get visualization data: ${response.status}`);
            }

            const titilerData = await response.json();

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
            console.error('Error getting visualization data:', error);
            setError(`Failed to load visualization data: ${error.message}`);
        }
    };

    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const handleStatusFilterChange = (event) => {
        setStatusFilter(event.target.value);
    };

    const clearFilters = () => {
        setSearchTerm('');
        setStatusFilter('all');
    };

    const handlePageChange = (event, newPage) => {
        setCurrentPage(newPage);
    };

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
                        <IconButton onClick={handleRefresh} disabled={loading}>
                            <RefreshIcon />
                        </IconButton>
                        <IconButton onClick={onClose}>
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

                    <Button
                        variant="outlined"
                        size="small"
                        onClick={clearFilters}
                        disabled={searchTerm === '' && statusFilter === 'all'}
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

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

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
                    <Card key={task.task_id} sx={{ mb: 2 }}>
                        <CardContent>
                            <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: 'repeat(6, 1fr)' }, alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Task ID
                                    </Typography>
                                    <Typography variant="body2" fontFamily="monospace">
                                        {task.task_id.slice(0, 8)}...
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Status
                                    </Typography>
                                    <Chip
                                        icon={getStatusIcon(task.status)}
                                        label={task.status.replace('_', ' ')}
                                        color={getStatusColor(task.status)}
                                        size="small"
                                    />
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Model Type
                                    </Typography>
                                    <Typography variant="body2">
                                        {task.model_type}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Bounding Boxes
                                    </Typography>
                                    <Typography variant="body2">
                                        {task.bboxes_count}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Created
                                    </Typography>
                                    <Typography variant="body2">
                                        {formatDate(task.created_at)}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Box display="flex" gap={1}>
                                        <Button
                                            size="small"
                                            onClick={() => handleTaskExpand(task.task_id)}
                                        >
                                            {expandedTask === task.task_id ? 'Hide' : 'Details'}
                                        </Button>
                                        {task.status === 'completed' && (
                                            <Button
                                                size="small"
                                                variant="contained"
                                                startIcon={<VisibilityIcon />}
                                                onClick={() => handleVisualize(task)}
                                                sx={{ minWidth: 'auto' }}
                                            >
                                                Visualize
                                            </Button>
                                        )}
                                    </Box>
                                </Box>
                            </Box>

                            {/* Progress Bar */}
                            <Box mt={2}>
                                <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                                    Progress
                                </Typography>
                                <LinearProgress
                                    variant="determinate"
                                    value={getTaskProgress(task)}
                                    sx={{ height: 8, borderRadius: 4 }}
                                />
                                <Typography variant="caption" color="textSecondary">
                                    {Math.round(getTaskProgress(task))}% Complete
                                </Typography>
                            </Box>

                            {/* Expanded Details */}
                            {expandedTask === task.task_id && (
                                <Box mt={2}>
                                    <Typography variant="h6" gutterBottom>
                                        Stage Details
                                    </Typography>
                                    <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)' } }}>
                                        {Object.entries(task.stages).map(([stageName, stageData]) => (
                                            <Box key={stageName}>
                                                <Card variant="outlined">
                                                    <CardContent>
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
                                                                Started: {formatDate(stageData.started_at)}
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
                                                                 !(stageName === 'model_prediction' && stageData.result.aod_values) &&
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
