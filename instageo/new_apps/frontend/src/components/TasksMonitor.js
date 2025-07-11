import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    IconButton,
    Chip,
    Typography,
    Box,
    CircularProgress,
    Alert,
    Button,
    LinearProgress,
    Grid,
    Card,
    CardContent,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Pagination,
} from '@mui/material';
import {
    Close as CloseIcon,
    Refresh as RefreshIcon,
    CheckCircle as CheckCircleIcon,
    Error as ErrorIcon,
    Schedule as ScheduleIcon,
    PlayArrow as PlayArrowIcon,
    Pause as PauseIcon,
    Visibility as VisibilityIcon,
    Search as SearchIcon,
} from '@mui/icons-material';
import { INSTAGEO_BACKEND_API_ENDPOINTS } from '../config';

const TasksMonitor = ({ open, onClose }) => {
    const [tasks, setTasks] = useState([]);
    const [filteredTasks, setFilteredTasks] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedTask, setExpandedTask] = useState(null);
    const [pollingInterval, setPollingInterval] = useState(null);

    // Filter states
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');

    // Pagination states
    const [currentPage, setCurrentPage] = useState(1);
    const tasksPerPage = 5;

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

        setFilteredTasks(filtered);
    }, [tasks, searchTerm, statusFilter]);

    // Reset to first page when filters change (but not when tasks data updates)
    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, statusFilter]);

    // Calculate pagination
    const totalPages = Math.ceil(filteredTasks.length / tasksPerPage);
    const startIndex = (currentPage - 1) * tasksPerPage;
    const endIndex = startIndex + tasksPerPage;
    const currentTasks = filteredTasks.slice(startIndex, endIndex);

    useEffect(() => {
        if (open) {
            fetchTasks();
            // Start polling every 10 seconds
            const interval = setInterval(fetchTasks, 10000);
            setPollingInterval(interval);
        } else {
            // Clear polling when dialog is closed
            if (pollingInterval) {
                clearInterval(pollingInterval);
                setPollingInterval(null);
            }
        }

        return () => {
            if (pollingInterval) {
                clearInterval(pollingInterval);
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
        return new Date(dateString).toLocaleString();
    };

    const getTaskProgress = (task) => {
        const stages = task.stages;
        const totalStages = 2; // data_processing and model_prediction
        let completedStages = 0;

        if (stages.data_processing?.status === 'completed') completedStages++;
        if (stages.model_prediction?.status === 'completed') completedStages++;

        return (completedStages / totalStages) * 100;
    };

    const handleRefresh = () => {
        fetchTasks();
    };

    const handleTaskExpand = (taskId) => {
        setExpandedTask(expandedTask === taskId ? null : taskId);
    };

    const handleVisualize = (taskId) => {
        // Dummy function for now - will be implemented later
        console.log(`Visualize results for task: ${taskId}`);
        alert(`Visualize functionality will be implemented later for task: ${taskId}`);
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
                            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                        }}
                        sx={{ minWidth: 250 }}
                    />

                    <FormControl size="small" sx={{ minWidth: 150 }}>
                        <InputLabel>Status Filter</InputLabel>
                        <Select
                            value={statusFilter}
                            label="Status Filter"
                            onChange={handleStatusFilterChange}
                        >
                            <MenuItem value="all">All Statuses</MenuItem>
                            <MenuItem value="data_processing">Data Processing</MenuItem>
                            <MenuItem value="model_prediction">Model Prediction</MenuItem>
                            <MenuItem value="completed">Completed</MenuItem>
                            <MenuItem value="failed">Failed</MenuItem>
                        </Select>
                    </FormControl>

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
                            <Grid container spacing={2} alignItems="center">
                                <Grid item xs={12} md={2}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Task ID
                                    </Typography>
                                    <Typography variant="body2" fontFamily="monospace">
                                        {task.task_id.slice(0, 8)}...
                                    </Typography>
                                </Grid>
                                <Grid item xs={12} md={2}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Status
                                    </Typography>
                                    <Chip
                                        icon={getStatusIcon(task.status)}
                                        label={task.status.replace('_', ' ')}
                                        color={getStatusColor(task.status)}
                                        size="small"
                                    />
                                </Grid>
                                <Grid item xs={12} md={2}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Model Type
                                    </Typography>
                                    <Typography variant="body2">
                                        {task.model_type}
                                    </Typography>
                                </Grid>
                                <Grid item xs={12} md={2}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Bounding Boxes
                                    </Typography>
                                    <Typography variant="body2">
                                        {task.bounding_boxes_count}
                                    </Typography>
                                </Grid>
                                <Grid item xs={12} md={2}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Created
                                    </Typography>
                                    <Typography variant="body2">
                                        {formatDate(task.created_at)}
                                    </Typography>
                                </Grid>
                                <Grid item xs={12} md={2}>
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
                                                onClick={() => handleVisualize(task.task_id)}
                                                sx={{ minWidth: 'auto' }}
                                            >
                                                Visualize
                                            </Button>
                                        )}
                                    </Box>
                                </Grid>
                            </Grid>

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
                                    <Grid container spacing={2}>
                                        {Object.entries(task.stages).map(([stageName, stageData]) => (
                                            <Grid item xs={12} md={6} key={stageName}>
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
                                                                {stageName.replace('_', ' ')}
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
                                                            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                                                                Result: {JSON.stringify(stageData.result).slice(0, 100)}...
                                                            </Typography>
                                                        )}
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        ))}
                                    </Grid>
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
    );
};

export default TasksMonitor;
