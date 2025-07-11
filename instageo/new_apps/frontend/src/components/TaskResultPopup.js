import React from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    Box,
    Chip,
    IconButton,
    Alert,
    CircularProgress,
    LinearProgress,
    Stepper,
    Step,
    StepLabel,
    StepContent
} from '@mui/material';
import {
    CheckCircle as SuccessIcon,
    Error as ErrorIcon,
    Close as CloseIcon,
    ContentCopy as CopyIcon,
    PlayArrow as RunningIcon,
} from '@mui/icons-material';

const TaskResultPopup = ({ open, onClose, result, error, onOpenTasksMonitor }) => {
    const handleCopyTaskId = () => {
        if (result?.task_id) {
            navigator.clipboard.writeText(result.task_id);
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'success';
            case 'failed':
                return 'error';
            case 'data_processing':
            case 'model_prediction':
                return 'primary';
            default:
                return 'default';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <SuccessIcon color="success" />;
            case 'failed':
                return <ErrorIcon color="error" />;
            case 'data_processing':
            case 'model_prediction':
                return <RunningIcon color="primary" />;
            default:
                return <CircularProgress size={20} />;
        }
    };

    const getTaskStageStatus = (taskStatus, stages) => {
        // If we have detailed stages information, use it
        if (stages && stages.data_processing && stages.model_prediction) {
            return {
                dataProcessing: stages.data_processing.status || 'pending',
                modelPrediction: stages.model_prediction.status || 'pending'
            };
        }

        // Fallback to task status mapping
        switch (taskStatus) {
            case 'data_processing':
                return { dataProcessing: 'running', modelPrediction: 'pending' };
            case 'model_prediction':
                return { dataProcessing: 'completed', modelPrediction: 'running' };
            case 'completed':
                return { dataProcessing: 'completed', modelPrediction: 'completed' };
            case 'failed':
                // Check if data processing failed or model prediction failed
                if (stages && stages.data_processing && stages.data_processing.status === 'failed') {
                    return { dataProcessing: 'failed', modelPrediction: 'pending' };
                } else {
                    return { dataProcessing: 'completed', modelPrediction: 'failed' };
                }
            default:
                return { dataProcessing: 'pending', modelPrediction: 'pending' };
        }
    };

    const getStatusDisplayText = (status) => {
        switch (status) {
            case 'completed':
                return 'Completed';
            case 'failed':
                return 'Failed';
            case 'data_processing':
                return 'Data Processing';
            case 'model_prediction':
                return 'Model Prediction';
            default:
                return status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    };

    if (error) {
        return (
            <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
                <DialogTitle sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    color: 'error.main'
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <ErrorIcon color="error" />
                        <Typography variant="h6">Task Submission Failed</Typography>
                    </Box>
                    <IconButton onClick={onClose} size="small">
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
                <DialogContent>
                    <Alert severity="error" sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                            Error Details:
                        </Typography>
                        <Typography variant="body2">
                            {error.message || 'An unexpected error occurred while submitting the task.'}
                        </Typography>
                    </Alert>
                    <Typography variant="body2" color="text.secondary">
                        Please check your input and try again. If the problem persists, contact support.
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={onClose} variant="contained" color="primary">
                        Close
                    </Button>
                </DialogActions>
            </Dialog>
        );
    }

    if (!result) return null;

    const stageStatus = getTaskStageStatus(result.status, result.stages);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                color: result.status === 'completed' ? 'success.main' :
                       result.status === 'failed' ? 'error.main' : 'primary.main'
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(result.status)}
                    <Typography variant="h6">Task Submitted Successfully</Typography>
                </Box>
                <IconButton onClick={onClose} size="small">
                    <CloseIcon />
                </IconButton>
            </DialogTitle>

            <DialogContent>
                <Box sx={{ mb: 3 }}>
                    <Alert severity="success" sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                            {result.message || 'Your task has been submitted successfully!'}
                        </Typography>
                    </Alert>
                </Box>

                <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Task ID
                    </Typography>
                    <Box sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        p: 1,
                        bgcolor: 'grey.50',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'grey.300'
                    }}>
                        <Typography
                            variant="body2"
                            sx={{
                                fontFamily: 'monospace',
                                flex: 1,
                                wordBreak: 'break-all'
                            }}
                        >
                            {result.task_id}
                        </Typography>
                        <IconButton
                            onClick={handleCopyTaskId}
                            size="small"
                            title="Copy Task ID"
                        >
                            <CopyIcon fontSize="small" />
                        </IconButton>
                    </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Overall Status
                    </Typography>
                    <Chip
                        label={getStatusDisplayText(result.status)}
                        color={getStatusColor(result.status)}
                        size="small"
                    />
                </Box>

                <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Task Stages
                    </Typography>
                    <Stepper orientation="vertical" sx={{ mt: 1 }}>
                        <Step active={stageStatus.dataProcessing === 'running'} completed={stageStatus.dataProcessing === 'completed'}>
                            <StepLabel>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    {getStatusIcon(stageStatus.dataProcessing)}
                                    <Typography variant="body2">Data Processing</Typography>
                                </Box>
                            </StepLabel>
                            <StepContent>
                                <Typography variant="body2" color="text.secondary">
                                    Downloading and preprocessing satellite data for the selected areas.
                                </Typography>
                                {result.stages?.data_processing?.started_at && (
                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                                        Started: {new Date(result.stages.data_processing.started_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.data_processing?.completed_at && (
                                    <Typography variant="caption" color="text.secondary" display="block">
                                        Completed: {new Date(result.stages.data_processing.completed_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.data_processing?.error && (
                                    <Alert severity="error" sx={{ mt: 1 }}>
                                        <Typography variant="caption">
                                            {result.stages.data_processing.error}
                                        </Typography>
                                    </Alert>
                                )}
                                {stageStatus.dataProcessing === 'running' && (
                                    <LinearProgress sx={{ mt: 1 }} />
                                )}
                            </StepContent>
                        </Step>

                        <Step active={stageStatus.modelPrediction === 'running'} completed={stageStatus.modelPrediction === 'completed'}>
                            <StepLabel>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    {getStatusIcon(stageStatus.modelPrediction)}
                                    <Typography variant="body2">Model Prediction</Typography>
                                </Box>
                            </StepLabel>
                            <StepContent>
                                <Typography variant="body2" color="text.secondary">
                                    Running model on processed data.
                                </Typography>
                                {result.stages?.model_prediction?.started_at && (
                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                                        Started: {new Date(result.stages.model_prediction.started_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.model_prediction?.completed_at && (
                                    <Typography variant="caption" color="text.secondary" display="block">
                                        Completed: {new Date(result.stages.model_prediction.completed_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.model_prediction?.error && (
                                    <Alert severity="error" sx={{ mt: 1 }}>
                                        <Typography variant="caption">
                                            {result.stages.model_prediction.error}
                                        </Typography>
                                    </Alert>
                                )}
                                {stageStatus.modelPrediction === 'running' && (
                                    <LinearProgress sx={{ mt: 1 }} />
                                )}
                            </StepContent>
                        </Step>
                    </Stepper>
                </Box>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                        <strong>Next Steps:</strong> You can monitor your task progress using the Task ID above.
                        The system will automatically process your data and run the model prediction.
                    </Typography>
                </Box>
            </DialogContent>

            <DialogActions>
                <Button onClick={onClose} variant="outlined">
                    Close
                </Button>
                <Button
                    onClick={onOpenTasksMonitor}
                    variant="contained"
                    color="primary"
                >
                    Monitor Tasks
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default TaskResultPopup;
