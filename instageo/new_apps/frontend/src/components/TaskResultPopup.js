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
    Login as LoginIcon,
} from '@mui/icons-material';
import { useAuth0 } from '@auth0/auth0-react';
import { isAuth0Configured } from '../auth0-config';
import { isAuthenticationError } from '../utils/authErrors';

const TaskResultPopup = ({ open, onClose, result, error, onOpenTasksMonitor }) => {
    const { loginWithRedirect } = useAuth0();
    const auth0Enabled = isAuth0Configured();
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
            case 'visualization_preparation':
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
            case 'visualization_preparation':
                return <RunningIcon color="primary" />;
            default:
                return <CircularProgress size={20} />;
        }
    };

    const getTaskStageStatus = (taskStatus, stages) => {
        // If we have detailed stages information, use it
        if (stages && stages.data_processing && stages.model_prediction && stages.visualization_preparation) {
            return {
                dataProcessing: stages.data_processing.status || 'pending',
                modelPrediction: stages.model_prediction.status || 'pending',
                visualizationPreparation: stages.visualization_preparation.status || 'pending'
            };
        }

        // Fallback to task status mapping
        switch (taskStatus) {
            case 'data_processing':
                return {
                    dataProcessing: 'running',
                    modelPrediction: 'pending',
                    visualizationPreparation: 'pending'
                };
            case 'model_prediction':
                return {
                    dataProcessing: 'completed',
                    modelPrediction: 'running',
                    visualizationPreparation: 'pending'
                };
            case 'visualization_preparation':
                return {
                    dataProcessing: 'completed',
                    modelPrediction: 'completed',
                    visualizationPreparation: 'running'
                };
            case 'completed':
                return {
                    dataProcessing: 'completed',
                    modelPrediction: 'completed',
                    visualizationPreparation: 'completed'
                };
            case 'failed':
                // Check which stage failed
                if (stages && stages.data_processing && stages.data_processing.status === 'failed') {
                    return {
                        dataProcessing: 'failed',
                        modelPrediction: 'pending',
                        visualizationPreparation: 'pending'
                    };
                } else if (stages && stages.model_prediction && stages.model_prediction.status === 'failed') {
                    return {
                        dataProcessing: 'completed',
                        modelPrediction: 'failed',
                        visualizationPreparation: 'pending'
                    };
                } else if (stages && stages.visualization_preparation && stages.visualization_preparation.status === 'failed') {
                    return {
                        dataProcessing: 'completed',
                        modelPrediction: 'completed',
                        visualizationPreparation: 'failed'
                    };
                }
                break;
            default:
                return {
                    dataProcessing: 'pending',
                    modelPrediction: 'pending',
                    visualizationPreparation: 'pending'
                };
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
            case 'visualization_preparation':
                return 'Visualization Preparation';
            default:
                return status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
    };

    const getStatusMessage = (status) => {
        switch (status) {
            case 'completed':
                return 'Task completed successfully! All data has been processed and is ready for visualization.';
            case 'data_processing':
                return 'Data processing is currently being run. Extracting and preprocessing satellite data for the selected area.';
            case 'model_prediction':
                return 'Model prediction is currently being run. Processing data through the machine learning model.';
            case 'visualization_preparation':
                return 'Visualization preparation is currently being run. Converting data to web-optimized format.';
            case 'pending':
                return 'Task submitted successfully. Data processing will start automatically.';
            default:
                return 'Your task has been submitted successfully!';
        }
    };

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
                        <Typography variant="h6">
                            {isAuthError ? 'Authentication Required' : 'Task Submission Failed'}
                        </Typography>
                    </Box>
                    <IconButton onClick={onClose} size="small">
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
                <DialogContent>
                    <Alert severity="error" sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                            {isAuthError ? 'Authentication Error:' : 'Error Details:'}
                        </Typography>
                        <Typography variant="body2">
                            {error.message || 'An unexpected error occurred while submitting the task.'}
                        </Typography>
                    </Alert>
                    <Typography variant="body2" color="text.secondary">
                        {isAuthError
                            ? 'Please sign in to continue. Your session may have expired.'
                            : 'Please check your input and try again. If the problem persists, contact support.'
                        }
                    </Typography>
                </DialogContent>
                <DialogActions>
                    {isAuthError && auth0Enabled ? (
                        <>
                            <Button onClick={onClose} variant="outlined" color="primary">
                                Close
                            </Button>
                            <Button
                                onClick={handleSignIn}
                                variant="contained"
                                color="primary"
                                startIcon={<LoginIcon />}
                            >
                                Sign In
                            </Button>
                        </>
                    ) : (
                        <Button onClick={onClose} variant="contained" color="primary">
                            Close
                        </Button>
                    )}
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
                    <Typography variant="h6">
                        {result.status === 'failed' ? 'Task Failed' : 'Task Submitted Successfully'}
                    </Typography>
                </Box>
                <IconButton onClick={onClose} size="small">
                    <CloseIcon />
                </IconButton>
            </DialogTitle>

            <DialogContent>
                <Box sx={{ mb: 3 }}>
                    <Alert
                        severity={result.status === 'failed' ? 'error' : 'info'}
                        sx={{
                            mb: 2,
                            backgroundColor: result.status === 'failed'
                                ? undefined
                                : (theme) => theme.palette.mode === 'dark'
                                    ? 'rgba(33, 150, 243, 0.1)'
                                    : 'rgba(33, 150, 243, 0.05)',
                            color: result.status === 'failed'
                                ? undefined
                                : (theme) => theme.palette.mode === 'dark'
                                    ? 'rgba(255, 255, 255, 0.9)'
                                    : 'rgba(0, 0, 0, 0.8)'
                        }}
                    >
                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                            {result.status === 'failed'
                                ? (result.message || 'Task failed. Please check the details in the tasks monitor panel or try again.')
                                : getStatusMessage(result.status)}
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
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'divider'
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
                                    Extracting and preprocessing satellite data for the selected areas.
                                </Typography>
                                {result.stages?.data_processing?.started_at && (
                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                                        Initiated: {new Date(result.stages.data_processing.started_at).toLocaleString()}
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

                        <Step active={stageStatus.visualizationPreparation === 'running'} completed={stageStatus.visualizationPreparation === 'completed'}>
                            <StepLabel>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    {getStatusIcon(stageStatus.visualizationPreparation)}
                                    <Typography variant="body2">Visualization Preparation</Typography>
                                </Box>
                            </StepLabel>
                            <StepContent>
                                <Typography variant="body2" color="text.secondary">
                                    Converting data to web-optimized format and preparing visualization endpoints.
                                </Typography>
                                {result.stages?.visualization_preparation?.started_at && (
                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                                        Started: {new Date(result.stages.visualization_preparation.started_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.visualization_preparation?.completed_at && (
                                    <Typography variant="caption" color="text.secondary" display="block">
                                        Completed: {new Date(result.stages.visualization_preparation.completed_at).toLocaleString()}
                                    </Typography>
                                )}
                                {result.stages?.visualization_preparation?.error && (
                                    <Alert severity="error" sx={{ mt: 1 }}>
                                        <Typography variant="caption">
                                            {result.stages.visualization_preparation.error}
                                        </Typography>
                                    </Alert>
                                )}
                                {stageStatus.visualizationPreparation === 'running' && (
                                    <LinearProgress sx={{ mt: 1 }} />
                                )}
                                {stageStatus.visualizationPreparation === 'completed' && result.visualization_data && (
                                    <Box sx={{ mt: 2, p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                                        <Typography variant="subtitle2" color="success.dark" gutterBottom>
                                            Visualization Data Ready
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary" gutterBottom>
                                            {result.visualization_data.datasets?.length || 0} datasets converted to web-optimized format
                                        </Typography>
                                        {result.visualization_data.endpoints && result.visualization_data.endpoints.length > 0 && (
                                            <Box sx={{ mt: 1 }}>
                                                <Typography variant="caption" color="text.secondary" display="block">
                                                    Available endpoints:
                                                </Typography>
                                                {result.visualization_data.endpoints.slice(0, 3).map((endpoint, index) => (
                                                    <Typography key={index} variant="caption" color="primary" display="block" sx={{ ml: 1 }}>
                                                        • {endpoint.type}: {endpoint.dataset_name}
                                                    </Typography>
                                                ))}
                                                {result.visualization_data.endpoints.length > 3 && (
                                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ ml: 1 }}>
                                                        ... and {result.visualization_data.endpoints.length - 3} more
                                                    </Typography>
                                                )}
                                            </Box>
                                        )}
                                    </Box>
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
