// API Configuration
const INSTAGEO_BACKEND_API_BASE_URL = process.env.REACT_APP_INSTAGEO_BACKEND_API_BASE_URL

// Helper function to prefix TiTiler URLs with API base URL
export const prefixTitilerUrl = (url) => {
    if (!url) return url;
    if (url.startsWith('http://') || url.startsWith('https://')) {
        return url; // Already absolute
    }
    if (url.startsWith('/')) {
        return `${INSTAGEO_BACKEND_API_BASE_URL}${url}`;
    }
    return url;
};

// API Endpoints
export const INSTAGEO_BACKEND_API_ENDPOINTS = {
    RUN_MODEL: `${INSTAGEO_BACKEND_API_BASE_URL}/api/run-model`,
    TASK_STATUS: (taskId) => `${INSTAGEO_BACKEND_API_BASE_URL}/api/task/${taskId}`,
    GET_ALL_TASKS: `${INSTAGEO_BACKEND_API_BASE_URL}/api/tasks`,
    GET_MODELS: `${INSTAGEO_BACKEND_API_BASE_URL}/api/models`,
    HEALTH: `${INSTAGEO_BACKEND_API_BASE_URL}/api/health`,
    VISUALIZE: (taskId) => `${INSTAGEO_BACKEND_API_BASE_URL}/api/visualize/${taskId}`,
    GET_TITILER_DATA: (url) => `${INSTAGEO_BACKEND_API_BASE_URL}${url}`,
};

// Environment configuration
export const CONFIG = {
    IS_DEV_STAGE: process.env.REACT_APP_ENV === 'dev',
    // Area validation limits (in km²)
    MIN_AREA_KM2: parseFloat(process.env.REACT_APP_MIN_AREA_KM2) || 50,
    MAX_AREA_KM2: parseFloat(process.env.REACT_APP_MAX_AREA_KM2) || 500,
};
