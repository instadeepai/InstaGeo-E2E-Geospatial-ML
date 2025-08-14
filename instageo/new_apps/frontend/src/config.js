// API Configuration
const INSTAGEO_BACKEND_API_BASE_URL = process.env.REACT_APP_INSTAGEO_BACKEND_API_BASE_URL

// API Endpoints
export const INSTAGEO_BACKEND_API_ENDPOINTS = {
    RUN_MODEL: `${INSTAGEO_BACKEND_API_BASE_URL}/api/run-model`,
    TASK_STATUS: (taskId) => `${INSTAGEO_BACKEND_API_BASE_URL}/api/task/${taskId}`,
    GET_ALL_TASKS: `${INSTAGEO_BACKEND_API_BASE_URL}/api/tasks`,
    GET_MODELS: `${INSTAGEO_BACKEND_API_BASE_URL}/api/models`,
    HEALTH: `${INSTAGEO_BACKEND_API_BASE_URL}/api/health`,
    VISUALIZE: (taskId) => `${INSTAGEO_BACKEND_API_BASE_URL}/api/visualize/${taskId}`,
};

// Environment configuration
export const CONFIG = {
    API_BASE_URL: INSTAGEO_BACKEND_API_BASE_URL,
    IS_DEVELOPMENT: process.env.NODE_ENV === 'development',
    IS_PRODUCTION: process.env.NODE_ENV === 'production',
};

// TODO: Add these to the environment variables
// REACT_APP_MAP_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
// REACT_APP_MAX_AREA_KM2=100000
// REACT_APP_MIN_AREA_KM2=1
