export const DEFAULT_TASK_PARAMS = {
    model_key: '',
    model_size: '',
    temporal_tolerance: 3,
    cloud_coverage: 20,
    date: new Date().toISOString().split('T')[0],
};

// Logo paths
export const LOGO_PATHS = {
    DEFAULT: '/static/images/logo.png',        // For white/light backgrounds
    DARK_BG: '/static/images/logo-dark.png'    // For darker backgrounds
};

// Map themes
export const APP_THEMES = {
    LIGHT: 'light',
    DARK: 'dark'
};

// Base map configuration
// For dark theme, we still use the default OSM tiles but apply a filter to make it darker
export const BASE_MAP_CONFIG = {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
};

// Dark mode filter for map tiles
export const DARK_MODE_MAP_FILTER = 'invert(0.94) hue-rotate(220deg) brightness(1.5) saturate(0.5)';

// Help text for parameters displayed in the Control Panel
export const PARAMS_HELP = {
    chip_size: 'Pixel width/height of the model input chip. Larger chips cover bigger areas per tile.',
    num_steps: 'Number of temporal steps (images) the model uses as context for a prediction. >1 means multi-temporal inference.',
    data_source: 'Satellite data source used to fetch imagery (e.g., HLS, Sentinel-2, Sentinel-1).',
    temporal_step: 'Spacing in days between temporal steps. 0 means single-date inference.',
    temporal_tolerance: 'Allowed ± days around the selected date to search for usable imagery. Larger windows increase availability but may shift seasonal conditions.',
    cloud_coverage: 'Maximum acceptable percentage of cloud cover in the original tile from which the chips are extracted. Lower values yield clearer imagery but fewer candidates.',
};
