export const DEFAULT_TASK_PARAMS = {
    model_key: '',
    model_size: '',
    temporal_tolerance: 3,
    cloud_coverage: 20,
    date: new Date().toISOString().split('T')[0],
};

// Help text for parameters displayed in the Control Panel
export const PARAMS_HELP = {
    chip_size: 'Pixel width/height of the model input chip. Larger chips cover bigger areas per tile.',
    num_steps: 'Number of temporal steps (images) the model uses as context for a prediction. >1 means multi-temporal inference.',
    data_source: 'Satellite data source used to fetch imagery (e.g., HLS, Sentinel-2, Sentinel-1).',
    temporal_step: 'Spacing in days between temporal steps. 0 means single-date inference.',
    temporal_tolerance: 'Allowed ± days around the selected date to search for usable imagery. Larger windows increase availability but may shift seasonal conditions.',
    cloud_coverage: 'Maximum acceptable percentage of cloud cover in the original tile from which the chips are extracted. Lower values yield clearer imagery but fewer candidates.',
};
