const BASE_PARAMS = {
    temporal_tolerance: 3,
    chip_size: 224,
    cloud_coverage: 20,
    date: new Date().toISOString().split('T')[0],
    data_source: 'HLS'
};

export const MODEL_CONFIGS = {
    aod: {
        ...BASE_PARAMS,
        temporal_step: 0,
        num_steps: 1
    },
    locust: {
        ...BASE_PARAMS,
        temporal_step: 30,
        num_steps: 3
    }
};

// Default to aod model params
export const DEFAULT_PARAMS = MODEL_CONFIGS.aod;
