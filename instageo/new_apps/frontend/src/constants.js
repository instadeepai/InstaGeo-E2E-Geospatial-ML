const BASE_PARAMS = {
    temporalTolerance: 3,
    chipSize: 224,
    cloudCover: 20,
    date: new Date().toISOString().split('T')[0],
    satelliteSource: 'HLS'
};

export const MODEL_CONFIGS = {
    aod: {
        ...BASE_PARAMS,
        temporalStep: 0,
        numSteps: 1
    },
    locust: {
        ...BASE_PARAMS,
        temporalStep: 30,
        numSteps: 3
    }
};

// Default to aod model params
export const DEFAULT_PARAMS = MODEL_CONFIGS.aod;
