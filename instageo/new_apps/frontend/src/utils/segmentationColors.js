// Utility functions for consistent segmentation layer coloring
// -----------------------------------------------------------
// 1. A large palette (30 colors) that are fairly distinct.
// 2. generateSegmentationColors:  maps class indices -> hex colors.
// 3. generateTiTilerColormap:     converts class indices directly into the
// colormap format TiTiler expects.

const SEGMENTATION_COLORS = [
    // Light variants
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    // Base
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    // Dark variants
    '#393b79', '#b35806', '#006d2c', '#a50f15', '#54278f',
    '#5d4037', '#c2185b', '#424242', '#827717', '#006064'
];

/**
 * Create a mapping from class index → hex color.
 * @param {number[]} classIndices – array of unique class ids.
 * @returns {Object} { [classIndex]: '#rrggbb' }
 */
export function generateSegmentationColors(classIndices = []) {
    const mapping = {};
    classIndices.forEach((idx, i) => {
        mapping[idx] = SEGMENTATION_COLORS[i % SEGMENTATION_COLORS.length];
    });
    return mapping;
}

/**
 * Convert class indices directly to a TiTiler colormap string.
 * Accepts either an array of indices OR an object mapping indices to colors.
 *
 * TiTiler format example:
 *   "1":[31,119,180,255],"2":[255,127,14,255],
 *
 * @param {number[]|Object} input - class indices array OR {index: '#rrggbb', ...} mapping
 * @returns {string} - json string of colormap definition
 */
export function generateTiTilerColormap(input) {
    let colorMap;
    if (Array.isArray(input)) {
        colorMap = generateSegmentationColors(input);
    } else if (typeof input === 'object' && input !== null) {
        colorMap = input;
    } else {
        throw new Error('generateTiTilerColormap expects array of indices or color mapping object');
    }

    const jsonObj = {};
    Object.entries(colorMap).forEach(([index, hex]) => {
        if (typeof hex !== 'string' || !hex.startsWith('#')) {
            throw new Error(`Invalid color value for class ${index}: ${hex}`);
        }
        // Expand shorthand #f00
        if (hex.length === 4) {
            hex = '#' + hex.slice(1).split('').map(ch => ch + ch).join('');
        }
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        jsonObj[index] = [r, g, b];
    });

    return JSON.stringify(jsonObj);
}
