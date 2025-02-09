
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        getDimensions: function() {
            const container = document.getElementById('plot-container');
            if (container) {
                const rect = container.getBoundingClientRect();
                return JSON.stringify({
                    width: rect.width,
                    height: window.innerHeight,
                    containerWidth: rect.width
                });
            }
            return JSON.stringify({
                width: window.innerWidth,
                height: window.innerHeight,
                containerWidth: window.innerWidth
            });
        }
    }
});
