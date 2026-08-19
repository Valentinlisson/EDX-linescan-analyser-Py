"""Static configuration shared across the application: color palettes, marker
styles, legend placements and the base dark theme colors."""

COLORS_DEFAULT = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#F0E442", "#FF0000", "#882255", "#117733",
]

MARKERS = {
    "Circle": "o", "Square": "s", "Triangle ▲": "^", "Triangle ▼": "v",
    "Diamond": "D", "Star": "*", "None": "None",
}

LEGEND_POSITIONS = {
    "Outside Right": "outside right",
    "Inside Top Right": "upper right",
    "Inside Top Left": "upper left",
    "Inside Bottom Right": "lower right",
}

# Quick-pick presets offered next to the background color pickers.
BACKGROUND_PRESETS = {
    "Dark (default)": "#282C34",
    "Figure Dark": "#1E2127",
    "White": "#FFFFFF",
    "Light Gray": "#F2F2F2",
    "Black": "#000000",
}

BG, ACCENT = "#1E2127", "#61AFEF"
