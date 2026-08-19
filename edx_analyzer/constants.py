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
    "Outside Top": "outside top",
    "Outside Bottom": "outside bottom",
    "Inside Top Right": "upper right",
    "Inside Top Left": "upper left",
    "Inside Bottom Right": "lower right",
    "Inside Bottom Left": "lower left",
    "None (hidden)": "none",
}

# Where a zone name is written, as (y in axes fraction, vertical alignment,
# clipped to the plotting area). "Above graph" / "Below axis" keep the label
# completely out of the curves.
ZONE_LABEL_POSITIONS = {
    "Above graph": (1.02, "bottom", False),
    "Inside top": (0.97, "top", True),
    "Inside middle": (0.50, "center", True),
    "Inside bottom": (0.03, "bottom", True),
    "Below axis": (-0.09, "top", False),
    "Hidden": None,
}

ZONE_LABEL_ORIENTATIONS = {"Horizontal": 0, "Vertical": 90}

# Generic matplotlib families: always available, whatever is installed.
ZONE_LABEL_FONTS = ["sans-serif", "serif", "monospace"]

DEFAULT_ZONE_ALPHA = 0.12

# Quick-pick presets offered next to the background color pickers.
BACKGROUND_PRESETS = {
    "Dark (default)": "#282C34",
    "Figure Dark": "#1E2127",
    "White": "#FFFFFF",
    "Light Gray": "#F2F2F2",
    "Black": "#000000",
}

BG, ACCENT = "#1E2127", "#61AFEF"
