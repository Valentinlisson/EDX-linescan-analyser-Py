"""Small color helpers used for keeping graph text readable on any background."""


def contrast_text_color(hex_color: str) -> str:
    """Return 'white' or 'black', whichever is more readable on hex_color."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "white"
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "black"
