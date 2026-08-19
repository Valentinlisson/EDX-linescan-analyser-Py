"""
Image Zone Analyser  --  CORE (no GUI dependency)
-------------------------------------------------------------------------
Segmentation and measurement of an optical micrograph of a polished
cross-section: tell the sound metal, the oxide / corrosion layer growing along
its edges and the mounting resin apart, then measure

  * the thickness of the oxide layer, perpendicular to the axis of the
    section, from the sound metal out to the end of the oxide, and
  * the area of every detected zone.

Zones are found by COLOUR (a pixel is assigned to the class whose sampled
colour it is closest to, in CIE-Lab) and by PROXIMITY (an oxide blob only
counts when it touches the metal, within a maximum distance): a dark speck
somewhere in the resin is not corrosion.

Only numpy / pandas / scipy / scikit-image are used here, so the whole
pipeline can be unit-tested without a screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:                                        # optional, heavy scientific stack
    import scipy.ndimage as ndi
    from skimage import color as skcolor
    from skimage import filters as skfilters
    from skimage import measure as skmeasure
    from skimage import morphology as skmorph
    from skimage import segmentation as skseg
    IMAGING_AVAILABLE, IMAGING_ERROR = True, None
except Exception as _exc:                   # noqa: BLE001 - reported in the UI
    ndi = skcolor = skfilters = skmeasure = skmorph = skseg = None
    IMAGING_AVAILABLE, IMAGING_ERROR = False, _exc


# ─────────────────────────────────────────────────────────────────────────────
#  Classes of the segmentation
# ─────────────────────────────────────────────────────────────────────────────
CLASS_METAL = "Sound metal"
CLASS_ZONE = "Oxide / corrosion"
CLASS_BACKGROUND = "Background (resin)"

# name, overlay colour, role ("metal" | "zone" | "background")
DEFAULT_CLASSES = [
    (CLASS_METAL, "#56B4E9", "metal"),
    (CLASS_ZONE, "#E69F00", "zone"),
    (CLASS_BACKGROUND, "#882255", "background"),
]

SIDES = ("top", "bottom")


class ImagingUnavailable(RuntimeError):
    """Raised when scipy / scikit-image are not installed."""


def require_imaging():
    if not IMAGING_AVAILABLE:
        raise ImagingUnavailable(
            "This module needs 'scipy' and 'scikit-image':\n\n"
            "    pip install scipy scikit-image\n\n"
            f"Import error: {IMAGING_ERROR}")


# ─────────────────────────────────────────────────────────────────────────────
#  Image preparation
# ─────────────────────────────────────────────────────────────────────────────
def to_rgb_float(image) -> np.ndarray:
    """Any image (uint8/float, grey/RGB/RGBA) -> float32 RGB in [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() > 1.001:
            arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    return np.clip(arr, 0.0, 1.0)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE-Lab: keeps the iridescent blue/brown of an oxide apart from the
    neutral grey of the resin, which a plain grey level cannot do."""
    require_imaging()
    return skcolor.rgb2lab(rgb).astype(np.float32)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.2125 * rgb[:, :, 0] + 0.7154 * rgb[:, :, 1] + 0.0721 * rgb[:, :, 2]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour classification
# ─────────────────────────────────────────────────────────────────────────────
def sample_stats(lab_pixels) -> dict:
    """Mean/std of the pixels the user sampled for one class."""
    px = np.asarray(lab_pixels, dtype=np.float32).reshape(-1, 3)
    if px.size == 0:
        return {"mean": np.zeros(3, np.float32), "std": np.ones(3, np.float32), "n": 0}
    std = px.std(axis=0)
    std = np.maximum(std, 1.0)              # never divide by ~0 on a flat sample
    return {"mean": px.mean(axis=0), "std": std.astype(np.float32), "n": int(len(px))}


def classify_lab(lab: np.ndarray, stats_list, reject_distance=None) -> np.ndarray:
    """
    Assign every pixel to the closest class (normalised Lab distance).
    Returns an int8 label image; -1 marks pixels rejected by `reject_distance`.
    Classes are compared one at a time so memory stays flat on big images.
    """
    require_imaging()
    h, w = lab.shape[:2]
    best_d = np.full((h, w), np.inf, dtype=np.float32)
    labels = np.full((h, w), -1, dtype=np.int8)
    for idx, st in enumerate(stats_list):
        if st is None or st.get("n", 0) == 0:
            continue
        d = np.zeros((h, w), dtype=np.float32)
        for c in range(3):
            diff = (lab[:, :, c] - st["mean"][c]) / st["std"][c]
            d += diff * diff
        np.sqrt(d, out=d)
        closer = d < best_d
        best_d[closer] = d[closer]
        labels[closer] = idx
    if reject_distance is not None:
        labels[best_d > float(reject_distance)] = -1
    return labels


def auto_classify(gray: np.ndarray, n_classes: int = 3):
    """
    Multi-Otsu on the grey levels.
    Returns (labels ordered from darkest to brightest, mean grey of each level).
    Kept as the fallback of `auto_class_stats`: on a polished cross-section the
    resin and the oxide are both dark, so grey levels alone cannot tell them
    apart - only their colour can.
    """
    require_imaging()
    n_classes = max(2, int(n_classes))
    thresholds = skfilters.threshold_multiotsu(gray, classes=n_classes)
    labels = np.digitize(gray, bins=thresholds).astype(np.int8)
    means = [float(gray[labels == i].mean()) if np.any(labels == i) else 0.0
             for i in range(n_classes)]
    return labels, means


def auto_class_stats(lab: np.ndarray, max_pixels: int = 60000):
    """
    Unsupervised seeding of the three classes: k-means in Lab.

    Colour is what does the work here - the oxide is chromatic (bluish/brown)
    where the mounting resin is a neutral grey of nearly the same darkness.
    Returns stats in the [metal, zone, background] order, ready to be handed
    to `classify_lab`, plus the centroids for information.
    """
    require_imaging()
    from scipy.cluster.vq import kmeans2

    flat = lab.reshape(-1, 3).astype(np.float64)
    step = max(1, flat.shape[0] // int(max_pixels))
    sample = flat[::step]

    # deterministic seeds taken along the lightness range (no RNG argument:
    # its name changed across scipy versions)
    order = np.argsort(sample[:, 0])
    init = np.vstack([sample[order[int(f * (len(order) - 1))]] for f in (0.02, 0.5, 0.98)])
    try:
        centroids, labels = kmeans2(sample, init, minit="matrix")
        if len(np.unique(labels)) < 3:
            raise ValueError("degenerate clustering")
    except Exception:                                     # noqa: BLE001 - fall back on grey levels
        gray = lab[:, :, 0] / 100.0
        lab_img, _means = auto_classify(gray, 3)
        flat_labels = lab_img.reshape(-1)[::step]
        centroids = np.vstack([sample[flat_labels == i].mean(axis=0) for i in range(3)])
        labels = flat_labels

    lightness = centroids[:, 0]
    chroma = np.hypot(centroids[:, 1], centroids[:, 2])
    metal = int(np.argmax(lightness))
    rest = [i for i in range(len(centroids)) if i != metal]
    # of the two dark classes, the coloured one is the oxide; if neither is
    # coloured, the darker one is
    if abs(chroma[rest[0]] - chroma[rest[1]]) > 3.0:
        zone = rest[int(np.argmax([chroma[rest[0]], chroma[rest[1]]]))]
    else:
        zone = rest[int(np.argmin([lightness[rest[0]], lightness[rest[1]]]))]
    background = [i for i in rest if i != zone][0]

    stats = [sample_stats(sample[labels == idx]) for idx in (metal, zone, background)]
    return stats, centroids[[metal, zone, background]]


# ─────────────────────────────────────────────────────────────────────────────
#  Masks: cleaning and the proximity rule
# ─────────────────────────────────────────────────────────────────────────────
def remove_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drop blobs smaller than `min_area_px`.

    Written with ndi.label rather than skimage.remove_small_objects: the
    signature of that one changed across versions, this does not.
    """
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    if min_area_px <= 0 or not m.any():
        return m
    lab, n = ndi.label(m)
    if n == 0:
        return m
    counts = np.bincount(lab.ravel())
    keep = np.flatnonzero(counts >= int(min_area_px))
    keep = keep[keep != 0]
    return np.isin(lab, keep) if keep.size else np.zeros_like(m)


def clean_mask(mask: np.ndarray, closing_radius: int = 0, opening_radius: int = 0,
               min_area_px: int = 0, fill_holes: bool = False) -> np.ndarray:
    require_imaging()
    out = np.asarray(mask, dtype=bool)
    if opening_radius > 0:
        out = skmorph.opening(out, skmorph.disk(int(opening_radius)))
    if closing_radius > 0:
        out = skmorph.closing(out, skmorph.disk(int(closing_radius)))
    if fill_holes:
        out = ndi.binary_fill_holes(out)
    return remove_small(out, min_area_px)


def largest_component(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """The specimen is one piece: keep the biggest blob only."""
    require_imaging()
    lab, n = ndi.label(np.asarray(mask, dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    out = lab == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(out) if fill_holes else out


def zones_touching(zone_mask: np.ndarray, metal_mask: np.ndarray,
                   max_distance_px: float = 0.0, min_area_px: int = 0) -> np.ndarray:
    """
    Keep only the parts of `zone_mask` that belong to the metal's edge: a blob
    counts when it touches the metal (or comes within `max_distance_px`), and
    only the portion closer than that distance is kept. This is the "proximity"
    half of the detection - it drops dark specks lying in the resin.
    """
    require_imaging()
    zone = np.asarray(zone_mask, dtype=bool)
    metal = np.asarray(metal_mask, dtype=bool)
    if not zone.any() or not metal.any():
        return np.zeros_like(zone)

    distance = ndi.distance_transform_edt(~metal)
    if max_distance_px and max_distance_px > 0:
        zone = zone & (distance <= float(max_distance_px))

    lab, n = ndi.label(zone)
    if n == 0:
        return np.zeros_like(zone)
    neighbourhood = skmorph.dilation(metal, skmorph.disk(2))
    keep = set(np.unique(lab[neighbourhood & zone])) - {0}
    out = np.isin(lab, list(keep)) if keep else np.zeros_like(zone)
    return remove_small(out, min_area_px)


# ─────────────────────────────────────────────────────────────────────────────
#  Straightening + thickness profile
# ─────────────────────────────────────────────────────────────────────────────
def band_orientation(mask: np.ndarray) -> float:
    """
    Tilt of the section in degrees (the micrographs are never straight).
    Positive = the band rises to the right. Straighten with rotate_mask(m, -tilt).
    """
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0
    props = skmeasure.regionprops(m.astype(np.uint8))
    if not props:
        return 0.0
    # regionprops measures from the row axis; fold the result back into
    # (-90, 90] so a band tilted by 3 deg never comes out as -177 deg
    angle = np.degrees(props[0].orientation) - 90.0
    return float((angle + 90.0) % 180.0 - 90.0)


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a boolean mask (nearest neighbour, so it stays boolean)."""
    require_imaging()
    if abs(angle_deg) < 1e-3:
        return np.asarray(mask, dtype=bool)
    rot = ndi.rotate(np.asarray(mask, dtype=np.uint8), angle_deg, order=0,
                     reshape=True, mode="constant", cval=0)
    return rot.astype(bool)


def _scan_side(zone_col: np.ndarray, y_metal: int, side: str, gap_tolerance: int) -> int:
    """Thickness in pixels of the zone lying outside the metal on one column."""
    step = -1 if side == "top" else 1
    limit = -1 if side == "top" else len(zone_col)
    outer, gap, y = y_metal, 0, y_metal + step
    while y != limit:
        if zone_col[y]:
            outer, gap = y, 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
        y += step
    return abs(y_metal - outer)


def thickness_profile(metal_mask: np.ndarray, zone_mask: np.ndarray, scale: float,
                      sides=SIDES, step_px: int = 1, gap_tolerance_px: int = 3,
                      keep_empty: bool = False) -> pd.DataFrame:
    """
    Walk the section column by column and measure, on each side, how far the
    oxide extends outwards from the first sound-metal pixel.

    `scale` is in real units per pixel; both masks must already be straightened.
    Returns: Position | Side | Thickness | Thickness (px) | Column (px)
    """
    require_imaging()
    metal = np.asarray(metal_mask, dtype=bool)
    zone = np.asarray(zone_mask, dtype=bool)
    if zone.shape != metal.shape:
        raise ValueError("metal and zone masks must have the same shape")

    columns = np.flatnonzero(metal.any(axis=0))
    if columns.size == 0:
        return pd.DataFrame(columns=["Position", "Side", "Thickness", "Thickness (px)", "Column (px)"])

    x0 = int(columns[0])
    step = max(1, int(step_px))
    rows = []
    for x in columns[::step]:
        col_metal = np.flatnonzero(metal[:, x])
        for side in sides:
            y_metal = int(col_metal[0]) if side == "top" else int(col_metal[-1])
            px = _scan_side(zone[:, x], y_metal, side, int(gap_tolerance_px))
            if px == 0 and not keep_empty:
                continue
            rows.append({"Position": (int(x) - x0) * scale,
                         "Side": side,
                         "Thickness": px * scale,
                         "Thickness (px)": px,
                         "Column (px)": int(x)})
    return pd.DataFrame(rows, columns=["Position", "Side", "Thickness", "Thickness (px)", "Column (px)"])


def thickness_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, **{k: float("nan") for k in
                           ("min", "max", "mean", "median", "std", "p10", "p90")}}
    return {
        "n": int(v.size),
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Areas
# ─────────────────────────────────────────────────────────────────────────────
def _axis_length(prop, which: str) -> float:
    """axis_major_length (new) / major_axis_length (old scikit-image)."""
    new = getattr(prop, f"axis_{which}_length", None)
    if new is not None:
        return float(new)
    return float(getattr(prop, f"{which}_axis_length", 0.0))


def region_table(mask: np.ndarray, scale: float, class_name: str = "") -> pd.DataFrame:
    """One row per connected object, in real units."""
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return pd.DataFrame(columns=["Class", "Object", "Area", "Perimeter",
                                     "Equivalent diameter", "Length", "Width",
                                     "Centroid X", "Centroid Y"])
    lab = skmeasure.label(m)
    props = skmeasure.regionprops(lab)
    rows = []
    for i, p in enumerate(props, start=1):
        area = p.area * scale * scale
        rows.append({
            "Class": class_name,
            "Object": i,
            "Area": area,
            "Perimeter": p.perimeter * scale,
            # computed here rather than read from regionprops: the property was
            # renamed across scikit-image versions
            "Equivalent diameter": 2.0 * np.sqrt(p.area / np.pi) * scale,
            "Length": _axis_length(p, "major") * scale,
            "Width": _axis_length(p, "minor") * scale,
            "Centroid X": p.centroid[1] * scale,
            "Centroid Y": p.centroid[0] * scale,
        })
    return pd.DataFrame(rows)


def class_summary(masks: dict, scale: float, reference_mask=None) -> pd.DataFrame:
    """Per class: pixel count, real area, share of the specimen, object count."""
    require_imaging()
    ref_px = int(np.count_nonzero(reference_mask)) if reference_mask is not None else None
    rows = []
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        px = int(np.count_nonzero(m))
        _lab, n_obj = ndi.label(m)
        rows.append({
            "Class": name,
            "Objects": int(n_obj),
            "Pixels": px,
            "Area": px * scale * scale,
            "Share of specimen (%)": (100.0 * px / ref_px) if ref_px else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay & scale bar
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgb01(hexcode: str):
    h = hexcode.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_overlay(rgb: np.ndarray, masks: dict, colors: dict, alpha: float = 0.45,
                  outline: bool = True) -> np.ndarray:
    """Blend the class masks over the picture (and outline them)."""
    require_imaging()
    out = to_rgb_float(rgb).copy()
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        col = np.array(hex_to_rgb01(colors.get(name, "#FF0000")), dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * col
        if outline:
            out[skseg.find_boundaries(m, mode="outer")] = col
    return np.clip(out, 0.0, 1.0)


def detect_scale_bar(rgb: np.ndarray, brightness: float = 0.85):
    """
    Look for the micrograph's scale bar: the longest isolated run of bright
    pixels in the bottom-right corner. Returns (length_px, (x0, y), (x1, y))
    or None. The user still types in what that length represents.
    """
    img = to_rgb_float(rgb)
    h, w = img.shape[:2]
    r0, c0 = int(h * 0.70), int(w * 0.50)
    bright = to_gray(img[r0:, c0:]) > brightness
    best = None
    for row in range(bright.shape[0]):
        line = bright[row]
        if not line.any():
            continue
        # longest run of True on this row
        idx = np.flatnonzero(np.diff(np.concatenate(([0], line.view(np.int8), [0]))))
        starts, ends = idx[::2], idx[1::2]
        if not len(starts):
            continue
        k = int(np.argmax(ends - starts))
        length = int(ends[k] - starts[k])
        if length < 20 or (best and length <= best[0]):
            continue
        # a bar is thin: the same span must be dark a few rows above and below,
        # otherwise it is the specimen itself
        span = slice(starts[k], ends[k])
        isolated = True
        for off in (-8, 8):
            probe = row + off
            if 0 <= probe < bright.shape[0] and bright[probe, span].mean() > 0.30:
                isolated = False
                break
        if isolated:
            best = (length, (int(starts[k] + c0), int(row + r0)), (int(ends[k] + c0), int(row + r0)))
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Whole pipeline (used by the panel and by the tests)
# ─────────────────────────────────────────────────────────────────────────────
def segment(rgb: np.ndarray, stats_list, roles, reject_distance=None,
            closing_radius: int = 1, min_area_px: int = 0,
            max_distance_px: float = 0.0) -> dict:
    """
    Colour classification + proximity rule.
    `stats_list` and `roles` are parallel lists (roles: metal / zone / background).
    Returns {"labels", "metal", "zone", "background", "masks"}.
    """
    require_imaging()
    lab = to_lab(rgb)
    labels = classify_lab(lab, stats_list, reject_distance)

    raw = {role: labels == i for i, role in enumerate(roles)}
    metal = raw.get("metal", np.zeros(labels.shape, bool))
    metal = clean_mask(metal, closing_radius=closing_radius, min_area_px=min_area_px)
    metal = largest_component(metal, fill_holes=True)

    zone = raw.get("zone", np.zeros(labels.shape, bool))
    zone = clean_mask(zone, closing_radius=closing_radius, min_area_px=min_area_px)
    zone = zones_touching(zone, metal, max_distance_px=max_distance_px,
                          min_area_px=min_area_px) & ~metal

    background = ~(metal | zone)
    return {"labels": labels, "metal": metal, "zone": zone, "background": background}


def analyse(rgb: np.ndarray, stats_list, roles, scale: float, **kwargs) -> dict:
    """segment() + straightening + thickness profile + areas, in one call."""
    seg = segment(rgb, stats_list, roles,
                  reject_distance=kwargs.get("reject_distance"),
                  closing_radius=kwargs.get("closing_radius", 1),
                  min_area_px=kwargs.get("min_area_px", 0),
                  max_distance_px=kwargs.get("max_distance_px", 0.0))

    angle = band_orientation(seg["metal"]) if kwargs.get("straighten", True) else 0.0
    # rotating by -tilt brings the section back to the horizontal
    metal_r, zone_r = rotate_mask(seg["metal"], -angle), rotate_mask(seg["zone"], -angle)
    profile = thickness_profile(metal_r, zone_r, scale,
                                sides=kwargs.get("sides", SIDES),
                                step_px=kwargs.get("step_px", 1),
                                gap_tolerance_px=kwargs.get("gap_tolerance_px", 3),
                                keep_empty=kwargs.get("keep_empty", False))
    return {
        **seg,
        "angle": angle,
        "profile": profile,
        "stats": thickness_stats(profile["Thickness"]) if len(profile) else thickness_stats([]),
        "objects": region_table(seg["zone"], scale, CLASS_ZONE),
        "summary": class_summary({CLASS_METAL: seg["metal"], CLASS_ZONE: seg["zone"]},
                                 scale, reference_mask=seg["metal"] | seg["zone"]),
    }
