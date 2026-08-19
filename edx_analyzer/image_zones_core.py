"""
Image Zone Analyser  --  CORE (no GUI dependency)
-------------------------------------------------------------------------
Segmentation and measurement of a polished cross-section seen under an optical
microscope.

The picture is split into an arbitrary number of ZONES. Each zone is defined by
its colour (the pixels the user sampled, compared in CIE-Lab) and carries a
ROLE that says what to do with it:

    reference   the sound material; its boundary is where the thicknesses
                start from. Exactly one zone.
    measure     the zone(s) whose thickness and area are wanted.
    background  the surrounding medium; defines "the outside".
    ignore      detected and displayed, but left out of the measurements.

Colour alone is never enough on a real cross-section: a layer growing along the
edge may hold black, iridescent and bright constituents at once, and the bright
ones share the colour of the specimen. Proximity settles it - a zone counts only
where it touches the reference, within a maximum distance - and the reference
itself is the largest connected component, so bright debris lying in the layer
is not mistaken for sound material.

Measurements are taken column by column on the straightened section: total
thickness of the layer on each edge, thickness of each measured zone inside it
(stratigraphy), areas, porosity of the specimen, thickness of the specimen
itself, attacked fraction of the edge and deepest penetration.

Only numpy / pandas / scipy / scikit-image are used, so everything here can be
tested without a screen.
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
#  Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
ROLE_REFERENCE, ROLE_MEASURE = "reference", "measure"
ROLE_BACKGROUND, ROLE_IGNORE = "background", "ignore"
ROLES = [ROLE_REFERENCE, ROLE_MEASURE, ROLE_BACKGROUND, ROLE_IGNORE]
ROLE_LABELS = {
    ROLE_REFERENCE: "Reference (sound material)",
    ROLE_MEASURE: "Measure (thickness + area)",
    ROLE_BACKGROUND: "Background (outside)",
    ROLE_IGNORE: "Ignore",
}

# name, overlay colour, role
DEFAULT_ZONES = [
    ("Specimen", "#56B4E9", ROLE_REFERENCE),
    ("Layer", "#E69F00", ROLE_MEASURE),
    ("Matrix", "#882255", ROLE_BACKGROUND),
]
DETACHED_ZONE = "Reference colour, detached"

LAYER_MODES = {
    "Selected zones": "selected",
    "Everything up to the background": "envelope",
    "Chromatic only": "chromatic",
}
DEFAULT_CHROMA = 12.0        # measured on the iridescent layer of the real images

SIDES = ("top", "bottom")
BLOCK_ROWS = 256             # rows processed at once, keeps memory flat


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


def decimate(image, factor: int):
    """Take one pixel out of `factor`. The micrographs are ~90 Mpx: analysing
    at 1:2 costs 12 s instead of 55 s and still leaves 0.96 µm/px."""
    factor = max(1, int(factor))
    return image if factor == 1 else np.ascontiguousarray(image[::factor, ::factor])


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE-Lab: keeps an iridescent film apart from a neutral grey of the same
    darkness, which a grey level cannot do."""
    require_imaging()
    return skcolor.rgb2lab(to_rgb_float(rgb)).astype(np.float32)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = to_rgb_float(rgb)
    return (0.2125 * rgb[:, :, 0] + 0.7154 * rgb[:, :, 1] + 0.0721 * rgb[:, :, 2]).astype(np.float32)


def median_smooth(rgb, radius: int = 1):
    """Small median filter before classification: the per-pixel classification
    is visibly noisy on real micrographs, and at 0.5 µm/px there is resolution
    to spare."""
    require_imaging()
    if radius <= 0:
        return rgb
    size = 2 * int(radius) + 1
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        out[:, :, c] = ndi.median_filter(rgb[:, :, c], size=size)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Colour classification
# ─────────────────────────────────────────────────────────────────────────────
def sample_stats(lab_pixels) -> dict:
    """Mean/std of the pixels sampled for one zone."""
    px = np.asarray(lab_pixels, dtype=np.float32).reshape(-1, 3)
    if px.size == 0:
        return {"mean": np.zeros(3, np.float32), "std": np.ones(3, np.float32), "n": 0}
    std = np.maximum(px.std(axis=0), 1.0)      # never divide by ~0 on a flat sample
    return {"mean": px.mean(axis=0), "std": std.astype(np.float32), "n": int(len(px))}


def classify(rgb, stats_list, reject_distance=None, block_rows: int = BLOCK_ROWS) -> np.ndarray:
    """
    Assign every pixel to the closest zone (normalised Lab distance).

    Lab is computed block of rows by block of rows and never kept whole: on a
    90 Mpx picture the Lab image alone would be 1 GB.
    Returns an int8 label image; -1 marks pixels rejected by `reject_distance`.
    """
    require_imaging()
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int8)
    active = [(i, st) for i, st in enumerate(stats_list) if st and st.get("n", 0)]
    if not active:
        return labels

    for r0 in range(0, h, int(block_rows)):
        r1 = min(h, r0 + int(block_rows))
        lab = to_lab(arr[r0:r1])
        best = np.full((r1 - r0, w), np.inf, dtype=np.float32)
        block = np.full((r1 - r0, w), -1, dtype=np.int8)
        for idx, st in active:
            d = np.zeros((r1 - r0, w), dtype=np.float32)
            for c in range(3):
                diff = (lab[:, :, c] - st["mean"][c]) / st["std"][c]
                d += diff * diff
            np.sqrt(d, out=d)
            closer = d < best
            best[closer] = d[closer]
            block[closer] = idx
        if reject_distance is not None:
            block[best > float(reject_distance)] = -1
        labels[r0:r1] = block
    return labels


def chroma_map(rgb, block_rows: int = BLOCK_ROWS) -> np.ndarray:
    """sqrt(a² + b²) of the Lab colour: how coloured a pixel is."""
    require_imaging()
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    out = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, int(block_rows)):
        r1 = min(h, r0 + int(block_rows))
        lab = to_lab(arr[r0:r1])
        out[r0:r1] = np.hypot(lab[:, :, 1], lab[:, :, 2])
    return out


def multiotsu_labels(gray: np.ndarray, n_classes: int = 3):
    """Grey-level split, kept as the fallback of the colour clustering."""
    require_imaging()
    n_classes = max(2, int(n_classes))
    thresholds = skfilters.threshold_multiotsu(gray, classes=n_classes)
    labels = np.digitize(gray, bins=thresholds).astype(np.int8)
    means = [float(gray[labels == i].mean()) if np.any(labels == i) else 0.0
             for i in range(n_classes)]
    return labels, means


def merge_close_clusters(sample, assign, centroids, threshold: float = 12.0):
    """
    Merge the clusters whose colours are nearly the same (single linkage on the
    Lab distance).

    k-means splits the biggest population first: on a real micrograph the
    specimen covers most of the frame and eats two or three clusters on its own
    lightness gradient. Merging them back frees those clusters for the thin
    layers, which is what we are after.
    """
    k = len(centroids)
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(centroids[i] - centroids[j]) < float(threshold):
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)

    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups, key=lambda g: centroids[groups[g]][:, 0].mean())

    new_assign = np.full(assign.shape, -1, dtype=np.int16)
    new_centroids = []
    for rank, g in enumerate(order):
        members = groups[g]
        selection = np.isin(assign, members)
        new_assign[selection] = rank
        pixels = sample[selection]
        new_centroids.append(pixels.mean(axis=0) if len(pixels) else centroids[members[0]])
    return new_assign, np.array(new_centroids), len(order)


def propose_roles(centroids, counts, band_shares=None):
    """
    Guess what each zone is.

    Colour alone is not enough: the surrounding medium is often several
    materials at once (a dark binder plus bright aggregates), and all of them
    belong to the background. What separates them from a real layer is WHERE
    they live - a layer hugs the boundary of the specimen, the medium is spread
    everywhere. `band_shares[i]` is the share of zone i lying in the interface
    band; when it is available it decides, and colour only breaks ties.
    """
    k = len(centroids)
    roles = [ROLE_IGNORE] * k
    if k == 0:
        return roles
    lightness = centroids[:, 0]
    chroma = np.hypot(centroids[:, 1], centroids[:, 2])
    reference = int(np.argmax(lightness))
    roles[reference] = ROLE_REFERENCE

    others = [i for i in range(k) if i != reference]
    if not others:
        return roles

    if band_shares is not None:
        shares = np.asarray(band_shares, dtype=float)
        near = [i for i in others if shares[i] >= 0.55]     # hugs the boundary
        far = [i for i in others if shares[i] < 0.55]
        for i in far:
            roles[i] = ROLE_BACKGROUND
        if not near:                                       # nothing hugs it: fall back
            near = [int(others[int(np.argmin(lightness[others]))])]
            roles[near[0]] = ROLE_MEASURE
        else:
            for i in near:
                roles[i] = ROLE_MEASURE
        if not far:
            widest = int(others[int(np.argmax(np.asarray(counts, float)[others]))])
            roles[widest] = ROLE_BACKGROUND
        return roles

    dark = [i for i in others if lightness[i] < lightness[reference] - 15.0] or others
    background = int(dark[int(np.argmax(np.asarray(counts, float)[dark]))])
    roles[background] = ROLE_BACKGROUND
    rest = [i for i in dark if i != background]
    if rest:
        measure = int(rest[int(np.argmax(chroma[rest]))]) if float(np.max(chroma[rest])) > 4.0 \
            else int(rest[int(np.argmin(lightness[rest]))])
        roles[measure] = ROLE_MEASURE
    return roles


def auto_zone_stats(rgb, k: int = 6, region_mask=None, max_pixels: int = 60000,
                    merge_threshold: float = 12.0):
    """
    Unsupervised zoning: k-means on the colours (Lab), near-duplicate clusters
    merged, then a role proposed for each zone.

    `region_mask` restricts the pixels the clustering learns from. Feeding it
    the interface band matters: clustering the whole picture spends most of its
    clusters on the specimen and misses the thin layers entirely.

    Returns (stats_list, centroids, roles) - the list can be shorter than k
    once near-identical clusters have been merged.
    """
    require_imaging()
    from scipy.cluster.vq import kmeans2

    lab = to_lab(rgb)
    flat = lab[np.asarray(region_mask, dtype=bool)] if region_mask is not None and np.any(region_mask) \
        else lab.reshape(-1, 3)
    flat = flat.astype(np.float64)
    step = max(1, flat.shape[0] // int(max_pixels))
    sample = flat[::step]
    k = int(max(2, min(k, 8)))
    if sample.shape[0] < k * 10:
        sample = lab.reshape(-1, 3).astype(np.float64)[::max(1, lab.size // (3 * max_pixels))]

    # deterministic seeds spread along the lightness range: the name of the RNG
    # argument of kmeans2 changed across scipy versions, so none is used
    order = np.argsort(sample[:, 0])
    init = np.vstack([sample[order[int(f * (len(order) - 1))]]
                      for f in np.linspace(0.02, 0.98, k)])
    try:
        centroids, assign = kmeans2(sample, init, minit="matrix")
        if len(np.unique(assign)) < 2:
            raise ValueError("degenerate clustering")
    except Exception:                                     # noqa: BLE001
        gray = lab[:, :, 0] / 100.0
        lab_img, _means = multiotsu_labels(gray, k)
        assign = lab_img.reshape(-1)[::step][:len(sample)]
        centroids = np.vstack([sample[assign == i].mean(axis=0) if np.any(assign == i)
                               else sample.mean(axis=0) for i in range(k)])

    assign, centroids, k = merge_close_clusters(sample, assign, centroids, merge_threshold)
    stats = [sample_stats(sample[assign == i]) for i in range(k)]
    counts = [st["n"] for st in stats]

    # where does each zone actually live? classify the picture once and measure
    # how much of every zone falls inside the interface band
    band_shares = None
    if region_mask is not None and np.any(region_mask):
        band = np.asarray(region_mask, dtype=bool)
        labels = classify(rgb, stats)
        band_shares = []
        for i in range(k):
            zone = labels == i
            total = int(np.count_nonzero(zone))
            band_shares.append(float(np.count_nonzero(zone & band)) / total if total else 0.0)
    return stats, centroids, propose_roles(centroids, counts, band_shares)


# ─────────────────────────────────────────────────────────────────────────────
#  Masks: cleaning, proximity, interface band
# ─────────────────────────────────────────────────────────────────────────────
def remove_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drop blobs smaller than `min_area_px` (written with ndi.label: the
    signature of skimage.remove_small_objects changed across versions)."""
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


def clean_mask(mask, closing_radius: int = 0, opening_radius: int = 0,
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


def largest_component(mask, fill_holes: bool = True) -> np.ndarray:
    """The specimen is one piece: keep the biggest blob only."""
    require_imaging()
    lab, n = ndi.label(np.asarray(mask, dtype=bool))
    if n == 0:
        return np.zeros(np.shape(mask), dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    out = lab == int(np.argmax(counts))
    return ndi.binary_fill_holes(out) if fill_holes else out


def distance_outside(mask, coarse: int = 4) -> np.ndarray:
    """
    Distance (in pixels of the original grid) to `mask`, computed on a coarser
    grid and blown back up: the exact transform costs 17 s and 700 MB on a
    90 Mpx picture, and this filter only needs a few pixels of accuracy.
    """
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    coarse = max(1, int(coarse))
    if coarse == 1:
        return ndi.distance_transform_edt(~m).astype(np.float32)
    small = m[::coarse, ::coarse]
    dist = ndi.distance_transform_edt(~small).astype(np.float32) * coarse
    out = np.repeat(np.repeat(dist, coarse, axis=0), coarse, axis=1)
    return out[:m.shape[0], :m.shape[1]]


def zones_touching(zone_mask, reference_mask, max_distance_px: float = 0.0,
                   min_area_px: int = 0) -> np.ndarray:
    """
    Keep only what belongs to the edge of the reference: a blob counts when it
    touches it (or comes within `max_distance_px`). This is the proximity half
    of the detection - it drops the dark specks and bright debris lying in the
    surrounding medium.
    """
    require_imaging()
    zone = np.asarray(zone_mask, dtype=bool)
    ref = np.asarray(reference_mask, dtype=bool)
    if not zone.any() or not ref.any():
        return np.zeros_like(zone)

    if max_distance_px and max_distance_px > 0:
        zone = zone & (distance_outside(ref) <= float(max_distance_px))

    lab, n = ndi.label(zone)
    if n == 0:
        return np.zeros_like(zone)
    neighbourhood = skmorph.dilation(ref, skmorph.disk(2))
    keep = set(np.unique(lab[neighbourhood & zone])) - {0}
    out = np.isin(lab, list(keep)) if keep else np.zeros_like(zone)
    return remove_small(out, min_area_px)


def interface_band(reference_mask, half_width_px: float, coarse: int = 4) -> np.ndarray:
    """Ribbon of ±half_width around the boundary of the reference: the only
    part of the picture where the layers live."""
    require_imaging()
    ref = np.asarray(reference_mask, dtype=bool)
    if not ref.any():
        return np.ones_like(ref)
    outside = distance_outside(ref, coarse)
    inside = distance_outside(~ref, coarse)
    return (outside <= half_width_px) & (inside <= half_width_px)


# ─────────────────────────────────────────────────────────────────────────────
#  Straightening
# ─────────────────────────────────────────────────────────────────────────────
def band_orientation(mask) -> float:
    """Tilt of the section in degrees, folded into (-90, 90].
    Straighten with rotate_mask(m, -tilt)."""
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0
    props = skmeasure.regionprops(m.astype(np.uint8))
    if not props:
        return 0.0
    angle = np.degrees(props[0].orientation) - 90.0
    return float((angle + 90.0) % 180.0 - 90.0)


def rotate_mask(mask, angle_deg: float, order: int = 0):
    """Rotate a mask or a label image without inventing values."""
    require_imaging()
    if abs(angle_deg) < 1e-3:
        return np.asarray(mask)
    src = np.asarray(mask)
    is_bool = src.dtype == bool
    rot = ndi.rotate(src.astype(np.int16) if is_bool else src, angle_deg,
                     order=order, reshape=True, mode="constant",
                     cval=0 if is_bool else -1)
    return rot.astype(bool) if is_bool else rot


# ─────────────────────────────────────────────────────────────────────────────
#  Column scan: total thickness and stratigraphy
# ─────────────────────────────────────────────────────────────────────────────
def _scan_side(layer_col, y_ref: int, side: str, gap_tolerance: int):
    """How far the layer extends outwards from the reference on one column.
    Returns (thickness_px, outer_index)."""
    step = -1 if side == "top" else 1
    limit = -1 if side == "top" else len(layer_col)
    outer, gap, y = y_ref, 0, y_ref + step
    while y != limit:
        if layer_col[y]:
            outer, gap = y, 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
        y += step
    return abs(y_ref - outer), outer


def thickness_profile(reference_mask, layer_mask, scale: float, sides=SIDES,
                      step_px: int = 1, gap_tolerance_px: int = 3,
                      keep_empty: bool = False, zone_labels=None,
                      zone_names=None) -> pd.DataFrame:
    """
    Walk the straightened section column by column and measure, on each edge,
    how far the layer extends outwards from the first sound-material pixel.

    `zone_labels` (an int array holding the index of the measured zone of each
    pixel, -1 elsewhere) adds one column per zone: the stratigraphy.
    Returns: Position | Side | Thickness | Thickness (px) | Column (px) [| zones]
    """
    require_imaging()
    ref = np.asarray(reference_mask, dtype=bool)
    layer = np.asarray(layer_mask, dtype=bool)
    if layer.shape != ref.shape:
        raise ValueError("reference and layer masks must have the same shape")
    names = list(zone_names or [])
    columns = np.flatnonzero(ref.any(axis=0))
    base = ["Position", "Side", "Thickness", "Thickness (px)", "Column (px)"]
    if columns.size == 0:
        return pd.DataFrame(columns=base + names)

    x0 = int(columns[0])
    rows = []
    for x in columns[::max(1, int(step_px))]:
        col_ref = np.flatnonzero(ref[:, x])
        for side in sides:
            y_ref = int(col_ref[0]) if side == "top" else int(col_ref[-1])
            px, outer = _scan_side(layer[:, x], y_ref, side, int(gap_tolerance_px))
            if px == 0 and not keep_empty:
                continue
            row = {"Position": (int(x) - x0) * scale, "Side": side,
                   "Thickness": px * scale, "Thickness (px)": px, "Column (px)": int(x)}
            if names and zone_labels is not None:
                lo, hi = (outer, y_ref) if side == "top" else (y_ref + 1, outer + 1)
                seg = zone_labels[lo:hi, x] if hi > lo else np.empty(0, dtype=np.int16)
                seg = seg[seg >= 0]
                counts = np.bincount(seg, minlength=len(names))[:len(names)] if seg.size \
                    else np.zeros(len(names), dtype=int)
                for name, count in zip(names, counts):
                    row[name] = float(count) * scale
            rows.append(row)
    return pd.DataFrame(rows, columns=base + names)


def specimen_profile(reference_mask, scale: float, step_px: int = 1) -> pd.DataFrame:
    """Thickness of the sound material itself, column by column."""
    require_imaging()
    ref = np.asarray(reference_mask, dtype=bool)
    columns = np.flatnonzero(ref.any(axis=0))
    if columns.size == 0:
        return pd.DataFrame(columns=["Position", "Specimen thickness"])
    x0 = int(columns[0])
    rows = []
    for x in columns[::max(1, int(step_px))]:
        col = np.flatnonzero(ref[:, x])
        rows.append({"Position": (int(x) - x0) * scale,
                     "Specimen thickness": (int(col[-1]) - int(col[0]) + 1) * scale})
    return pd.DataFrame(rows)


def thickness_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, **{k: float("nan") for k in
                           ("min", "max", "mean", "median", "std", "p10", "p90")}}
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def attacked_fraction(profile: pd.DataFrame, threshold: float, step_px: int,
                      scale: float, total_columns: int = 0) -> pd.DataFrame:
    """
    Share of each edge actually covered by the layer, above `threshold`.

    The share is taken over EVERY column of the edge, not only over the columns
    where something was found: a layer covering a third of the edge must read
    33 %, not 100 %.
    """
    cols = ["Side", "Columns", "Attacked columns", "Attacked (%)", "Attacked length"]
    if profile is None or not len(profile):
        return pd.DataFrame(columns=cols)
    span = max(1, int(step_px)) * scale
    rows = []
    for side in sorted(set(profile["Side"])):
        part = profile[profile["Side"] == side]
        attacked = int((part["Thickness"] >= float(threshold)).sum())
        total = int(total_columns) if total_columns else int(len(part))
        rows.append({"Side": side, "Columns": total, "Attacked columns": attacked,
                     "Attacked (%)": 100.0 * attacked / max(1, total),
                     "Attacked length": attacked * span})
    return pd.DataFrame(rows, columns=cols)


def max_penetration(profile: pd.DataFrame):
    """Deepest point of the layer: value, edge, position and column."""
    if profile is None or not len(profile):
        return None
    row = profile.loc[profile["Thickness"].idxmax()]
    return {"thickness": float(row["Thickness"]), "side": str(row["Side"]),
            "position": float(row["Position"]), "column": int(row["Column (px)"])}


# ─────────────────────────────────────────────────────────────────────────────
#  Porosity and areas
# ─────────────────────────────────────────────────────────────────────────────
def pore_mask(raw_reference, filled_reference, min_area_px: int = 0) -> np.ndarray:
    """The pores of the specimen are exactly the holes of its own mask."""
    require_imaging()
    pores = np.asarray(filled_reference, dtype=bool) & ~np.asarray(raw_reference, dtype=bool)
    return remove_small(pores, min_area_px)


def _axis_length(prop, which: str) -> float:
    """axis_major_length (new) / major_axis_length (old scikit-image)."""
    new = getattr(prop, f"axis_{which}_length", None)
    if new is not None:
        return float(new)
    return float(getattr(prop, f"{which}_axis_length", 0.0))


def region_table(mask, scale: float, class_name: str = "") -> pd.DataFrame:
    """One row per connected object, in real units."""
    require_imaging()
    m = np.asarray(mask, dtype=bool)
    cols = ["Class", "Object", "Area", "Perimeter", "Equivalent diameter",
            "Length", "Width", "Centroid X", "Centroid Y"]
    if not m.any():
        return pd.DataFrame(columns=cols)
    props = skmeasure.regionprops(skmeasure.label(m))
    rows = [{
        "Class": class_name, "Object": i,
        "Area": p.area * scale * scale,
        "Perimeter": p.perimeter * scale,
        # computed here: the regionprops property was renamed across versions
        "Equivalent diameter": 2.0 * np.sqrt(p.area / np.pi) * scale,
        "Length": _axis_length(p, "major") * scale,
        "Width": _axis_length(p, "minor") * scale,
        "Centroid X": p.centroid[1] * scale, "Centroid Y": p.centroid[0] * scale,
    } for i, p in enumerate(props, start=1)]
    return pd.DataFrame(rows, columns=cols)


def zone_summary(masks: dict, scale: float, reference_mask=None) -> pd.DataFrame:
    """Per zone: object count, real area, share of the specimen."""
    require_imaging()
    ref_px = int(np.count_nonzero(reference_mask)) if reference_mask is not None else None
    rows = []
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        px = int(np.count_nonzero(m))
        _lab, n_obj = ndi.label(m)
        rows.append({"Zone": name, "Objects": int(n_obj), "Pixels": px,
                     "Area": px * scale * scale,
                     "Share of specimen (%)": (100.0 * px / ref_px) if ref_px else float("nan")})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay & scale bar
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgb01(hexcode: str):
    h = hexcode.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_overlay(rgb, masks: dict, colors: dict, alpha: float = 0.45,
                  outline: bool = True) -> np.ndarray:
    """Blend the zone masks over the picture (and outline them)."""
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


def detect_scale_bar(rgb, brightness: float = 0.85, max_thickness_px: int = 40):
    """
    Find the micrograph's scale bar: the longest bright run of the bottom-right
    corner that belongs to a THIN, STRAIGHT and FILLED horizontal band.

    The three tests matter, and each was written against a real failure:
      * thickness, measured on nine columns and taken as the median, tells the
        bar (15-16 px on the real pictures) from the specimen (125-312 px) - a
        single column can fall on a pore and make the specimen look thin;
      * the band must exist on most of those columns, which rejects the wavy
        bright filaments running along the edge of the specimen;
      * the rectangle it spans must be almost fully bright.
    Returns (length_px, (x0, y), (x1, y)) or None; the caller still asks what
    that length represents.
    """
    require_imaging()
    gray_full = to_gray(rgb)
    h, w = gray_full.shape
    r0, c0 = int(h * 0.70), int(w * 0.50)
    region = gray_full[r0:, c0:] > brightness
    limit = int(max(8, min(max_thickness_px, h * 0.01)))

    best = None
    for row in range(region.shape[0]):
        line = region[row]
        if not line.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], line.view(np.int8), [0]))))
        starts, ends = idx[::2], idx[1::2]
        if not len(starts):
            continue
        k = int(np.argmax(ends - starts))
        length = int(ends[k] - starts[k])
        if length < 20 or (best and length <= best[0]):
            continue

        y = row + r0
        x0a, x1a = int(starts[k] + c0), int(ends[k] + c0)
        tops, bottoms = [], []
        for probe in np.linspace(x0a + 0.1 * length, x1a - 0.1 * length, 9).astype(int):
            column = gray_full[:, probe] > brightness
            if not column[y]:
                continue
            a = b = y
            while a > 0 and column[a - 1]:
                a -= 1
            while b < h - 1 and column[b + 1]:
                b += 1
            tops.append(a)
            bottoms.append(b)
        if len(tops) < 5:
            continue                                  # not a continuous band
        up, down = int(np.median(tops)), int(np.median(bottoms))
        if (down - up + 1) > limit:
            continue                                  # thick: this is the specimen
        rect = gray_full[up:down + 1, x0a:x1a] > brightness
        if rect.size == 0 or rect.mean() < 0.90:
            continue                                  # holes: not a solid bar
        best = (length, (x0a, y), (x1a, y))
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Whole pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_masks(rgb, zones, reject_distance=None, closing_radius: int = 1,
                min_area_px: int = 0, max_distance_px: float = 0.0,
                layer_mode: str = "selected", chroma_threshold: float = DEFAULT_CHROMA):
    """
    Classify, apply the roles and the proximity rule.

    `zones` is a list of dicts {name, color, role, stats}. Returns a dict with
    the reference / layer / background masks, the per-zone masks, the raw label
    image and the detached reference-coloured pixels.
    """
    require_imaging()
    labels = classify(rgb, [z.get("stats") for z in zones], reject_distance)

    raw = {i: labels == i for i in range(len(zones))}
    ref_index = next((i for i, z in enumerate(zones) if z["role"] == ROLE_REFERENCE), None)
    if ref_index is None:
        raise ValueError("One zone must have the 'reference' role.")

    raw_reference = clean_mask(raw[ref_index], closing_radius=closing_radius,
                               min_area_px=min_area_px)
    reference = largest_component(raw_reference, fill_holes=True)
    detached = raw_reference & ~reference          # bright debris lying in the layer

    background = np.zeros(labels.shape, dtype=bool)
    for i, z in enumerate(zones):
        if z["role"] == ROLE_BACKGROUND:
            background |= raw[i]

    measure_indices = [i for i, z in enumerate(zones) if z["role"] == ROLE_MEASURE]
    if layer_mode == "envelope":
        layer = ~(background | reference)
    elif layer_mode == "chromatic":
        layer = (chroma_map(rgb) >= float(chroma_threshold)) & ~reference
    else:
        layer = np.zeros(labels.shape, dtype=bool)
        for i in measure_indices:
            layer |= raw[i]

    layer = clean_mask(layer, closing_radius=closing_radius, min_area_px=min_area_px)
    layer = zones_touching(layer, reference, max_distance_px=max_distance_px,
                           min_area_px=min_area_px) & ~reference

    zone_masks = {}
    for i, z in enumerate(zones):
        if z["role"] == ROLE_REFERENCE:
            zone_masks[z["name"]] = reference
        elif z["role"] == ROLE_MEASURE:
            zone_masks[z["name"]] = raw[i] & layer
        elif z["role"] == ROLE_IGNORE:
            zone_masks[z["name"]] = raw[i] & ~reference
    if detached.any():
        zone_masks[DETACHED_ZONE] = detached & layer

    return {"labels": labels, "reference": reference, "raw_reference": raw_reference,
            "detached": detached, "background": background, "layer": layer,
            "zone_masks": zone_masks, "measure_indices": measure_indices}


def analyse(rgb, zones, scale: float, factor: int = 1, smooth_radius: int = 0, **kwargs) -> dict:
    """
    Full analysis: classification, roles, straightening, thickness profile,
    stratigraphy, areas, porosity, specimen thickness, attack coverage.

    `factor` decimates the picture first; `scale` is the real size of a pixel of
    the ORIGINAL picture, so the effective scale becomes scale * factor and every
    result stays in real units.
    """
    require_imaging()
    image = decimate(np.asarray(rgb), factor)
    if smooth_radius:
        image = median_smooth(to_rgb_float(image), smooth_radius)
    px = float(scale) * max(1, int(factor))

    masks = build_masks(
        image, zones,
        reject_distance=kwargs.get("reject_distance"),
        closing_radius=kwargs.get("closing_radius", 1),
        min_area_px=int(kwargs.get("min_area", 0) / (px * px)) if kwargs.get("min_area") else 0,
        max_distance_px=(kwargs.get("max_distance", 0.0) / px) if kwargs.get("max_distance") else 0.0,
        layer_mode=kwargs.get("layer_mode", "selected"),
        chroma_threshold=kwargs.get("chroma_threshold", DEFAULT_CHROMA))

    reference, layer = masks["reference"], masks["layer"]
    angle = band_orientation(reference) if kwargs.get("straighten", True) else 0.0
    reference_r = rotate_mask(reference, -angle)
    layer_r = rotate_mask(layer, -angle)

    # per-pixel index of the measured zones, for the stratigraphy
    names = [zones[i]["name"] for i in masks["measure_indices"]]
    zone_index = None
    if names and kwargs.get("stratigraphy", True):
        zone_index = np.full(masks["labels"].shape, -1, dtype=np.int16)
        for rank, i in enumerate(masks["measure_indices"]):
            zone_index[masks["labels"] == i] = rank
        if DETACHED_ZONE in masks["zone_masks"] and masks["detached"].any():
            names = names + [DETACHED_ZONE]
            zone_index[masks["detached"]] = len(names) - 1
        zone_index = rotate_mask(zone_index, -angle, order=0)

    step = int(kwargs.get("step_px", 1))
    profile = thickness_profile(reference_r, layer_r, px,
                                sides=kwargs.get("sides", SIDES), step_px=step,
                                gap_tolerance_px=kwargs.get("gap_tolerance_px", 3),
                                keep_empty=kwargs.get("keep_empty", False),
                                zone_labels=zone_index, zone_names=names)

    scanned = np.flatnonzero(reference_r.any(axis=0))
    total_columns = int(len(scanned[::max(1, step)]))

    result = {
        **masks,
        "image": image, "scale": px, "factor": max(1, int(factor)), "angle": angle,
        "total_columns": total_columns,
        "profile": profile, "zone_names": names,
        "stats": thickness_stats(profile["Thickness"]) if len(profile) else thickness_stats([]),
        "attack": attacked_fraction(profile, kwargs.get("attack_threshold", 0.0), step, px,
                                    total_columns=total_columns),
        "max_penetration": max_penetration(profile),
        "objects": region_table(layer, px, "Layer"),
        "summary": zone_summary(masks["zone_masks"], px, reference_mask=reference),
    }

    if kwargs.get("specimen_thickness", True):
        result["specimen"] = specimen_profile(reference_r, px, step)
    if kwargs.get("porosity", True):
        pores = pore_mask(masks["raw_reference"], reference,
                          int(kwargs.get("min_pore_area", 0) / (px * px)))
        result["pores"] = pores
        result["pore_table"] = region_table(pores, px, "Pore")
        ref_px = int(np.count_nonzero(reference))
        result["porosity_percent"] = 100.0 * np.count_nonzero(pores) / ref_px if ref_px else float("nan")
    return result


def compare_modes(rgb, zones, scale: float, factor: int = 1, **kwargs) -> pd.DataFrame:
    """Run the three layer definitions with the same settings, side by side."""
    rows = []
    for label, mode in LAYER_MODES.items():
        try:
            res = analyse(rgb, zones, scale, factor=factor, layer_mode=mode,
                          porosity=False, specimen_thickness=False,
                          stratigraphy=False, **kwargs)
            st = res["stats"]
            area = float(res["summary"]["Area"].sum()) if len(res["summary"]) else 0.0
            rows.append({"Definition": label, "n": st["n"], "Mean": st["mean"],
                         "Median": st["median"], "Max": st["max"], "Layer area": area})
        except Exception as exc:                          # noqa: BLE001
            rows.append({"Definition": label, "n": 0, "Mean": float("nan"),
                         "Median": float("nan"), "Max": float("nan"),
                         "Layer area": float("nan"), "Error": str(exc)})
    return pd.DataFrame(rows)
