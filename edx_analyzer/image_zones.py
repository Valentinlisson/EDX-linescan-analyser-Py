"""
Image Zone Analyser  --  "Image Zone Analyser" module of the analysis suite
-------------------------------------------------------------------------
Splits a micrograph of a polished cross-section into as many ZONES as needed,
each defined by its colour and given a ROLE (reference material, zone to
measure, background, ignored), then measures:

  * the thickness of the layer growing along the edges of the reference,
    perpendicular to the section, column after column,
  * the thickness of each measured zone inside that layer (stratigraphy),
  * the area of every zone and of every object,
  * the porosity of the reference material, its own thickness, the share of
    the edge actually attacked and the deepest penetration.

Zones come either from sampling the picture by hand or from an automatic
colour clustering restricted to the interface band. Thicknesses can be pushed
into the LOM Depth Analyser module, which draws their distribution.

The UI is a plain CTkFrame so it can fill a tab of the suite; the maths lives
in image_zones_core.py and is tested without a screen.
"""

import os

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .constants import COLORS_DEFAULT, BACKGROUND_PRESETS, ACCENT
from .color_utils import contrast_text_color
from .legend_utils import apply_legend, fit_layout
from .widgets import ColorPickerDialog, add_tooltip
from .image_measurement import CalibrationDialog
from .lom_depth_core import parse_number
from . import image_zones_core as core

VIEW_MODES = ["Overlay", "Original image", "Thickness profile"]
SIDE_CHOICES = {"Both edges": ("top", "bottom"), "Top edge": ("top",), "Bottom edge": ("bottom",)}
RESOLUTIONS = {"1:2 (recommended)": 2, "1:1 (full, slow)": 1, "1:4 (fast preview)": 4}
MIN_RELIABLE_PX = 5.0          # below that a thickness is only a few pixels wide


class ImageZoneAnalyserPanel(ctk.CTkFrame):
    """The whole 'Image Zone Analyser' module, as an embeddable frame."""

    def __init__(self, master, status_callback=None, send_to_lom=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._status = status_callback if callable(status_callback) else (lambda _m: None)
        self._send_to_lom = send_to_lom

        # ---- data ------------------------------------------------------
        self.image_path = None
        self.image = None                    # uint8 RGB, full resolution
        self.preview = None                  # decimated copy used for display
        self.preview_factor = 1
        self.result = None
        self.comparison = None
        self.scale = None                    # real units per pixel, full resolution
        self.scale_unit = "µm"
        self.calibration_arrow = None
        self.zones = [{"name": n, "color": c, "role": r, "samples": [], "stats": None,
                       "auto": False} for n, c, r in core.DEFAULT_ZONES]

        # ---- interaction ----------------------------------------------
        self._tool = None                    # None | "calibrate" | ("sample", idx)
        self._drag_start = None
        self._temp_artist = None

        # ---- settings --------------------------------------------------
        self.resolution_var = tk.StringVar(value=list(RESOLUTIONS)[0])
        self.auto_k_var = tk.StringVar(value="6")
        self.band_var = tk.StringVar(value="150")           # real units, each side
        self.layer_mode_var = tk.StringVar(value=list(core.LAYER_MODES)[0])
        self.chroma_var = tk.StringVar(value=f"{core.DEFAULT_CHROMA:g}")
        self.smooth_var = tk.StringVar(value="1")
        self.closing_var = tk.StringVar(value="1")
        self.min_area_var = tk.StringVar(value="20")        # real units squared
        self.max_distance_var = tk.StringVar(value="300")   # real units
        self.attack_var = tk.StringVar(value="0")           # real units
        self.side_var = tk.StringVar(value="Both edges")
        self.step_var = tk.StringVar(value="1")
        self.gap_var = tk.StringVar(value="3")
        self.keep_empty = ctk.BooleanVar(value=False)
        self.straighten = ctk.BooleanVar(value=True)
        self.stratigraphy = ctk.BooleanVar(value=True)
        self.want_porosity = ctk.BooleanVar(value=True)
        self.want_specimen = ctk.BooleanVar(value=True)

        # ---- display ---------------------------------------------------
        self.view_var = tk.StringVar(value=VIEW_MODES[0])
        self.overlay_alpha = ctk.DoubleVar(value=0.45)
        self.show_outlines = ctk.BooleanVar(value=True)
        self.font_size = ctk.IntVar(value=10)
        self.legend_pos_var = tk.StringVar(value="Outside Right")
        self.plot_bg_color = tk.StringVar(value=BACKGROUND_PRESETS["Dark (default)"])
        self.fig_bg_color = tk.StringVar(value=BACKGROUND_PRESETS["Figure Dark"])
        self.scale_status = tk.StringVar(value="Not calibrated.")
        self.image_name_var = tk.StringVar(value="No image loaded")

        self._build_ui()
        self._refresh_zones()
        self._refresh_results()
        self._render()

    def on_shown(self):
        """A canvas built inside a hidden tab is laid out 0x0: redraw for real."""
        self.update_idletasks()
        self._render()

    # ─────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────
    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Helvetica", 12, "bold"),
                     text_color=ACCENT).pack(anchor="w", pady=(15, 5))

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkScrollableFrame(self, width=340, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        canvas_frame = ctk.CTkFrame(center)
        canvas_frame.pack(fill="both", expand=True)

        self.right_panel = ctk.CTkScrollableFrame(self, width=380, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._build_left(self.left_panel)
        self._build_canvas(canvas_frame)
        self._build_right(self.right_panel)

    def _entry_row(self, parent, label, var, tooltip=None, width=70):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        entry = ctk.CTkEntry(row, textvariable=var, width=width)
        entry.pack(side="right")
        if tooltip:
            add_tooltip(entry, tooltip)
        return entry

    def _build_left(self, p):
        ctk.CTkLabel(p, text="Image Zone Analyser",
                     font=("Helvetica", 20, "bold")).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(p, text="Zones by colour and proximity: thickness, stratigraphy, areas",
                     font=("Helvetica", 11), text_color="gray", wraplength=310,
                     justify="left").pack(anchor="w")

        self._section(p, "IMAGE")
        ctk.CTkButton(p, text="🖼 Load micrograph", command=self._load_image).pack(fill="x", pady=2)
        ctk.CTkLabel(p, textvariable=self.image_name_var, font=("Helvetica", 11),
                     text_color="gray", wraplength=310, justify="left").pack(anchor="w")
        ctk.CTkLabel(p, text="Analysis resolution :").pack(anchor="w", pady=(6, 0))
        res = ctk.CTkComboBox(p, variable=self.resolution_var, values=list(RESOLUTIONS))
        res.pack(fill="x", pady=2)
        add_tooltip(res, "These micrographs reach 90 Mpx. 1:2 keeps ~1 µm/px\n"
                         "and runs in seconds; 1:1 is four times slower and\n"
                         "needs several GB of memory.")

        self._section(p, "SCALE CALIBRATION")
        cal_btn = ctk.CTkButton(p, text="📏 Set scale (draw arrow)", fg_color="#E69F00",
                                text_color="black", command=self._activate_calibration)
        cal_btn.pack(fill="x", pady=2)
        add_tooltip(cal_btn, "Click, then drag an arrow over the scale bar\nand type the length it represents.")
        det_btn = ctk.CTkButton(p, text="🔍 Detect the scale bar", fg_color="gray35",
                                command=self._detect_scale_bar)
        det_btn.pack(fill="x", pady=2)
        add_tooltip(det_btn, "Finds the thin straight bright bar of the bottom-right\ncorner and asks what length it stands for.")
        ctk.CTkLabel(p, textvariable=self.scale_status, text_color=ACCENT,
                     font=("Arial", 11, "bold"), wraplength=310, justify="left").pack(anchor="w", pady=(4, 0))

        self._section(p, "ZONES")
        ctk.CTkLabel(p, text="Sample each material on the image, then say what to do "
                             "with it. Add as many zones as the picture has materials.",
                     text_color="gray", font=("Arial", 10), wraplength=310,
                     justify="left").pack(anchor="w")
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", pady=4)
        auto_btn = ctk.CTkButton(row, text="⚡ Auto", width=90, fg_color="gray35",
                                 command=self._auto_zones)
        auto_btn.pack(side="left", padx=(0, 4))
        add_tooltip(auto_btn, "Cluster the colours of the interface band and propose\n"
                              "one zone per material, with a role for each.\n"
                              "Restricting it to the band matters: on the whole picture\n"
                              "the clusters are eaten by the specimen.")
        ctk.CTkLabel(row, text="k :").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.auto_k_var, width=40).pack(side="left", padx=(2, 8))
        ctk.CTkButton(row, text="＋ Add zone", width=110,
                      command=self._add_zone).pack(side="right")
        self._entry_row(p, "Interface band (± ) :", self.band_var,
                        "Half-width of the ribbon around the boundary of the\n"
                        "reference the automatic clustering learns from.")
        self.zones_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.zones_frame.pack(fill="x", pady=2)

        self._section(p, "LAYER DEFINITION")
        mode = ctk.CTkComboBox(p, variable=self.layer_mode_var, values=list(core.LAYER_MODES))
        mode.pack(fill="x", pady=2)
        add_tooltip(mode, "What counts in the thickness:\n"
                          "• the zones you marked 'measure'\n"
                          "• everything between the reference and the background\n"
                          "• only the coloured (iridescent) pixels")
        self._entry_row(p, "Chroma threshold :", self.chroma_var,
                        "Used by the 'chromatic only' definition.\n"
                        "12 separates an iridescent film from a neutral grey.")
        ctk.CTkButton(p, text="⚖ Compare the three definitions", fg_color="gray35",
                      command=self._compare_modes).pack(fill="x", pady=4)

        self._section(p, "SEGMENTATION")
        self._entry_row(p, "Pre-smoothing (px) :", self.smooth_var,
                        "Median filter applied before classification.\n0 disables it.")
        self._entry_row(p, "Smoothing radius (px) :", self.closing_var,
                        "Closes the small holes and jagged edges of the masks.")
        self._entry_row(p, "Min. object area :", self.min_area_var,
                        "Objects smaller than this (squared real units) are dropped.")
        self._entry_row(p, "Max. distance to reference :", self.max_distance_var,
                        "A zone counts as a layer only within this distance of the\n"
                        "reference. This is what rejects specks lying further away.")

        self._section(p, "MEASUREMENT")
        ctk.CTkLabel(p, text="Edge measured :").pack(anchor="w")
        ctk.CTkComboBox(p, variable=self.side_var, values=list(SIDE_CHOICES)).pack(fill="x", pady=2)
        self._entry_row(p, "Column step (px) :", self.step_var,
                        "1 = one measurement per pixel column.")
        self._entry_row(p, "Gap tolerance (px) :", self.gap_var,
                        "How many pixels of something else may interrupt the layer\nbefore the scan stops.")
        self._entry_row(p, "Attacked above :", self.attack_var,
                        "A column counts as attacked above this thickness.\n"
                        "Use it to ignore the thin mounting gap of an intact edge.")
        ctk.CTkCheckBox(p, text="Count columns without layer as 0",
                        variable=self.keep_empty).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(p, text="Straighten the section before measuring",
                        variable=self.straighten).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(p, text="Thickness of each zone (stratigraphy)",
                        variable=self.stratigraphy).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(p, text="Porosity of the reference",
                        variable=self.want_porosity).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(p, text="Thickness of the reference itself",
                        variable=self.want_specimen).pack(anchor="w", pady=2)

        ctk.CTkButton(p, text="▶ Analyse", fg_color="#009E73", hover_color="#007755",
                      height=36, font=("Helvetica", 13, "bold"),
                      command=self._run_analysis).pack(fill="x", pady=(12, 15))

    def _build_canvas(self, frame):
        self.fig = Figure(figsize=(8.5, 6), facecolor=self.fig_bg_color.get())
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = ctk.CTkFrame(frame, fg_color="transparent")
        tb.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tb).update()
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _build_right(self, p):
        self._section(p, "RESULTS")
        self.results_box = ctk.CTkTextbox(p, height=420, font=("Consolas", 11), wrap="none")
        self.results_box.pack(fill="both", expand=True, pady=5)

        self._section(p, "DISPLAY")
        ctk.CTkComboBox(p, variable=self.view_var, values=VIEW_MODES,
                        command=lambda _v: self._render()).pack(fill="x", pady=2)
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="Overlay opacity :").pack(side="left")
        self.alpha_label = ctk.CTkLabel(row, text="0.45", text_color=ACCENT)
        self.alpha_label.pack(side="right")
        ctk.CTkSlider(p, from_=0.0, to=1.0, variable=self.overlay_alpha,
                      command=self._on_alpha).pack(fill="x")
        ctk.CTkCheckBox(p, text="Outline the zones", variable=self.show_outlines,
                        command=self._render).pack(anchor="w", pady=3)

        self._section(p, "EXPORTS")
        ctk.CTkButton(p, text="🖼 Save current view (PNG/SVG/PDF)",
                      command=self._export_image).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📄 Export measurements (CSV)",
                      command=self._export_csv).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📊 Export measurements (Excel)",
                      command=self._export_excel).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📑 Generate PDF report", fg_color="#882255",
                      hover_color="#551133", command=self._export_pdf).pack(fill="x", pady=2)
        send = ctk.CTkButton(p, text="➡ Send thicknesses to LOM Depth Analyser",
                             fg_color="#0072B2", hover_color="#005588",
                             command=self._send_thicknesses)
        send.pack(fill="x", pady=(10, 15))
        add_tooltip(send, "Create a group in the LOM Depth Analyser tab holding every\n"
                          "thickness measured here, for the distribution and statistics.")

    def _on_alpha(self, value):
        self.alpha_label.configure(text=f"{float(value):.2f}")
        self._render()

    # ─────────────────────────────────────────────────────────────────────
    #  Zones
    # ─────────────────────────────────────────────────────────────────────
    def _refresh_zones(self):
        for w in self.zones_frame.winfo_children():
            w.destroy()
        if not self.zones:
            ctk.CTkLabel(self.zones_frame, text="No zone yet.", text_color="gray",
                         font=("Arial", 11)).pack(anchor="w", pady=4)
            return
        for idx, zone in enumerate(self.zones):
            card = ctk.CTkFrame(self.zones_frame, corner_radius=6)
            card.pack(fill="x", pady=4)

            head = ctk.CTkFrame(card, fg_color="transparent")
            head.pack(fill="x", padx=6, pady=(6, 2))
            btn = ctk.CTkButton(head, text="", width=22, height=22, fg_color=zone["color"])
            btn.configure(command=lambda i=idx, b=btn: self._pick_zone_color(i, b))
            btn.pack(side="left", padx=(0, 6))
            entry = ctk.CTkEntry(head, width=130)
            entry.insert(0, zone["name"])
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<FocusOut>", lambda _e, i=idx, en=entry: self._rename_zone(i, en))
            entry.bind("<Return>", lambda _e, i=idx, en=entry: self._rename_zone(i, en))
            ctk.CTkButton(head, text="✕", width=26, fg_color="gray35", hover_color="#7a2222",
                          command=lambda i=idx: self._remove_zone(i)).pack(side="right")

            role = tk.StringVar(value=core.ROLE_LABELS[zone["role"]])
            ctk.CTkComboBox(card, variable=role, values=list(core.ROLE_LABELS.values()),
                            height=26, command=lambda v, i=idx: self._set_role(i, v)
                            ).pack(fill="x", padx=8, pady=2)

            if zone.get("auto") and zone["stats"]:
                info = f"automatic · {zone['stats']['n']} px"
            else:
                n_px = sum(len(s) for s in zone["samples"])
                info = f"{len(zone['samples'])} sample(s) · {n_px} px"
            ctk.CTkLabel(card, text=info, font=("Arial", 10), text_color="#9aa0a6",
                         anchor="w").pack(fill="x", padx=10)

            actions = ctk.CTkFrame(card, fg_color="transparent")
            actions.pack(fill="x", padx=8, pady=6)
            ctk.CTkButton(actions, text="🖉 Sample", height=24, fg_color="gray30",
                          command=lambda i=idx: self._activate_sampling(i)
                          ).pack(side="left", expand=True, padx=(0, 4))
            ctk.CTkButton(actions, text="✕ samples", width=76, height=24, fg_color="gray35",
                          command=lambda i=idx: self._clear_samples(i)).pack(side="right")

    def _set_role(self, idx, label):
        for role, text in core.ROLE_LABELS.items():
            if text == label:
                self.zones[idx]["role"] = role
                break

    def _add_zone(self):
        colour = COLORS_DEFAULT[len(self.zones) % len(COLORS_DEFAULT)]
        name = simpledialog.askstring("New zone", "Name of the zone :",
                                      initialvalue=f"Zone {len(self.zones) + 1}", parent=self)
        if name is None:
            return
        self.zones.append({"name": name.strip() or f"Zone {len(self.zones) + 1}",
                           "color": colour, "role": core.ROLE_MEASURE,
                           "samples": [], "stats": None, "auto": False})
        self._refresh_zones()

    def _remove_zone(self, idx):
        if 0 <= idx < len(self.zones):
            self.zones.pop(idx)
            self._refresh_zones()

    def _pick_zone_color(self, idx, button):
        col = ColorPickerDialog.ask_color(self.winfo_toplevel(), initial_color=self.zones[idx]["color"],
                                          title=f"Colour of '{self.zones[idx]['name']}'")
        if col:
            self.zones[idx]["color"] = col
            button.configure(fg_color=col)
            self._render()

    def _rename_zone(self, idx, entry):
        try:
            new = entry.get().strip()
        except Exception:                                  # noqa: BLE001 - widget gone
            return
        if new and new != self.zones[idx]["name"]:
            self.zones[idx]["name"] = new
            self.after(60, self._refresh_zones)

    def _clear_samples(self, idx):
        self.zones[idx].update({"samples": [], "stats": None, "auto": False})
        self._refresh_zones()

    def _activate_sampling(self, idx):
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        self._tool = ("sample", idx)
        self._status(f"Sampling '{self.zones[idx]['name']}': drag a small box over that material.")

    def _auto_zones(self):
        """Cluster the colours of the interface band into zones with roles."""
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        if not core.IMAGING_AVAILABLE:
            messagebox.showerror("Missing packages", str(core.IMAGING_ERROR), parent=self)
            return
        try:
            self._status("Clustering the colours of the interface band…")
            self.update_idletasks()
            rgb = core.to_rgb_float(self.preview)
            px = (self.scale or 1.0) * self.preview_factor
            reference = core.largest_component(core.to_gray(rgb) > 0.6, fill_holes=True)
            band = None
            half = parse_number(self.band_var.get())
            if reference.any() and half:
                band = core.interface_band(reference, half_width_px=float(half) / px)
            k = int(parse_number(self.auto_k_var.get()) or 6)
            stats, centroids, roles = core.auto_zone_stats(rgb, k=k, region_mask=band)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Automatic zoning failed", str(exc), parent=self)
            return

        self.zones = []
        for i, (centroid, role, st) in enumerate(zip(centroids, roles, stats)):
            self.zones.append({
                "name": f"Zone {i + 1} (L={centroid[0]:.0f})",
                "color": COLORS_DEFAULT[i % len(COLORS_DEFAULT)],
                "role": role, "samples": [], "stats": st, "auto": True})
        self._refresh_zones()
        self._status(f"{len(self.zones)} zones proposed — check their roles, "
                     "or sample a zone by hand to refine it.")

    # ─────────────────────────────────────────────────────────────────────
    #  Image & calibration
    # ─────────────────────────────────────────────────────────────────────
    def _load_image(self):
        if not core.IMAGING_AVAILABLE:
            messagebox.showerror("Missing packages", str(core.IMAGING_ERROR), parent=self)
            return
        path = filedialog.askopenfilename(
            parent=self, title="Open a micrograph",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            image = mpimg.imread(path)
            if image.dtype != np.uint8:
                image = (np.clip(core.to_rgb_float(image), 0, 1) * 255).astype(np.uint8)
            elif image.ndim == 3 and image.shape[2] > 3:
                image = image[:, :, :3]
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Error", f"Could not read the image:\n{exc}", parent=self)
            return
        self.image = image
        h, w = image.shape[:2]
        # the display copy never exceeds ~4 Mpx, whatever the file holds
        self.preview_factor = max(1, int(np.ceil(np.sqrt((h * w) / 4.0e6))))
        self.preview = core.decimate(image, self.preview_factor)
        self.image_path = path
        self.result = self.comparison = None
        self.calibration_arrow = None
        self.scale = None
        self.scale_status.set("Not calibrated.")
        for zone in self.zones:
            zone.update({"samples": [], "stats": None, "auto": False})
        self.image_name_var.set(f"{os.path.basename(path)}  ({w} × {h} px, "
                                f"{w * h / 1e6:.0f} Mpx — preview 1:{self.preview_factor})")
        self._refresh_zones()
        self._refresh_results()
        self._render(reset_view=True)
        self._status(f"Micrograph loaded: {os.path.basename(path)}")

    def _activate_calibration(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        self._tool = "calibrate"
        self._status("Calibration: drag an arrow over the scale bar.")

    def _detect_scale_bar(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        self._status("Looking for the scale bar…")
        self.update_idletasks()
        found = core.detect_scale_bar(self.image)
        if not found:
            messagebox.showinfo("Scale bar", "No scale bar found in the bottom-right corner.\n"
                                             "Use 'Set scale (draw arrow)' instead.", parent=self)
            return
        length_px, p1, p2 = found
        f = self.preview_factor
        shown = ((p1[0] / f, p1[1] / f), (p2[0] / f, p2[1] / f))
        CalibrationDialog(self.winfo_toplevel(), length_px,
                          on_confirm=lambda real, unit: self._apply_calibration(
                              shown[0], shown[1], length_px, real, unit))

    def _apply_calibration(self, p1, p2, pixel_len, real_len, unit):
        """`pixel_len` is expressed in pixels of the full-resolution picture."""
        self.scale = float(real_len) / float(pixel_len)
        self.scale_unit = unit
        self.calibration_arrow = {"p1": p1, "p2": p2, "real_length": real_len}
        self.scale_status.set(f"Calibrated: {self.scale:.4g} {unit}/px "
                              f"({pixel_len:.0f} px = {real_len:g} {unit})")
        self._refresh_results()
        self._render()
        self._status(f"Scale set: {self.scale:.4g} {unit}/px")

    # ─────────────────────────────────────────────────────────────────────
    #  Mouse interaction (only active once a tool is armed, so the toolbar's
    #  own pan/zoom keeps working the rest of the time)
    # ─────────────────────────────────────────────────────────────────────
    def _on_press(self, event):
        if self._tool is None or event.button != 1 or event.inaxes != self.ax:
            return
        self._drag_start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._drag_start is None or event.inaxes != self.ax:
            return
        if self._temp_artist is not None:
            self._temp_artist.remove()
            self._temp_artist = None
        x0, y0 = self._drag_start
        if self._tool == "calibrate":
            self._temp_artist = self.ax.annotate(
                "", xy=(event.xdata, event.ydata), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color="white", linewidth=1.5, linestyle="--"))
        else:
            self._temp_artist = self.ax.add_patch(Rectangle(
                (min(x0, event.xdata), min(y0, event.ydata)),
                abs(event.xdata - x0), abs(event.ydata - y0),
                fill=False, edgecolor="white", linestyle="--", linewidth=1.5))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_start is None:
            return
        p1, self._drag_start = self._drag_start, None
        if self._temp_artist is not None:
            self._temp_artist.remove()
            self._temp_artist = None
        tool, self._tool = self._tool, None
        if event.xdata is None or event.ydata is None:
            self.canvas.draw_idle()
            return
        p2 = (event.xdata, event.ydata)

        if tool == "calibrate":
            pixel_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if pixel_len < 2:
                self.canvas.draw_idle()
                return
            # the arrow is drawn on the preview: convert to full-resolution pixels
            CalibrationDialog(self.winfo_toplevel(), pixel_len * self.preview_factor,
                              on_confirm=lambda real, unit: self._apply_calibration(
                                  p1, p2, pixel_len * self.preview_factor, real, unit))
        elif isinstance(tool, tuple) and tool[0] == "sample":
            self._collect_sample(tool[1], p1, p2)
        else:
            self.canvas.draw_idle()

    def _collect_sample(self, idx, p1, p2):
        h, w = self.preview.shape[:2]
        c0, c1 = sorted((int(round(p1[0])), int(round(p2[0]))))
        r0, r1 = sorted((int(round(p1[1])), int(round(p2[1]))))
        c0, c1 = max(0, c0), min(w, max(c1, c0 + 1))
        r0, r1 = max(0, r0), min(h, max(r1, r0 + 1))
        pixels = core.to_lab(self.preview[r0:r1, c0:c1]).reshape(-1, 3)
        if pixels.size == 0:
            return
        zone = self.zones[idx]
        if zone.get("auto"):            # a hand-picked sample replaces the automatic seed
            zone.update({"samples": [], "auto": False})
        zone["samples"].append(pixels)
        zone["stats"] = core.sample_stats(np.concatenate(zone["samples"]))
        self._refresh_zones()
        self._status(f"'{zone['name']}': {len(pixels)} pixels sampled "
                     f"({sum(len(s) for s in zone['samples'])} in total).")

    def _on_scroll(self, event):
        if event.inaxes is None or self.view_var.get() == "Thickness profile":
            return
        factor = 1.0 / 1.15 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim([cx - (cx - xlim[0]) * factor, cx + (xlim[1] - cx) * factor])
        self.ax.set_ylim([cy - (cy - ylim[0]) * factor, cy + (ylim[1] - cy) * factor])
        self.canvas.draw()

    # ─────────────────────────────────────────────────────────────────────
    #  Analysis
    # ─────────────────────────────────────────────────────────────────────
    def _number(self, var, default=0.0):
        value = parse_number(var.get())
        return default if value is None else float(value)

    def _settings(self):
        return {
            "reject_distance": None,
            "closing_radius": int(self._number(self.closing_var, 1)),
            "min_area": self._number(self.min_area_var, 0),
            "max_distance": self._number(self.max_distance_var, 0),
            "layer_mode": core.LAYER_MODES.get(self.layer_mode_var.get(), "selected"),
            "chroma_threshold": self._number(self.chroma_var, core.DEFAULT_CHROMA),
            "straighten": bool(self.straighten.get()),
            "sides": SIDE_CHOICES.get(self.side_var.get(), core.SIDES),
            "step_px": int(self._number(self.step_var, 1)),
            "gap_tolerance_px": int(self._number(self.gap_var, 3)),
            "keep_empty": bool(self.keep_empty.get()),
            "attack_threshold": self._number(self.attack_var, 0),
            "stratigraphy": bool(self.stratigraphy.get()),
            "porosity": bool(self.want_porosity.get()),
            "specimen_thickness": bool(self.want_specimen.get()),
            "min_pore_area": self._number(self.min_area_var, 0),
        }

    def _ready(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return False
        if not core.IMAGING_AVAILABLE:
            messagebox.showerror("Missing packages", str(core.IMAGING_ERROR), parent=self)
            return False
        if self.scale is None:
            messagebox.showwarning("Not calibrated",
                                   "Set the scale first, otherwise the measurements have no unit.",
                                   parent=self)
            return False
        roles = [z["role"] for z in self.zones if z["stats"]]
        if core.ROLE_REFERENCE not in roles:
            messagebox.showwarning("No reference",
                                   "One sampled zone must carry the 'Reference' role: it is "
                                   "where the thicknesses are measured from.", parent=self)
            return False
        return True

    def _sampled_zones(self):
        return [z for z in self.zones if z["stats"]]

    def _run_analysis(self):
        if not self._ready():
            return
        factor = RESOLUTIONS.get(self.resolution_var.get(), 2)
        try:
            self._status(f"Analysing at 1:{factor}…")
            self.update_idletasks()
            self.result = core.analyse(self.image, self._sampled_zones(), self.scale,
                                       factor=factor,
                                       smooth_radius=int(self._number(self.smooth_var, 0)),
                                       **self._settings())
            self.comparison = None
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Analysis failed", str(exc), parent=self)
            self._status("Analysis failed.")
            return
        self._refresh_results()
        self._render(reset_view=True)
        st = self.result["stats"]
        self._status(f"Analysis done: {st['n']} thickness measurement(s), "
                     f"mean {st['mean']:.4g} {self.scale_unit}.")

    def _compare_modes(self):
        if not self._ready():
            return
        factor = RESOLUTIONS.get(self.resolution_var.get(), 2)
        settings = self._settings()
        settings.pop("layer_mode")
        for key in ("porosity", "specimen_thickness", "stratigraphy"):
            settings.pop(key, None)
        try:
            self._status("Comparing the three layer definitions…")
            self.update_idletasks()
            self.comparison = core.compare_modes(self.image, self._sampled_zones(),
                                                 self.scale, factor=factor, **settings)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Comparison failed", str(exc), parent=self)
            return
        self._refresh_results()
        self._status("Comparison done — see the results panel.")

    # ─────────────────────────────────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────────────────────────────────
    def _display_image(self):
        """The picture shown, and the factor between it and the analysis grid."""
        if self.result is not None:
            return self.result["image"], self.result["factor"]
        return self.preview, self.preview_factor

    def _render(self, reset_view=False, ax=None, fig=None, light=False):
        target_ax = ax if ax is not None else self.ax
        target_fig = fig if fig is not None else self.fig
        embedded = ax is None

        xlim = ylim = None
        if embedded and not reset_view and target_ax.has_data():
            xlim, ylim = target_ax.get_xlim(), target_ax.get_ylim()

        target_ax.clear()
        face = "white" if light else self.plot_bg_color.get()
        text_color = "black" if light else contrast_text_color(face)
        target_fig.patch.set_facecolor("white" if light else self.fig_bg_color.get())
        target_ax.set_facecolor(face)

        if self.view_var.get() == "Thickness profile":
            self._render_profile(target_ax, target_fig, face, text_color)
            return

        target_ax.axis("off")
        if self.image is None:
            target_ax.text(0.5, 0.5, "Load a micrograph to begin", ha="center", va="center",
                           transform=target_ax.transAxes, color="gray",
                           fontsize=self.font_size.get() + 2)
            target_fig.tight_layout()
            if embedded:
                self.canvas.draw()
            return

        picture, factor = self._display_image()
        masks, colors = {}, {}
        if self.view_var.get() == "Overlay" and self.result:
            for name, mask in self.result["zone_masks"].items():
                if np.any(mask):
                    masks[name] = mask
                    colors[name] = next((z["color"] for z in self.zones if z["name"] == name), "#FFFFFF")
            picture = core.build_overlay(picture, masks, colors,
                                         alpha=float(self.overlay_alpha.get()),
                                         outline=bool(self.show_outlines.get()))
        target_ax.imshow(picture)

        if self.calibration_arrow:
            ratio = self.preview_factor / max(1, factor)
            p1 = (self.calibration_arrow["p1"][0] * ratio, self.calibration_arrow["p1"][1] * ratio)
            p2 = (self.calibration_arrow["p2"][0] * ratio, self.calibration_arrow["p2"][1] * ratio)
            target_ax.annotate("", xy=p2, xytext=p1,
                               arrowprops=dict(arrowstyle="<->", color="#E69F00", linewidth=2))
            target_ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2,
                           f"{self.calibration_arrow['real_length']:g} {self.scale_unit}",
                           color="#E69F00", fontsize=self.font_size.get(), fontweight="bold",
                           ha="center", va="bottom",
                           bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

        deepest = (self.result or {}).get("max_penetration")
        if deepest and self.view_var.get() == "Overlay":
            column = deepest["column"]
            rows = np.flatnonzero(self.result["reference"][:, min(column, self.result["reference"].shape[1] - 1)]) \
                if column < self.result["reference"].shape[1] else np.array([])
            if rows.size:
                y = rows[0] if deepest["side"] == "top" else rows[-1]
                depth = deepest["thickness"] / self.result["scale"]
                y2 = y - depth if deepest["side"] == "top" else y + depth
                target_ax.annotate("", xy=(column, y2), xytext=(column, y),
                                   arrowprops=dict(arrowstyle="<->", color="#FF0000", linewidth=2))
                target_ax.text(column, (y + y2) / 2, f"  max {deepest['thickness']:.4g} {self.scale_unit}",
                               color="#FF0000", fontsize=self.font_size.get(), fontweight="bold",
                               ha="left", va="center",
                               bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

        if masks:
            handles = [Rectangle((0, 0), 1, 1, facecolor=colors[name], edgecolor="none")
                       for name in masks]
            target_ax.legend(handles, list(masks), loc="upper right",
                             fontsize=self.font_size.get(), framealpha=0.8)

        if xlim is not None:
            target_ax.set_xlim(xlim)
            target_ax.set_ylim(ylim)
        target_fig.tight_layout()
        if embedded:
            self.canvas.draw()

    def _render_profile(self, ax, fig, face, text_color):
        fs = self.font_size.get()
        # imshow leaves an equal aspect and a top-down y axis behind, and
        # ax.clear() keeps both: undo them or the curve is squashed flat
        ax.set_aspect("auto")
        ax.axis("on")
        if ax.yaxis_inverted():
            ax.invert_yaxis()
        if ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.set_facecolor(face)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors=text_color, labelsize=fs)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.set_title("Layer thickness along the section", fontsize=fs + 2, fontweight="bold")
        ax.set_xlabel(f"Position along the section ({self.scale_unit})", fontsize=fs)
        ax.set_ylabel(f"Thickness ({self.scale_unit})", fontsize=fs)
        ax.grid(color="gray", linestyle="--", alpha=0.3)

        profile = self.result["profile"] if self.result else None
        if profile is None or not len(profile):
            ax.text(0.5, 0.5, "Run the analysis to get a thickness profile", ha="center",
                    va="center", transform=ax.transAxes, color="gray", fontsize=fs + 2)
        else:
            for side, color in (("top", "#56B4E9"), ("bottom", "#E69F00")):
                part = profile[profile["Side"] == side]
                if len(part):
                    ax.plot(part["Position"], part["Thickness"], color=color, linewidth=1.4,
                            label=f"{side} edge (n={len(part)}, mean {part['Thickness'].mean():.3g})")
                    ax.axhline(part["Thickness"].mean(), color=color, linestyle="--", linewidth=1)
            apply_legend(ax, self.legend_pos_var.get(), facecolor=face,
                         labelcolor=text_color, fontsize=fs)
        fit_layout(fig, self.legend_pos_var.get())
        if ax is self.ax:
            self.canvas.draw()

    # ─────────────────────────────────────────────────────────────────────
    #  Results panel
    # ─────────────────────────────────────────────────────────────────────
    def _results_text(self):
        unit = self.scale_unit
        lines = ["IMAGE",
                 f"  File          : {os.path.basename(self.image_path) if self.image_path else '-'}"]
        if self.image is not None:
            h, w = self.image.shape[:2]
            lines.append(f"  Size          : {w} x {h} px ({w * h / 1e6:.0f} Mpx)")
        lines.append("  Scale         : " + (f"{self.scale:.4g} {unit}/px" if self.scale else "not calibrated"))

        if self.comparison is not None and len(self.comparison):
            lines += ["", "LAYER DEFINITIONS COMPARED", self.comparison.round(3).to_string(index=False)]

        if not self.result:
            lines += ["", "Sample the zones, give them a role, then press 'Analyse'."]
            return "\n".join(lines)

        res, st = self.result, self.result["stats"]
        lines += ["", "THICKNESS OF THE LAYER",
                  f"  Analysed at   : 1:{res['factor']}  ({res['scale']:.4g} {unit}/px)",
                  f"  Definition    : {self.layer_mode_var.get()}",
                  f"  Section tilt  : {res['angle']:+.2f} deg (corrected)",
                  f"  Measurements  : {st['n']}"]
        if st["n"]:
            lines += [f"  Min / Max     : {st['min']:.4g} / {st['max']:.4g} {unit}",
                      f"  Mean          : {st['mean']:.4g} {unit}",
                      f"  Median        : {st['median']:.4g} {unit}",
                      f"  Std dev (n-1) : {st['std']:.4g} {unit}",
                      f"  P10 / P90     : {st['p10']:.4g} / {st['p90']:.4g} {unit}"]
            profile = res["profile"]
            for side in ("top", "bottom"):
                part = profile[profile["Side"] == side]
                if len(part):
                    lines.append(f"  {side:<6} edge   : n={len(part)}  mean={part['Thickness'].mean():.4g}"
                                 f"  max={part['Thickness'].max():.4g} {unit}")
            mean_px = st["mean"] / res["scale"]
            lines.append(f"  Resolution    : mean thickness = {mean_px:.1f} px")
            if mean_px < MIN_RELIABLE_PX:
                lines += ["  /!\\ fewer than 5 pixels: the measurement is dominated by",
                          "      the pixel size — analyse at a finer resolution."]

        if res.get("zone_names") and len(res["profile"]):
            lines += ["", "STRATIGRAPHY (mean thickness of each zone)"]
            for name in res["zone_names"]:
                if name in res["profile"].columns:
                    column = res["profile"][name]
                    lines.append(f"  {name[:26]:<26} : mean {column.mean():.4g}  max {column.max():.4g} {unit}")

        if len(res["attack"]):
            lines += ["", f"ATTACK COVERAGE (above {self._number(self.attack_var, 0):g} {unit})"]
            for _, row in res["attack"].iterrows():
                lines.append(f"  {row['Side']:<6} : {row['Attacked (%)']:5.1f} % of the edge, "
                             f"{row['Attacked length']:.4g} {unit} in total")

        deepest = res.get("max_penetration")
        if deepest:
            lines += ["", "DEEPEST PENETRATION",
                      f"  {deepest['thickness']:.4g} {unit} on the {deepest['side']} edge, "
                      f"at {deepest['position']:.4g} {unit} along the section"]

        if res.get("porosity_percent") is not None and np.isfinite(res.get("porosity_percent", np.nan)):
            lines += ["", "POROSITY OF THE REFERENCE",
                      f"  Pores         : {len(res['pore_table'])}",
                      f"  Porosity      : {res['porosity_percent']:.2f} % of its area"]
            if len(res["pore_table"]):
                area = res["pore_table"]["Area"]
                lines.append(f"  Pore area     : mean {area.mean():.4g}, max {area.max():.4g} {unit}^2")

        if res.get("specimen") is not None and len(res["specimen"]):
            thick = res["specimen"]["Specimen thickness"]
            lines += ["", "THICKNESS OF THE REFERENCE ITSELF",
                      f"  Mean / Median    : {thick.mean():.4g} / {thick.median():.4g} {unit}",
                      f"  Min / Max        : {thick.min():.4g} / {thick.max():.4g} {unit}",
                      "  (the extreme columns hold only a few pixels: read the median)"]

        lines += ["", "AREAS"]
        for _, row in res["summary"].iterrows():
            lines.append(f"  {row['Zone'][:26]:<26} : {row['Area']:.6g} {unit}^2  "
                         f"({int(row['Objects'])} objects)")
        return "\n".join(lines)

    def _refresh_results(self):
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", "end")
        self.results_box.insert("1.0", self._results_text())

    # ─────────────────────────────────────────────────────────────────────
    #  Exports
    # ─────────────────────────────────────────────────────────────────────
    def _has_result(self):
        if not self.result:
            messagebox.showwarning("No result", "Run the analysis first.", parent=self)
            return False
        return True

    def _export_frames(self):
        unit, res = self.scale_unit, self.result
        summary = res["summary"].rename(columns={"Area": f"Area ({unit}^2)"})
        profile = res["profile"].rename(columns={"Position": f"Position ({unit})",
                                                 "Thickness": f"Thickness ({unit})"})
        stats = pd.DataFrame([{"Statistic": k, f"Value ({unit})": v} for k, v in res["stats"].items()])
        frames = {"Areas": summary, "Statistics": stats, "Thickness profile": profile,
                  "Attack coverage": res["attack"],
                  "Objects": res["objects"].rename(columns={"Area": f"Area ({unit}^2)"})}
        if res.get("specimen") is not None:
            frames["Specimen profile"] = res["specimen"]
        if res.get("pore_table") is not None and len(res["pore_table"]):
            frames["Pores"] = res["pore_table"].rename(columns={"Area": f"Area ({unit}^2)"})
        if self.comparison is not None and len(self.comparison):
            frames["Definitions compared"] = self.comparison
        return frames

    def _header_lines(self):
        res, st = self.result, self.result["stats"]
        return [
            "Image Zone Analyser export",
            f"Image: {os.path.basename(self.image_path) if self.image_path else '-'}",
            f"Scale: {self.scale:.6g} {self.scale_unit}/px (analysed at 1:{res['factor']}"
            f" = {res['scale']:.6g} {self.scale_unit}/px)",
            f"Layer definition: {self.layer_mode_var.get()}",
            f"Zones: " + "; ".join(f"{z['name']} [{z['role']}]" for z in self._sampled_zones()),
            f"Section tilt corrected: {res['angle']:+.3f} deg",
            f"Edge measured: {self.side_var.get()}",
            f"Column step: {self.step_var.get()} px, gap tolerance: {self.gap_var.get()} px",
            f"Max distance to reference: {self.max_distance_var.get()} {self.scale_unit}",
            f"Min object area: {self.min_area_var.get()} {self.scale_unit}^2",
            f"Measurements: {st['n']}",
        ]

    def _export_image(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF", "*.pdf")])
        if not path:
            return
        fig = Figure(figsize=(11, 7), facecolor="white")
        ax = fig.add_subplot(111)
        self._render(ax=ax, fig=fig, light=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        self._status(f"View saved: {os.path.basename(path)}")
        messagebox.showinfo("Success", f"Saved:\n{path}", parent=self)

    def _export_csv(self):
        if not self._has_result():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                for line in self._header_lines():
                    f.write(f"# {line}\n")
                for title, df in self._export_frames().items():
                    f.write(f"\n# --- {title.upper()} ---\n")
                    if len(df):
                        df.round(6).to_csv(f, sep=";", index=False, decimal=",", lineterminator="\n")
                    else:
                        f.write("(empty)\n")
            self._status(f"Measurements exported: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Data exported:\n{path}", parent=self)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Export error", str(exc), parent=self)

    def _export_excel(self):
        if not self._has_result():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame({"Parameter": self._header_lines()}).to_excel(
                    writer, sheet_name="Parameters", index=False)
                for title, df in self._export_frames().items():
                    if len(df):
                        df.to_excel(writer, sheet_name=title[:31], index=False)
            self._status(f"Measurements exported: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Data exported:\n{path}", parent=self)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Export error",
                                 f"Excel export failed:\n{exc}\n\n('openpyxl' is required)", parent=self)

    def _export_pdf(self):
        if not self._has_result():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".pdf",
                                            filetypes=[("PDF Report", "*.pdf")])
        if not path:
            return
        try:
            text = ("Image Zone Analysis Report\n" + "=" * 52 + "\n\n"
                    + "\n".join(self._header_lines()) + "\n\n" + self._results_text())
            lines = text.split("\n")
            chunks = [lines[i:i + 70] for i in range(0, len(lines), 70)] or [[""]]
            keep_view = self.view_var.get()
            with PdfPages(path) as pdf:
                for chunk in chunks:
                    page = Figure(figsize=(8.5, 11), facecolor="white")
                    page.text(0.05, 0.96, "\n".join(chunk), fontsize=8, va="top",
                              ha="left", fontfamily="monospace")
                    pdf.savefig(page)
                for view in ("Overlay", "Thickness profile"):
                    self.view_var.set(view)
                    fig = Figure(figsize=(11, 7), facecolor="white")
                    ax = fig.add_subplot(111)
                    self._render(ax=ax, fig=fig, light=True)
                    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            self.view_var.set(keep_view)
            self._render()
            self._status(f"PDF report generated: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"PDF report generated:\n{path}", parent=self)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Export error", f"Failed to build the PDF:\n{exc}", parent=self)

    def _send_thicknesses(self):
        if not self._has_result():
            return
        values = self.result["profile"]["Thickness"].to_numpy(dtype=float)
        if values.size == 0:
            messagebox.showinfo("Nothing to send", "No thickness was measured.", parent=self)
            return
        default = os.path.splitext(os.path.basename(self.image_path or "Image"))[0]
        name = simpledialog.askstring("Send to LOM Depth Analyser",
                                      "Name of the group to create:",
                                      initialvalue=default, parent=self)
        if name is None:
            return
        if not callable(self._send_to_lom):
            messagebox.showerror("Not available",
                                 "The LOM Depth Analyser module is not reachable from here.",
                                 parent=self)
            return
        self._send_to_lom(name.strip() or default, values, self.scale_unit)
        self._status(f"{values.size} thickness(es) sent to the LOM Depth Analyser.")
