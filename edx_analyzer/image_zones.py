"""
Image Zone Analyser  --  "Image Zone Analyser" module of the analysis suite
-------------------------------------------------------------------------
Finds the zones of an optical micrograph by COLOUR and by PROXIMITY, then
measures them:

  * the oxide / corrosion layer growing along the edges of the section is
    detected as the class of pixels touching the sound metal,
  * its thickness is measured perpendicular to the axis of the section - from
    the first sound-metal pixel out to the end of the oxide - column after
    column, giving hundreds of measurements and a thickness profile,
  * the area of every detected object is reported in real units.

The thicknesses can be pushed straight into the LOM Depth Analyser module,
which already draws their distribution and computes the statistics.

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

from .constants import BACKGROUND_PRESETS, ACCENT
from .color_utils import contrast_text_color
from .legend_utils import apply_legend, fit_layout
from .widgets import ColorPickerDialog, add_tooltip
from .image_measurement import CalibrationDialog
from .lom_depth_core import parse_number
from . import image_zones_core as core

VIEW_MODES = ["Overlay", "Original image", "Thickness profile"]
SIDE_CHOICES = {"Both edges": ("top", "bottom"), "Top edge": ("top",), "Bottom edge": ("bottom",)}
ROLES = ["metal", "zone", "background"]
ROLE_LABELS = {"metal": "Sound metal (reference)", "zone": "Zone to measure",
               "background": "Background (ignored)"}
MIN_RELIABLE_PX = 5.0          # below that, a thickness is only a few pixels wide


class ImageZoneAnalyserPanel(ctk.CTkFrame):
    """The whole 'Image Zone Analyser' module, as an embeddable frame."""

    def __init__(self, master, status_callback=None, send_to_lom=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._status = status_callback if callable(status_callback) else (lambda _m: None)
        self._send_to_lom = send_to_lom

        # ---- data ------------------------------------------------------
        self.image_path = None
        self.rgb = None
        self.lab = None
        self.result = None
        self.scale = None                    # real units per pixel
        self.scale_unit = "µm"
        self.calibration_arrow = None
        self.classes = [{"name": n, "color": c, "role": r, "samples": [],
                         "stats": None, "auto": False}
                        for n, c, r in core.DEFAULT_CLASSES]

        # ---- interaction ----------------------------------------------
        self._tool = None                    # None | "calibrate" | ("sample", idx)
        self._drag_start = None
        self._temp_artist = None

        # ---- settings --------------------------------------------------
        self.closing_var = tk.StringVar(value="1")
        self.min_area_var = tk.StringVar(value="20")        # real units squared
        self.max_distance_var = tk.StringVar(value="300")   # real units
        self.reject_var = tk.StringVar(value="")            # Lab distance, empty = off
        self.side_var = tk.StringVar(value="Both edges")
        self.step_var = tk.StringVar(value="1")
        self.gap_var = tk.StringVar(value="3")
        self.keep_empty = ctk.BooleanVar(value=False)
        self.straighten = ctk.BooleanVar(value=True)

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
        self._refresh_classes()
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

        self.left_panel = ctk.CTkScrollableFrame(self, width=330, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        canvas_frame = ctk.CTkFrame(center)
        canvas_frame.pack(fill="both", expand=True)

        self.right_panel = ctk.CTkScrollableFrame(self, width=370, corner_radius=0)
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
        ctk.CTkLabel(p, text="Oxide / corrosion thickness and area from a micrograph",
                     font=("Helvetica", 11), text_color="gray", wraplength=300,
                     justify="left").pack(anchor="w")

        self._section(p, "IMAGE")
        ctk.CTkButton(p, text="🖼 Load micrograph", command=self._load_image).pack(fill="x", pady=2)
        ctk.CTkLabel(p, textvariable=self.image_name_var, font=("Helvetica", 11),
                     text_color="gray", wraplength=300, justify="left").pack(anchor="w")

        self._section(p, "SCALE CALIBRATION")
        cal_btn = ctk.CTkButton(p, text="📏 Set scale (draw arrow)", fg_color="#E69F00",
                                text_color="black", command=self._activate_calibration)
        cal_btn.pack(fill="x", pady=2)
        add_tooltip(cal_btn, "Click, then drag an arrow over the scale bar\nand type the length it represents.")
        det_btn = ctk.CTkButton(p, text="🔍 Detect the scale bar", fg_color="gray35",
                                command=self._detect_scale_bar)
        det_btn.pack(fill="x", pady=2)
        add_tooltip(det_btn, "Look for the bright scale bar in the bottom-right corner\nand ask for the length it stands for.")
        ctk.CTkLabel(p, textvariable=self.scale_status, text_color=ACCENT,
                     font=("Arial", 11, "bold"), wraplength=300, justify="left").pack(anchor="w", pady=(4, 0))

        self._section(p, "COLOUR CLASSES")
        ctk.CTkLabel(p, text="Sample a few spots of each class on the image, "
                             "or start from the automatic split.",
                     text_color="gray", font=("Arial", 10), wraplength=300,
                     justify="left").pack(anchor="w")
        auto_btn = ctk.CTkButton(p, text="⚡ Automatic split (colour clustering)", fg_color="gray35",
                                 command=self._auto_classes)
        auto_btn.pack(fill="x", pady=4)
        add_tooltip(auto_btn, "Cluster the colours of the picture in three groups:\n"
                              "brightest = metal, the coloured dark one = oxide,\n"
                              "the neutral dark one = resin. Sampling a class by hand\n"
                              "afterwards replaces its automatic seed.")
        self.classes_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.classes_frame.pack(fill="x", pady=2)

        self._section(p, "SEGMENTATION")
        self._entry_row(p, "Smoothing radius (px) :", self.closing_var,
                        "Closes the small holes and jagged edges of the masks.")
        self._entry_row(p, "Min. object area :", self.min_area_var,
                        "Objects smaller than this (in squared real units) are dropped.")
        self._entry_row(p, "Max. distance to metal :", self.max_distance_var,
                        "A zone is corrosion only if it lies within this distance\nof the sound metal. This is what rejects dark specks in the resin.")
        self._entry_row(p, "Colour tolerance :", self.reject_var,
                        "Optional. Pixels further than this normalised Lab distance\nfrom every class are left unclassified.")

        self._section(p, "THICKNESS MEASUREMENT")
        ctk.CTkLabel(p, text="Edge measured :").pack(anchor="w")
        ctk.CTkComboBox(p, variable=self.side_var, values=list(SIDE_CHOICES.keys())).pack(fill="x", pady=2)
        self._entry_row(p, "Column step (px) :", self.step_var,
                        "1 = one measurement per pixel column.")
        self._entry_row(p, "Gap tolerance (px) :", self.gap_var,
                        "How many non-zone pixels may interrupt the layer\nbefore the scan stops.")
        ctk.CTkCheckBox(p, text="Count columns without oxide as 0",
                        variable=self.keep_empty).pack(anchor="w", pady=3)
        ctk.CTkCheckBox(p, text="Straighten the section before measuring",
                        variable=self.straighten).pack(anchor="w", pady=3)

        run_btn = ctk.CTkButton(p, text="▶ Analyse", fg_color="#009E73", hover_color="#007755",
                                height=36, font=("Helvetica", 13, "bold"), command=self._run_analysis)
        run_btn.pack(fill="x", pady=(12, 15))

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
        self.results_box = ctk.CTkTextbox(p, height=360, font=("Consolas", 11), wrap="none")
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
        send_btn = ctk.CTkButton(p, text="➡ Send thicknesses to LOM Depth Analyser",
                                 fg_color="#0072B2", hover_color="#005588",
                                 command=self._send_thicknesses)
        send_btn.pack(fill="x", pady=(10, 15))
        add_tooltip(send_btn, "Create a group in the LOM Depth Analyser tab holding\nevery thickness measured here, for the distribution and statistics.")

    def _on_alpha(self, value):
        self.alpha_label.configure(text=f"{float(value):.2f}")
        self._render()

    # ─────────────────────────────────────────────────────────────────────
    #  Classes
    # ─────────────────────────────────────────────────────────────────────
    def _refresh_classes(self):
        for w in self.classes_frame.winfo_children():
            w.destroy()
        for idx, cl in enumerate(self.classes):
            card = ctk.CTkFrame(self.classes_frame, corner_radius=6)
            card.pack(fill="x", pady=4)

            head = ctk.CTkFrame(card, fg_color="transparent")
            head.pack(fill="x", padx=6, pady=(6, 2))
            btn = ctk.CTkButton(head, text="", width=22, height=22, fg_color=cl["color"])
            btn.configure(command=lambda i=idx, b=btn: self._pick_class_color(i, b))
            btn.pack(side="left", padx=(0, 6))
            entry = ctk.CTkEntry(head, width=140)
            entry.insert(0, cl["name"])
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<FocusOut>", lambda _e, i=idx, en=entry: self._rename_class(i, en))
            entry.bind("<Return>", lambda _e, i=idx, en=entry: self._rename_class(i, en))

            ctk.CTkLabel(card, text=ROLE_LABELS.get(cl["role"], cl["role"]),
                         font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", padx=10)

            if cl.get("auto") and cl["stats"]:
                info = f"automatic split · {cl['stats']['n']} px"
            else:
                n_px = sum(len(sample) for sample in cl["samples"])
                info = f"{len(cl['samples'])} sample(s) · {n_px} px"
            ctk.CTkLabel(card, text=info, font=("Arial", 10),
                         text_color="#9aa0a6", anchor="w").pack(fill="x", padx=10)

            actions = ctk.CTkFrame(card, fg_color="transparent")
            actions.pack(fill="x", padx=8, pady=6)
            ctk.CTkButton(actions, text="🖉 Sample", height=24, fg_color="gray30",
                          command=lambda i=idx: self._activate_sampling(i)).pack(side="left", expand=True, padx=(0, 4))
            ctk.CTkButton(actions, text="✕", width=32, height=24, fg_color="gray35",
                          hover_color="#7a2222",
                          command=lambda i=idx: self._clear_samples(i)).pack(side="right")

    def _pick_class_color(self, idx, button):
        col = ColorPickerDialog.ask_color(self.winfo_toplevel(), initial_color=self.classes[idx]["color"],
                                          title=f"Colour of '{self.classes[idx]['name']}'")
        if col:
            self.classes[idx]["color"] = col
            button.configure(fg_color=col)
            self._render()

    def _rename_class(self, idx, entry):
        try:
            new = entry.get().strip()
        except Exception:                                  # noqa: BLE001 - widget gone
            return
        if new and new != self.classes[idx]["name"]:
            self.classes[idx]["name"] = new
            self.after(60, self._refresh_classes)

    def _clear_samples(self, idx):
        self.classes[idx].update({"samples": [], "stats": None, "auto": False})
        self._refresh_classes()

    def _activate_sampling(self, idx):
        if self.rgb is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        self._tool = ("sample", idx)
        self._status(f"Sampling '{self.classes[idx]['name']}': drag a small box over that material.")

    def _auto_classes(self):
        """Seed the three classes without any click, by clustering the colours."""
        if self.rgb is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        try:
            stats, _centroids = core.auto_class_stats(self.lab)
            for cl, st in zip(self.classes, stats):
                cl.update({"stats": st, "samples": [], "auto": True})
            self._refresh_classes()
            self._render()
            self._status("Automatic split done (colour clustering) — "
                         "sample a few spots to refine it if needed.")
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Automatic split failed", str(exc), parent=self)

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
            self.rgb = core.to_rgb_float(mpimg.imread(path))
            self.lab = core.to_lab(self.rgb)
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Error", f"Could not read the image:\n{exc}", parent=self)
            return
        self.image_path = path
        self.result = None
        self.calibration_arrow = None
        self.scale = None
        self.scale_status.set("Not calibrated.")
        for cl in self.classes:
            cl.update({"samples": [], "stats": None, "auto": False})
        h, w = self.rgb.shape[:2]
        self.image_name_var.set(f"{os.path.basename(path)}  ({w} × {h} px)")
        self._refresh_classes()
        self._refresh_results()
        self._render(reset_view=True)
        self._status(f"Micrograph loaded: {os.path.basename(path)}")

    def _activate_calibration(self):
        if self.rgb is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        self._tool = "calibrate"
        self._status("Calibration: drag an arrow over the scale bar.")

    def _detect_scale_bar(self):
        if self.rgb is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        found = core.detect_scale_bar(self.rgb)
        if not found:
            messagebox.showinfo("Scale bar", "No scale bar found in the bottom-right corner.\n"
                                             "Use 'Set scale (draw arrow)' instead.", parent=self)
            return
        length_px, p1, p2 = found
        CalibrationDialog(self.winfo_toplevel(), length_px,
                          on_confirm=lambda real, unit: self._apply_calibration(p1, p2, length_px, real, unit))

    def _apply_calibration(self, p1, p2, pixel_len, real_len, unit):
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
            CalibrationDialog(self.winfo_toplevel(), pixel_len,
                              on_confirm=lambda real, unit: self._apply_calibration(p1, p2, pixel_len, real, unit))
        elif isinstance(tool, tuple) and tool[0] == "sample":
            self._collect_sample(tool[1], p1, p2)
        else:
            self.canvas.draw_idle()

    def _collect_sample(self, idx, p1, p2):
        h, w = self.rgb.shape[:2]
        c0, c1 = sorted((int(round(p1[0])), int(round(p2[0]))))
        r0, r1 = sorted((int(round(p1[1])), int(round(p2[1]))))
        c0, c1 = max(0, c0), min(w, max(c1, c0 + 1))
        r0, r1 = max(0, r0), min(h, max(r1, r0 + 1))
        pixels = self.lab[r0:r1, c0:c1].reshape(-1, 3)
        if pixels.size == 0:
            return
        cl = self.classes[idx]
        if cl.get("auto"):              # a hand-picked sample replaces the automatic seed
            cl.update({"samples": [], "auto": False})
        cl["samples"].append(pixels)
        cl["stats"] = core.sample_stats(np.concatenate(cl["samples"]))
        self._refresh_classes()
        self._status(f"'{cl['name']}': {len(pixels)} pixels sampled "
                     f"({sum(len(s) for s in cl['samples'])} in total).")
        self._render()

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

    def _run_analysis(self):
        if self.rgb is None:
            messagebox.showwarning("No image", "Load a micrograph first.", parent=self)
            return
        if not core.IMAGING_AVAILABLE:
            messagebox.showerror("Missing packages", str(core.IMAGING_ERROR), parent=self)
            return
        if self.scale is None:
            messagebox.showwarning("Not calibrated",
                                   "Set the scale first, otherwise the measurements have no unit.",
                                   parent=self)
            return
        missing = [c["name"] for c in self.classes if c["role"] in ("metal", "zone") and not c["stats"]]
        if missing:
            messagebox.showwarning("Classes not defined",
                                   "Sample these classes on the image first:\n• " + "\n• ".join(missing)
                                   + "\n\n(or use the automatic split)", parent=self)
            return

        px_area = self.scale * self.scale
        try:
            self._status("Analysing…")
            self.update_idletasks()
            self.result = core.analyse(
                self.rgb,
                [c["stats"] for c in self.classes],
                [c["role"] for c in self.classes],
                scale=self.scale,
                reject_distance=parse_number(self.reject_var.get()),
                closing_radius=int(self._number(self.closing_var, 1)),
                min_area_px=int(self._number(self.min_area_var, 0) / px_area),
                max_distance_px=self._number(self.max_distance_var, 0) / self.scale,
                straighten=bool(self.straighten.get()),
                sides=SIDE_CHOICES.get(self.side_var.get(), core.SIDES),
                step_px=int(self._number(self.step_var, 1)),
                gap_tolerance_px=int(self._number(self.gap_var, 3)),
                keep_empty=bool(self.keep_empty.get()))
        except Exception as exc:                           # noqa: BLE001
            messagebox.showerror("Analysis failed", str(exc), parent=self)
            self._status("Analysis failed.")
            return

        n = self.result["stats"]["n"]
        self.view_var.set("Overlay" if n else self.view_var.get())
        self._refresh_results()
        self._render()
        self._status(f"Analysis done: {n} thickness measurement(s), "
                     f"mean {self.result['stats']['mean']:.3g} {self.scale_unit}.")

    def _masks_for_display(self):
        if not self.result:
            return {}, {}
        masks, colors = {}, {}
        for cl in self.classes:
            if cl["role"] == "background":
                continue
            mask = self.result.get(cl["role"])
            if mask is not None and np.any(mask):
                masks[cl["name"]] = mask
                colors[cl["name"]] = cl["color"]
        return masks, colors

    # ─────────────────────────────────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────────────────────────────────
    def _style_axes(self, light=False):
        face = "white" if light else self.plot_bg_color.get()
        self.fig.patch.set_facecolor("white" if light else self.fig_bg_color.get())
        self.ax.set_facecolor(face)
        return face, ("black" if light else contrast_text_color(face))

    def _render(self, reset_view=False, ax=None, fig=None, light=False):
        target_ax = ax if ax is not None else self.ax
        target_fig = fig if fig is not None else self.fig
        embedded = ax is None

        xlim = ylim = None
        if embedded and not reset_view and target_ax.has_data():
            xlim, ylim = target_ax.get_xlim(), target_ax.get_ylim()

        target_ax.clear()
        if embedded:
            face, text_color = self._style_axes(light)
        else:
            face = "white" if light else self.plot_bg_color.get()
            text_color = "black" if light else contrast_text_color(face)
            target_fig.patch.set_facecolor("white" if light else self.fig_bg_color.get())
            target_ax.set_facecolor(face)

        view = self.view_var.get()
        if view == "Thickness profile":
            self._render_profile(target_ax, target_fig, face, text_color)
            return

        target_ax.axis("off")
        if self.rgb is None:
            target_ax.text(0.5, 0.5, "Load a micrograph to begin", ha="center", va="center",
                           transform=target_ax.transAxes, color="gray", fontsize=self.font_size.get() + 2)
            target_fig.tight_layout()
            if embedded:
                self.canvas.draw()
            return

        picture = self.rgb
        masks, colors = self._masks_for_display()
        if view == "Overlay" and masks:
            picture = core.build_overlay(self.rgb, masks, colors,
                                         alpha=float(self.overlay_alpha.get()),
                                         outline=bool(self.show_outlines.get()))
        target_ax.imshow(picture)

        if self.calibration_arrow:
            p1, p2 = self.calibration_arrow["p1"], self.calibration_arrow["p2"]
            target_ax.annotate("", xy=p2, xytext=p1,
                               arrowprops=dict(arrowstyle="<->", color="#E69F00", linewidth=2))
            target_ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2,
                           f"{self.calibration_arrow['real_length']:g} {self.scale_unit}",
                           color="#E69F00", fontsize=self.font_size.get(), fontweight="bold",
                           ha="center", va="bottom",
                           bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

        if view == "Overlay" and masks:
            handles = [Rectangle((0, 0), 1, 1, facecolor=colors[name], edgecolor="none")
                       for name in masks]
            target_ax.legend(handles, list(masks.keys()), loc="upper right",
                             fontsize=self.font_size.get(), framealpha=0.8)

        if xlim is not None:
            target_ax.set_xlim(xlim)
            target_ax.set_ylim(ylim)
        target_fig.tight_layout()
        if embedded:
            self.canvas.draw()

    def _render_profile(self, ax, fig, face, text_color):
        fs = self.font_size.get()
        # imshow leaves the axes with an equal aspect and a top-down y axis,
        # and ax.clear() keeps both: undo them or the curve is squashed flat
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
            ax.text(0.5, 0.5, "Run the analysis to get a thickness profile",
                    ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=fs + 2)
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
        lines = ["IMAGE", f"  File          : {os.path.basename(self.image_path) if self.image_path else '-'}"]
        if self.rgb is not None:
            h, w = self.rgb.shape[:2]
            lines.append(f"  Size          : {w} x {h} px")
        lines.append(f"  Scale         : " + (f"{self.scale:.4g} {unit}/px" if self.scale else "not calibrated"))
        lines.append("")

        if not self.result:
            lines.append("Sample the classes, then press 'Analyse'.")
            return "\n".join(lines)

        st = self.result["stats"]
        lines.append("THICKNESS OF THE MEASURED ZONE")
        lines.append(f"  Section tilt  : {self.result['angle']:+.2f} deg (corrected)")
        lines.append(f"  Measurements  : {st['n']}")
        if st["n"]:
            lines.append(f"  Min / Max     : {st['min']:.4g} / {st['max']:.4g} {unit}")
            lines.append(f"  Mean          : {st['mean']:.4g} {unit}")
            lines.append(f"  Median        : {st['median']:.4g} {unit}")
            lines.append(f"  Std dev (n-1) : {st['std']:.4g} {unit}")
            lines.append(f"  P10 / P90     : {st['p10']:.4g} / {st['p90']:.4g} {unit}")
            profile = self.result["profile"]
            for side in ("top", "bottom"):
                part = profile[profile["Side"] == side]
                if len(part):
                    lines.append(f"  {side:<6} edge   : n={len(part)}  mean={part['Thickness'].mean():.4g}  "
                                 f"max={part['Thickness'].max():.4g} {unit}")
            mean_px = st["mean"] / self.scale if self.scale else 0
            lines.append("")
            lines.append(f"  Resolution    : mean thickness = {mean_px:.1f} px")
            if mean_px < MIN_RELIABLE_PX:
                lines.append("  /!\\ WARNING: fewer than 5 pixels. At this magnification the")
                lines.append("      measurement is dominated by the pixel size — use a more")
                lines.append("      magnified picture for a layer this thin.")
        lines.append("")

        lines.append("AREAS")
        for _, row in self.result["summary"].iterrows():
            lines.append(f"  {row['Class']}")
            lines.append(f"     objects    : {int(row['Objects'])}")
            lines.append(f"     area       : {row['Area']:.6g} {unit}^2")
            if np.isfinite(row["Share of specimen (%)"]):
                lines.append(f"     share      : {row['Share of specimen (%)']:.2f} % of the specimen")
        objects = self.result["objects"]
        if len(objects):
            lines.append("")
            lines.append(f"  Largest objects ({min(5, len(objects))} of {len(objects)}):")
            for _, row in objects.nlargest(5, "Area").iterrows():
                lines.append(f"     #{int(row['Object']):<4} area={row['Area']:.5g} {unit}^2  "
                             f"length={row['Length']:.4g} {unit}")
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
        unit = self.scale_unit
        summary = self.result["summary"].rename(columns={"Area": f"Area ({unit}^2)"})
        profile = self.result["profile"].rename(columns={
            "Position": f"Position ({unit})", "Thickness": f"Thickness ({unit})"})
        objects = self.result["objects"].rename(columns={
            "Area": f"Area ({unit}^2)", "Perimeter": f"Perimeter ({unit})",
            "Equivalent diameter": f"Equivalent diameter ({unit})",
            "Length": f"Length ({unit})", "Width": f"Width ({unit})",
            "Centroid X": f"Centroid X ({unit})", "Centroid Y": f"Centroid Y ({unit})"})
        st = self.result["stats"]
        stats = pd.DataFrame([{"Statistic": k, f"Value ({unit})": v} for k, v in st.items()])
        return summary, stats, profile, objects

    def _header_lines(self):
        st = self.result["stats"]
        return [
            "Image Zone Analyser export",
            f"Image: {os.path.basename(self.image_path) if self.image_path else '-'}",
            f"Scale: {self.scale:.6g} {self.scale_unit}/px",
            f"Section tilt corrected: {self.result['angle']:+.3f} deg",
            f"Edge measured: {self.side_var.get()}",
            f"Column step: {self.step_var.get()} px",
            f"Gap tolerance: {self.gap_var.get()} px",
            f"Max distance to metal: {self.max_distance_var.get()} {self.scale_unit}",
            f"Min object area: {self.min_area_var.get()} {self.scale_unit}^2",
            f"Measurements: {st['n']}",
        ]

    def _export_image(self):
        if self.rgb is None:
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
        summary, stats, profile, objects = self._export_frames()
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                for line in self._header_lines():
                    f.write(f"# {line}\n")
                for title, df in (("AREAS", summary), ("THICKNESS STATISTICS", stats),
                                  ("THICKNESS PROFILE", profile), ("OBJECTS", objects)):
                    f.write(f"\n# --- {title} ---\n")
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
        summary, stats, profile, objects = self._export_frames()
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame({"Parameter": self._header_lines()}).to_excel(
                    writer, sheet_name="Parameters", index=False)
                summary.to_excel(writer, sheet_name="Areas", index=False)
                stats.to_excel(writer, sheet_name="Statistics", index=False)
                profile.to_excel(writer, sheet_name="Thickness profile", index=False)
                if len(objects):
                    objects.to_excel(writer, sheet_name="Objects", index=False)
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
                    page.text(0.07, 0.96, "\n".join(chunk), fontsize=9, va="top",
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
        name = name.strip() or default
        if not callable(self._send_to_lom):
            messagebox.showerror("Not available",
                                 "The LOM Depth Analyser module is not reachable from here.", parent=self)
            return
        self._send_to_lom(name, values, self.scale_unit)
        self._status(f"{values.size} thickness(es) sent to the LOM Depth Analyser as '{name}'.")
