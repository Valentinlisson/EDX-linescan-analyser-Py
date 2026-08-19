"""Main application window: UI layout, plotting, and Matplotlib event
handling for the EDX Line Scan Viewer."""

import os
import json

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.widgets import SpanSelector

from .constants import COLORS_DEFAULT, MARKERS, LEGEND_POSITIONS, BACKGROUND_PRESETS, BG, ACCENT
from .color_utils import contrast_text_color
from .data_processing import parse_edx_file, get_elements, get_pos_col, normalize_to_100
from .phase_manager import PhaseManagerWindow
from .manual_zones import ManualZoneDialog, ManualZoneManagerWindow
from .widgets import ColorPickerDialog, add_tooltip
from .history import ChangeHistory
from .image_measurement import ImageMeasurementMixin
from .report import ReportInfoDialog, generate_pdf_report

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except ImportError:
    TkinterDnD, DND_FILES = None, None
    _DND_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Drag-and-drop support requires mixing in TkinterDnD's wrapper alongside
# CTk's root window class; fall back to plain CTk if the optional
# tkinterdnd2 dependency isn't installed.
if _DND_AVAILABLE:
    class _AppBase(ctk.CTk, TkinterDnD.DnDWrapper):
        pass
else:
    _AppBase = ctk.CTk


class EDXApp(_AppBase, ImageMeasurementMixin):
    def __init__(self):
        super().__init__()
        self.title("EDX Line Scan Viewer - Lab Edition (PDF Reports)")
        self.geometry("1600x900")

        self.df_raw, self.df_norm = None, None
        self.elements = []
        self.is_normalized = False
        self.phase_presets = []
        self.detected_zones = []  # Auto-detected zones (from the phase presets)
        self.manual_zones = []  # User-defined zones (drag + precise numeric edit)

        self.filepath_var = tk.StringVar(value="No file loaded")
        self.enable_crosshair = ctk.BooleanVar(value=False)

        self.graph_title = "EDX — Analysis Profile"
        self.graph_xlabel = "Position (µm)"
        self.graph_ylabel = "Intensity / Concentration"

        self.el_vars, self.el_colors, self.el_markers = {}, {}, {}

        self.smooth_window = ctk.IntVar(value=1)
        self.plot_bg_color, self.fig_bg_color = tk.StringVar(value="#282C34"), tk.StringVar(value="#1E2127")
        self.line_width, self.marker_size = ctk.DoubleVar(value=1.8), ctk.DoubleVar(value=5.0)
        self.show_grid = ctk.BooleanVar(value=True)
        self.font_size = ctk.IntVar(value=10)
        self.legend_pos_var = ctk.StringVar(value="Outside Right")

        self.scale_x, self.scale_y = ctk.DoubleVar(value=1.0), ctk.DoubleVar(value=1.0)
        self._slider_interacting, self._pan_start = False, None
        self.roi_selector = None

        self.appearance_var = ctk.StringVar(value="Dark")
        self.status_var = tk.StringVar(value="Ready. Load an EDX .txt file to begin.")

        self.change_history = ChangeHistory()
        self._last_style_snapshot = None

        self._init_image_state()

        self._build_ui()
        self._init_style_baseline()

        if _DND_AVAILABLE:
            self.TkdndVersion = TkinterDnD._require(self)
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._on_file_drop)

        self.bind_all("<Control-z>", self._undo)
        self.bind_all("<Control-y>", self._redo)
        self.bind_all("<Control-Shift-Z>", self._redo)

    # ─────────────────────────────────────────────────────────────────────
    #  Layout
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.tabview = ctk.CTkTabview(self, corner_radius=0)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        edx_tab = self.tabview.add("EDX Analysis")
        image_tab = self.tabview.add("SEM Image & Measurements")

        self._build_edx_tab(edx_tab)
        self._build_image_tab(image_tab)
        # A tab built while hidden gets a matplotlib canvas sized 0x0; force a
        # redraw once it's actually shown so the figure renders at real size.
        self.tabview.configure(command=self._on_tab_changed)

        self.status_bar = ctk.CTkFrame(self, height=28, corner_radius=0)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(self.status_bar, textvariable=self.status_var, anchor="w", font=("Arial", 11), text_color="gray70").pack(side="left", padx=10, pady=3)

    def _build_edx_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # A resizable PanedWindow so the side panels can be dragged wider/narrower.
        self.paned = tk.PanedWindow(tab, orient="horizontal", sashrelief="raised", sashwidth=6, bg=BG, bd=0)
        self.paned.grid(row=0, column=0, sticky="nsew")

        # CTkScrollableFrame renders itself inside an internal canvas, so its
        # own Tk widget isn't a direct child of the pane — wrap each side in
        # a plain CTkFrame that the PanedWindow can hold directly.
        left_wrapper = ctk.CTkFrame(self.paned, fg_color="transparent")
        self.paned.add(left_wrapper, minsize=220, width=280)
        self.left_panel = ctk.CTkScrollableFrame(left_wrapper, corner_radius=0)
        self.left_panel.pack(fill="both", expand=True)

        center_frame = ctk.CTkFrame(self.paned, fg_color="transparent")
        self.paned.add(center_frame, minsize=400, stretch="always")

        graph_inside_frame = ctk.CTkFrame(center_frame)
        graph_inside_frame.pack(fill="both", expand=True, padx=5, pady=10)

        right_wrapper = ctk.CTkFrame(self.paned, fg_color="transparent")
        self.paned.add(right_wrapper, minsize=280, width=350)
        self.right_panel = ctk.CTkScrollableFrame(right_wrapper, corner_radius=0)
        self.right_panel.pack(fill="both", expand=True)

        self._build_left(self.left_panel)
        self._build_graph(graph_inside_frame)
        self._build_right(self.right_panel)

    def _set_status(self, message):
        self.status_var.set(message)

    def _on_tab_changed(self):
        # A tab's matplotlib canvas / CTkScrollableFrame can be laid out at
        # 0x0 while hidden; force a layout pass before redrawing so the
        # first switch doesn't show a stale/blank frame.
        self.update_idletasks()
        if self.tabview.get() == "SEM Image & Measurements":
            self._redraw_image()
        elif self.df_norm is not None:
            self._plot()

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Helvetica", 12, "bold"), text_color=ACCENT).pack(anchor="w", pady=(15, 5))

    def _build_left(self, p):
        ctk.CTkLabel(p, text="EDX Lab Viewer", font=("Helvetica", 20, "bold")).pack(anchor="w", pady=(5, 10))

        self._section(p, "MAIN DATA")
        load_btn = ctk.CTkButton(p, text="📂 Load File", command=self._open_file)
        load_btn.pack(fill="x", pady=2)
        drop_hint = " Drag & drop a .txt file onto the window also works." if _DND_AVAILABLE else ""
        add_tooltip(load_btn, "Open an EDX line-scan .txt file." + drop_hint)
        ctk.CTkLabel(p, textvariable=self.filepath_var, font=("Helvetica", 11), text_color="gray", wraplength=240).pack(anchor="w")

        self._section(p, "DATA PROCESSING")
        row_btn = ctk.CTkFrame(p, fg_color="transparent")
        row_btn.pack(fill="x", pady=5)
        norm_btn = ctk.CTkButton(row_btn, text="✔ Normalize 100%", command=self._apply_normalization, width=130)
        norm_btn.pack(side="left", padx=(0, 5))
        add_tooltip(norm_btn, "Rescale the checked elements so they sum to 100% at every point.")
        raw_btn = ctk.CTkButton(row_btn, text="⏪ Raw", command=self._reset_to_raw, width=80, fg_color="gray40")
        raw_btn.pack(side="right")
        add_tooltip(raw_btn, "Discard normalization, go back to raw intensities.")

        self._section(p, "SCIENTIFIC ANALYSIS")
        self._slider_generic(p, "Smoothing (Pts):", self.smooth_window, 1, 20)
        roi_btn = ctk.CTkButton(p, text="📊 Extract Stats (ROI)", command=self._activate_roi, fg_color="#E69F00", text_color="black")
        roi_btn.pack(fill="x", pady=5)
        add_tooltip(roi_btn, "Drag a region on the graph to get per-element average/max stats.")

        phase_btn = ctk.CTkButton(p, text="⚙ Phase Editor (JSON)", fg_color="#009E73", hover_color="#007755", command=self._open_phase_editor)
        phase_btn.pack(fill="x", pady=2)
        add_tooltip(phase_btn, "Define chemical composition thresholds that identify a phase.")
        zones_btn = ctk.CTkButton(p, text="📋 Identify Multi-Zones", fg_color="#56B4E9", text_color="black", hover_color="#3399CC", command=self._generate_phase_report)
        zones_btn.pack(fill="x", pady=2)
        add_tooltip(zones_btn, "Scan the profile and mark phase boundaries using the presets above.")
        clear_btn = ctk.CTkButton(p, text="❌ Clear Auto Zones", fg_color="gray40", hover_color="gray30", command=self._clear_auto_zones)
        clear_btn.pack(fill="x", pady=2)
        add_tooltip(clear_btn, "Remove the auto-detected zone markers from the graph.")

        self._section(p, "MANUAL ZONES")
        ctk.CTkLabel(p, text="For noisy data where auto-detection struggles.", text_color="gray", font=("Arial", 10), wraplength=240, justify="left").pack(anchor="w")
        add_zone_btn = ctk.CTkButton(p, text="📐 Add Manual Zone", fg_color="#CC79A7", text_color="black", hover_color="#A5628A", command=self._activate_manual_zone_tool)
        add_zone_btn.pack(fill="x", pady=(5, 2))
        add_tooltip(add_zone_btn, "Drag on the graph to place a zone, then name it and fine-tune its exact bounds.")
        manage_zone_btn = ctk.CTkButton(p, text="📝 Manage Manual Zones", fg_color="gray35", command=self._open_manual_zone_manager)
        manage_zone_btn.pack(fill="x", pady=2)
        add_tooltip(manage_zone_btn, "Review, edit or delete the manual zones you've placed.")
        clear_manual_btn = ctk.CTkButton(p, text="❌ Clear Manual Zones", fg_color="gray40", hover_color="gray30", command=self._clear_manual_zones)
        clear_manual_btn.pack(fill="x", pady=2)
        add_tooltip(clear_manual_btn, "Remove all manual zone markers from the graph.")

        self._section(p, "ELEMENTS")
        self.el_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.el_frame.pack(fill="x")
        row_tous = ctk.CTkFrame(p, fg_color="transparent")
        row_tous.pack(fill="x", pady=10)
        all_btn = ctk.CTkButton(row_tous, text="✅ All", command=self._show_all, width=100)
        all_btn.pack(side="left", expand=True)
        add_tooltip(all_btn, "Show every element's curve.")
        none_btn = ctk.CTkButton(row_tous, text="⬜ None", command=self._hide_all, width=100, fg_color="gray40")
        none_btn.pack(side="right", expand=True)
        add_tooltip(none_btn, "Hide every element's curve.")

    def _build_right(self, p):
        self._section(p, "APPEARANCE")
        theme_row = ctk.CTkFrame(p, fg_color="transparent")
        theme_row.pack(fill="x", pady=2)
        ctk.CTkLabel(theme_row, text="App theme :").pack(side="left")
        ctk.CTkSegmentedButton(theme_row, values=["Light", "Dark"], variable=self.appearance_var,
                               command=self._toggle_appearance).pack(side="right")

        history_row = ctk.CTkFrame(p, fg_color="transparent")
        history_row.pack(fill="x", pady=2)
        self.undo_btn = ctk.CTkButton(history_row, text="↶ Undo", command=self._undo, state="disabled", width=100)
        self.undo_btn.pack(side="left", expand=True)
        add_tooltip(self.undo_btn, "Undo the last style change (Ctrl+Z).")
        self.redo_btn = ctk.CTkButton(history_row, text="↷ Redo", command=self._redo, state="disabled", width=100, fg_color="gray40")
        self.redo_btn.pack(side="right", expand=True)
        add_tooltip(self.redo_btn, "Redo the last undone style change (Ctrl+Y).")

        self._section(p, "INTERACTIVE TOOLS")
        ctk.CTkLabel(p, text="Double-click Graph:", text_color="gray", font=("Arial", 10)).pack(anchor="w")
        ctk.CTkLabel(p, text="Top = Title | Bottom/Left = Axes | Center = Reset", text_color="gray", font=("Arial", 10)).pack(anchor="w")
        ctk.CTkCheckBox(p, text="Enable Dynamic Crosshair", variable=self.enable_crosshair, command=self._plot).pack(anchor="w", pady=10)

        self._section(p, "VISUAL PRESETS")
        row_pre = ctk.CTkFrame(p, fg_color="transparent")
        row_pre.pack(fill="x")
        save_pre_btn = ctk.CTkButton(row_pre, text="💾 Save", command=self._save_presets, width=100)
        save_pre_btn.pack(side="left", expand=True)
        add_tooltip(save_pre_btn, "Save the current colors/markers/backgrounds/legend layout to a JSON file.")
        load_pre_btn = ctk.CTkButton(row_pre, text="📂 Load", command=self._load_presets, width=100, fg_color="gray40")
        load_pre_btn.pack(side="right", expand=True)
        add_tooltip(load_pre_btn, "Load a previously saved visual layout.")

        self._section(p, "ZOOM / SCALE")
        self._slider_zoom(p, "Zoom X :", self.scale_x, 1.0, 10.0)
        self._slider_zoom(p, "Zoom Y :", self.scale_y, 1.0, 10.0)
        autoscale_btn = ctk.CTkButton(p, text="🔍 Auto-Scale", command=self._autoscale_graph)
        autoscale_btn.pack(fill="x", pady=5)
        add_tooltip(autoscale_btn, "Reset zoom/pan and fit the whole profile in view.")

        self._section(p, "DESIGN & LEGEND")
        ctk.CTkComboBox(p, variable=self.legend_pos_var, values=list(LEGEND_POSITIONS.keys()), command=self._on_legend_change).pack(fill="x", pady=5)
        self._slider_generic(p, "Line Thickness :", self.line_width, 0.5, 5.0)
        self._slider_generic(p, "Point Size :", self.marker_size, 0.0, 20.0)
        self._slider_generic(p, "Font Size :", self.font_size, 7, 18)
        ctk.CTkCheckBox(p, text="Show Grid", variable=self.show_grid, command=self._on_grid_toggle).pack(anchor="w", pady=5)

        self._section(p, "GRAPH BACKGROUND")
        self._build_background_controls(p)

        self._section(p, "EXPORTS & REPORTS")
        img_btn = ctk.CTkButton(p, text="🖼 Save Image (PNG/SVG...)", command=self._save_fig)
        img_btn.pack(fill="x", pady=2)
        add_tooltip(img_btn, "Export the current graph as an image or vector file.")
        csv_btn = ctk.CTkButton(p, text="📄 Export Data (CSV)", command=self._export_csv)
        csv_btn.pack(fill="x", pady=2)
        add_tooltip(csv_btn, "Export the currently displayed (raw or normalized) data as CSV.")
        xlsx_btn = ctk.CTkButton(p, text="📊 Export Data (Excel)", command=self._export_excel)
        xlsx_btn.pack(fill="x", pady=2)
        add_tooltip(xlsx_btn, "Export both raw and processed data as an Excel workbook.")
        pdf_btn = ctk.CTkButton(p, text="📑 Generate PDF Report", command=self._export_pdf_report, fg_color="#882255", hover_color="#551133")
        pdf_btn.pack(fill="x", pady=(10, 5))
        add_tooltip(pdf_btn, "Build a multi-page PDF: parameters, detected zones table, and the graph.")

        self._section(p, "ELEMENT CUSTOMIZATION")
        self.custom_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.custom_frame.pack(fill="x")

    def _toggle_appearance(self, value):
        ctk.set_appearance_mode(value.lower())

    def _build_background_controls(self, p):
        """Color pickers + quick presets for the plot area and figure backgrounds."""
        row_plot = ctk.CTkFrame(p, fg_color="transparent")
        row_plot.pack(fill="x", pady=2)
        ctk.CTkLabel(row_plot, text="Plot Area :").pack(side="left")
        self.plot_bg_btn = ctk.CTkButton(row_plot, text="", width=32, height=25, fg_color=self.plot_bg_color.get(), border_width=1, border_color="gray50")
        self.plot_bg_btn.configure(command=lambda: self._pick_bg_color(self.plot_bg_color, self.plot_bg_btn))
        self.plot_bg_btn.pack(side="right")
        add_tooltip(self.plot_bg_btn, "Background color behind the curves.")

        row_fig = ctk.CTkFrame(p, fg_color="transparent")
        row_fig.pack(fill="x", pady=2)
        ctk.CTkLabel(row_fig, text="Figure / Border :").pack(side="left")
        self.fig_bg_btn = ctk.CTkButton(row_fig, text="", width=32, height=25, fg_color=self.fig_bg_color.get(), border_width=1, border_color="gray50")
        self.fig_bg_btn.configure(command=lambda: self._pick_bg_color(self.fig_bg_color, self.fig_bg_btn))
        self.fig_bg_btn.pack(side="right")
        add_tooltip(self.fig_bg_btn, "Outer figure background (around the plot area, incl. title/labels).")

        ctk.CTkLabel(p, text="Quick presets :", text_color="gray", font=("Arial", 10)).pack(anchor="w", pady=(6, 2))
        preset_row = ctk.CTkFrame(p, fg_color="transparent")
        preset_row.pack(fill="x")
        for name, hexcode in BACKGROUND_PRESETS.items():
            preset_btn = ctk.CTkButton(preset_row, text="", width=22, height=22, fg_color=hexcode, border_width=1,
                                       border_color="gray50", command=lambda h=hexcode: self._apply_bg_preset(h))
            preset_btn.pack(side="left", padx=2)
            add_tooltip(preset_btn, name)

        reset_bg_btn = ctk.CTkButton(p, text="↺ Reset Backgrounds", fg_color="gray40", command=self._reset_backgrounds)
        reset_bg_btn.pack(fill="x", pady=(6, 2))
        add_tooltip(reset_bg_btn, "Restore the default dark backgrounds.")

    def _pick_bg_color(self, str_var, btn_widget):
        color = ColorPickerDialog.ask_color(self, initial_color=str_var.get(), title="Select Background Color")
        if color:
            str_var.set(color)
            btn_widget.configure(fg_color=color)
            self._record_style_change()
            self._plot()

    def _apply_bg_preset(self, hexcode):
        """A single preset swatch sets both plot-area and figure backgrounds together."""
        self.plot_bg_color.set(hexcode)
        self.fig_bg_color.set(hexcode)
        self.plot_bg_btn.configure(fg_color=hexcode)
        self.fig_bg_btn.configure(fg_color=hexcode)
        self._record_style_change()
        self._plot()

    def _reset_backgrounds(self):
        self.plot_bg_color.set("#282C34")
        self.fig_bg_color.set("#1E2127")
        self.plot_bg_btn.configure(fg_color="#282C34")
        self.fig_bg_btn.configure(fg_color="#1E2127")
        self._record_style_change()
        self._plot()

    def _on_legend_change(self, value):
        self._record_style_change()
        self._plot()

    def _on_grid_toggle(self):
        self._record_style_change()
        self._plot()

    def _on_marker_change(self, value):
        self._record_style_change()
        self._plot()

    def _slider_zoom(self, parent, label, var, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{var.get():.1f}x", text_color=ACCENT)
        val_lbl.pack(side="right")

        def _upd(v):
            val_lbl.configure(text=f"{float(v):.1f}x")
            self._slider_interacting = True
            self._plot()
            self._slider_interacting = False

        ctk.CTkSlider(parent, from_=from_, to=to, variable=var, command=_upd).pack(fill="x")

    def _slider_generic(self, parent, label, var, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{var.get():.1f}", text_color=ACCENT)
        val_lbl.pack(side="right")

        def _upd(v):
            val_lbl.configure(text=f"{int(v) if isinstance(var, ctk.IntVar) else float(v):.1f}")
            self._plot()

        ctk.CTkSlider(parent, from_=from_, to=to, variable=var, number_of_steps=int(to - from_) if isinstance(var, ctk.IntVar) else None, command=_upd).pack(fill="x")

    def _pick_element_color(self, el, btn_widget):
        color = ColorPickerDialog.ask_color(self, initial_color=self.el_colors.get(el, "#FFFFFF"), title=f"Select Color — {el}")
        if color:
            self.el_colors[el] = color
            btn_widget.configure(fg_color=color)
            self._record_style_change()
            self._plot()

    def _save_presets(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = {
            "plot_bg": self.plot_bg_color.get(), "fig_bg": self.fig_bg_color.get(),
            "lw": self.line_width.get(), "ms": self.marker_size.get(), "fs": self.font_size.get(),
            "legend": self.legend_pos_var.get(), "colors": self.el_colors,
            "markers": {k: v.get() for k, v in self.el_markers.items()}
        }
        with open(p, 'w') as f:
            json.dump(data, f)
        messagebox.showinfo("Success", "Visual preset layout saved.")

    def _load_presets(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            self.plot_bg_color.set(data.get("plot_bg", "#282C34"))
            self.fig_bg_color.set(data.get("fig_bg", "#1E2127"))
            self.plot_bg_btn.configure(fg_color=self.plot_bg_color.get())
            self.fig_bg_btn.configure(fg_color=self.fig_bg_color.get())
            self.line_width.set(data.get("lw", 1.8)); self.marker_size.set(data.get("ms", 5.0)); self.font_size.set(data.get("fs", 10))
            self.legend_pos_var.set(data.get("legend", "Outside Right"))
            if "colors" in data:
                self.el_colors.update(data["colors"])
            if "markers" in data:
                for k, v in data["markers"].items():
                    if k in self.el_markers:
                        self.el_markers[k].set(v)
            self._build_element_rows()
            self._plot()
            self._init_style_baseline()
            self._set_status(f"Visual layout loaded from {os.path.basename(p)}.")
        except Exception:
            messagebox.showerror("Error", "File corrupted or invalid layout.")

    def _build_graph(self, frame):
        self.fig = Figure(figsize=(8.5, 6), facecolor=BG)
        self.ax = self.fig.add_subplot(111)

        self.crosshair_v = self.ax.axvline(0, color='gray', linestyle='--', linewidth=1, visible=False, zorder=10)
        self.crosshair_text = self.ax.text(0, 0, '', bbox=dict(facecolor=BG, edgecolor=ACCENT, alpha=0.9), color="white", visible=False, zorder=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tb_frame).update()

        self.canvas.mpl_connect("scroll_event", self._on_mpl_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_mpl_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
        self.canvas.mpl_connect("button_release_event", lambda e: setattr(self, '_pan_start', None))

        self._style_axes()
        self.canvas.draw()

    def _build_element_rows(self):
        for w in self.el_frame.winfo_children():
            w.destroy()
        self.el_vars.clear()
        for el in self.elements:
            self.el_vars[el] = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(self.el_frame, text=el, variable=self.el_vars[el], command=self._plot).pack(anchor="w", pady=2)

        for w in self.custom_frame.winfo_children():
            w.destroy()
        for i, el in enumerate(self.elements):
            if el not in self.el_colors:
                self.el_colors[el] = COLORS_DEFAULT[i % len(COLORS_DEFAULT)]
            if el not in self.el_markers:
                self.el_markers[el] = ctk.StringVar(value="Circle")

            r = ctk.CTkFrame(self.custom_frame, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=el, width=30).pack(side="left")

            btn = ctk.CTkButton(r, text="", width=25, height=25, fg_color=self.el_colors[el])
            btn.configure(command=lambda e=el, b=btn: self._pick_element_color(e, b))
            btn.pack(side="left", padx=5)
            add_tooltip(btn, f"Line color for {el}.")

            cb = ctk.CTkComboBox(r, variable=self.el_markers[el], values=list(MARKERS.keys()), width=100, command=self._on_marker_change)
            cb.pack(side="right", padx=5)

    def _show_all(self):
        [v.set(True) for v in self.el_vars.values()]
        self._plot()

    def _hide_all(self):
        [v.set(False) for v in self.el_vars.values()]
        self._plot()

    # ─────────────────────────────────────────────────────────────────────
    #  Style Undo / Redo
    #  Scope is intentionally limited to discrete, easy-to-mistake actions:
    #  colors, markers, legend position and grid. Continuous sliders (line
    #  width, point size, font size, zoom) are excluded so that an undo
    #  doesn't silently revert an unrelated in-progress drag.
    # ─────────────────────────────────────────────────────────────────────
    def _capture_style_state(self):
        return {
            "plot_bg": self.plot_bg_color.get(),
            "fig_bg": self.fig_bg_color.get(),
            "legend": self.legend_pos_var.get(),
            "grid": self.show_grid.get(),
            "colors": dict(self.el_colors),
            "markers": {k: v.get() for k, v in self.el_markers.items()},
        }

    def _restore_style_state(self, state):
        self.plot_bg_color.set(state["plot_bg"])
        self.fig_bg_color.set(state["fig_bg"])
        self.plot_bg_btn.configure(fg_color=state["plot_bg"])
        self.fig_bg_btn.configure(fg_color=state["fig_bg"])
        self.legend_pos_var.set(state["legend"])
        self.show_grid.set(state["grid"])
        self.el_colors.clear()
        self.el_colors.update(state["colors"])
        for k, v in state["markers"].items():
            if k in self.el_markers:
                self.el_markers[k].set(v)
        self._build_element_rows()
        self._plot()

    def _init_style_baseline(self):
        self._last_style_snapshot = self._capture_style_state()
        self.change_history.reset()
        self._update_undo_redo_buttons()

    def _record_style_change(self):
        """Call AFTER a discrete style mutation has already been applied."""
        if self._last_style_snapshot is None:
            self._last_style_snapshot = self._capture_style_state()
            return
        self.change_history.push(self._last_style_snapshot)
        self._last_style_snapshot = self._capture_style_state()
        self._update_undo_redo_buttons()

    def _undo(self, event=None):
        new_state = self.change_history.undo(self._last_style_snapshot)
        if new_state is None:
            return
        self._restore_style_state(new_state)
        self._last_style_snapshot = new_state
        self._update_undo_redo_buttons()
        self._set_status("Undid last style change.")

    def _redo(self, event=None):
        new_state = self.change_history.redo(self._last_style_snapshot)
        if new_state is None:
            return
        self._restore_style_state(new_state)
        self._last_style_snapshot = new_state
        self._update_undo_redo_buttons()
        self._set_status("Redid style change.")

    def _update_undo_redo_buttons(self):
        if hasattr(self, "undo_btn"):
            self.undo_btn.configure(state="normal" if self.change_history.can_undo else "disabled")
        if hasattr(self, "redo_btn"):
            self.redo_btn.configure(state="normal" if self.change_history.can_redo else "disabled")

    def _style_axes(self):
        self.ax.set_facecolor(self.plot_bg_color.get())
        self.fig.patch.set_facecolor(self.fig_bg_color.get())
        tc = contrast_text_color(self.fig_bg_color.get())
        for s in self.ax.spines.values():
            s.set_edgecolor("gray")

        fs = self.font_size.get()
        self.ax.tick_params(colors=tc, labelsize=fs)
        self.ax.xaxis.label.set_color(tc)
        self.ax.yaxis.label.set_color(tc)
        self.ax.title.set_color(tc)

        # INCREASE PAD TO MAKE ROOM FOR ZONE LABELS
        pad_amount = 30 if self.detected_zones else 10
        self.ax.set_title(self.graph_title, fontsize=fs + 2, fontweight='bold', pad=pad_amount)

        self.ax.set_xlabel(self.graph_xlabel, fontsize=fs)
        self.ax.set_ylabel(self.graph_ylabel, fontsize=fs)

        if self.show_grid.get():
            self.ax.grid(color="gray", linestyle="--", alpha=0.3)
        else:
            self.ax.grid(False)

    def _open_file(self):
        p = filedialog.askopenfilename(filetypes=[("Text File", "*.txt"), ("All Files", "*.*")])
        if p:
            self._load_edx_file(p)

    def _on_file_drop(self, event):
        paths = self.tk.splitlist(event.data)
        if not paths:
            return
        self._load_edx_file(paths[0])

    def _load_edx_file(self, path):
        try:
            df_raw = parse_edx_file(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load EDX file:\n{e}")
            self._set_status(f"Failed to load {os.path.basename(path)}.")
            return
        self.df_raw = df_raw
        self.elements = get_elements(self.df_raw)
        self.df_norm = self.df_raw.copy()
        self.is_normalized = False
        self.detected_zones = []
        self.manual_zones = []
        self.filepath_var.set(os.path.basename(path))
        self._build_element_rows()
        self._autoscale_graph()
        self._init_style_baseline()
        self._set_status(f"Loaded {os.path.basename(path)} — {len(self.df_raw)} points, {len(self.elements)} elements.")

    def _apply_normalization(self):
        if self.df_raw is None:
            return
        act = [el for el in self.elements if self.el_vars[el].get()]
        self.df_norm = normalize_to_100(self.df_raw, self.elements, act)
        self.is_normalized = True
        self._autoscale_graph()
        self._set_status("Data normalized to 100%.")

    def _reset_to_raw(self):
        if self.df_raw is None:
            return
        self.df_norm = self.df_raw.copy()
        self.is_normalized = False
        self._autoscale_graph()
        self._set_status("Back to raw intensities.")

    def _autoscale_graph(self):
        if self.df_norm is None:
            return
        self.scale_x.set(1.0)
        self.scale_y.set(1.0)
        self._plot(force_reset=True)

    def _apply_smoothing(self, y_values):
        w = self.smooth_window.get()
        if w > 1:
            return pd.Series(y_values).rolling(window=w, center=True, min_periods=1).mean().values
        return y_values

    # ─────────────────────────────────────────────────────────────────────
    #  Phase Detection Logic & PDF Report Generation
    # ─────────────────────────────────────────────────────────────────────
    def _open_phase_editor(self):
        if self.df_norm is None:
            messagebox.showwarning("File Missing", "Please load an EDX working file first.")
            return
        PhaseManagerWindow(self, self.elements, self.phase_presets, self._callback_update_phases)

    def _callback_update_phases(self, updated_presets):
        self.phase_presets = updated_presets
        messagebox.showinfo("Update", f"Presets applied ({len(self.phase_presets)} active phase(s)).")

    def _eval_phase_at_row(self, row_data):
        for phase in self.phase_presets:
            match = True
            for el, lims in phase.get("conditions", {}).items():
                v = row_data.get(el, 0.0)
                if "min" in lims and v < lims["min"]:
                    match = False
                    break
                if "max" in lims and v > lims["max"]:
                    match = False
                    break
            if match:
                return phase["nom_phase"]
        return "Undetermined Zone"

    def _generate_phase_report(self):
        if self.df_norm is None:
            return
        if not self.phase_presets:
            messagebox.showinfo("Presets Missing", "Open the editor to configure your custom phase conditions.")
            return

        pos_col = get_pos_col(self.df_norm)
        positions = self.df_norm[pos_col].values

        self.detected_zones = []
        current_phase = None
        start_pos = positions[0]

        for idx, row in self.df_norm.iterrows():
            smoothed_row = {}
            for el in self.elements:
                smoothed_row[el] = self._apply_smoothing(self.df_norm[el].values)[idx]

            matched = self._eval_phase_at_row(smoothed_row)

            if current_phase is None:
                current_phase = matched
                start_pos = row[pos_col]
            elif matched != current_phase:
                self.detected_zones.append({"start": start_pos, "end": row[pos_col], "name": current_phase})
                current_phase = matched
                start_pos = row[pos_col]

        self.detected_zones.append({"start": start_pos, "end": positions[-1], "name": current_phase})

        # Redraw the plot to show the vertical lines and titles safely
        self._plot()
        self._set_status(f"Identified {len(self.detected_zones)} zone(s).")

        # Show quick textual summary
        report_lines = [f" ◼ [{z['start']:.2f} to {z['end']:.2f} µm]  ➡  {z['name']}" for z in self.detected_zones]
        rep_win = ctk.CTkToplevel(self)
        rep_win.title("Structural Segmentation Summary")
        rep_win.geometry("550x450")

        ctk.CTkLabel(rep_win, text="Spatial Chemical Profiling", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=10)
        txt_box = tk.Text(rep_win, bg="#282C34", fg="white", font=("Consolas", 11), wrap="word", padx=10, pady=10)
        txt_box.pack(fill="both", expand=True, padx=15, pady=10)

        txt_box.insert("1.0", f"File: {self.filepath_var.get()}\n" + "=" * 40 + "\n\n" + "\n".join(report_lines))
        txt_box.configure(state="disabled")

    def _clear_auto_zones(self):
        self.detected_zones = []
        self._plot()

    # ─────────────────────────────────────────────────────────────────────
    #  Manual Zones
    # ─────────────────────────────────────────────────────────────────────
    def _activate_manual_zone_tool(self):
        if self.df_norm is None:
            messagebox.showwarning("File Missing", "Please load an EDX working file first.")
            return
        messagebox.showinfo("Manual Zone", "Drag on the graph to select the approximate zone extent — you'll be able to fine-tune the exact bounds next.")

        def onselect(xmin, xmax):
            ManualZoneDialog(self, xmin, xmax, on_confirm=self._add_manual_zone)

        self._set_span_selector(onselect)

    def _add_manual_zone(self, zone):
        self.manual_zones.append(zone)
        self._plot()
        self._set_status(f"Manual zone '{zone['name']}' added.")

    def _open_manual_zone_manager(self):
        if not self.manual_zones:
            messagebox.showinfo("No Zones", "No manual zones yet. Use 'Add Manual Zone' first.")
            return
        ManualZoneManagerWindow(self, self.manual_zones, self._callback_update_manual_zones)

    def _callback_update_manual_zones(self, updated_zones):
        self.manual_zones = updated_zones
        self._plot()
        self._set_status(f"Manual zones updated ({len(self.manual_zones)}).")

    def _clear_manual_zones(self):
        self.manual_zones = []
        self._plot()
        self._set_status("Manual zones cleared.")

    def _export_pdf_report(self):
        if self.df_norm is None:
            return

        def on_confirm(meta):
            path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Report", "*.pdf")])
            if not path:
                return
            try:
                generate_pdf_report(self, path, meta)
                self._set_status(f"PDF report saved to {os.path.basename(path)}.")
                messagebox.showinfo("Success", "Comprehensive PDF Report successfully exported.")
            except Exception as e:
                self._set_status("PDF export failed.")
                messagebox.showerror("Export Error", f"Failed to build PDF:\n{str(e)}")

        ReportInfoDialog(self, on_confirm, has_image=(self.sem_image_array is not None))

    def _save_fig(self):
        if self.df_norm is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF Plot Only", "*.pdf")])
        if p:
            self.fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=self.fig_bg_color.get())
            self._set_status(f"Image saved to {os.path.basename(p)}.")

    def _export_csv(self):
        if self.df_norm is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.df_norm.to_csv(p, index=False, sep=";", float_format="%.4f")
            self._set_status(f"Data exported to {os.path.basename(p)}.")

    def _export_excel(self):
        if self.df_norm is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            try:
                with pd.ExcelWriter(p, engine="openpyxl") as w:
                    self.df_norm.to_excel(w, sheet_name="Processed", index=False)
                    self.df_raw.to_excel(w, sheet_name="Raw", index=False)
                self._set_status(f"Data exported to {os.path.basename(p)}.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ─────────────────────────────────────────────────────────────────────
    #  Figure Rendering & Matplotlib Events
    # ─────────────────────────────────────────────────────────────────────
    def _plot(self, force_reset=False):
        if self.df_norm is None:
            return
        ox, oy = self.ax.get_xlim() if self.ax.get_lines() and not force_reset else None, self.ax.get_ylim() if self.ax.get_lines() and not force_reset else None

        self.ax.clear()
        self.crosshair_v = self.ax.axvline(0, color='gray', linestyle='--', visible=False, zorder=10)
        self.crosshair_text = self.ax.text(0, 0, '', bbox=dict(facecolor=BG, edgecolor=ACCENT, alpha=0.9), color="white", visible=False, zorder=10)
        self._style_axes()

        x = self.df_raw[get_pos_col(self.df_raw)].values
        lw, ms, fs = self.line_width.get(), self.marker_size.get(), self.font_size.get()
        zone_text_color = contrast_text_color(self.plot_bg_color.get())

        # Draw the phase zones over the background, but behind the curves
        if self.detected_zones:
            for zone in self.detected_zones:
                self.ax.axvline(x=zone["start"], color="#E69F00", linestyle=":", linewidth=1.5, zorder=1)

                # Safe text placement (doesn't touch the curves nor the title)
                mid_x = (zone["start"] + zone["end"]) / 2
                self.ax.text(mid_x, 1.02, zone["name"], transform=self.ax.get_xaxis_transform(),
                             ha="center", va="bottom", fontsize=max(8, fs - 1), color=zone_text_color,
                             bbox=dict(facecolor=self.plot_bg_color.get(), edgecolor='none', alpha=0.7),
                             clip_on=False, zorder=5)

        # Manual zones use a shaded span + in-plot rotated label so they stay
        # visually distinct from the auto-detected zones' floating labels.
        if self.manual_zones:
            for zone in self.manual_zones:
                zcolor = zone.get("color", "#CC79A7")
                self.ax.axvspan(zone["start"], zone["end"], color=zcolor, alpha=0.12, zorder=0)
                self.ax.axvline(x=zone["start"], color=zcolor, linestyle="-", linewidth=1.3, zorder=2)
                self.ax.axvline(x=zone["end"], color=zcolor, linestyle="-", linewidth=1.3, zorder=2)
                mid_x = (zone["start"] + zone["end"]) / 2
                self.ax.text(mid_x, 0.96, zone["name"], transform=self.ax.get_xaxis_transform(),
                             ha="center", va="top", rotation=90, fontsize=max(8, fs - 1), color=zcolor,
                             fontweight="bold", clip_on=True, zorder=6)

        for el in self.elements:
            if not self.el_vars.get(el, ctk.BooleanVar(value=True)).get():
                continue
            y = self._apply_smoothing(self.df_norm[el].values)
            col = self.el_colors.get(el, "#FFF")
            mk = MARKERS.get(self.el_markers.get(el).get(), "o")
            if mk == "None":
                mk = None
            self.ax.plot(x, y, color=col, linewidth=lw, marker=mk, markersize=ms if mk else 0, label=el, zorder=3)

        xcen, hs = (x.min() + x.max()) / 2, ((x.max() - x.min()) / 2) / self.scale_x.get()
        ymax = max([self.df_norm[el].max() for el in self.elements if self.el_vars[el].get()] + [1])

        if not force_reset and ox and not self._slider_interacting:
            self.ax.set_xlim(ox)
            self.ax.set_ylim(oy)
        else:
            self.ax.set_xlim([xcen - hs, xcen + hs])
            self.ax.set_ylim([0, (ymax * 1.05) / self.scale_y.get()])

        if any(v.get() for v in self.el_vars.values()):
            pos = LEGEND_POSITIONS.get(self.legend_pos_var.get(), "outside right")
            args = {"frameon": True, "facecolor": self.plot_bg_color.get(), "labelcolor": contrast_text_color(self.plot_bg_color.get())}
            if pos == "outside right":
                self.ax.legend(**args, loc="upper left", bbox_to_anchor=(1.02, 1))
            else:
                self.ax.legend(**args, loc=pos)

        self.ax.xaxis.set_major_locator(ticker.AutoLocator())
        self.ax.yaxis.set_major_locator(ticker.AutoLocator())
        self.fig.tight_layout()
        self.canvas.draw()

    # --- ROI and Events ---
    def _set_span_selector(self, onselect):
        """(Re)activate the graph's drag-select tool. Only one drag tool
        (ROI stats vs. manual zone placement) can usefully be active at a
        time, so any previous selector is disconnected first."""
        if self.roi_selector is not None:
            try:
                self.roi_selector.disconnect_events()
            except Exception:
                pass
        self.roi_selector = SpanSelector(self.ax, onselect, 'horizontal', useblit=True, props=dict(alpha=0.3, facecolor=ACCENT))

    def _activate_roi(self):
        if self.df_norm is None:
            messagebox.showwarning("File Missing", "Please load an EDX working file first.")
            return
        messagebox.showinfo("Mode ROI", "Drag on the graph to select an X region.")

        def onselect(xmin, xmax):
            pos_col = get_pos_col(self.df_norm)
            mask = (self.df_norm[pos_col] >= xmin) & (self.df_norm[pos_col] <= xmax)
            sub = self.df_norm[mask]
            stats = f"Zone : {xmin:.2f} à {xmax:.2f}\n\n"
            for el in self.elements:
                if self.el_vars[el].get():
                    stats += f"• {el}: Avg={sub[el].mean():.2f} | Max={sub[el].max():.2f}\n"
            messagebox.showinfo("ROI Stats", stats)

        self._set_span_selector(onselect)

    def _on_mpl_motion(self, event):
        if self._pan_start and event.x and event.y:
            sx, sy, sxl, syl = self._pan_start
            inv = self.ax.transData.inverted()
            ps, pc = inv.transform((sx, sy)), inv.transform((event.x, event.y))
            self.ax.set_xlim([sxl[0] - (pc[0] - ps[0]), sxl[1] - (pc[0] - ps[0])])
            self.ax.set_ylim([syl[0] - (pc[1] - ps[1]), syl[1] - (pc[1] - ps[1])])
            self.canvas.draw()
            return

        if self.enable_crosshair.get() and event.inaxes and self.df_norm is not None:
            x_val = event.xdata
            pos_col = get_pos_col(self.df_norm)
            idx = (np.abs(self.df_raw[pos_col].values - x_val)).argmin()
            real_x = self.df_raw[pos_col].values[idx]

            txt = f"X: {real_x:.2f}\n"
            for el in self.elements:
                if self.el_vars[el].get():
                    txt += f"{el}: {self.df_norm[el].values[idx]:.1f}\n"

            self.crosshair_v.set_xdata([real_x])
            self.crosshair_v.set_visible(True)
            self.crosshair_text.set_text(txt.strip())
            self.crosshair_text.set_position((real_x, event.ydata))
            self.crosshair_text.set_visible(True)
            self.canvas.draw_idle()
        elif self.crosshair_v.get_visible():
            self.crosshair_v.set_visible(False)
            self.crosshair_text.set_visible(False)
            self.canvas.draw_idle()

    def _on_mpl_scroll(self, event):
        if self.df_norm is None or event.inaxes is None:
            return
        sf = 1.0 / 1.15 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim([cx - (cx - self.ax.get_xlim()[0]) * sf, cx + (self.ax.get_xlim()[1] - cx) * sf])
        self.ax.set_ylim([cy - (cy - self.ax.get_ylim()[0]) * sf, cy + (self.ax.get_ylim()[1] - cy) * sf])
        self.canvas.draw()

    def _on_mpl_button_press(self, event):
        if event.dblclick and event.button == 1:
            bbox = self.fig.bbox
            if event.y > bbox.height * 0.9:
                t = simpledialog.askstring("Title", "New title :", initialvalue=self.graph_title)
                if t is not None:
                    self.graph_title = t
                    self._plot()
            elif event.y < bbox.height * 0.1:
                x = simpledialog.askstring("X-Axis", "X-axis label :", initialvalue=self.graph_xlabel)
                if x is not None:
                    self.graph_xlabel = x
                    self._plot()
            elif event.x < bbox.width * 0.1:
                y = simpledialog.askstring("Y-Axis", "Y-axis label :", initialvalue=self.graph_ylabel)
                if y is not None:
                    self.graph_ylabel = y
                    self._plot()
            elif event.inaxes:
                self._autoscale_graph()
        elif event.button in (2, 3) and event.inaxes:
            self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
