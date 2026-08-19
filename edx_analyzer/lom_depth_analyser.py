"""
LOM Depth Analyser  --  "LOM Depth Analyser" module of the analysis suite
-------------------------------------------------------------------------
Loads one or several LOM measurement CSV files ("No.";"Measure";"Result";"Unit"),
sorts / filters the depth values, computes the statistics (min, max, mean,
median, std, quartiles) and draws a distribution curve of the depths versus
their occurrence, using a user defined grouping factor (bin width, e.g. 0.05).

Several CSV files can be gathered into a "group" (= one experiment) and several
groups can be plotted on the same graph (e.g. 3 files for experiment A and
4 files for experiment B).

Everything can be exported: data (CSV / Excel), graph (PNG / SVG / PDF) and a
complete PDF report.

The UI is packaged as `LOMDepthAnalyserPanel`, a plain CTkFrame, so it can be
embedded in the suite's tab bar next to the EDX and SEM modules.
`LOMDepthAnalyserWindow` wraps the very same panel in its own window, and the
module stays runnable alone:   python -m edx_analyzer.lom_depth_analyser
"""

import os

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .constants import COLORS_DEFAULT, BACKGROUND_PRESETS, LEGEND_POSITIONS, ACCENT
from .color_utils import contrast_text_color
from .legend_utils import apply_legend, fit_layout
from .widgets import ColorPickerDialog, add_tooltip
from .lom_depth_core import (
    DepthGroup,
    histogram_error,
    normal_curve,
    LomParseError,
    bin_centers,
    build_histogram_frame,
    build_summary_frame,
    build_values_frame,
    compute_stats,
    histogram,
    make_bin_edges,
    parse_number,
    smooth_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Look & feel (shared with the rest of the suite, see constants.py)
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG = BACKGROUND_PRESETS["Dark (default)"]
GROUP_COLORS = COLORS_DEFAULT

Y_MODES = {"Count (occurrences)": "count", "Percentage (%)": "percent", "Density": "density"}
CURVE_STYLES = ["Bars", "Line", "Bars + Line", "Step"]
MAX_BINS = 5000                      # safety limit for a very small grouping factor
ORIGIN_MODES = ["Auto (min of data)", "Zero", "Custom"]


class LOMDepthAnalyserPanel(ctk.CTkFrame):
    """The whole 'LOM Depth Analyser' module, as an embeddable frame."""

    def __init__(self, master, status_callback=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._status = status_callback if callable(status_callback) else (lambda _m: None)

        # ---- data ------------------------------------------------------
        self.groups = []                 # list[DepthGroup]
        self.measure_vars = {}           # measure label -> BooleanVar
        self._group_name_entries = {}    # id(group) -> CTkEntry

        # ---- filters ---------------------------------------------------
        self.min_var = tk.StringVar(value="")
        self.max_var = tk.StringVar(value="")

        # ---- distribution ---------------------------------------------
        self.bin_width_var = tk.StringVar(value="0.05")
        self.origin_mode_var = tk.StringVar(value=ORIGIN_MODES[0])
        self.origin_value_var = tk.StringVar(value="0")
        self.ymode_var = tk.StringVar(value=list(Y_MODES.keys())[0])
        self.style_var = tk.StringVar(value="Bars + Line")
        self.smooth_var = ctk.IntVar(value=1)
        self.show_stat_lines = ctk.BooleanVar(value=False)
        self.show_error_bars = ctk.BooleanVar(value=False)
        self.show_normal_fit = ctk.BooleanVar(value=False)
        self.show_sigma_bands = ctk.BooleanVar(value=False)
        self._bin_warning_for = None

        # ---- design ----------------------------------------------------
        self.graph_title = tk.StringVar(value="LOM — Depth Distribution")
        self.graph_xlabel = tk.StringVar(value="Depth (µm)")
        self.graph_ylabel = tk.StringVar(value="Number of measurements")
        self.bar_alpha = ctk.DoubleVar(value=0.55)
        self.line_width = ctk.DoubleVar(value=1.8)
        self.font_size = ctk.IntVar(value=10)
        self.show_grid = ctk.BooleanVar(value=True)
        self.show_markers = ctk.BooleanVar(value=False)
        self.legend_pos_var = tk.StringVar(value="Outside Right")
        self.light_export = ctk.BooleanVar(value=True)
        self.plot_bg_color = tk.StringVar(value=PLOT_BG)
        self.fig_bg_color = tk.StringVar(value=BACKGROUND_PRESETS["Figure Dark"])
        self.bg_preset_var = tk.StringVar(value="Dark (default)")

        self._build_ui()
        self._refresh_all()

    def on_shown(self):
        """Called when the module becomes visible: a matplotlib canvas built
        inside a hidden tab is laid out 0x0, so force a real redraw."""
        self.update_idletasks()
        self._plot()

    # ─────────────────────────────────────────────────────────────────────
    #  UI construction
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
        graph_frame = ctk.CTkFrame(center)
        graph_frame.pack(fill="both", expand=True)

        self.right_panel = ctk.CTkScrollableFrame(self, width=370, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._build_left(self.left_panel)
        self._build_graph(graph_frame)
        self._build_right(self.right_panel)

    # -- left panel ------------------------------------------------------
    def _build_left(self, p):
        ctk.CTkLabel(p, text="LOM Depth Analyser",
                     font=("Helvetica", 20, "bold")).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(p, text='CSV format : "No.";"Measure";"Result";"Unit"',
                     font=("Helvetica", 11), text_color="gray").pack(anchor="w")

        self._section(p, "GROUPS / EXPERIMENTS")
        btn_new = ctk.CTkButton(p, text="➕ New group (load CSV files)", command=self._new_group)
        btn_new.pack(fill="x", pady=2)
        add_tooltip(btn_new, "One group = one experiment.\nSelect every CSV file of that "
                             "experiment at once;\nthey are pooled into a single curve.")
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkButton(row, text="🎨 Auto colors", width=120, fg_color="gray40",
                      command=self._reset_colors).pack(side="left", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="🗑 Clear all", width=120, fg_color="gray40",
                      hover_color="#7a2222", command=self._clear_all).pack(side="right", expand=True)

        self.groups_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.groups_frame.pack(fill="x", pady=5)

        self._section(p, "MEASURE TYPES")
        ctk.CTkLabel(p, text="Keep only the measurement types below:",
                     font=("Helvetica", 11), text_color="gray").pack(anchor="w")
        self.measures_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.measures_frame.pack(fill="x", pady=3)

        self._section(p, "VALUE FILTERS (min / max)")
        ctk.CTkLabel(p, text="Values kept :  min ≤ value ≤ max   (leave empty = no limit)",
                     font=("Helvetica", 11), text_color="gray", wraplength=300).pack(anchor="w")
        f = ctk.CTkFrame(p, fg_color="transparent")
        f.pack(fill="x", pady=5)
        ctk.CTkLabel(f, text="Min :", width=40).pack(side="left")
        e_min = ctk.CTkEntry(f, textvariable=self.min_var, width=80, placeholder_text="none")
        e_min.pack(side="left", padx=(0, 10))
        add_tooltip(e_min, "Values strictly below this limit are excluded\n"
                           "(useful to drop near-zero measurement noise).")
        ctk.CTkLabel(f, text="Max :", width=40).pack(side="left")
        e_max = ctk.CTkEntry(f, textvariable=self.max_var, width=80, placeholder_text="none")
        e_max.pack(side="left")
        for e in (e_min, e_max):
            e.bind("<Return>", lambda _e: self._refresh_all())
        row2 = ctk.CTkFrame(p, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkButton(row2, text="✔ Apply filters", command=self._refresh_all,
                      width=130).pack(side="left", expand=True, padx=(0, 4))
        ctk.CTkButton(row2, text="✖ Reset", width=90, fg_color="gray40",
                      command=self._reset_filters).pack(side="right", expand=True)
        self.filter_info = ctk.CTkLabel(p, text="", font=("Helvetica", 11),
                                        text_color="gray", wraplength=300, justify="left")
        self.filter_info.pack(anchor="w", pady=(4, 0))

        self._section(p, "DISTRIBUTION SETTINGS")
        f2 = ctk.CTkFrame(p, fg_color="transparent")
        f2.pack(fill="x", pady=3)
        ctk.CTkLabel(f2, text="Grouping factor (bin width) :").pack(side="left")
        e_bin = ctk.CTkEntry(f2, textvariable=self.bin_width_var, width=70)
        e_bin.pack(side="right")
        e_bin.bind("<Return>", lambda _e: self._refresh_all())
        add_tooltip(e_bin, "Width of one column of the distribution.\nClose depths fall in the "
                           "same column\n(0.05 groups 5.54 and 5.55 together).")
        ctk.CTkLabel(p, text="e.g. 0.05 → 5.54 and 5.55 fall in the same column",
                     font=("Helvetica", 11), text_color="gray", wraplength=300).pack(anchor="w")
        ctk.CTkButton(p, text="🎯 Suggest bin width", fg_color="gray40",
                      command=self._suggest_bin_width).pack(fill="x", pady=3)

        f3 = ctk.CTkFrame(p, fg_color="transparent")
        f3.pack(fill="x", pady=3)
        ctk.CTkLabel(f3, text="Bins aligned on :").pack(side="left")
        self.origin_entry = ctk.CTkEntry(f3, textvariable=self.origin_value_var, width=70)
        self.origin_entry.pack(side="right")
        add_tooltip(self.origin_entry, "Reference value the column grid starts from\n"
                                       "(used when 'Custom' is selected).")
        self.origin_entry.bind("<Return>", lambda _e: self._refresh_all())
        ctk.CTkComboBox(p, variable=self.origin_mode_var, values=ORIGIN_MODES,
                        command=lambda _v: self._refresh_all()).pack(fill="x", pady=3)

        ctk.CTkLabel(p, text="Y axis :").pack(anchor="w", pady=(8, 0))
        ctk.CTkComboBox(p, variable=self.ymode_var, values=list(Y_MODES.keys()),
                        command=self._on_ymode_change).pack(fill="x", pady=2)
        ctk.CTkLabel(p, text="Curve style :").pack(anchor="w", pady=(8, 0))
        ctk.CTkComboBox(p, variable=self.style_var, values=CURVE_STYLES,
                        command=lambda _v: self._plot()).pack(fill="x", pady=2)
        self._slider(p, "Curve smoothing (bins) :", self.smooth_var, 1, 15)
        ctk.CTkCheckBox(p, text="Show mean / median lines", variable=self.show_stat_lines,
                        command=self._plot).pack(anchor="w", pady=6)

        self._section(p, "STATISTICS ON THE GRAPH")
        err_chk = ctk.CTkCheckBox(p, text="Probability error bars (±√n)",
                                  variable=self.show_error_bars, command=self._plot)
        err_chk.pack(anchor="w", pady=3)
        add_tooltip(err_chk, "Counting uncertainty of each column: a column holding\n"
                             "n measurements is known to about ±√n. Tells a real peak\n"
                             "from one built on three measurements.")
        fit_chk = ctk.CTkCheckBox(p, text="Normal law fit", variable=self.show_normal_fit,
                                  command=self._plot)
        fit_chk.pack(anchor="w", pady=3)
        add_tooltip(fit_chk, "Draws the normal law of same mean and same standard\n"
                             "deviation as the data, scaled to the histogram.")
        sig_chk = ctk.CTkCheckBox(p, text="Standard deviation bands (±1σ, ±2σ)",
                                  variable=self.show_sigma_bands, command=self._plot)
        sig_chk.pack(anchor="w", pady=3)
        add_tooltip(sig_chk, "Shades mean ±1σ and marks mean ±2σ: about 68 % and 95 %\n"
                             "of a normal population fall inside them.")

        ctk.CTkButton(p, text="↻ Update graph & statistics", command=self._refresh_all,
                      fg_color="#009E73", hover_color="#007755").pack(fill="x", pady=(10, 15))

    # -- graph -----------------------------------------------------------
    def _build_graph(self, frame):
        self.fig = Figure(figsize=(8.5, 6), facecolor=self.fig_bg_color.get())
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = ctk.CTkFrame(frame, fg_color="transparent")
        tb.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tb).update()

    # -- right panel -----------------------------------------------------
    def _build_right(self, p):
        self._section(p, "STATISTICS")
        self.stats_box = ctk.CTkTextbox(p, height=330, font=("Consolas", 11), wrap="none")
        self.stats_box.pack(fill="both", expand=True, pady=5)

        self._section(p, "GRAPH LABELS")
        for label, var in (("Title :", self.graph_title),
                           ("X axis :", self.graph_xlabel),
                           ("Y axis :", self.graph_ylabel)):
            r = ctk.CTkFrame(p, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=label, width=60, anchor="w").pack(side="left")
            e = ctk.CTkEntry(r, textvariable=var)
            e.pack(side="right", fill="x", expand=True)
            e.bind("<Return>", lambda _e: self._plot())
        ctk.CTkButton(p, text="✔ Apply labels", command=self._plot,
                      fg_color="gray40").pack(fill="x", pady=3)

        self._section(p, "DESIGN & LEGEND")
        ctk.CTkLabel(p, text="Legend position :").pack(anchor="w")
        ctk.CTkComboBox(p, variable=self.legend_pos_var, values=list(LEGEND_POSITIONS.keys()),
                        command=lambda _v: self._plot()).pack(fill="x", pady=5)
        ctk.CTkLabel(p, text="Graph background :").pack(anchor="w")
        ctk.CTkComboBox(p, variable=self.bg_preset_var, values=list(BACKGROUND_PRESETS.keys()),
                        command=self._apply_bg_preset).pack(fill="x", pady=5)
        self._slider(p, "Bar opacity :", self.bar_alpha, 0.1, 1.0)
        self._slider(p, "Line thickness :", self.line_width, 0.5, 5.0)
        self._slider(p, "Font size :", self.font_size, 7, 18)
        ctk.CTkCheckBox(p, text="Show grid", variable=self.show_grid,
                        command=self._plot).pack(anchor="w", pady=3)
        ctk.CTkCheckBox(p, text="Show points on curve", variable=self.show_markers,
                        command=self._plot).pack(anchor="w", pady=3)

        self._section(p, "EXPORTS")
        ctk.CTkCheckBox(p, text="White background for exported graph",
                        variable=self.light_export).pack(anchor="w", pady=3)
        ctk.CTkButton(p, text="🖼 Save graph image (PNG/SVG/PDF)",
                      command=self._export_image).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📄 Export data (CSV)",
                      command=self._export_csv).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📊 Export data (Excel)",
                      command=self._export_excel).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📑 Generate PDF report", fg_color="#882255",
                      hover_color="#551133", command=self._export_pdf).pack(fill="x", pady=(10, 15))

    def _apply_bg_preset(self, name):
        hexcode = BACKGROUND_PRESETS.get(name, PLOT_BG)
        self.plot_bg_color.set(hexcode)
        self.fig_bg_color.set(BACKGROUND_PRESETS["Figure Dark"] if name == "Dark (default)" else hexcode)
        self._plot()

    def _slider(self, parent, label, var, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        val = ctk.CTkLabel(row, text=f"{var.get():.2f}", text_color=ACCENT)
        val.pack(side="right")

        def _upd(v):
            val.configure(text=f"{int(float(v))}" if isinstance(var, ctk.IntVar) else f"{float(v):.2f}")
            self._plot()

        steps = int(to - from_) if isinstance(var, ctk.IntVar) else None
        ctk.CTkSlider(parent, from_=from_, to=to, variable=var,
                      number_of_steps=steps, command=_upd).pack(fill="x")

    # ─────────────────────────────────────────────────────────────────────
    #  Groups management
    # ─────────────────────────────────────────────────────────────────────
    def _ask_files(self):
        return filedialog.askopenfilenames(
            parent=self, title="Select LOM measurement CSV file(s)",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])

    def _load_into(self, group, paths):
        added, errors, skipped = 0, [], []
        for p in paths:
            if group.has_file(p):
                skipped.append(os.path.basename(p))
                continue
            try:
                n = group.add_file(p)
                added += n
            except LomParseError as exc:
                errors.append(f"• {os.path.basename(p)} : {exc}")
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"• {os.path.basename(p)} : {exc}")
        msg = ""
        if skipped:
            msg += "Already loaded (ignored):\n" + "\n".join(skipped) + "\n\n"
        if errors:
            msg += "Files that could not be read:\n" + "\n".join(errors)
        if msg:
            messagebox.showwarning("Loading report", msg, parent=self)
        return added

    def _new_group(self):
        paths = self._ask_files()
        if not paths:
            return
        default = f"Experiment {len(self.groups) + 1}"
        name = simpledialog.askstring("New group",
                                      "Name of this group / experiment :",
                                      initialvalue=default, parent=self)
        if name is None:
            return
        name = name.strip() or default
        color = GROUP_COLORS[len(self.groups) % len(GROUP_COLORS)]
        group = DepthGroup(name, color)
        if self._load_into(group, paths) == 0 and not group.files:
            return
        self.groups.append(group)
        self._refresh_all()

    def add_measurement_group(self, name, values, unit: str = "µm",
                              measure: str = "Image thickness"):
        """
        Public entry point used by the Image Zone Analyser: build a group out of
        values measured elsewhere, with no CSV file involved.
        """
        color = GROUP_COLORS[len(self.groups) % len(GROUP_COLORS)]
        group = DepthGroup(name, color)
        if group.add_values(values, name, unit=unit, measure=measure) == 0:
            return None
        self.groups.append(group)
        self._refresh_all()
        return group

    def _add_files(self, group):
        paths = self._ask_files()
        if paths:
            self._load_into(group, paths)
            self._refresh_all()

    def _remove_group(self, group):
        if messagebox.askyesno("Remove group", f"Remove '{group.name}' ?", parent=self):
            self.groups = [g for g in self.groups if g is not group]
            self._refresh_all()

    def _remove_file(self, group, path):
        group.remove_file(path)
        self._refresh_all()

    def _clear_all(self):
        if not self.groups:
            return
        if messagebox.askyesno("Clear all", "Remove every loaded group ?", parent=self):
            self.groups = []
            self.measure_vars = {}
            self._refresh_all()

    def _reset_colors(self):
        for i, g in enumerate(self.groups):
            g.color = GROUP_COLORS[i % len(GROUP_COLORS)]
        self._refresh_all()

    def _pick_color(self, group, button):
        col = ColorPickerDialog.ask_color(self.winfo_toplevel(), initial_color=group.color,
                                          title=f"Color of '{group.name}'")
        if col:
            group.color = col
            button.configure(fg_color=col)
            self._plot()

    def _rename(self, group, entry):
        try:
            new = entry.get().strip()
        except Exception:                                 # noqa: BLE001 - widget destroyed
            return
        if not new or new == group.name:
            return
        group.name = new
        self.after(60, self._refresh_all)                 # deferred: we are in a widget event

    def _refresh_groups(self):
        for w in self.groups_frame.winfo_children():
            w.destroy()
        self._group_name_entries.clear()

        if not self.groups:
            ctk.CTkLabel(self.groups_frame, text="No group loaded yet.",
                         text_color="gray", font=("Helvetica", 11)).pack(anchor="w", pady=5)
            return

        measures, vmin, vmax = self._current_filters()
        for g in self.groups:
            card = ctk.CTkFrame(self.groups_frame, corner_radius=6)
            card.pack(fill="x", pady=4)

            head = ctk.CTkFrame(card, fg_color="transparent")
            head.pack(fill="x", padx=6, pady=(6, 2))

            vis = ctk.BooleanVar(value=g.visible)
            ctk.CTkCheckBox(head, text="", width=24, variable=vis,
                            command=lambda gr=g, v=vis: (setattr(gr, "visible", v.get()),
                                                         self._refresh_all())).pack(side="left")
            btn = ctk.CTkButton(head, text="", width=22, height=22, fg_color=g.color)
            btn.configure(command=lambda gr=g, b=btn: self._pick_color(gr, b))
            btn.pack(side="left", padx=4)

            ent = ctk.CTkEntry(head, width=120)
            ent.insert(0, g.name)
            ent.pack(side="left", fill="x", expand=True, padx=4)
            ent.bind("<Return>", lambda _e, gr=g, en=ent: self._rename(gr, en))
            ent.bind("<FocusOut>", lambda _e, gr=g, en=ent: self._rename(gr, en))
            self._group_name_entries[id(g)] = ent

            ctk.CTkButton(head, text="✕", width=26, fg_color="gray35",
                          hover_color="#7a2222",
                          command=lambda gr=g: self._remove_group(gr)).pack(side="right")

            kept, total, excluded = g.values(measures, vmin, vmax)
            info = (f"{len(g.files)} file(s) · {total} value(s) · "
                    f"{len(kept)} kept · {excluded} excluded")
            ctk.CTkLabel(card, text=info, font=("Helvetica", 10), text_color="gray",
                         anchor="w").pack(fill="x", padx=10)

            for f in g.files:
                fr = ctk.CTkFrame(card, fg_color="transparent")
                fr.pack(fill="x", padx=10)
                ctk.CTkLabel(fr, text=f"• {f['name']}  ({len(f['df'])})",
                             font=("Helvetica", 10), text_color="#9aa0a6",
                             anchor="w", wraplength=210, justify="left").pack(side="left")
                ctk.CTkButton(fr, text="✕", width=20, height=18, fg_color="transparent",
                              hover_color="#7a2222", text_color="gray",
                              command=lambda gr=g, p=f["path"]: self._remove_file(gr, p)
                              ).pack(side="right")

            ctk.CTkButton(card, text="＋ Add CSV file(s) to this group", height=24,
                          fg_color="gray30", command=lambda gr=g: self._add_files(gr)
                          ).pack(fill="x", padx=8, pady=6)

    def _refresh_measures(self):
        labels = sorted({m for g in self.groups for m in g.measures()})
        for lab in labels:
            self.measure_vars.setdefault(lab, ctk.BooleanVar(value=True))
        for lab in list(self.measure_vars):
            if lab not in labels:
                del self.measure_vars[lab]

        for w in self.measures_frame.winfo_children():
            w.destroy()
        if not labels:
            ctk.CTkLabel(self.measures_frame, text="(no data loaded)", text_color="gray",
                         font=("Helvetica", 11)).pack(anchor="w")
            return
        for lab in labels:
            ctk.CTkCheckBox(self.measures_frame, text=lab, variable=self.measure_vars[lab],
                            font=("Helvetica", 11),
                            command=self._refresh_all).pack(anchor="w", pady=1)

    # ─────────────────────────────────────────────────────────────────────
    #  Filters & computation
    # ─────────────────────────────────────────────────────────────────────
    def _reset_filters(self):
        self.min_var.set("")
        self.max_var.set("")
        self._refresh_all()

    def _current_filters(self):
        selected = [m for m, v in self.measure_vars.items() if v.get()]
        measures = None if (not self.measure_vars or len(selected) == len(self.measure_vars)) else selected
        vmin = parse_number(self.min_var.get())
        vmax = parse_number(self.max_var.get())
        if vmin is not None and vmax is not None and vmin > vmax:
            vmin, vmax = vmax, vmin
        return measures, vmin, vmax

    def _bin_width(self):
        w = parse_number(self.bin_width_var.get())
        if w is None or w <= 0:
            return 0.05
        return float(w)

    def _unit(self):
        units = [u for g in self.groups for u in g.units()]
        return max(set(units), key=units.count) if units else "µm"

    def _visible_data(self):
        """[(group, kept_values, n_total, n_excluded)] for visible, non-empty groups."""
        measures, vmin, vmax = self._current_filters()
        out = []
        for g in self.groups:
            if not g.visible:
                continue
            kept, total, excluded = g.values(measures, vmin, vmax)
            if kept.size:
                out.append((g, kept, total, excluded))
        return out

    def _compute_series(self):
        """Shared bin grid + histogram of every visible group. None when nothing to draw."""
        data = self._visible_data()
        if not data:
            return None
        width = requested = self._bin_width()
        gmin = min(float(v.min()) for _, v, _, _ in data)
        gmax = max(float(v.max()) for _, v, _, _ in data)

        span = max(gmax - gmin, 0.0)
        if span / width > MAX_BINS:                       # keep the graph usable
            width = span / MAX_BINS
            if self._bin_warning_for != requested:
                self._bin_warning_for = requested
                messagebox.showwarning(
                    "Grouping factor too small",
                    f"A grouping factor of {requested:g} would create more than {MAX_BINS} "
                    f"columns for this data range.\n\nThe graph uses {width:.4g} instead.",
                    parent=self)

        mode = self.origin_mode_var.get()
        if mode.startswith("Zero"):
            origin = 0.0
        elif mode.startswith("Custom"):
            o = parse_number(self.origin_value_var.get())
            origin = 0.0 if o is None else float(o)
        else:
            origin = gmin

        edges = make_bin_edges(gmin, gmax, width, origin)
        ymode = Y_MODES.get(self.ymode_var.get(), "count")
        series = [(g, v, histogram(v, edges, ymode)) for g, v, _, _ in data]
        return {"edges": edges, "centers": bin_centers(edges), "series": series,
                "mode": ymode, "width": width, "min": gmin, "max": gmax}

    def _suggest_bin_width(self):
        data = self._visible_data()
        if not data:
            messagebox.showinfo("No data", "Load at least one CSV file first.", parent=self)
            return
        allv = np.concatenate([v for _, v, _, _ in data])
        span = float(allv.max() - allv.min())
        if span <= 0:
            w = 0.05
        else:
            raw = span / max(10.0, min(40.0, np.sqrt(allv.size) * 2))
            exp = np.floor(np.log10(raw))
            frac = raw / (10 ** exp)
            nice = 1 if frac < 1.5 else (2 if frac < 3.5 else (5 if frac < 7.5 else 10))
            w = nice * (10 ** exp)
        self.bin_width_var.set(f"{w:g}")
        self._refresh_all()

    def _on_ymode_change(self, _value=None):
        labels = {"count": "Number of measurements", "percent": "Frequency (%)",
                  "density": "Probability density"}
        self.graph_ylabel.set(labels.get(Y_MODES.get(self.ymode_var.get(), "count"), "Count"))
        self._plot()

    # ─────────────────────────────────────────────────────────────────────
    #  Plotting
    # ─────────────────────────────────────────────────────────────────────
    def _draw_distribution(self, ax, fig, light=False):
        face = "white" if light else self.plot_bg_color.get()
        figface = "white" if light else self.fig_bg_color.get()
        tc = "black" if light else contrast_text_color(face)
        fs = self.font_size.get()

        ax.clear()
        ax.set_facecolor(face)
        fig.patch.set_facecolor(figface)
        for s in ax.spines.values():
            s.set_edgecolor("gray")
        ax.tick_params(colors=tc, labelsize=fs)
        ax.xaxis.label.set_color(tc)
        ax.yaxis.label.set_color(tc)
        ax.title.set_color(tc)
        ax.set_title(self.graph_title.get(), fontsize=fs + 2, fontweight="bold", pad=12)
        ax.set_xlabel(self.graph_xlabel.get(), fontsize=fs)
        ax.set_ylabel(self.graph_ylabel.get(), fontsize=fs)
        ax.grid(color="gray", linestyle="--", alpha=0.3) if self.show_grid.get() else ax.grid(False)

        res = self._compute_series()
        if res is None:
            ax.text(0.5, 0.5, "Load one or several LOM CSV files\n(button  + New group)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=fs + 2)
            return

        style = self.style_var.get()
        centers, width = res["centers"], res["width"]
        alpha, lw = self.bar_alpha.get(), self.line_width.get()
        marker = "o" if self.show_markers.get() else None

        mode, width = res["mode"], res["width"]
        for g, values, counts in res["series"]:
            label = f"{g.name} (n={values.size})"
            raw_counts = histogram(values, res["edges"], "count")
            if style in ("Bars", "Bars + Line"):
                errors = histogram_error(raw_counts, mode, values.size, width) \
                    if self.show_error_bars.get() else None
                ax.bar(centers, counts, width=width * 0.92, color=g.color,
                       alpha=alpha, edgecolor=g.color, linewidth=0.5, zorder=2,
                       yerr=errors, ecolor=tc, capsize=2,
                       error_kw={"elinewidth": 0.8, "alpha": 0.7} if errors is not None else None,
                       label=label if style == "Bars" else None)
            elif self.show_error_bars.get():
                errors = histogram_error(raw_counts, mode, values.size, width)
                ax.errorbar(centers, counts, yerr=errors, fmt="none", ecolor=g.color,
                            elinewidth=0.8, capsize=2, alpha=0.7, zorder=2)
            if style in ("Line", "Bars + Line"):
                ax.plot(centers, smooth_curve(counts, self.smooth_var.get()), color=g.color,
                        linewidth=lw, marker=marker, markersize=max(3.0, lw * 2),
                        label=label, zorder=3)
            if style == "Step":
                ax.step(centers, counts, where="mid", color=g.color, linewidth=lw,
                        marker=marker, markersize=max(3.0, lw * 2), label=label, zorder=3)

            if self.show_normal_fit.get():
                fit = normal_curve(values, res["edges"], mode)
                if fit is not None:
                    ax.plot(fit[0], fit[1], color=g.color, linewidth=lw + 0.4,
                            linestyle="-.", zorder=5,
                            label=f"{g.name} — normal fit")

            if self.show_sigma_bands.get():
                st = compute_stats(values)
                if np.isfinite(st["std"]) and st["std"] > 0:
                    ax.axvspan(st["mean"] - st["std"], st["mean"] + st["std"],
                               color=g.color, alpha=0.12, zorder=1)
                    for k in (-2, 2):
                        ax.axvline(st["mean"] + k * st["std"], color=g.color,
                                   linestyle=(0, (1, 3)), linewidth=1.0, zorder=4)

            if self.show_stat_lines.get():
                st = compute_stats(values)
                ax.axvline(st["mean"], color=g.color, linestyle="--", linewidth=1.2, zorder=4)
                ax.axvline(st["median"], color=g.color, linestyle=":", linewidth=1.4, zorder=4)

        apply_legend(ax, self.legend_pos_var.get(), facecolor=face, labelcolor=tc, fontsize=fs)

        hints = []
        if self.show_stat_lines.get():
            hints.append("-- mean   ·· median")
        if self.show_sigma_bands.get():
            hints.append("shaded ±1σ   dotted ±2σ")
        if self.show_error_bars.get():
            hints.append("error bars ±√n")
        if hints:
            ax.text(0.01, 0.99, "   |   ".join(hints), transform=ax.transAxes,
                    va="top", ha="left", fontsize=max(8, fs - 2), color=tc, alpha=0.8)

    def _plot(self):
        self._draw_distribution(self.ax, self.fig, light=False)
        fit_layout(self.fig, self.legend_pos_var.get())
        self.canvas.draw()

    def _make_export_figure(self):
        light = self.light_export.get()
        fig = Figure(figsize=(10, 6.5), facecolor="white" if light else self.fig_bg_color.get())
        ax = fig.add_subplot(111)
        self._draw_distribution(ax, fig, light=light)
        fit_layout(fig, self.legend_pos_var.get())
        return fig

    # ─────────────────────────────────────────────────────────────────────
    #  Statistics panel
    # ─────────────────────────────────────────────────────────────────────
    def _stats_rows(self):
        """[(group, n_files, n_total, n_excluded, stats, unit)] for every group."""
        measures, vmin, vmax = self._current_filters()
        rows = []
        for g in self.groups:
            kept, total, excluded = g.values(measures, vmin, vmax)
            rows.append((g, len(g.files), total, excluded, compute_stats(kept), self._unit()))
        return rows

    def _stats_text(self):
        measures, vmin, vmax = self._current_filters()
        unit = self._unit()
        lines = []
        lines.append("FILTERS")
        lines.append(f"  Min limit      : {'none' if vmin is None else f'{vmin:g} {unit}'}")
        lines.append(f"  Max limit      : {'none' if vmax is None else f'{vmax:g} {unit}'}")
        lines.append(f"  Measure types  : {'all' if measures is None else ', '.join(measures)}")
        lines.append(f"  Grouping factor: {self._bin_width():g} {unit}")
        lines.append("")

        if not self.groups:
            lines.append("No data loaded.")
            return "\n".join(lines)

        for g, n_files, total, excluded, st, _u in self._stats_rows():
            lines.append("=" * 52)
            lines.append(f"GROUP : {g.name}{'' if g.visible else '   (hidden)'}")
            lines.append("-" * 52)
            lines.append(f"  Files          : {n_files}")
            lines.append(f"  Values loaded  : {total}")
            lines.append(f"  Kept / excluded: {st['n']} / {excluded}")
            if st["n"]:
                lines.append(f"  Min            : {st['min']:.4g} {unit}")
                lines.append(f"  Max            : {st['max']:.4g} {unit}")
                lines.append(f"  Mean           : {st['mean']:.4g} {unit}")
                lines.append(f"  Median         : {st['median']:.4g} {unit}")
                lines.append(f"  Std dev (n-1)  : {st['std']:.4g} {unit}")
                lines.append(f"  Variation coef.: {st['cv']:.3g} %")
                lines.append(f"  Std error mean : {st['sem']:.4g} {unit}")
                lines.append(f"  Mean 95% CI    : {st['mean'] - st['ci95']:.4g} … "
                             f"{st['mean'] + st['ci95']:.4g} {unit}")
                lines.append(f"  Q1 / Q3        : {st['q1']:.4g} / {st['q3']:.4g} {unit}")
                lines.append(f"  IQR / Range    : {st['iqr']:.4g} / {st['range']:.4g} {unit}")
            else:
                lines.append("  (no value left after filtering)")
            for f in g.files:
                lines.append(f"    • {f['name']} ({len(f['df'])} values)")
            lines.append("")

        visible = self._visible_data()
        if len(visible) > 1:
            pooled = np.concatenate([v for _, v, _, _ in visible])
            st = compute_stats(pooled)
            lines.append("=" * 52)
            lines.append("ALL VISIBLE GROUPS POOLED")
            lines.append("-" * 52)
            lines.append(f"  Values         : {st['n']}")
            lines.append(f"  Min / Max      : {st['min']:.4g} / {st['max']:.4g} {unit}")
            lines.append(f"  Mean / Median  : {st['mean']:.4g} / {st['median']:.4g} {unit}")
            lines.append(f"  Std dev (n-1)  : {st['std']:.4g} {unit}  (CV {st['cv']:.3g} %)")
            lines.append(f"  Mean 95% CI    : {st['mean'] - st['ci95']:.4g} … "
                         f"{st['mean'] + st['ci95']:.4g} {unit}")
        return "\n".join(lines)

    def _refresh_stats(self):
        self.stats_box.configure(state="normal")
        self.stats_box.delete("1.0", "end")
        self.stats_box.insert("1.0", self._stats_text())

        measures, vmin, vmax = self._current_filters()
        tot = kept = 0
        for g in self.groups:
            k, t, _e = g.values(measures, vmin, vmax)
            tot += t
            kept += len(k)
        self.filter_info.configure(
            text=f"{kept} value(s) kept / {tot} loaded  ({tot - kept} excluded)")
        if self.groups:
            self._status(f"LOM Depth Analyser — {len(self.groups)} group(s), "
                         f"{kept} value(s) kept / {tot} loaded.")

    def _refresh_all(self):
        self._refresh_measures()
        self._refresh_groups()
        self._refresh_stats()
        self._plot()

    # ─────────────────────────────────────────────────────────────────────
    #  Exports
    # ─────────────────────────────────────────────────────────────────────
    def _has_data(self):
        if not self.groups:
            messagebox.showwarning("No data", "Load at least one CSV file first.", parent=self)
            return False
        return True

    def _export_image(self):
        if not self._has_data():
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"),
                       ("PDF", "*.pdf"), ("JPEG", "*.jpg")])
        if not path:
            return
        fig = self._make_export_figure()
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        self._status(f"LOM graph saved: {os.path.basename(path)}")
        messagebox.showinfo("Success", f"Graph saved:\n{path}", parent=self)

    def _export_frames(self):
        """(summary_df, histogram_df, values_df) built from the current settings."""
        measures, vmin, vmax = self._current_filters()
        rows = [(g.name, n_files, total, excluded, st, unit)
                for g, n_files, total, excluded, st, unit in self._stats_rows()]
        summary = build_summary_frame(rows)

        res = self._compute_series()
        if res is None:
            hist = pd.DataFrame()
        else:
            hist = build_histogram_frame(res["edges"],
                                         [(g.name, c) for g, _v, c in res["series"]],
                                         res["mode"])
        values = [build_values_frame(g, measures, vmin, vmax) for g in self.groups]
        values = pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        return summary, hist, values

    def _header_lines(self):
        measures, vmin, vmax = self._current_filters()
        return [
            "LOM Depth Analyser export",
            f"Unit: {self._unit()}",
            f"Min limit: {'none' if vmin is None else vmin}",
            f"Max limit: {'none' if vmax is None else vmax}",
            f"Measure types: {'all' if measures is None else ', '.join(measures)}",
            f"Grouping factor (bin width): {self._bin_width():g}",
            f"Bin alignment: {self.origin_mode_var.get()}",
            f"Y axis: {self.ymode_var.get()}",
        ]

    def _export_csv(self):
        if not self._has_data():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        summary, hist, values = self._export_frames()
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                for line in self._header_lines():
                    f.write(f"# {line}\n")
                for title, df in (("STATISTICS", summary), ("DISTRIBUTION (histogram)", hist),
                                  ("VALUES", values)):
                    f.write(f"\n# --- {title} ---\n")
                    if len(df):
                        df.round(6).to_csv(f, sep=";", index=False, decimal=",",
                                           lineterminator="\n")
                    else:
                        f.write("(empty)\n")
            self._status(f"LOM data exported: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Data exported:\n{path}", parent=self)
        except Exception as exc:                          # noqa: BLE001
            messagebox.showerror("Export error", str(exc), parent=self)

    def _export_excel(self):
        if not self._has_data():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        summary, hist, values = self._export_frames()
        measures, vmin, vmax = self._current_filters()
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                pd.DataFrame({"Parameter": self._header_lines()}).to_excel(
                    w, sheet_name="Parameters", index=False)
                summary.to_excel(w, sheet_name="Statistics", index=False)
                if len(hist):
                    hist.to_excel(w, sheet_name="Distribution", index=False)
                used = set()
                for g in self.groups:
                    name = "".join(c for c in g.name if c not in "[]:*?/\\")[:28] or "Group"
                    base, i = name, 1
                    while name in used:
                        name = f"{base[:26]}_{i}"
                        i += 1
                    used.add(name)
                    build_values_frame(g, measures, vmin, vmax).to_excel(
                        w, sheet_name=name, index=False)
            self._status(f"LOM data exported: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Data exported:\n{path}", parent=self)
        except Exception as exc:                          # noqa: BLE001
            messagebox.showerror("Export error",
                                 f"Excel export failed:\n{exc}\n\n"
                                 "(the 'openpyxl' package is required)", parent=self)

    def _export_pdf(self):
        if not self._has_data():
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".pdf",
                                            filetypes=[("PDF Report", "*.pdf")])
        if not path:
            return
        try:
            text = ("LOM Depth Analysis Report\n" + "=" * 52 + "\n\n"
                    + f"Bin alignment  : {self.origin_mode_var.get()}\n"
                    + f"Y axis         : {self.ymode_var.get()}\n\n"
                    + self._stats_text())
            lines = text.split("\n")
            chunks = [lines[i:i + 70] for i in range(0, len(lines), 70)] or [[""]]

            with PdfPages(path) as pdf:
                for chunk in chunks:
                    fig_t = Figure(figsize=(8.5, 11), facecolor="white")
                    fig_t.text(0.07, 0.96, "\n".join(chunk), fontsize=9, va="top",
                               ha="left", fontfamily="monospace")
                    pdf.savefig(fig_t)
                fig = self._make_export_figure()
                pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            self._status(f"LOM PDF report generated: {os.path.basename(path)}")
            messagebox.showinfo("Success", f"PDF report generated:\n{path}", parent=self)
        except Exception as exc:                          # noqa: BLE001
            messagebox.showerror("Export error", f"Failed to build PDF:\n{exc}", parent=self)


# ─────────────────────────────────────────────────────────────────────────────
#  Same module in its own window (optional) + stand-alone launcher
# ─────────────────────────────────────────────────────────────────────────────
class LOMDepthAnalyserWindow(ctk.CTkToplevel):
    """Hosts the exact same panel in a separate window."""

    def __init__(self, master=None, status_callback=None):
        super().__init__(master)
        self.title("LOM Depth Analyser — depth distribution & statistics")
        self.geometry("1650x920")
        self.minsize(1150, 700)
        self.panel = LOMDepthAnalyserPanel(self, status_callback=status_callback)
        self.panel.pack(fill="both", expand=True)
        if master is not None:
            try:
                if master.winfo_viewable():
                    self.transient(master)
            except Exception:                             # noqa: BLE001
                pass
        self.after(200, self.lift)


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.withdraw()
    win = LOMDepthAnalyserWindow(root)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
