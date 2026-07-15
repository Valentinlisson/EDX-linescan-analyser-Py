"""
EDX Line Scan Viewer  –  v6.2 (English Lab Edition + Visual Zones & PDF Reports)
-------------------------------------------------------------------------
• BASE: Version 6.1 (English interface, JSON presets, crosshair, etc.).
• ADDED: Visual delimitation of phases on the graph (vertical dashed lines).
• ADDED: Phase names displayed floating *above* the graph (no overlapping with curves or titles).
• ADDED: Comprehensive PDF Report Generator (Graph + Parameters + Zone Data Table).
• PRESERVED: Save graph as Image (PNG/SVG/PDF), export data as CSV and Excel (.xlsx).
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.widgets import SpanSelector
import os
import json

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS_DEFAULT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#FF0000", "#882255", "#117733"]
MARKERS = {"Circle": "o", "Square": "s", "Triangle ▲": "^", "Triangle ▼": "v", "Diamond": "D", "Star": "*", "None": "None"}
LEGEND_POSITIONS = {"Outside Right": "outside right", "Inside Top Right": "upper right", "Inside Top Left": "upper left", "Inside Bottom Right": "lower right"}

BG, ACCENT = "#1E2127", "#61AFEF"

# ─────────────────────────────────────────────────────────────────────────────
#  Mathematical Processing & Parsing Functions
# ─────────────────────────────────────────────────────────────────────────────
def parse_edx_file(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f: lines = f.readlines()
    header_idx = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("index")), None)
    if header_idx is None: raise ValueError("Header line starting with 'Index' not found.")
    header = lines[header_idx].split()
    rows = [[float(p) for p in l.split()] for l in lines[header_idx + 1:] if l.strip() and len(l.split()) >= 3]
    return pd.DataFrame(rows, columns=header).fillna(0)

def get_elements(df): return [c for c in df.columns if not c.lower().startswith(("index", "pos"))]
def get_pos_col(df): return next((c for c in df.columns if c.lower().startswith("pos")), df.columns[1])

def normalize_to_100(df, all_elements, active_elements):
    df_norm = df.copy()
    if not active_elements: return df_norm
    sum_vals = df_norm[active_elements].sum(axis=1)
    sum_vals = sum_vals.replace(0, 1) 
    for el in active_elements:
        df_norm[el] = (df_norm[el] / sum_vals) * 100
    for el in all_elements:
        if el not in active_elements:
            df_norm[el] = 0.0
    return df_norm

# ─────────────────────────────────────────────────────────────────────────────
#  Phase Management & Graphic Editor Window (Pop-up)
# ─────────────────────────────────────────────────────────────────────────────
class PhaseManagerWindow(ctk.CTkToplevel):
    def __init__(self, master, elements, current_presets, on_save_callback):
        super().__init__(master)
        self.title("Phase Configuration Editor (JSON)")
        self.geometry("850x650")
        self.transient(master)
        self.grab_set()
        
        self.elements = elements
        self.presets = list(current_presets)
        self.on_save_callback = on_save_callback
        self._build_ui()
        
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
        left_frame = ctk.CTkFrame(self, corner_radius=8)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(left_frame, text="Current Preset Phases", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=5)
        
        self.scroll_list = ctk.CTkScrollableFrame(left_frame)
        self.scroll_list.pack(fill="both", expand=True, padx=5, pady=5)
        self._refresh_presets_display()
        
        right_frame = ctk.CTkFrame(self, corner_radius=8)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(right_frame, text="Create / Edit a Phase", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="Phase Name (e.g., Oxide, 304 Steel, etc.) :").pack(anchor="w", padx=15, pady=(5,0))
        self.phase_name_entry = ctk.CTkEntry(right_frame, placeholder_text="Enter identification name...")
        self.phase_name_entry.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(right_frame, text="Chemical thresholds per element :", font=("Helvetica", 12, "italic")).pack(anchor="w", padx=15, pady=(10,5))
        
        self.scroll_elements = ctk.CTkScrollableFrame(right_frame)
        self.scroll_elements.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.element_inputs = {}
        for el in self.elements:
            row = ctk.CTkFrame(self.scroll_elements, fg_color="transparent")
            row.pack(fill="x", pady=3)
            
            chk_var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(row, text=f" {el}", variable=chk_var, width=70, font=("Helvetica", 12, "bold"))
            chk.pack(side="left", padx=5)
            
            ctk.CTkLabel(row, text="Min (%) :").pack(side="left", padx=2)
            min_ent = ctk.CTkEntry(row, width=60, placeholder_text="0")
            min_ent.pack(side="left", padx=5)
            
            ctk.CTkLabel(row, text="Max (%) :").pack(side="left", padx=2)
            max_ent = ctk.CTkEntry(row, width=60, placeholder_text="100")
            max_ent.pack(side="left", padx=5)
            
            self.element_inputs[el] = {"active": chk_var, "min": min_ent, "max": max_ent}
            
        ctk.CTkButton(right_frame, text="➕ Add / Validate Phase", fg_color="#0072B2", command=self._add_phase_to_list).pack(fill="x", padx=15, pady=12)
        
        bottom_frame = ctk.CTkFrame(self, height=60, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkButton(bottom_frame, text="📂 Import JSON", fg_color="gray35", command=self._load_json_file).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(bottom_frame, text="💾 Export JSON", fg_color="gray35", command=self._save_json_file).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(bottom_frame, text="✔ Apply & Return to Graph", fg_color="#009E73", hover_color="#007755", font=("Helvetica", 12, "bold"), command=self._apply_and_close).pack(side="right", padx=5, pady=10)

    def _refresh_presets_display(self):
        for widget in self.scroll_list.winfo_children(): widget.destroy()
        for idx, phase in enumerate(self.presets):
            box = ctk.CTkFrame(self.scroll_list, fg_color="gray23", corner_radius=6)
            box.pack(fill="x", pady=3, padx=2)
            
            desc_text = f"⭐ {phase['nom_phase']}\n"
            conds = []
            for el, lims in phase.get("conditions", {}).items():
                c_str = f"{el} ["
                if "min" in lims: c_str += f"≥{lims['min']}%"
                if "max" in lims: c_str += f" ≤{lims['max']}%"
                c_str += "]"
                conds.append(c_str)
            desc_text += " | ".join(conds) if conds else "No restriction (Default Phase)"
            
            lbl = ctk.CTkLabel(box, text=desc_text, justify="left", font=("Arial", 11), anchor="w")
            lbl.pack(side="left", fill="x", expand=True, padx=8, pady=4)
            
            ctk.CTkButton(box, text="❌", width=28, height=28, fg_color="#D55E00", hover_color="#AA3300", command=lambda i=idx: self._remove_phase(i)).pack(side="right", padx=6)

    def _remove_phase(self, idx):
        self.presets.pop(idx)
        self._refresh_presets_display()

    def _add_phase_to_list(self):
        name = self.phase_name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please give an identifiable name to the phase.")
            return
            
        conditions = {}
        for el, widgets in self.element_inputs.items():
            if widgets["active"].get():
                lims = {}
                min_v = widgets["min"].get().strip()
                max_v = widgets["max"].get().strip()
                if min_v:
                    try: lims["min"] = float(min_v)
                    except ValueError: pass
                if max_v:
                    try: lims["max"] = float(max_v)
                    except ValueError: pass
                if lims:
                    conditions[el] = lims
                    
        self.presets.append({"nom_phase": name, "conditions": conditions})
        self.phase_name_entry.delete(0, tk.END)
        for w in self.element_inputs.values():
            w["active"].set(False); w["min"].delete(0, tk.END); w["max"].delete(0, tk.END)
        self._refresh_presets_display()

    def _load_json_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON File", "*.json")])
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            if isinstance(data, list):
                self.presets = data
                self._refresh_presets_display()
            else: messagebox.showerror("Incorrect Structure", "The file must be structured as a list of phases.")
        except Exception as e: messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _save_json_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON File", "*.json")])
        if not path: return
        try:
            with open(path, 'w', encoding='utf-8') as f: json.dump(self.presets, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Success", "The identification preset has been successfully encoded and saved.")
        except Exception as e: messagebox.showerror("Error", f"Unable to save file: {str(e)}")

    def _apply_and_close(self):
        self.on_save_callback(self.presets)
        self.destroy()

# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────
class EDXApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EDX Line Scan Viewer - Lab Edition (PDF Reports)")
        self.geometry("1600x900")
        
        self.df_raw, self.df_norm = None, None
        self.elements = []
        self.is_normalized = False 
        self.phase_presets = []
        self.detected_zones = [] # Stocke les zones calculées pour les tracer

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

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)

        graph_inside_frame = ctk.CTkFrame(center_frame)
        graph_inside_frame.pack(fill="both", expand=True)

        self.right_panel = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._build_left(self.left_panel)
        self._build_graph(graph_inside_frame)
        self._build_right(self.right_panel)

    def _section(self, parent, text): 
        ctk.CTkLabel(parent, text=text, font=("Helvetica", 12, "bold"), text_color=ACCENT).pack(anchor="w", pady=(15, 5))

    def _build_left(self, p):
        ctk.CTkLabel(p, text="EDX Lab Viewer", font=("Helvetica", 20, "bold")).pack(anchor="w", pady=(5, 10))

        self._section(p, "MAIN DATA")
        ctk.CTkButton(p, text="📂 Load File", command=self._open_file).pack(fill="x", pady=2)
        ctk.CTkLabel(p, textvariable=self.filepath_var, font=("Helvetica", 11), text_color="gray", wraplength=240).pack(anchor="w")

        self._section(p, "DATA PROCESSING")
        row_btn = ctk.CTkFrame(p, fg_color="transparent")
        row_btn.pack(fill="x", pady=5)
        ctk.CTkButton(row_btn, text="✔ Normalize 100%", command=self._apply_normalization, width=130).pack(side="left", padx=(0, 5))
        ctk.CTkButton(row_btn, text="⏪ Raw", command=self._reset_to_raw, width=80, fg_color="gray40").pack(side="right")

        self._section(p, "SCIENTIFIC ANALYSIS")
        self._slider_generic(p, "Smoothing (Pts):", self.smooth_window, 1, 20)
        ctk.CTkButton(p, text="📊 Extract Stats (ROI)", command=self._activate_roi, fg_color="#E69F00", text_color="black").pack(fill="x", pady=5)
        
        ctk.CTkButton(p, text="⚙ Phase Editor (JSON)", fg_color="#009E73", hover_color="#007755", command=self._open_phase_editor).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📋 Identify Multi-Zones", fg_color="#56B4E9", text_color="black", hover_color="#3399CC", command=self._generate_phase_report).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="❌ Clear Zones Overlay", fg_color="gray40", hover_color="gray30", command=self._clear_zones).pack(fill="x", pady=2)

        self._section(p, "ELEMENTS")
        self.el_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.el_frame.pack(fill="x")
        row_tous = ctk.CTkFrame(p, fg_color="transparent")
        row_tous.pack(fill="x", pady=10)
        ctk.CTkButton(row_tous, text="✅ All", command=self._show_all, width=100).pack(side="left", expand=True)
        ctk.CTkButton(row_tous, text="⬜ None", command=self._hide_all, width=100, fg_color="gray40").pack(side="right", expand=True)

    def _build_right(self, p):
        self._section(p, "INTERACTIVE TOOLS")
        ctk.CTkLabel(p, text="Double-click Graph:", text_color="gray", font=("Arial", 10)).pack(anchor="w")
        ctk.CTkLabel(p, text="Top = Title | Bottom/Left = Axes | Center = Reset", text_color="gray", font=("Arial", 10)).pack(anchor="w")
        ctk.CTkCheckBox(p, text="Enable Dynamic Crosshair", variable=self.enable_crosshair, command=self._plot).pack(anchor="w", pady=10)
        
        self._section(p, "VISUAL PRESETS")
        row_pre = ctk.CTkFrame(p, fg_color="transparent")
        row_pre.pack(fill="x")
        ctk.CTkButton(row_pre, text="💾 Save", command=self._save_presets, width=100).pack(side="left", expand=True)
        ctk.CTkButton(row_pre, text="📂 Load", command=self._load_presets, width=100, fg_color="gray40").pack(side="right", expand=True)

        self._section(p, "ZOOM / SCALE")
        self._slider_zoom(p, "Zoom X :", self.scale_x, 1.0, 10.0)
        self._slider_zoom(p, "Zoom Y :", self.scale_y, 1.0, 10.0)
        ctk.CTkButton(p, text="🔍 Auto-Scale", command=self._autoscale_graph).pack(fill="x", pady=5)

        self._section(p, "DESIGN & LEGEND")
        ctk.CTkComboBox(p, variable=self.legend_pos_var, values=list(LEGEND_POSITIONS.keys()), command=lambda v: self._plot()).pack(fill="x", pady=5)
        self._slider_generic(p, "Line Thickness :", self.line_width, 0.5, 5.0)
        self._slider_generic(p, "Point Size :", self.marker_size, 0.0, 20.0)
        self._slider_generic(p, "Font Size :", self.font_size, 7, 18)

        self._section(p, "EXPORTS & REPORTS")
        ctk.CTkButton(p, text="🖼 Save Image (PNG/SVG...)", command=self._save_fig).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📄 Export Data (CSV)", command=self._export_csv).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📊 Export Data (Excel)", command=self._export_excel).pack(fill="x", pady=2)
        ctk.CTkButton(p, text="📑 Generate PDF Report", command=self._export_pdf_report, fg_color="#882255", hover_color="#551133").pack(fill="x", pady=(10,5))

        self._section(p, "ELEMENT CUSTOMIZATION")
        self.custom_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.custom_frame.pack(fill="x")

    def _slider_zoom(self, parent, label, var, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{var.get():.1f}x", text_color=ACCENT)
        val_lbl.pack(side="right")
        def _upd(v):
            val_lbl.configure(text=f"{float(v):.1f}x")
            self._slider_interacting = True; self._plot(); self._slider_interacting = False
        ctk.CTkSlider(parent, from_=from_, to=to, variable=var, command=_upd).pack(fill="x")

    def _slider_generic(self, parent, label, var, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{var.get():.1f}", text_color=ACCENT)
        val_lbl.pack(side="right")
        def _upd(v): val_lbl.configure(text=f"{int(v) if isinstance(var, ctk.IntVar) else float(v):.1f}"); self._plot()
        ctk.CTkSlider(parent, from_=from_, to=to, variable=var, number_of_steps=int(to-from_) if isinstance(var, ctk.IntVar) else None, command=_upd).pack(fill="x")

    def _pick_color(self, str_var, btn_widget):
        color = colorchooser.askcolor(color=str_var.get(), title="Select Color")
        if color[1]: str_var.set(color[1]); btn_widget.configure(fg_color=color[1]); self._plot()

    def _save_presets(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p: return
        data = {
            "plot_bg": self.plot_bg_color.get(), "fig_bg": self.fig_bg_color.get(),
            "lw": self.line_width.get(), "ms": self.marker_size.get(), "fs": self.font_size.get(),
            "legend": self.legend_pos_var.get(), "colors": self.el_colors, 
            "markers": {k: v.get() for k, v in self.el_markers.items()}
        }
        with open(p, 'w') as f: json.dump(data, f)
        messagebox.showinfo("Success", "Visual preset layout saved.")

    def _load_presets(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p: return
        try:
            with open(p, 'r') as f: data = json.load(f)
            self.plot_bg_color.set(data.get("plot_bg", "#282C34")); self.fig_bg_color.set(data.get("fig_bg", "#1E2127"))
            self.line_width.set(data.get("lw", 1.8)); self.marker_size.set(data.get("ms", 5.0)); self.font_size.set(data.get("fs", 10))
            self.legend_pos_var.set(data.get("legend", "Outside Right"))
            if "colors" in data: self.el_colors.update(data["colors"])
            if "markers" in data: 
                for k, v in data["markers"].items(): 
                    if k in self.el_markers: self.el_markers[k].set(v)
            self._build_element_rows() 
            self._plot()
        except Exception as e: messagebox.showerror("Error", "File corrupted or invalid layout.")

    def _build_graph(self, frame):
        self.fig = Figure(figsize=(8.5, 6), facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        
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
        for w in self.el_frame.winfo_children(): w.destroy()
        self.el_vars.clear()
        for el in self.elements:
            self.el_vars[el] = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(self.el_frame, text=el, variable=self.el_vars[el], command=self._plot).pack(anchor="w", pady=2)

        for w in self.custom_frame.winfo_children(): w.destroy()
        for i, el in enumerate(self.elements):
            if el not in self.el_colors: self.el_colors[el] = COLORS_DEFAULT[i % len(COLORS_DEFAULT)]
            if el not in self.el_markers: self.el_markers[el] = ctk.StringVar(value="Circle")

            r = ctk.CTkFrame(self.custom_frame, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=el, width=30).pack(side="left")
            
            btn = ctk.CTkButton(r, text="", width=25, height=25, fg_color=self.el_colors[el])
            btn.configure(command=lambda e=el, b=btn: self._pick_color(tk.StringVar(value=self.el_colors[e]), b) or self.el_colors.update({e: b.cget("fg_color")}) or self._plot())
            btn.pack(side="left", padx=5)

            cb = ctk.CTkComboBox(r, variable=self.el_markers[el], values=list(MARKERS.keys()), width=100, command=lambda v: self._plot())
            cb.pack(side="right", padx=5)

    def _show_all(self): [v.set(True) for v in self.el_vars.values()]; self._plot()
    def _hide_all(self): [v.set(False) for v in self.el_vars.values()]; self._plot()

    def _style_axes(self):
        self.ax.set_facecolor(self.plot_bg_color.get())
        self.fig.patch.set_facecolor(self.fig_bg_color.get())
        tc = "white" if self.fig_bg_color.get().lower() in [BG.lower(), "#1e2127"] else "black"
        for s in self.ax.spines.values(): s.set_edgecolor("gray")
        
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

        if self.show_grid.get(): self.ax.grid(color="gray", linestyle="--", alpha=0.3)
        else: self.ax.grid(False)

    def _open_file(self):
        p = filedialog.askopenfilename()
        if p:
            self.df_raw = parse_edx_file(p); self.elements = get_elements(self.df_raw)
            self.df_norm = self.df_raw.copy(); self.filepath_var.set(os.path.basename(p))
            self._build_element_rows(); self._autoscale_graph()

    def _apply_normalization(self):
        if self.df_raw is None: return
        act = [el for el in self.elements if self.el_vars[el].get()]
        self.df_norm = normalize_to_100(self.df_raw, self.elements, act)
        self.is_normalized = True
        self._autoscale_graph()

    def _reset_to_raw(self):
        if self.df_raw is None: return
        self.df_norm = self.df_raw.copy(); self.is_normalized = False; self._autoscale_graph()

    def _autoscale_graph(self):
        if self.df_norm is None: return
        self.scale_x.set(1.0); self.scale_y.set(1.0); self._plot(force_reset=True)

    def _apply_smoothing(self, y_values):
        w = self.smooth_window.get()
        if w > 1: return pd.Series(y_values).rolling(window=w, center=True, min_periods=1).mean().values
        return y_values

    # ─────────────────────────────────────────────────────────────────────────────
    #  Phase Detection Logic & PDF Report Generation
    # ─────────────────────────────────────────────────────────────────────────────
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
                if "min" in lims and v < lims["min"]: match = False; break
                if "max" in lims and v > lims["max"]: match = False; break
            if match: return phase["nom_phase"]
        return "Undetermined Zone"

    def _generate_phase_report(self):
        if self.df_norm is None: return
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
        
        # Show quick textual summary
        report_lines = [f" ◼ [{z['start']:.2f} to {z['end']:.2f} µm]  ➡  {z['name']}" for z in self.detected_zones]
        rep_win = ctk.CTkToplevel(self)
        rep_win.title("Structural Segmentation Summary")
        rep_win.geometry("550x450")
        
        ctk.CTkLabel(rep_win, text="Spatial Chemical Profiling", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=10)
        txt_box = tk.Text(rep_win, bg="#282C34", fg="white", font=("Consolas", 11), wrap="word", padx=10, pady=10)
        txt_box.pack(fill="both", expand=True, padx=15, pady=10)
        
        txt_box.insert("1.0", f"File: {self.filepath_var.get()}\n" + "="*40 + "\n\n" + "\n".join(report_lines))
        txt_box.configure(state="disabled")

    def _clear_zones(self):
        self.detected_zones = []
        self._plot()

    def _export_pdf_report(self):
        if self.df_norm is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Report", "*.pdf")])
        if not path: return
        
        try:
            with PdfPages(path) as pdf:
                # Page 1: Parameters and Zones Table
                fig_text = Figure(figsize=(8.5, 11))
                ax_text = fig_text.add_subplot(111)
                ax_text.axis('off')
                
                content = "EDX Structural Analysis Report\n"
                content += "="*50 + "\n\n"
                content += f"File Analyzed: {self.filepath_var.get()}\n"
                content += f"Data Mode: {'100% Normalized' if self.is_normalized else 'Raw Intensity'}\n"
                content += f"Smoothing Applied: {self.smooth_window.get()} points\n\n"
                
                content += "Detected Zones & Phases Summary:\n"
                content += "-"*50 + "\n"
                if self.detected_zones:
                    for z in self.detected_zones:
                        content += f"Start: {z['start']:.2f} µm | End: {z['end']:.2f} µm | Phase: {z['name']}\n"
                else:
                    content += "No multi-zones mapped. Run 'Identify Multi-Zones' first.\n"
                    
                ax_text.text(0.05, 0.95, content, transform=ax_text.transAxes, fontsize=11, va='top', fontfamily='monospace')
                pdf.savefig(fig_text)
                
                # Page 2: The current graphical plot
                # Temporary adjust layout specifically for PDF sizing if needed
                self.fig.patch.set_facecolor("white")
                self.ax.set_facecolor("white")
                self.ax.tick_params(colors="black")
                for s in self.ax.spines.values(): s.set_edgecolor("black")
                self.ax.xaxis.label.set_color("black")
                self.ax.yaxis.label.set_color("black")
                self.ax.title.set_color("black")
                if any(v.get() for v in self.el_vars.values()):
                    pos = LEGEND_POSITIONS.get(self.legend_pos_var.get(), "outside right")
                    args = {"frameon":True, "facecolor":"white", "labelcolor":"black"}
                    if pos == "outside right": self.ax.legend(**args, loc="upper left", bbox_to_anchor=(1.02, 1))
                    else: self.ax.legend(**args, loc=pos)
                
                pdf.savefig(self.fig, bbox_inches="tight")
                
                # Restore UI dark theme colors
                self._style_axes()
                self._plot()
                
            messagebox.showinfo("Success", "Comprehensive PDF Report successfully exported.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to build PDF:\n{str(e)}")

    def _save_fig(self):
        if self.df_norm is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF Plot Only", "*.pdf")])
        if p: 
            self.fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=self.fig_bg_color.get())
            messagebox.showinfo("Success", "Image saved.")

    def _export_csv(self):
        if self.df_norm is None: return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.df_norm.to_csv(p, index=False, sep=";", float_format="%.4f")
            messagebox.showinfo("Success", "Data exported to CSV.")

    def _export_excel(self):
        if self.df_norm is None: return
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            try:
                with pd.ExcelWriter(p, engine="openpyxl") as w:
                    self.df_norm.to_excel(w, sheet_name="Processed", index=False)
                    self.df_raw.to_excel(w, sheet_name="Raw", index=False)
                messagebox.showinfo("Success", "Data exported to Excel.")
            except Exception as e: messagebox.showerror("Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────────
    #  Tracé de la Figure et Événements Matplotlib
    # ─────────────────────────────────────────────────────────────────────────────
    def _plot(self, force_reset=False):
        if self.df_norm is None: return
        ox, oy = self.ax.get_xlim() if self.ax.get_lines() and not force_reset else None, self.ax.get_ylim() if self.ax.get_lines() and not force_reset else None
        
        self.ax.clear()
        self.crosshair_v = self.ax.axvline(0, color='gray', linestyle='--', visible=False, zorder=10)
        self.crosshair_text = self.ax.text(0, 0, '', bbox=dict(facecolor=BG, edgecolor=ACCENT, alpha=0.9), color="white", visible=False, zorder=10)
        self._style_axes()

        x = self.df_raw[get_pos_col(self.df_raw)].values
        lw, ms, fs = self.line_width.get(), self.marker_size.get(), self.font_size.get()

        # Dessin des zones par-dessus le fond, mais derrière les courbes
        if self.detected_zones:
            for zone in self.detected_zones:
                self.ax.axvline(x=zone["start"], color="#E69F00", linestyle=":", linewidth=1.5, zorder=1)
                
                # Placement sécurisé du texte (Ne touche pas les courbes, ne touche pas le titre)
                mid_x = (zone["start"] + zone["end"]) / 2
                self.ax.text(mid_x, 1.02, zone["name"], transform=self.ax.get_xaxis_transform(),
                             ha="center", va="bottom", fontsize=max(8, fs-1), color=ACCENT,
                             bbox=dict(facecolor=self.plot_bg_color.get(), edgecolor='none', alpha=0.7),
                             clip_on=False, zorder=5)

        for el in self.elements:
            if not self.el_vars.get(el, ctk.BooleanVar(value=True)).get(): continue
            y = self._apply_smoothing(self.df_norm[el].values)
            col = self.el_colors.get(el, "#FFF")
            mk = MARKERS.get(self.el_markers.get(el).get(), "o")
            if mk == "None": mk = None
            self.ax.plot(x, y, color=col, linewidth=lw, marker=mk, markersize=ms if mk else 0, label=el, zorder=3)

        xcen, hs = (x.min() + x.max()) / 2, ((x.max() - x.min()) / 2) / self.scale_x.get()
        ymax = max([self.df_norm[el].max() for el in self.elements if self.el_vars[el].get()] + [1])
        
        if not force_reset and ox and not self._slider_interacting: self.ax.set_xlim(ox); self.ax.set_ylim(oy)
        else: self.ax.set_xlim([xcen - hs, xcen + hs]); self.ax.set_ylim([0, (ymax * 1.05) / self.scale_y.get()])

        if any(v.get() for v in self.el_vars.values()):
            pos = LEGEND_POSITIONS.get(self.legend_pos_var.get(), "outside right")
            args = {"frameon":True, "facecolor":self.plot_bg_color.get(), "labelcolor":"white" if self.plot_bg_color.get() == "#282C34" else "black"}
            if pos == "outside right": self.ax.legend(**args, loc="upper left", bbox_to_anchor=(1.02, 1))
            else: self.ax.legend(**args, loc=pos)

        self.ax.xaxis.set_major_locator(ticker.AutoLocator()); self.ax.yaxis.set_major_locator(ticker.AutoLocator())
        self.fig.tight_layout(); self.canvas.draw()

    # --- ROI et Events ---
    def _activate_roi(self):
        messagebox.showinfo("Mode ROI", "Drag on the graph to select an X region.")
        def onselect(xmin, xmax):
            if self.df_norm is None: return
            pos_col = get_pos_col(self.df_norm)
            mask = (self.df_norm[pos_col] >= xmin) & (self.df_norm[pos_col] <= xmax)
            sub = self.df_norm[mask]
            stats = f"Zone : {xmin:.2f} à {xmax:.2f}\n\n"
            for el in self.elements:
                if self.el_vars[el].get(): stats += f"• {el}: Avg={sub[el].mean():.2f} | Max={sub[el].max():.2f}\n"
            messagebox.showinfo("ROI Stats", stats)
        self.roi_selector = SpanSelector(self.ax, onselect, 'horizontal', useblit=True, props=dict(alpha=0.3, facecolor=ACCENT))

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
                if self.el_vars[el].get(): txt += f"{el}: {self.df_norm[el].values[idx]:.1f}\n"
                
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
        if self.df_norm is None or event.inaxes is None: return
        sf = 1.0 / 1.15 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim([cx - (cx - self.ax.get_xlim()[0])*sf, cx + (self.ax.get_xlim()[1] - cx)*sf])
        self.ax.set_ylim([cy - (cy - self.ax.get_ylim()[0])*sf, cy + (self.ax.get_ylim()[1] - cy)*sf])
        self.canvas.draw()

    def _on_mpl_button_press(self, event):
        if event.dblclick and event.button == 1:
            bbox = self.fig.bbox
            if event.y > bbox.height * 0.9: 
                t = simpledialog.askstring("Title", "New title :", initialvalue=self.graph_title)
                if t is not None: self.graph_title = t; self._plot()
            elif event.y < bbox.height * 0.1: 
                x = simpledialog.askstring("X-Axis", "X-axis label :", initialvalue=self.graph_xlabel)
                if x is not None: self.graph_xlabel = x; self._plot()
            elif event.x < bbox.width * 0.1: 
                y = simpledialog.askstring("Y-Axis", "Y-axis label :", initialvalue=self.graph_ylabel)
                if y is not None: self.graph_ylabel = y; self._plot()
            elif event.inaxes: 
                self._autoscale_graph()
        elif event.button in (2, 3) and event.inaxes: 
            self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

if __name__ == "__main__":
    app = EDXApp()
    app.mainloop()