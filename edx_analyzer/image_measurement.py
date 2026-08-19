"""SEM image import, pan/zoom display, scale calibration and distance
measurement. Packaged as a mixin (`ImageMeasurementMixin`) that `EDXApp`
also inherits, so this whole feature area lives in its own file instead of
bloating app.py further."""

import os
import math

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .constants import BG, ACCENT
from .widgets import add_tooltip

CALIBRATION_COLOR = "#E69F00"
MEASUREMENT_COLOR = "#56B4E9"


class CalibrationDialog(ctk.CTkToplevel):
    """Asks for the real-world length a just-drawn reference arrow represents."""

    def __init__(self, master, pixel_length, on_confirm):
        super().__init__(master)
        self.title("Calibrate Scale")
        self.geometry("320x240")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.on_confirm = on_confirm

        ctk.CTkLabel(self, text="Scale Calibration", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=(15, 10))
        ctk.CTkLabel(self, text=f"Arrow length: {pixel_length:.1f} px", text_color="gray").pack()

        ctk.CTkLabel(self, text="This arrow represents :").pack(anchor="w", padx=15, pady=(15, 0))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=5)
        self.len_entry = ctk.CTkEntry(row, placeholder_text="e.g. 10")
        self.len_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.unit_var = ctk.StringVar(value="µm")
        ctk.CTkComboBox(row, variable=self.unit_var, values=["nm", "µm", "mm"], width=80).pack(side="right")

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=15, side="bottom")
        ctk.CTkButton(btn_row, text="Cancel", fg_color="gray40", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_row, text="✔ Confirm", fg_color="#009E73", hover_color="#007755", command=self._confirm).pack(side="right", padx=5)

    def _confirm(self):
        try:
            real_len = float(self.len_entry.get().strip())
            if real_len <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a positive number.")
            return
        self.on_confirm(real_len, self.unit_var.get())
        self.destroy()


class ImageMeasurementMixin:
    """Provides the whole 'SEM Picture Analyser' tab. Expects the host
    class to also provide `_set_status(message)`."""

    def _init_image_state(self):
        self.sem_image_path = None
        self.sem_image_array = None
        self.image_scale = None  # real units per pixel, None until calibrated
        self.image_scale_unit = "µm"
        self.calibration_arrow = None  # {"p1", "p2", "real_length"}
        self.measurements = []  # [{"p1","p2","pixel_length","real_length","label"}]
        self._image_tool_mode = None  # None | "calibrate" | "measure"
        self._image_drag_start = None
        self._image_temp_artist = None

    # ─────────────────────────────────────────────────────────────────────
    #  Layout
    # ─────────────────────────────────────────────────────────────────────
    def _build_image_tab(self, frame):
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        left_wrapper = ctk.CTkFrame(frame, width=300, fg_color="transparent")
        left_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        left = ctk.CTkScrollableFrame(left_wrapper, corner_radius=0)
        left.pack(fill="both", expand=True)

        center = ctk.CTkFrame(frame, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self._build_image_controls(left)
        self._build_image_canvas(center)

    def _section_img(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Helvetica", 12, "bold"), text_color=ACCENT).pack(anchor="w", pady=(15, 5))

    def _build_image_controls(self, p):
        ctk.CTkLabel(p, text="SEM Image", font=("Helvetica", 20, "bold")).pack(anchor="w", pady=(5, 10))

        self._section_img(p, "IMPORT")
        load_btn = ctk.CTkButton(p, text="🖼 Load SEM Image", command=self._load_sem_image)
        load_btn.pack(fill="x", pady=2)
        add_tooltip(load_btn, "Import the SEM micrograph this EDX line scan was taken on.")
        self.sem_image_var = tk.StringVar(value="No image loaded")
        ctk.CTkLabel(p, textvariable=self.sem_image_var, font=("Helvetica", 11), text_color="gray", wraplength=260).pack(anchor="w")

        self._section_img(p, "SCALE CALIBRATION")
        ctk.CTkLabel(p, text="Draw an arrow over a known reference length (e.g. a scale bar), then enter its real length.",
                     text_color="gray", font=("Arial", 10), wraplength=260, justify="left").pack(anchor="w")
        self.calibrate_btn = ctk.CTkButton(p, text="📏 Set Scale (draw arrow)", fg_color=CALIBRATION_COLOR, text_color="black", command=self._activate_calibration_tool)
        self.calibrate_btn.pack(fill="x", pady=(5, 2))
        add_tooltip(self.calibrate_btn, "Click, then drag an arrow over a known reference length on the image.")
        self.scale_status_var = tk.StringVar(value="Not calibrated.")
        ctk.CTkLabel(p, textvariable=self.scale_status_var, text_color=ACCENT, font=("Arial", 11, "bold"), wraplength=260, justify="left").pack(anchor="w", pady=(4, 5))
        clear_cal_btn = ctk.CTkButton(p, text="↺ Clear Calibration", fg_color="gray40", command=self._clear_calibration)
        clear_cal_btn.pack(fill="x", pady=2)

        self._section_img(p, "DISTANCE MEASUREMENT")
        self.measure_btn = ctk.CTkButton(p, text="📐 Measure Distance", fg_color=MEASUREMENT_COLOR, text_color="black", command=self._activate_measure_tool)
        self.measure_btn.pack(fill="x", pady=(5, 2))
        add_tooltip(self.measure_btn, "Click, then drag an arrow between two points to measure the real distance (needs calibration first).")
        self.measurements_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.measurements_frame.pack(fill="x", pady=5)
        self._refresh_measurements_list()
        clear_meas_btn = ctk.CTkButton(p, text="❌ Clear Measurements", fg_color="gray40", command=self._clear_measurements)
        clear_meas_btn.pack(fill="x", pady=2)

    def _build_image_canvas(self, frame):
        self.img_fig = Figure(figsize=(8, 6), facecolor=BG)
        self.img_ax = self.img_fig.add_subplot(111)
        self.img_ax.set_facecolor("#111111")
        self.img_ax.axis("off")
        self.img_ax.text(0.5, 0.5, "Load a SEM image to begin", ha="center", va="center", color="gray", transform=self.img_ax.transAxes)

        self.img_canvas = FigureCanvasTkAgg(self.img_fig, master=frame)
        self.img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self.img_canvas, tb_frame).update()

        self.img_canvas.mpl_connect("button_press_event", self._on_image_press)
        self.img_canvas.mpl_connect("motion_notify_event", self._on_image_motion)
        self.img_canvas.mpl_connect("button_release_event", self._on_image_release)
        self.img_canvas.mpl_connect("scroll_event", self._on_image_scroll)

        self.img_canvas.draw()

    # ─────────────────────────────────────────────────────────────────────
    #  Import & rendering
    # ─────────────────────────────────────────────────────────────────────
    def _load_sem_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All Files", "*.*")])
        if not p:
            return
        try:
            img = mpimg.imread(p)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        self.sem_image_path = p
        self.sem_image_array = img
        self.image_scale = None
        self.calibration_arrow = None
        self.measurements = []
        self.sem_image_var.set(os.path.basename(p))
        self.scale_status_var.set("Not calibrated.")
        self._refresh_measurements_list()
        self._redraw_image(reset_view=True)
        self._set_status(f"SEM image loaded: {os.path.basename(p)}")

    def _redraw_image(self, reset_view=False):
        xlim, ylim = (self.img_ax.get_xlim(), self.img_ax.get_ylim()) if not reset_view and self.img_ax.has_data() else (None, None)

        self.img_ax.clear()
        self.img_ax.axis("off")
        if self.sem_image_array is not None:
            self.img_ax.imshow(self.sem_image_array)
        else:
            self.img_ax.text(0.5, 0.5, "Load a SEM image to begin", ha="center", va="center", color="gray", transform=self.img_ax.transAxes)

        if self.calibration_arrow:
            self._draw_arrow(self.calibration_arrow["p1"], self.calibration_arrow["p2"], CALIBRATION_COLOR,
                              f"Ref: {self.calibration_arrow['real_length']:.3g} {self.image_scale_unit}")
        for m in self.measurements:
            self._draw_arrow(m["p1"], m["p2"], MEASUREMENT_COLOR, f"{m['label']}: {m['real_length']:.3g} {self.image_scale_unit}")

        if xlim is not None:
            self.img_ax.set_xlim(xlim)
            self.img_ax.set_ylim(ylim)

        self.img_fig.tight_layout()
        self.img_canvas.draw()

    def _draw_arrow(self, p1, p2, color, label):
        self.img_ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="<->", color=color, linewidth=2))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.img_ax.text(mid[0], mid[1], label, color=color, fontsize=10, fontweight="bold",
                         ha="center", va="bottom", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    # ─────────────────────────────────────────────────────────────────────
    #  Tool activation
    # ─────────────────────────────────────────────────────────────────────
    def _activate_calibration_tool(self):
        if self.sem_image_array is None:
            messagebox.showwarning("No Image", "Please load a SEM image first.")
            return
        self._image_tool_mode = "calibrate"
        self._set_status("Calibration mode: drag an arrow over a known reference length.")

    def _activate_measure_tool(self):
        if self.sem_image_array is None:
            messagebox.showwarning("No Image", "Please load a SEM image first.")
            return
        if self.image_scale is None:
            messagebox.showwarning("Not Calibrated", "Please set the scale first (Set Scale button above).")
            return
        self._image_tool_mode = "measure"
        self._set_status("Measure mode: drag an arrow between the two points to measure.")

    # ─────────────────────────────────────────────────────────────────────
    #  Mouse interaction (left-drag draws an arrow; only active once a tool
    #  is selected, so the toolbar's own pan/zoom keep working otherwise)
    # ─────────────────────────────────────────────────────────────────────
    def _on_image_press(self, event):
        if self._image_tool_mode is None or event.button != 1 or event.inaxes != self.img_ax:
            return
        self._image_drag_start = (event.xdata, event.ydata)

    def _on_image_motion(self, event):
        if self._image_drag_start is None or event.inaxes != self.img_ax:
            return
        if self._image_temp_artist:
            self._image_temp_artist.remove()
        self._image_temp_artist = self.img_ax.annotate(
            "", xy=(event.xdata, event.ydata), xytext=self._image_drag_start,
            arrowprops=dict(arrowstyle="->", color="white", linewidth=1.5, linestyle="--"))
        self.img_canvas.draw_idle()

    def _on_image_release(self, event):
        if self._image_drag_start is None:
            return
        p1 = self._image_drag_start
        self._image_drag_start = None
        if self._image_temp_artist:
            self._image_temp_artist.remove()
            self._image_temp_artist = None

        mode, self._image_tool_mode = self._image_tool_mode, None
        if event.xdata is None or event.ydata is None:
            self.img_canvas.draw_idle()
            return

        p2 = (event.xdata, event.ydata)
        pixel_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if pixel_len < 2:
            self.img_canvas.draw_idle()
            return  # ignore accidental clicks

        if mode == "calibrate":
            CalibrationDialog(self, pixel_len, on_confirm=lambda real_len, unit: self._apply_calibration(p1, p2, pixel_len, real_len, unit))
        elif mode == "measure":
            self._finish_measurement(p1, p2, pixel_len)
        else:
            self.img_canvas.draw_idle()

    def _apply_calibration(self, p1, p2, pixel_len, real_len, unit):
        self.image_scale = real_len / pixel_len
        self.image_scale_unit = unit
        self.calibration_arrow = {"p1": p1, "p2": p2, "real_length": real_len}
        self.scale_status_var.set(f"Calibrated: {self.image_scale:.4g} {unit}/px")
        self._redraw_image()
        self._set_status("Scale calibration set.")

    def _finish_measurement(self, p1, p2, pixel_len):
        real_len = pixel_len * self.image_scale
        label = f"M{len(self.measurements) + 1}"
        self.measurements.append({"p1": p1, "p2": p2, "pixel_length": pixel_len, "real_length": real_len, "label": label})
        self._refresh_measurements_list()
        self._redraw_image()
        self._set_status(f"Measured {label}: {real_len:.3g} {self.image_scale_unit}")

    def _refresh_measurements_list(self):
        for w in self.measurements_frame.winfo_children():
            w.destroy()
        if not self.measurements:
            ctk.CTkLabel(self.measurements_frame, text="No measurements yet.", text_color="gray", font=("Arial", 10)).pack(anchor="w")
            return
        for idx, m in enumerate(self.measurements):
            row = ctk.CTkFrame(self.measurements_frame, fg_color="gray23", corner_radius=6)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"{m['label']}: {m['real_length']:.3g} {self.image_scale_unit}", anchor="w").pack(side="left", fill="x", expand=True, padx=6, pady=4)
            ctk.CTkButton(row, text="❌", width=24, height=24, fg_color="#D55E00", hover_color="#AA3300",
                         command=lambda i=idx: self._delete_measurement(i)).pack(side="right", padx=4)

    def _delete_measurement(self, idx):
        self.measurements.pop(idx)
        self._refresh_measurements_list()
        self._redraw_image()

    def _clear_measurements(self):
        self.measurements = []
        self._refresh_measurements_list()
        self._redraw_image()
        self._set_status("Measurements cleared.")

    def _clear_calibration(self):
        self.image_scale = None
        self.calibration_arrow = None
        self.scale_status_var.set("Not calibrated.")
        self._redraw_image()
        self._set_status("Calibration cleared.")

    def _on_image_scroll(self, event):
        if self.sem_image_array is None or event.inaxes is None:
            return
        sf = 1.0 / 1.15 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        xlim, ylim = self.img_ax.get_xlim(), self.img_ax.get_ylim()
        self.img_ax.set_xlim([cx - (cx - xlim[0]) * sf, cx + (xlim[1] - cx) * sf])
        self.img_ax.set_ylim([cy - (cy - ylim[0]) * sf, cy + (ylim[1] - cy) * sf])
        self.img_canvas.draw()
