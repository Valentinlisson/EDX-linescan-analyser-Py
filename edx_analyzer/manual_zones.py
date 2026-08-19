"""Manual zone creation/editing: a dialog to name a zone and fine-tune its
numeric bounds (since a mouse drag on noisy data is rarely pixel-precise),
and a manager window to review/edit/delete the zones already placed."""

import customtkinter as ctk
from tkinter import messagebox

from .constants import ACCENT, DEFAULT_ZONE_ALPHA
from .widgets import ColorPickerDialog

DEFAULT_ZONE_COLOR = "#CC79A7"


class ManualZoneDialog(ctk.CTkToplevel):
    """Prompts for a zone name, precise start/end position, a color and how
    transparent its shading is. `on_confirm` is called with the resulting
    zone dict when saved."""

    def __init__(self, master, x_min, x_max, initial_name="", initial_color=DEFAULT_ZONE_COLOR,
                 initial_alpha=DEFAULT_ZONE_ALPHA, on_confirm=None):
        super().__init__(master)
        self.title("Manual Zone")
        self.geometry("320x430")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.on_confirm = on_confirm
        self.color = initial_color

        ctk.CTkLabel(self, text="Manual Zone", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=(15, 10))

        ctk.CTkLabel(self, text="Zone Name :").pack(anchor="w", padx=15)
        self.name_entry = ctk.CTkEntry(self, placeholder_text="e.g. Interface, Oxide layer...")
        self.name_entry.insert(0, initial_name)
        self.name_entry.pack(fill="x", padx=15, pady=(2, 10))

        ctk.CTkLabel(self, text="Start Position :").pack(anchor="w", padx=15)
        self.start_entry = ctk.CTkEntry(self)
        self.start_entry.insert(0, f"{x_min:.3f}")
        self.start_entry.pack(fill="x", padx=15, pady=(2, 10))

        ctk.CTkLabel(self, text="End Position :").pack(anchor="w", padx=15)
        self.end_entry = ctk.CTkEntry(self)
        self.end_entry.insert(0, f"{x_max:.3f}")
        self.end_entry.pack(fill="x", padx=15, pady=(2, 10))

        color_row = ctk.CTkFrame(self, fg_color="transparent")
        color_row.pack(fill="x", padx=15, pady=(0, 10))
        ctk.CTkLabel(color_row, text="Color :").pack(side="left")
        self.color_btn = ctk.CTkButton(color_row, text="", width=32, height=25, fg_color=initial_color,
                                       border_width=1, border_color="gray50", command=self._pick_color)
        self.color_btn.pack(side="right")

        alpha_row = ctk.CTkFrame(self, fg_color="transparent")
        alpha_row.pack(fill="x", padx=15)
        ctk.CTkLabel(alpha_row, text="Shading opacity :").pack(side="left")
        self.alpha_var = ctk.DoubleVar(value=float(initial_alpha))
        self.alpha_label = ctk.CTkLabel(alpha_row, text=f"{int(float(initial_alpha) * 100)} %", text_color=ACCENT)
        self.alpha_label.pack(side="right")
        ctk.CTkSlider(self, from_=0.0, to=1.0, variable=self.alpha_var,
                      command=lambda v: self.alpha_label.configure(text=f"{int(float(v) * 100)} %")
                      ).pack(fill="x", padx=15, pady=(2, 10))
        ctk.CTkLabel(self, text="0 % = outline only, no shading over the curves.",
                     text_color="gray", font=("Arial", 10)).pack(anchor="w", padx=15)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=15, side="bottom")
        ctk.CTkButton(btn_row, text="Cancel", fg_color="gray40", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_row, text="✔ Save Zone", fg_color="#009E73", hover_color="#007755", command=self._confirm).pack(side="right", padx=5)

    def _pick_color(self):
        color = ColorPickerDialog.ask_color(self, initial_color=self.color, title="Zone Color")
        if color:
            self.color = color
            self.color_btn.configure(fg_color=color)

    def _confirm(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please give the zone a name.")
            return
        try:
            start = float(self.start_entry.get().strip())
            end = float(self.end_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Start/End must be numbers.")
            return
        if start >= end:
            messagebox.showerror("Error", "Start must be smaller than End.")
            return
        if self.on_confirm:
            self.on_confirm({"start": start, "end": end, "name": name, "color": self.color,
                             "alpha": round(float(self.alpha_var.get()), 3)})
        self.destroy()


class ManualZoneManagerWindow(ctk.CTkToplevel):
    """Lists existing manual zones with edit/delete actions."""

    def __init__(self, master, zones, on_update_callback):
        super().__init__(master)
        self.title("Manual Zones")
        self.geometry("480x500")
        self.transient(master)
        self.grab_set()
        self.zones = list(zones)
        self.on_update_callback = on_update_callback
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Manual Zones", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=10)
        self.scroll_list = ctk.CTkScrollableFrame(self)
        self.scroll_list.pack(fill="both", expand=True, padx=10, pady=5)
        self._refresh()

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(bottom, text="✔ Close & Apply", fg_color="#009E73", hover_color="#007755", command=self._apply_and_close).pack(side="right")

    def _refresh(self):
        for w in self.scroll_list.winfo_children():
            w.destroy()
        if not self.zones:
            ctk.CTkLabel(self.scroll_list, text="No manual zones yet.", text_color="gray").pack(pady=10)
            return
        for idx, z in enumerate(self.zones):
            box = ctk.CTkFrame(self.scroll_list, fg_color="gray23", corner_radius=6)
            box.pack(fill="x", pady=3, padx=2)
            ctk.CTkFrame(box, width=14, height=14, fg_color=z.get("color", DEFAULT_ZONE_COLOR), corner_radius=3).pack(side="left", padx=(8, 4), pady=8)
            alpha = float(z.get("alpha", DEFAULT_ZONE_ALPHA))
            desc = f"{z['name']}   [{z['start']:.2f} → {z['end']:.2f}]   opacity {int(alpha * 100)}%"
            ctk.CTkLabel(box, text=desc, anchor="w").pack(side="left", fill="x", expand=True, padx=4, pady=6)
            ctk.CTkButton(box, text="❌", width=28, height=28, fg_color="#D55E00", hover_color="#AA3300",
                         command=lambda i=idx: self._delete(i)).pack(side="right", padx=3)
            ctk.CTkButton(box, text="✏", width=28, height=28, fg_color="gray35",
                         command=lambda i=idx: self._edit(i)).pack(side="right", padx=3)

    def _edit(self, idx):
        z = self.zones[idx]

        def on_confirm(new_zone):
            self.zones[idx] = new_zone
            self._refresh()

        ManualZoneDialog(self, z["start"], z["end"], initial_name=z["name"],
                         initial_color=z.get("color", DEFAULT_ZONE_COLOR),
                         initial_alpha=z.get("alpha", DEFAULT_ZONE_ALPHA), on_confirm=on_confirm)

    def _delete(self, idx):
        self.zones.pop(idx)
        self._refresh()

    def _apply_and_close(self):
        self.on_update_callback(self.zones)
        self.destroy()
