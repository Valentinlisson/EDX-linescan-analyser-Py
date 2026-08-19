"""Small reusable UI widgets: hover tooltips and a live-preview color picker
that replaces tkinter's blocking colorchooser dialog."""

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk


class Tooltip:
    """Lightweight hover tooltip attached to any tkinter/customtkinter widget."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._tip_window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel_pending()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_pending(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background="#2B2B2B", foreground="white",
                 relief="solid", borderwidth=1, font=("Arial", 10), padx=6, pady=3).pack()

    def _hide(self, event=None):
        self._cancel_pending()
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


def add_tooltip(widget, text):
    return Tooltip(widget, text)


def hex_to_rgb(hexcode):
    hexcode = hexcode.lstrip("#")
    if len(hexcode) != 6:
        raise ValueError(f"Invalid hex color: {hexcode}")
    return tuple(int(hexcode[i:i + 2], 16) for i in (0, 2, 4))


class ColorPickerDialog(ctk.CTkToplevel):
    """Modal color picker with RGB sliders, a hex entry and a live preview
    swatch, so the effect of a color is visible before it's confirmed."""

    def __init__(self, master, initial_color="#FFFFFF", title="Pick a color"):
        super().__init__(master)
        self.title(title)
        self.geometry("300x300")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.result = None
        try:
            r, g, b = hex_to_rgb(initial_color)
        except ValueError:
            r, g, b = (255, 255, 255)

        self.preview = ctk.CTkFrame(self, height=60, fg_color=initial_color, corner_radius=8)
        self.preview.pack(padx=15, pady=15, fill="x")

        self.r_var, self.g_var, self.b_var = ctk.IntVar(value=r), ctk.IntVar(value=g), ctk.IntVar(value=b)
        self._slider_row("R", self.r_var)
        self._slider_row("G", self.g_var)
        self._slider_row("B", self.b_var)

        hex_row = ctk.CTkFrame(self, fg_color="transparent")
        hex_row.pack(fill="x", padx=15, pady=(10, 0))
        ctk.CTkLabel(hex_row, text="Hex :").pack(side="left")
        self.hex_entry = ctk.CTkEntry(hex_row, width=100)
        self.hex_entry.insert(0, f"#{r:02X}{g:02X}{b:02X}")
        self.hex_entry.pack(side="left", padx=5)
        ctk.CTkButton(hex_row, text="Apply", width=60, command=self._apply_hex).pack(side="left")

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=15, side="bottom")
        ctk.CTkButton(btn_row, text="Cancel", fg_color="gray40", command=self._cancel).pack(side="right", padx=5)
        ctk.CTkButton(btn_row, text="OK", command=self._confirm).pack(side="right", padx=5)

    def _slider_row(self, label, var):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=2)
        ctk.CTkLabel(row, text=label, width=15).pack(side="left")
        ctk.CTkSlider(row, from_=0, to=255, variable=var, number_of_steps=255,
                      command=lambda v: self._update_preview()).pack(side="left", fill="x", expand=True, padx=5)

    def _current_hex(self):
        return "#{:02X}{:02X}{:02X}".format(int(self.r_var.get()), int(self.g_var.get()), int(self.b_var.get()))

    def _update_preview(self):
        hexcode = self._current_hex()
        self.preview.configure(fg_color=hexcode)
        self.hex_entry.delete(0, "end")
        self.hex_entry.insert(0, hexcode)

    def _apply_hex(self):
        try:
            r, g, b = hex_to_rgb(self.hex_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid hex color code (expected format #RRGGBB).")
            return
        self.r_var.set(r)
        self.g_var.set(g)
        self.b_var.set(b)
        self.preview.configure(fg_color=self._current_hex())

    def _confirm(self):
        self.result = self._current_hex()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    @classmethod
    def ask_color(cls, master, initial_color="#FFFFFF", title="Pick a color"):
        dlg = cls(master, initial_color, title)
        master.wait_window(dlg)
        return dlg.result
