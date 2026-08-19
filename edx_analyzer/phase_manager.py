"""Popup window for creating, editing and persisting chemical phase presets
used to auto-segment a line scan into zones."""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json

from .constants import ACCENT


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

        ctk.CTkLabel(right_frame, text="Phase Name (e.g., Oxide, 304 Steel, etc.) :").pack(anchor="w", padx=15, pady=(5, 0))
        self.phase_name_entry = ctk.CTkEntry(right_frame, placeholder_text="Enter identification name...")
        self.phase_name_entry.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(right_frame, text="Chemical thresholds per element :", font=("Helvetica", 12, "italic")).pack(anchor="w", padx=15, pady=(10, 5))

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
        for widget in self.scroll_list.winfo_children():
            widget.destroy()
        for idx, phase in enumerate(self.presets):
            box = ctk.CTkFrame(self.scroll_list, fg_color="gray23", corner_radius=6)
            box.pack(fill="x", pady=3, padx=2)

            desc_text = f"⭐ {phase['nom_phase']}\n"
            conds = []
            for el, lims in phase.get("conditions", {}).items():
                c_str = f"{el} ["
                if "min" in lims:
                    c_str += f"≥{lims['min']}%"
                if "max" in lims:
                    c_str += f" ≤{lims['max']}%"
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
                    try:
                        lims["min"] = float(min_v)
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid Min value for {el}: '{min_v}' is not a number.")
                        return
                if max_v:
                    try:
                        lims["max"] = float(max_v)
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid Max value for {el}: '{max_v}' is not a number.")
                        return
                if "min" in lims and "max" in lims and lims["min"] > lims["max"]:
                    messagebox.showerror("Error", f"For {el}: Min ({lims['min']}%) cannot be greater than Max ({lims['max']}%).")
                    return
                if lims:
                    conditions[el] = lims

        self.presets.append({"nom_phase": name, "conditions": conditions})
        self.phase_name_entry.delete(0, tk.END)
        for w in self.element_inputs.values():
            w["active"].set(False)
            w["min"].delete(0, tk.END)
            w["max"].delete(0, tk.END)
        self._refresh_presets_display()

    def _load_json_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON File", "*.json")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                self.presets = data
                self._refresh_presets_display()
            else:
                messagebox.showerror("Incorrect Structure", "The file must be structured as a list of phases.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _save_json_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON File", "*.json")])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Success", "The identification preset has been successfully encoded and saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to save file: {str(e)}")

    def _apply_and_close(self):
        self.on_save_callback(self.presets)
        self.destroy()
