"""PDF report generation, built from scratch as a standalone module: a
cover page with sample metadata, per-zone chemical statistics (auto and
manual zones together), the graph, an optional SEM image page with
calibration/measurement overlays, and a distance measurements table."""

import datetime

import customtkinter as ctk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .constants import ACCENT, LEGEND_POSITIONS
from .data_processing import get_pos_col

LINES_PER_PAGE = 55


class ReportInfoDialog(ctk.CTkToplevel):
    """Collects cover-page metadata right before generating the PDF."""

    def __init__(self, master, on_confirm, has_image=False):
        super().__init__(master)
        self.title("PDF Report — Details")
        self.geometry("380x460")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.on_confirm = on_confirm

        ctk.CTkLabel(self, text="Report Details", font=("Helvetica", 14, "bold"), text_color=ACCENT).pack(pady=(15, 10))

        ctk.CTkLabel(self, text="Sample Name :").pack(anchor="w", padx=15)
        self.sample_entry = ctk.CTkEntry(self, placeholder_text="e.g. Sample A-12")
        self.sample_entry.pack(fill="x", padx=15, pady=(2, 10))

        ctk.CTkLabel(self, text="Operator :").pack(anchor="w", padx=15)
        self.operator_entry = ctk.CTkEntry(self, placeholder_text="e.g. J. Dupont")
        self.operator_entry.pack(fill="x", padx=15, pady=(2, 10))

        ctk.CTkLabel(self, text="Notes :").pack(anchor="w", padx=15)
        self.notes_box = ctk.CTkTextbox(self, height=100)
        self.notes_box.pack(fill="x", padx=15, pady=(2, 10))

        self.include_image_var = ctk.BooleanVar(value=has_image)
        chk = ctk.CTkCheckBox(self, text="Include SEM image page", variable=self.include_image_var)
        chk.pack(anchor="w", padx=15, pady=(0, 10))
        if not has_image:
            chk.configure(state="disabled")

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=15, side="bottom")
        ctk.CTkButton(btn_row, text="Cancel", fg_color="gray40", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_row, text="📑 Generate PDF", fg_color="#882255", hover_color="#551133", command=self._confirm).pack(side="right", padx=5)

    def _confirm(self):
        meta = {
            "sample": self.sample_entry.get().strip() or "—",
            "operator": self.operator_entry.get().strip() or "—",
            "notes": self.notes_box.get("1.0", "end").strip(),
            "include_image": self.include_image_var.get(),
        }
        self.on_confirm(meta)
        self.destroy()


def _new_text_page():
    fig = Figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    return fig, ax


def _zone_stats(df, pos_col, elements, zone):
    mask = (df[pos_col] >= zone["start"]) & (df[pos_col] <= zone["end"])
    sub = df[mask]
    stats = {}
    for el in elements:
        if len(sub) == 0:
            stats[el] = (float("nan"), float("nan"), float("nan"))
        else:
            stats[el] = (sub[el].mean(), sub[el].min(), sub[el].max())
    return stats


def generate_pdf_report(app, path, meta):
    """Builds the multi-page PDF report for the current app state. `app` is
    the EDXApp instance; `meta` is the dict produced by ReportInfoDialog."""
    pos_col = get_pos_col(app.df_norm)
    all_zones = (
        [dict(z, kind="Auto") for z in app.detected_zones]
        + [dict(z, kind="Manual") for z in app.manual_zones]
    )
    all_zones.sort(key=lambda z: z["start"])

    with PdfPages(path) as pdf:
        _add_cover_page(pdf, app, meta, all_zones)
        if all_zones:
            _add_zones_pages(pdf, app, pos_col, all_zones)
        _add_graph_page(pdf, app)
        if meta.get("include_image") and app.sem_image_array is not None:
            _add_image_page(pdf, app)
        if app.measurements:
            _add_measurements_page(pdf, app)


def _add_cover_page(pdf, app, meta, all_zones):
    fig, ax = _new_text_page()
    lines = [
        "EDX Line Scan — Analysis Report",
        "=" * 60,
        "",
        f"Sample Name     : {meta['sample']}",
        f"Operator        : {meta['operator']}",
        f"Report Date     : {datetime.date.today().isoformat()}",
        "",
        f"File Analyzed   : {app.filepath_var.get()}",
        f"Data Mode       : {'100% Normalized' if app.is_normalized else 'Raw Intensity'}",
        f"Smoothing       : {app.smooth_window.get()} point(s)",
        f"Elements        : {', '.join(app.elements)}",
        f"Zones           : {len(all_zones)} total ({len(app.detected_zones)} auto, {len(app.manual_zones)} manual)",
        f"SEM Image       : {('Yes — ' + str(app.sem_image_path)) if app.sem_image_array is not None else 'None'}",
        f"Measurements    : {len(app.measurements)}",
        "",
        "Notes:",
        meta["notes"] or "(none)",
    ]
    ax.text(0.06, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11, va="top", fontfamily="monospace")
    pdf.savefig(fig)


def _add_zones_pages(pdf, app, pos_col, all_zones):
    lines = ["Detected & Manual Zones — Chemical Summary", "=" * 60, ""]
    for z in all_zones:
        tag = "[AUTO]  " if z["kind"] == "Auto" else "[MANUAL]"
        lines.append(f"{tag} {z['name']}   [{z['start']:.2f} → {z['end']:.2f}]")
        stats = _zone_stats(app.df_norm, pos_col, app.elements, z)
        for el, (mean, mn, mx) in stats.items():
            lines.append(f"    {el:<6} mean={mean:6.2f}  min={mn:6.2f}  max={mx:6.2f}")
        lines.append("")

    # Paginate by line count so a long zone list doesn't overflow one page.
    for i in range(0, len(lines), LINES_PER_PAGE):
        fig, ax = _new_text_page()
        chunk = "\n".join(lines[i:i + LINES_PER_PAGE])
        ax.text(0.05, 0.95, chunk, transform=ax.transAxes, fontsize=10, va="top", fontfamily="monospace")
        pdf.savefig(fig)


def _add_graph_page(pdf, app):
    # Temporarily swap to print-friendly (white) axis colors, then restore
    # the app's normal dark theme via its own redraw logic.
    try:
        app.fig.patch.set_facecolor("white")
        app.ax.set_facecolor("white")
        app.ax.tick_params(colors="black")
        for s in app.ax.spines.values():
            s.set_edgecolor("black")
        app.ax.xaxis.label.set_color("black")
        app.ax.yaxis.label.set_color("black")
        app.ax.title.set_color("black")
        if any(v.get() for v in app.el_vars.values()):
            pos = LEGEND_POSITIONS.get(app.legend_pos_var.get(), "outside right")
            args = {"frameon": True, "facecolor": "white", "labelcolor": "black"}
            if pos == "outside right":
                app.ax.legend(**args, loc="upper left", bbox_to_anchor=(1.02, 1))
            else:
                app.ax.legend(**args, loc=pos)
        pdf.savefig(app.fig, bbox_inches="tight")
    finally:
        app._style_axes()
        app._plot()


def _add_image_page(pdf, app):
    fig = Figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(app.sem_image_array)

    caption = f"SEM Image: {app.sem_image_path or ''}"
    if app.image_scale is not None:
        caption += f"\nScale: {app.image_scale:.4g} {app.image_scale_unit}/px"
    ax.set_title(caption, fontsize=10)

    if app.calibration_arrow:
        ax.annotate("", xy=app.calibration_arrow["p2"], xytext=app.calibration_arrow["p1"],
                    arrowprops=dict(arrowstyle="<->", color="#E69F00", linewidth=1.5))
    for m in app.measurements:
        ax.annotate("", xy=m["p2"], xytext=m["p1"], arrowprops=dict(arrowstyle="<->", color="#0072B2", linewidth=1.5))
        mid = ((m["p1"][0] + m["p2"][0]) / 2, (m["p1"][1] + m["p2"][1]) / 2)
        ax.text(mid[0], mid[1], f"{m['label']}: {m['real_length']:.3g} {app.image_scale_unit}",
                color="#0072B2", fontsize=8, ha="center", va="bottom")

    pdf.savefig(fig, bbox_inches="tight")


def _add_measurements_page(pdf, app):
    fig, ax = _new_text_page()
    lines = ["Distance Measurements", "=" * 60, "", f"{'Label':<8}{'Pixels':>12}    {'Real Length':>14}", "-" * 44]
    for m in app.measurements:
        lines.append(f"{m['label']:<8}{m['pixel_length']:>10.1f}px    {m['real_length']:>10.3f} {app.image_scale_unit}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11, va="top", fontfamily="monospace")
    pdf.savefig(fig)
