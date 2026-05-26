"""
AquaSense — Carp Biomass Estimator
====================================
Single-screen two-stage pipeline:
  1. Presence detection  (bream_carp_presence ensemble, NMS)
  2. Weight regression   (combined or salinity-aware model)

Usage:
    python run.py
"""

import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.infer_twostage import run_twostage_inference

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#1565C0"
C_LBLUE  = "#1976D2"
C_GREEN  = "#2E7D32"
C_WHITE  = "#FFFFFF"
C_LBG    = "#F5F7FA"
C_DGREY  = "#444444"
C_MGREY  = "#888888"
C_ACCENT = "#0288D1"

C_SEG_ON  = C_BLUE
C_SEG_OFF = "#CFD8DC"
C_SEG_TXT_ON  = C_WHITE
C_SEG_TXT_OFF = C_DGREY


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AquaSense  |  Carp Biomass Estimator")
        self.configure(bg=C_LBG)
        self.resizable(True, True)
        self.minsize(700, 660)
        self._weight_variant = tk.StringVar(value="no_sal")
        self._build_ui()


    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_header()
        main = tk.Frame(self, bg=C_LBG, padx=24, pady=16)
        main.pack(fill="both", expand=True)
        self._build_folder(main)
        self._build_model_toggle(main)
        self._build_run_row(main)
        self._build_progress(main)
        self._build_results(main)
        self._build_per_file(main)

    def _build_header(self):
        hdr = tk.Frame(self, bg=C_BLUE, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="AquaSense  |  Carp Biomass Estimator",
                 bg=C_BLUE, fg=C_WHITE,
                 font=("Helvetica", 17, "bold")).pack()
        tk.Label(hdr, text="Two-stage detection · presence + weight estimation",
                 bg=C_BLUE, fg="#BBDEFB",
                 font=("Helvetica", 10)).pack(pady=(2, 0))

    def _build_folder(self, parent):
        tk.Label(parent, text="Data Folder", font=("Helvetica", 11, "bold"),
                 bg=C_LBG, fg=C_DGREY, anchor="w").pack(fill="x", pady=(0, 2))

        row = tk.Frame(parent, bg=C_LBG)
        row.pack(fill="x", pady=(0, 4))
        self._folder_var = tk.StringVar()
        tk.Entry(row, textvariable=self._folder_var,
                 font=("Courier", 10), bg=C_WHITE, relief="solid", bd=1
                 ).pack(side="left", fill="x", expand=True)
        tk.Button(row, text=" Browse… ", command=self._browse,
                  bg=C_BLUE, fg=C_WHITE, relief="flat",
                  font=("Helvetica", 10), padx=6
                  ).pack(side="left", padx=(6, 0))

        tk.Label(parent,
                 text="Point to a folder with *.csv recordings.",
                 font=("Helvetica", 9), fg=C_MGREY, bg=C_LBG
                 ).pack(anchor="w", pady=(0, 14))

    def _build_model_toggle(self, parent):
        tk.Label(parent, text="Weight Model", font=("Helvetica", 11, "bold"),
                 bg=C_LBG, fg=C_DGREY, anchor="w").pack(fill="x", pady=(0, 6))

        seg_frame = tk.Frame(parent, bg=C_LBG)
        seg_frame.pack(anchor="w", pady=(0, 14))

        self._btn_no_sal = tk.Button(
            seg_frame,
            text="No Salinity",
            command=lambda: self._set_variant("no_sal"),
            relief="flat", font=("Helvetica", 10, "bold"),
            padx=18, pady=6, cursor="hand2", bd=0,
        )
        self._btn_no_sal.pack(side="left")

        self._btn_sal = tk.Button(
            seg_frame,
            text="Salinity-Aware  (reads _S400 from filename)",
            command=lambda: self._set_variant("filename_sal"),
            relief="flat", font=("Helvetica", 10, "bold"),
            padx=18, pady=6, cursor="hand2", bd=0,
        )
        self._btn_sal.pack(side="left", padx=(2, 0))

        self._set_variant("no_sal")   # initialise colours

    def _set_variant(self, v: str):
        self._weight_variant.set(v)
        on, off = (self._btn_no_sal, self._btn_sal) if v == "no_sal" else (self._btn_sal, self._btn_no_sal)
        on.config(bg=C_SEG_ON,  fg=C_SEG_TXT_ON)
        off.config(bg=C_SEG_OFF, fg=C_SEG_TXT_OFF)

    def _build_run_row(self, parent):
        tk.Frame(parent, height=1, bg="#CCCCCC").pack(fill="x", pady=(0, 10))
        self._run_btn = tk.Button(
            parent, text="▶   Run Analysis",
            command=self._start_inference,
            bg=C_GREEN, fg=C_WHITE,
            font=("Helvetica", 13, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2",
        )
        self._run_btn.pack(pady=(0, 8))

    def _build_progress(self, parent):
        self._prog_lbl = tk.Label(parent, text="", fg=C_MGREY,
                                   font=("Helvetica", 9), bg=C_LBG)
        self._prog_lbl.pack(anchor="w")
        self._prog_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Aqua.Horizontal.TProgressbar",
                        troughcolor="#DDD", background=C_ACCENT)
        ttk.Progressbar(parent, variable=self._prog_var,
                        maximum=100, style="Aqua.Horizontal.TProgressbar",
                        length=600).pack(fill="x", pady=(2, 12))

    def _build_results(self, parent):
        frame = tk.LabelFrame(parent, text="  Results  ",
                               font=("Helvetica", 11, "bold"),
                               bg=C_LBG, fg=C_DGREY, padx=16, pady=10)
        frame.pack(fill="x", pady=(0, 8))

        self._fish_var   = tk.StringVar(value="—")
        self._avgwt_var  = tk.StringVar(value="—")
        self._biomass_var = tk.StringVar(value="—")

        grid = tk.Frame(frame, bg=C_LBG)
        grid.pack(fill="x")

        for col, (label, var, big) in enumerate([
            ("Fish Detected",    self._fish_var,    False),
            ("Avg Fish Weight",  self._avgwt_var,   False),
            ("Total Biomass",    self._biomass_var,  True),
        ]):
            tk.Label(grid, text=label, font=("Helvetica", 9),
                     fg=C_MGREY, bg=C_LBG).grid(row=0, column=col, padx=32, pady=(0, 2))
            fnt = ("Helvetica", 18, "bold") if big else ("Helvetica", 14, "bold")
            clr = C_BLUE if big else C_DGREY
            tk.Label(grid, textvariable=var, font=fnt, fg=clr,
                     bg=C_LBG).grid(row=1, column=col, padx=32)

    def _build_per_file(self, parent):
        frame = tk.LabelFrame(parent, text="  Per-File Breakdown  ",
                               font=("Helvetica", 10, "bold"),
                               bg=C_LBG, fg=C_DGREY)
        frame.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("File", "Fish Count", "Avg Weight (g)", "Biomass (g)", "Biomass (kg)")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", height=6)
        for col in cols:
            self._tree.heading(col, text=col)
            width = 260 if col == "File" else 110
            self._tree.column(col, width=width,
                              anchor="w" if col == "File" else "center")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse(self):
        d = filedialog.askdirectory(title="Select folder containing CSV recordings")
        if d:
            self._folder_var.set(d)

    def _start_inference(self):
        folder = self._folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder selected",
                                   "Please select a data folder first.")
            return
        if not Path(folder).is_dir():
            messagebox.showerror("Folder not found",
                                 f"Cannot find directory:\n{folder}")
            return

        self._run_btn.config(state="disabled")
        self._fish_var.set("—")
        self._avgwt_var.set("—")
        self._biomass_var.set("—")
        self._prog_var.set(0)
        self._prog_lbl.config(text="Starting…")
        for row in self._tree.get_children():
            self._tree.delete(row)

        variant = self._weight_variant.get()

        def progress(cur, _total, msg):
            self._prog_var.set(cur)
            self._prog_lbl.config(text=msg)
            self.update_idletasks()

        def worker():
            try:
                result = run_twostage_inference(
                    data_dir=folder,
                    weight_variant=variant,
                    progress_callback=progress,
                )
                self.after(0, lambda r=result: self._show_results(r))
            except Exception as exc:
                self.after(0, lambda e=exc: (
                    messagebox.showerror("Error during inference", str(e)),
                    self._run_btn.config(state="normal"),
                    self._prog_lbl.config(text="Error — see dialog above."),
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, result: dict):
        n    = result["n_fish"]
        avg  = result["avg_weight_g"]
        bio  = result["biomass_g"]

        self._fish_var.set(f"{n:,}")
        self._avgwt_var.set(f"{avg:,.0f} g")
        self._biomass_var.set(f"{bio/1000:.2f} kg  ({bio:,.0f} g)")
        self._prog_var.set(100)
        model_lbl = "No Salinity" if result["weight_variant"] == "no_sal" else "Salinity-Aware"
        self._prog_lbl.config(text=f"Done  [{model_lbl}]")
        self._run_btn.config(state="normal")

        for row in self._tree.get_children():
            self._tree.delete(row)
        for pf in result["per_file"]:
            bg_g  = pf["biomass_g"]
            self._tree.insert("", "end", values=(
                pf["file"],
                f"{pf['n_fish']:,}",
                f"{pf['avg_weight_g']:,.0f}" if pf["n_fish"] > 0 else "—",
                f"{bg_g:,.0f}",
                f"{bg_g / 1000:.3f}",
            ))


# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
