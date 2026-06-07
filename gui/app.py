"""Tkinter GUI for the Yokogawa CV8000 compiler.

Wraps the same orchestration as `compiler.cli:run`, but with a form and a
live log pane. Long-running compilation runs in a background thread; stdout,
stderr, and the `compiler` logger are routed into the log widget via a queue.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# tqdm bar lines look like:
#   "Processing field stacks:  45%|████▌     | 9/20 [00:30<00:35,  3.18s/it]"
_TQDM_RE = re.compile(
    r"(?P<n>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<]+)<(?P<eta>[^,\]]+)"
)

AUTHOR = "Daniel Härtter"
AUTHOR_AFFILIATION = "University Medical Center Göttingen"
AUTHOR_EMAIL = "dani.hae@posteo.de"
REPO_URL = "https://github.com/danihae/yokogawa-cv8000-compiler"

# Allow running from a source checkout (e.g. `python gui/app.py`) by adding
# the package's src/ to sys.path. In a PyInstaller bundle the package is
# already collected as a top-level module, so this is a no-op.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from compiler.discovery import find_measurements  # noqa: E402
from compiler.export import process_fieldstacks_parallel  # noqa: E402
from compiler.processing import parse_measurements  # noqa: E402

Z_MODES = ["keep", "mip", "maxz", "osbm", "max_entropy", "min_entropy"]
Z_MODES_BF = ["keep", "osbm"]
TILE_MODES = ["per-field", "stitch"]
FORMATS = ["tiff", "zarr", "both"]


class _QueueWriter:
    """File-like object that pushes writes onto a queue.

    Carriage returns are preserved so the GUI can collapse tqdm progress
    updates into a single line.
    """

    def __init__(self, q: "queue.Queue[str]") -> None:
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover - required by file API
        pass


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self._q = q
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put(self.format(record) + "\n")
        except Exception:  # pragma: no cover
            self.handleError(record)


class CompilerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Yokogawa CV8000 Compiler")
        self.geometry("780x700")
        self.minsize(640, 560)

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None

        self._root_dir = tk.StringVar()
        self._out_dir = tk.StringVar()
        self._title = tk.StringVar()
        self._z_mode = tk.StringVar(value="maxz")
        self._z_mode_bf = tk.StringVar(value="keep")
        self._tile_mode = tk.StringVar(value="per-field")
        self._format = tk.StringVar(value="tiff")
        self._no_merge = tk.StringVar()
        self._exclude = tk.StringVar()
        self._workers = tk.IntVar(value=os.cpu_count() or 4)
        self._overwrite = tk.BooleanVar(value=True)

        self._build_form()
        self._build_footer()
        self._build_progress()
        self._build_log()
        self._poll_log_queue()

    # ----- layout --------------------------------------------------------

    def _build_form(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(side="top", fill="x")
        frm.columnconfigure(1, weight=1)

        self._dir_row(frm, 0, "Source folder", self._root_dir, self._pick_root)
        self._dir_row(frm, 1, "Output folder", self._out_dir, self._pick_out)

        ttk.Label(frm, text="Title prefix").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self._title).grid(
            row=2, column=1, columnspan=2, sticky="we", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="Z-mode (fluorescence)").grid(row=3, column=0, sticky="w", pady=3)
        ttk.OptionMenu(frm, self._z_mode, self._z_mode.get(), *Z_MODES).grid(
            row=3, column=1, sticky="w", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="Z-mode (BF3D)").grid(row=4, column=0, sticky="w", pady=3)
        ttk.OptionMenu(frm, self._z_mode_bf, self._z_mode_bf.get(), *Z_MODES_BF).grid(
            row=4, column=1, sticky="w", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="Tile mode").grid(row=5, column=0, sticky="w", pady=3)
        ttk.OptionMenu(frm, self._tile_mode, self._tile_mode.get(), *TILE_MODES).grid(
            row=5, column=1, sticky="w", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="Output format").grid(row=6, column=0, sticky="w", pady=3)
        ttk.OptionMenu(frm, self._format, self._format.get(), *FORMATS).grid(
            row=6, column=1, sticky="w", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="No-merge actions").grid(row=7, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self._no_merge).grid(
            row=7, column=1, columnspan=2, sticky="we", pady=3, padx=(6, 0)
        )
        ttk.Label(
            frm,
            text="Comma-separated, e.g. BF,2D — keep rapid bursts unmerged",
            foreground="#888",
        ).grid(row=8, column=1, columnspan=2, sticky="w", padx=(6, 0))

        ttk.Label(frm, text="Exclude keyword").grid(row=9, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self._exclude).grid(
            row=9, column=1, columnspan=2, sticky="we", pady=3, padx=(6, 0)
        )

        ttk.Label(frm, text="Workers").grid(row=10, column=0, sticky="w", pady=3)
        ttk.Spinbox(
            frm, from_=1, to=128, textvariable=self._workers, width=6
        ).grid(row=10, column=1, sticky="w", pady=3, padx=(6, 0))

        ttk.Checkbutton(
            frm, text="Overwrite existing output", variable=self._overwrite
        ).grid(row=11, column=1, sticky="w", pady=(6, 0), padx=(6, 0))

        btns = ttk.Frame(self, padding=(12, 0, 12, 8))
        btns.pack(side="top", fill="x")
        self._run_btn = ttk.Button(btns, text="Run", command=self._on_run)
        self._run_btn.pack(side="right")

    def _dir_row(self, frm, row, label, var, command) -> None:
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var).grid(
            row=row, column=1, sticky="we", pady=3, padx=(6, 6)
        )
        ttk.Button(frm, text="Browse...", command=command).grid(
            row=row, column=2, sticky="e", pady=3
        )

    def _build_log(self) -> None:
        wrap = ttk.Frame(self, padding=(12, 0, 12, 12))
        wrap.pack(side="top", fill="both", expand=True)
        ttk.Label(wrap, text="Log").pack(anchor="w")
        self._log = tk.Text(wrap, wrap="word", height=18, state="disabled")
        ysb = ttk.Scrollbar(wrap, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=ysb.set)
        self._log.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

    def _build_progress(self) -> None:
        frm = ttk.Frame(self, padding=(12, 0, 12, 6))
        frm.pack(side="top", fill="x")
        self._progress = ttk.Progressbar(frm, mode="determinate", maximum=1, value=0)
        self._progress.pack(side="top", fill="x")
        self._progress_status = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self._progress_status, foreground="#555").pack(
            side="top", anchor="w", pady=(2, 0)
        )

    def _build_footer(self) -> None:
        bar = ttk.Frame(self, padding=(12, 4, 12, 8))
        bar.pack(side="bottom", fill="x")
        ttk.Separator(bar, orient="horizontal").pack(fill="x", pady=(0, 4))
        row = ttk.Frame(bar)
        row.pack(fill="x")
        ttk.Label(
            row,
            text=f"Developed by {AUTHOR} ({AUTHOR_AFFILIATION}) — questions: ",
            foreground="#555",
        ).pack(side="left")
        email = ttk.Label(
            row, text=AUTHOR_EMAIL, foreground="#1a5fb4", cursor="hand2"
        )
        email.pack(side="left")
        email.bind("<Button-1>", lambda _e: webbrowser.open(f"mailto:{AUTHOR_EMAIL}"))
        repo = ttk.Label(
            row, text="GitHub", foreground="#1a5fb4", cursor="hand2"
        )
        repo.pack(side="right")
        repo.bind("<Button-1>", lambda _e: webbrowser.open(REPO_URL))

    # ----- pickers -------------------------------------------------------

    def _pick_root(self) -> None:
        p = filedialog.askdirectory(title="Select source folder")
        if p:
            self._root_dir.set(p)

    def _pick_out(self) -> None:
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)
        if p:
            self._out_dir.set(p)

    # ----- log streaming -------------------------------------------------

    def _append_log(self, s: str) -> None:
        m = _TQDM_RE.search(s)
        if m:
            self._update_progress(
                int(m["n"]), int(m["total"]), m["elapsed"].strip(), m["eta"].strip()
            )
        self._log.configure(state="normal")
        # Handle tqdm-style carriage returns: replace the current line.
        if "\r" in s and "\n" not in s:
            self._log.delete("end-1l", "end-1c")
            self._log.insert("end", s.replace("\r", ""))
        else:
            self._log.insert("end", s)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _update_progress(self, n: int, total: int, elapsed: str, eta: str) -> None:
        if self._progress["mode"] != "determinate":
            self._progress.stop()
            self._progress.configure(mode="determinate")
        self._progress.configure(maximum=total, value=n)
        pct = (n / total * 100) if total else 0
        self._progress_status.set(
            f"{n}/{total} field stacks · {pct:0.0f}% · elapsed {elapsed} · ETA {eta}"
        )

    def _poll_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    # ----- run -----------------------------------------------------------

    def _on_run(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        root_dir = self._root_dir.get().strip()
        out_dir = self._out_dir.get().strip()
        if not root_dir or not Path(root_dir).is_dir():
            messagebox.showerror("Invalid input", "Source folder is not set or does not exist.")
            return
        if not out_dir:
            messagebox.showerror("Invalid input", "Output folder is not set.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        fmt = self._format.get()
        if fmt in ("zarr", "both"):
            try:
                import zarr  # noqa: F401
                import ome_zarr  # noqa: F401
            except ImportError:
                messagebox.showerror(
                    "Missing dependency",
                    "OME-Zarr output requires the [zarr] extra:\n\n"
                    "    uv pip install -e '.[zarr]'",
                )
                return

        no_merge_raw = self._no_merge.get().strip()
        no_merge_actions = (
            [a.strip() for a in no_merge_raw.split(",") if a.strip()]
            if no_merge_raw
            else None
        )

        params = {
            "root_dir": Path(root_dir),
            "out_dir": Path(out_dir),
            "title": self._title.get().strip() or None,
            "z_mode": self._z_mode.get(),
            "z_mode_bf": self._z_mode_bf.get(),
            "tile_mode": self._tile_mode.get(),
            "format": fmt,
            "no_merge_actions": no_merge_actions,
            "exclude": self._exclude.get().strip() or None,
            "max_workers": int(self._workers.get()),
            "overwrite": bool(self._overwrite.get()),
        }

        self._run_btn.configure(state="disabled", text="Running...")
        self._progress.configure(mode="indeterminate", maximum=100, value=0)
        self._progress.start(80)
        self._progress_status.set("Scanning measurements...")
        self._worker = threading.Thread(
            target=self._run_worker, args=(params,), daemon=True
        )
        self._worker.start()

    def _run_worker(self, p: dict) -> None:
        writer = _QueueWriter(self._log_queue)
        log_handler = _QueueLogHandler(self._log_queue)
        compiler_logger = logging.getLogger("compiler")
        prev_level = compiler_logger.level
        compiler_logger.addHandler(log_handler)
        compiler_logger.setLevel(logging.INFO)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        try:
            print(
                f"Starting compilation\n"
                f"  Source: {p['root_dir']}\n"
                f"  Destination: {p['out_dir']}\n"
                f"  Tile mode: {p['tile_mode']}\n"
                f"  Format: {p['format']}"
            )
            wpi_paths = list(find_measurements(p["root_dir"]))
            if not wpi_paths:
                print("No .wpi files found. Nothing to do.")
                return
            print(f"Found {len(wpi_paths)} measurement(s).")
            merged_records_df, acquisitions = parse_measurements(
                wpi_paths, exclude_keyword=p["exclude"]
            )
            process_fieldstacks_parallel(
                merged_records_df,
                acquisitions,
                p["out_dir"],
                title=p["title"],
                z_mode=p["z_mode"],
                z_mode_BF=p["z_mode_bf"],
                overwrite=p["overwrite"],
                max_workers=p["max_workers"],
                tile_mode=p["tile_mode"],
                no_merge_actions=p["no_merge_actions"],
                format=p["format"],
            )
            print("Done.")
        except Exception as exc:  # surface the error in the log pane
            import traceback

            print("ERROR: " + str(exc))
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            compiler_logger.removeHandler(log_handler)
            compiler_logger.setLevel(prev_level)
            self.after(0, self._on_finished)

    def _on_finished(self) -> None:
        self._run_btn.configure(state="normal", text="Run")
        self._progress.stop()
        self._progress.configure(mode="determinate")
        if self._progress["maximum"] and self._progress["value"] >= self._progress["maximum"]:
            self._progress_status.set("Done")
        else:
            cur = self._progress_status.get()
            self._progress_status.set(cur if cur and cur != "Scanning measurements..." else "Stopped")


def main() -> None:
    CompilerApp().mainloop()


if __name__ == "__main__":
    main()
