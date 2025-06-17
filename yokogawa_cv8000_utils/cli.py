# src/ycvcompiler/cli.py
"""
Command-line interface for Yokogawa CV8000 compiler.
"""
from __future__ import annotations
import logging, os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from yokogawa_cv8000_utils import discovery, processing, export   # local modules

# ──────────────────────────────────────────────
# 1.  Enum to replace Literal["keep","mip","maxz"]
# ──────────────────────────────────────────────
class ZMode(str, Enum):
    keep = "keep"      # keep full Z
    mip  = "mip"       # maximum-intensity projection
    maxz = "maxz"      # pick single brightest slice


console = Console()
app = typer.Typer(
    name="cv8000-compile",
    help="Fast compiler for Yokogawa CV8000 data → FAIR OME-TIFF",
    add_completion=False,
)


# ──────────────────────────────────────────────
# 2.  logging helper
# ──────────────────────────────────────────────
def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tifffile").setLevel(logging.WARNING)


def _auto_workers() -> int:
    return max(1, os.cpu_count() or 4)


# ──────────────────────────────────────────────
# 3.  compile command
# ──────────────────────────────────────────────
@app.command()
def compile(
    src: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        readable=True,
        help="Folder with CV8000 measurements (.zip or dirs)",
    ),
    output: Path = typer.Argument(..., help="Destination folder for OME-TIFFs"),
    z_mode: ZMode = typer.Option(         # ← Enum instead of Literal
        ZMode.keep,
        "--z-mode",
        "-z",
        help="Z handling: keep | mip | maxz",
    ),
    compress: str = typer.Option(
        "zstd",
        "--compress",
        "-c",
        help="tifffile compression ('zstd', 'lzma', 'deflate', 'none')",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        min=1,
        help="Parallel writer processes (default: all cores)",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List tasks, don’t write"),
) -> None:
    """
    Compile Yokogawa CV8000 measurements → one OME-TIFF per Action.
    """
    _setup_logging(verbose)
    log = logging.getLogger("cli")

    workers = workers or _auto_workers()
    output.mkdir(parents=True, exist_ok=True)

    console.print("[blue]Scanning for measurements…[/blue]")
    mlf_paths = list(discovery.find_measurements(src))
    if not mlf_paths:
        console.print(f"[yellow]No MeasurementData.mlf under {src}[/yellow]")
        raise typer.Exit()

    console.print(f"[green]Found {len(mlf_paths)} measurements[/green]")
    if dry_run:
        for p in mlf_paths:
            console.print(f" └─ {p}")
        console.print("[yellow]Dry-run finished.[/yellow]")
        raise typer.Exit()

    console.print("[blue]Building xarray stacks…[/blue]")
    plate_dict = processing.compile_measurements(mlf_paths)
    console.print(f"[green]{len(plate_dict)} FoV stacks ready[/green]")

    console.print(f"[blue]Writing OME-TIFFs ({workers} workers)…[/blue]")
    export.write_plate(
        plate_dict,
        out_dir=output,
        z_mode=z_mode.value,     # pass plain string to exporter
        compress=None if compress.lower() == "none" else compress,
        workers=workers,
        overwrite=overwrite,
    )
    console.print("[bold green]✓ Finished[/bold green]")


# ──────────────────────────────────────────────
# 4.  discover command (unchanged except Enum import)
# ──────────────────────────────────────────────
@app.command()
def discover(
    src: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Folder to search",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """List MeasurementData.mlf instances (plain & zipped)."""
    _setup_logging(verbose)
    paths = list(discovery.find_measurements(src))
    if not paths:
        console.print("[yellow]No measurement metadata found[/yellow]")
        return
    console.print(f"[green]{len(paths)} measurements:[/green]")
    for p in paths:
        tag = "ZIP" if discovery.is_zip_measurement(p) else "DIR"
        console.print(f" [{tag}] {p}")


if __name__ == "__main__":
    app()
