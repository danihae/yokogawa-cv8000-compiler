import typer
from pathlib import Path
import os

from .discovery import find_measurements
from .processing import parse_measurements
from .export import process_fieldstacks_parallel

app = typer.Typer(
    name="cellvoyager-compiler",
    help="A tool to process and compile Yokogawa CV8000 microscopy data."
)


@app.command()
def run(
    root_dir: Path = typer.Option(..., "--root-dir", help="The root directory to search for measurements."),
    out_dir: Path = typer.Option(..., "--out-dir", help="The directory where OME-TIFF files will be saved."),
    title: str = typer.Option(None, help="Optional prefix for output filenames. If unset, files are named like A01_F01_L1_A1_BF3D_60x.ome.tif."),
    z_mode: str = typer.Option("maxz", help="Z-projection mode for fluorescence channels."),
    z_mode_bf: str = typer.Option(
        "keep",
        "--z-mode-bf",
        help="Z-projection mode for brightfield (BF3D) channels. One of: keep, osbm.",
    ),
    tile_mode: str = typer.Option(
        "per-field",
        "--tile-mode",
        help=(
            "How to handle tiled acquisitions. 'per-field' writes one OME-TIFF "
            "per field (current default). 'stitch' blends each PartialTileIndex "
            "tile grid into a single mosaic OME-TIFF (M{idx} in the filename)."
        ),
    ),
    format: str = typer.Option(
        "tiff",
        "--format",
        help=(
            "Output format: 'tiff' (default, OME-TIFF), 'zarr' (OME-NGFF v0.4 "
            ".ome.zarr directory), or 'both'. The 'zarr' and 'both' modes "
            "require the [zarr] extra: uv pip install -e '.[zarr]'."
        ),
    ),
    no_merge_actions: str = typer.Option(
        None,
        "--no-merge-actions",
        help=(
            "Comma-separated action names (e.g. 'BF,2D') for which timepoints "
            "should NOT be merged across multiple WPI acquisitions. Useful for "
            "rapid (~20 Hz) BF/2D bursts: each WPI run becomes its own file "
            "with an _R{idx:02d} suffix. Default: merge all (long-term "
            "timelapses behaviour)."
        ),
    ),
    exclude_keyword: str = typer.Option(None, "--exclude", help="Keyword to exclude measurements."),
    overwrite: bool = typer.Option(True, help="Overwrite existing output files."),
    max_workers: int = typer.Option(os.cpu_count() or 4, help="Number of parallel workers to use.")
):
    """
    Finds, parses, and compiles all Yokogawa measurements in a directory.
    """
    typer.echo(f"Starting compilation...")
    typer.echo(f"  Source: {root_dir}")
    typer.echo(f"  Destination: {out_dir}")
    typer.echo(f"  Tile mode: {tile_mode}")
    typer.echo(f"  Format: {format}")

    if tile_mode not in ("per-field", "stitch"):
        typer.secho(
            f"Invalid --tile-mode '{tile_mode}'. Must be 'per-field' or 'stitch'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    if format not in ("tiff", "zarr", "both"):
        typer.secho(
            f"Invalid --format '{format}'. Must be 'tiff', 'zarr', or 'both'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    if format in ("zarr", "both"):
        try:
            import zarr  # noqa: F401
            import ome_zarr  # noqa: F401
        except ImportError:
            typer.secho(
                "OME-Zarr output requires the [zarr] extra: "
                "uv pip install -e '.[zarr]'",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

    no_merge_list = (
        [a.strip() for a in no_merge_actions.split(",") if a.strip()]
        if no_merge_actions
        else None
    )

    try:
        wpi_paths = list(find_measurements(root_dir))
        if not wpi_paths:
            typer.secho("No .wpi files found. Exiting.", fg=typer.colors.YELLOW)
            return

        merged_records_df, acquisitions = parse_measurements(
            wpi_paths, exclude_keyword=exclude_keyword
        )

        process_fieldstacks_parallel(
            merged_records_df,
            acquisitions,
            out_dir,
            title=title,
            z_mode=z_mode,
            z_mode_BF=z_mode_bf,
            overwrite=overwrite,
            max_workers=max_workers,
            tile_mode=tile_mode,
            no_merge_actions=no_merge_list,
            format=format,
        )

        typer.secho(f"Successfully finished compilation!", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"An error occurred during compilation: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
