import typer
from pathlib import Path
import os

from .discovery import find_measurements
from .processing import parse_measurements
from .export import process_fieldstacks_parallel

# Create a Typer application
app = typer.Typer(
    name="cellvoyager-compiler",
    help="A tool to process and compile Yokogawa CV8000 microscopy data."
)

@app.command()
def run(
    root_dir: Path = typer.Option(..., "--root-dir", help="The root directory to search for measurements."),
    out_dir: Path = typer.Option(..., "--out-dir", help="The directory where OME-TIFF files will be saved."),
    title: str = typer.Option("compiled_data", help="The base name for the output files."),
    z_mode: str = typer.Option("maxz", help="Z-projection mode for fluorescence channels."),
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

    try:
        # 1. Find measurements
        wpi_paths = list(find_measurements(root_dir))
        if not wpi_paths:
            typer.secho("No .wpi files found. Exiting.", fg=typer.colors.YELLOW)
            return

        # 2. Parse measurements
        merged_records_df, acquisitions = parse_measurements(
            wpi_paths, exclude_keyword=exclude_keyword
        )

        # 3. Group into field stacks
        group_cols = ["row", "column", "field_index", "timeline_index", "action_index", "action"]
        fieldstacks = merged_records_df.groupby(group_cols, sort=False, dropna=True)

        # 4. Process in parallel
        process_fieldstacks_parallel(
            fieldstacks,
            acquisitions,
            out_dir,
            title=title,
            z_mode=z_mode,
            overwrite=overwrite,
            max_workers=max_workers,
        )

        typer.secho(f"Successfully finished compilation!", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"An error occurred during compilation: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
