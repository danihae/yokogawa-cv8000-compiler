"""Generate compressed MP4 previews from compiled OME-TIFF timelapses.

Walks ``in_dir`` for ``*.ome.tif`` files, skips brightfield acquisitions
(any stem containing ``BF``), and writes one 8-bit H.264 MP4 per remaining
timelapse into ``<in_dir>/mp4``. Each frame is percentile-normalized to 8-bit,
2x downscaled, and encoded with libx264 — turning ~1 TB of raw fluorescence
timelapses into a ~12 GiB browsable archive (~14x smaller in practice).

The previews are **lossy and for viewing only** (8-bit, downscaled, per-file
contrast stretch). The raw ``.ome.tif`` remain the quantitative source; nothing
in the analysis path should read the MP4s.

Encoding runs one file per worker process (I/O-bound over network storage) and
is idempotent: existing outputs are skipped, and each file is written to a
``.partial.mp4`` temp then atomically renamed, so an interrupted run never
leaves a corrupt MP4.

Requires the ``[video]`` extra (bundles a static ffmpeg — no system ffmpeg
needed)::

    uv pip install -e '.[video]'   # or: uv sync --extra video

Usage
-----
::

    uv run python experiments/build_mp4_previews.py /path/to/out_dir

    # keep native resolution, higher quality, only well A01:
    uv run python experiments/build_mp4_previews.py /path/to/out_dir \\
        --scale 1 --crf 20 --glob 'A01_*.ome.tif'
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
import typer

app = typer.Typer(
    name="build-mp4-previews",
    help="Generate 8-bit H.264 MP4 previews from compiled OME-TIFF timelapses.",
)


def _subsample_percentile(tf, pages, plo, phi, k=24):
    """Per-file contrast window from an even ~k-frame subsample.

    Mirrors the percentile convention of ``compiler.export_zarr._percentile_windows``
    (which uses (1, 99) for zarr display windows); we default to a gentler
    (0.5, 99.5) for the 8-bit preview stretch — both are configurable.
    """
    idx = np.unique(np.linspace(0, len(pages) - 1, min(k, len(pages))).astype(int))
    stack = np.stack([tf.pages[pages[i]].asarray() for i in idx])
    return np.percentile(stack, (plo, phi))


def _downscale_mean(frame, factor):
    """Anti-aliased YX downscale by an integer factor via block averaging.

    For factor 2 this equals cv2.INTER_AREA — used instead of stride decimation
    (``export_zarr._decimate_yx``) to avoid aliasing fine sarcomere structure,
    and instead of OpenCV to avoid adding a dependency.
    """
    if factor == 1:
        return frame
    h, w = frame.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    frame = frame[:h2, :w2]
    return (
        frame.reshape(h2 // factor, factor, w2 // factor, factor)
        .mean(axis=(1, 3))
        .astype(np.uint8)
    )


def _convert(src, dst, crf, fps, scale, plo, phi):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = str(dst) + ".partial.mp4"
    with tifffile.TiffFile(src) as tf:
        H, W = tf.series[0].shape[-2:]  # frame size from the OME series
        # Keep only full-size frames; drops degenerate/empty trailing pages seen
        # in a few single-frame files (e.g. B06-B09 F01 L3_A2_2D).
        pages = [i for i in range(len(tf.pages)) if tf.pages[i].shape == (H, W)]
        if not pages:
            raise RuntimeError("no full-size frames")
        lo, hi = _subsample_percentile(tf, pages, plo, phi)
        rng = max(float(hi - lo), 1.0)
        oH, oW = H // scale, W // scale
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{oW}x{oH}",
            "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", tmp,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in pages:
            f = tf.pages[i].asarray().astype(np.float32)
            f = np.clip((f - lo) / rng * 255, 0, 255).astype(np.uint8)
            f = _downscale_mean(f, scale)
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc}")
    os.replace(tmp, dst)
    return os.path.getsize(dst)


def _worker(task):
    src, dst, crf, fps, scale, plo, phi = task
    insz = os.path.getsize(src)
    try:
        outsz = _convert(src, dst, crf, fps, scale, plo, phi)
        return (src, insz, outsz, None)
    except Exception as exc:  # noqa: BLE001 — report per-file, keep the batch going
        return (src, insz, 0, str(exc))


def _keep(stem, excludes, includes):
    low = stem.lower()
    if any(x and x in low for x in excludes):
        return False
    if includes:
        return any(x in low for x in includes)
    return True


@app.command()
def main(
    in_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Compiled output directory containing *.ome.tif timelapses.",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Directory for the MP4s (default: <in_dir>/mp4).",
    ),
    glob: str = typer.Option("*.ome.tif", "--glob", help="Input filename glob."),
    exclude: str = typer.Option(
        "BF", "--exclude",
        help="Comma-separated substrings; stems containing any are skipped "
             "(default 'BF' → skips brightfield acquisitions).",
    ),
    include: str = typer.Option(
        "", "--include",
        help="Comma-separated substrings; if set, only matching stems are kept.",
    ),
    crf: int = typer.Option(23, "--crf", help="libx264 quality (lower = better/bigger)."),
    fps: int = typer.Option(10, "--fps", help="Playback frame rate."),
    scale: int = typer.Option(2, "--scale", help="Integer YX downscale factor (1 = native)."),
    plo: float = typer.Option(0.5, "--plo", help="Low percentile for 8-bit contrast stretch."),
    phi: float = typer.Option(99.5, "--phi", help="High percentile for 8-bit contrast stretch."),
    workers: int = typer.Option(4, "--workers", help="Parallel encoder processes."),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Re-encode files that already exist."
    ),
):
    """Convert OME-TIFF timelapses under IN_DIR to 8-bit H.264 MP4 previews.

    Previews are lossy and for viewing only — keep the raw ``.ome.tif`` as the
    quantitative source.
    """
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        typer.secho(
            "MP4 previews require the [video] extra: uv pip install -e '.[video]'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    in_dir = in_dir.resolve()
    out_dir = (output.resolve() if output is not None else in_dir / "mp4")
    out_dir.mkdir(parents=True, exist_ok=True)

    excludes = [s.strip().lower() for s in exclude.split(",") if s.strip()]
    includes = [s.strip().lower() for s in include.split(",") if s.strip()]

    files = sorted(in_dir.glob(glob))
    kept = [f for f in files if _keep(f.name[: -len(".ome.tif")]
            if f.name.endswith(".ome.tif") else f.stem, excludes, includes)]

    tasks = []
    for src in kept:
        stem = src.name[: -len(".ome.tif")] if src.name.endswith(".ome.tif") else src.stem
        dst = out_dir / f"{stem}.mp4"
        if overwrite or not dst.exists():
            tasks.append((str(src), dst, crf, fps, scale, plo, phi))

    typer.secho(
        f"{len(kept)} kept ({len(files) - len(kept)} excluded), "
        f"{len(kept) - len(tasks)} already done, {len(tasks)} to encode "
        f"→ {out_dir} ({workers} workers)",
        fg=typer.colors.BLUE,
    )
    if not tasks:
        raise typer.Exit()

    tot_in = tot_out = done = errs = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fu in as_completed(futs):
            src, insz, outsz, err = fu.result()
            done += 1
            name = os.path.basename(src)
            if err:
                errs += 1
                typer.secho(f"[{done}/{len(tasks)}] ERROR {name}: {err}", fg=typer.colors.RED)
            else:
                tot_in += insz
                tot_out += outsz
                typer.echo(
                    f"[{done}/{len(tasks)}] {name}  "
                    f"{insz / 1048576:.0f}→{outsz / 1048576:.1f} MiB "
                    f"({insz / max(outsz, 1):.1f}x)"
                )

    color = typer.colors.GREEN if errs == 0 else typer.colors.YELLOW
    typer.secho(
        f"DONE. converted={done - errs} errors={errs} "
        f"in={tot_in / 1073741824:.2f} GiB out={tot_out / 1073741824:.2f} GiB "
        f"ratio={tot_in / max(tot_out, 1):.1f}x",
        fg=color,
    )


if __name__ == "__main__":
    app()
