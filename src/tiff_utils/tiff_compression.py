"""Utilities for compressing TIFF files from zip archives, directories, or loose files.

Accepts any mix of ``.zip`` archives, directories (scanned recursively for
TIFFs), and individual ``.tif``/``.tiff`` files.  Each TIFF is compressed,
verified, and (for zips) cleaned up with minimal temporary disk usage.

A ``ProcessPoolExecutor`` is used because Python's ``zipfile`` does *not*
release the GIL during decompression; separate processes give true parallelism
for both extraction and compression.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import tifffile

logger = logging.getLogger(__name__)

VALID_COMPRESSIONS = ("zlib", "lzw", "lzma", "zstd")

# OME <Pixels>/<Plane> attributes carried over when recompressing an OME-TIFF.
_OME_PIXELS_FLOAT = ("PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement")
_OME_PIXELS_STR = (
    "PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit", "TimeIncrementUnit",
)
_OME_PLANE_INT = ("TheT", "TheC", "TheZ")
_OME_PLANE_FLOAT = ("DeltaT", "ExposureTime", "PositionX", "PositionY", "PositionZ")
_OME_PLANE_STR = ("DeltaTUnit", "ExposureTimeUnit")


def _ome_rewrite_metadata(tif: tifffile.TiffFile) -> dict | None:
    """Reconstruct an ``imwrite`` ``metadata`` dict from a source OME-TIFF.

    The dimension order (``axes``) is the essential piece: without it tifffile
    cannot tell time / z / channel apart on read-back and either mislabels the
    stack axis (a movie silently read as channels) or leaves it unclassifiable
    ('I'/'Q'), which OME-aware readers such as SarcAsM then reject with
    ``Invalid axis letter(s): I``. Pixel size, per-frame ``DeltaT`` timing,
    channel names and the compiler's custom JSON ``Description`` are carried
    over too so the compressed copy is calibrated identically to the source.

    The source OME-XML cannot simply be re-emitted verbatim (it embeds ``µm``,
    and TIFF descriptions must be 7-bit ASCII), so tifffile is asked to
    regenerate it from these values.

    Returns ``None`` when the source is not an OME-TIFF.
    """
    if not (tif.is_ome and tif.ome_metadata):
        return None

    meta: dict = {"axes": tif.series[0].axes}
    try:
        root = ET.fromstring(tif.ome_metadata)
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            for key in _OME_PIXELS_FLOAT:
                if (val := pixels.get(key)) is not None:
                    meta[key] = float(val)
            for key in _OME_PIXELS_STR:
                if (val := pixels.get(key)) is not None:
                    meta[key] = val

            planes: list = []
            for plane in pixels.findall("{*}Plane"):
                entry: dict = {}
                for key in _OME_PLANE_INT:
                    if (val := plane.get(key)) is not None:
                        entry[key] = int(val)
                for key in _OME_PLANE_FLOAT:
                    if (val := plane.get(key)) is not None:
                        entry[key] = float(val)
                for key in _OME_PLANE_STR:
                    if (val := plane.get(key)) is not None:
                        entry[key] = val
                if entry:
                    planes.append(entry)
            if planes:
                meta["Plane"] = planes

            channels = [
                {"Name": ch.get("Name", f"Channel_{i}")}
                for i, ch in enumerate(pixels.findall("{*}Channel"))
            ]
            if channels:
                meta["Channels"] = channels

        description = root.findtext(".//{*}Image/{*}Description")
        if description:
            meta["Description"] = description
    except Exception as exc:  # a partial parse must never lose the axes
        logger.debug("Partial OME metadata parse: %s", exc)

    return meta


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def compress_tiff(
    input_path: Path,
    output_path: Path,
    compression: str = "zlib",
) -> dict:
    """Read a TIFF and rewrite it with the specified compression.

    Preserves resolution and photometric interpretation, plus the dimension
    metadata: ImageJ hyperstack metadata for ImageJ files, or the OME axis
    order / pixel size / per-frame timing / channels for OME-TIFFs. Without
    the latter a compressed movie loses its time axis (see
    :func:`_ome_rewrite_metadata`).

    Parameters:
        input_path: Path to the input TIFF file.
        output_path: Path for the compressed output TIFF file.
        compression: Compression codec (one of 'zlib', 'lzw', 'lzma', 'zstd').

    Returns:
        dict with keys ``success`` (bool), ``input_size`` (int),
        ``output_size`` (int), and optionally ``error`` (str).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    result: dict = {
        "success": False,
        "input_size": input_path.stat().st_size,
        "output_size": 0,
    }

    try:
        with tifffile.TiffFile(input_path) as tif:
            data = tif.asarray()
            kwargs: dict = {}
            page = tif.pages[0]

            # Preserve photometric interpretation
            kwargs["photometric"] = page.photometric

            # Preserve resolution tags
            if "XResolution" in page.tags and "YResolution" in page.tags:
                xres = page.tags["XResolution"].value
                yres = page.tags["YResolution"].value
                # Rational values may come back as (numerator, denominator)
                if isinstance(xres, tuple):
                    xres = xres[0] / xres[1]
                if isinstance(yres, tuple):
                    yres = yres[0] / yres[1]
                kwargs["resolution"] = (xres, yres)
                if "ResolutionUnit" in page.tags:
                    kwargs["resolutionunit"] = page.tags["ResolutionUnit"].value

            # Preserve dimension metadata so the compressed copy stays a usable
            # stack. ImageJ and OME store it differently; dropping the OME axes
            # corrupts the file -- a movie is silently re-read as channels, or
            # the stack axis becomes unclassifiable ('I'/'Q') and OME-aware
            # readers reject it. ImageJ keeps its own hyperstack metadata.
            if tif.is_imagej and tif.imagej_metadata:
                kwargs["imagej"] = True
                kwargs["metadata"] = tif.imagej_metadata
            else:
                ome_meta = _ome_rewrite_metadata(tif)
                if ome_meta is not None:
                    # Force OME regardless of the output extension; tifffile only
                    # auto-selects OME for a .ome.tif name.
                    kwargs["ome"] = True
                    kwargs["metadata"] = ome_meta

        tifffile.imwrite(output_path, data, compression=compression, **kwargs)
        result["success"] = True
        result["output_size"] = output_path.stat().st_size

    except Exception as e:
        result["error"] = str(e)
        logger.error("Error compressing %s: %s", input_path.name, e)

    return result


def _is_tiff_member(name: str) -> bool:
    """Return True if *name* looks like a TIFF file (excluding macOS artifacts)."""
    if name.startswith("__MACOSX") or name.endswith("/"):
        return False
    return name.lower().endswith((".tif", ".tiff"))


def _is_real_member(name: str) -> bool:
    """Return True if *name* is a real file entry (not a directory or macOS artifact)."""
    return not name.startswith("__MACOSX") and not name.endswith("/")


def _is_tiff_path(path: Path) -> bool:
    """Return True if *path* is an existing file with a TIFF extension."""
    return path.is_file() and path.suffix.lower() in (".tif", ".tiff")


# ---------------------------------------------------------------------------
# Pipelined per-file worker (runs in a child process)
# ---------------------------------------------------------------------------


def _process_single_tiff(
    zip_path: str,
    member: str,
    tmp_dir: str,
    output_path: str,
    compression: str,
) -> dict:
    """Extract one TIFF from a zip, compress it, verify, and clean up.

    This is the main worker function, designed for ``ProcessPoolExecutor``.
    All arguments are plain strings for pickle-friendliness.  Each call opens
    its own ``ZipFile`` handle so workers can run truly in parallel (Python's
    ``zipfile`` holds the GIL during decompression, so threads would not help).

    Returns:
        dict with keys ``member``, ``success``, ``input_size``,
        ``output_size``, and optionally ``error``.
    """
    import tifffile as _tifffile  # import per-process to avoid fork issues

    tmp_path = Path(tmp_dir) / Path(member).name
    output_p = Path(output_path)
    result: dict = {"member": member, "success": False, "input_size": 0, "output_size": 0}

    try:
        # --- Extract single member ---------------------------------------
        with zipfile.ZipFile(zip_path, "r") as zf:
            info = zf.getinfo(member)
            with zf.open(info) as src, open(tmp_path, "wb") as dst:
                while chunk := src.read(1 << 20):  # 1 MiB chunks
                    dst.write(chunk)

        # --- Compress (reuses core helper) -------------------------------
        output_p.parent.mkdir(parents=True, exist_ok=True)
        res = compress_tiff(tmp_path, output_p, compression)
        result.update(res)

        # --- Verify ------------------------------------------------------
        if result["success"]:
            with _tifffile.TiffFile(output_p) as tif:
                _ = tif.pages[0].shape

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    finally:
        # --- Clean up temp file immediately ------------------------------
        tmp_path.unlink(missing_ok=True)

    return result


def _extract_member(
    zip_path: str,
    member: str,
    output_path: str,
) -> dict:
    """Extract a single non-TIFF member from a zip and copy it to output.

    Designed for ``ProcessPoolExecutor``.  All arguments are plain strings.
    """
    output_p = Path(output_path)
    result: dict = {"member": member, "success": False}

    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            info = zf.getinfo(member)
            with zf.open(info) as src, open(output_p, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Zip-level processing
# ---------------------------------------------------------------------------


def process_zip(
    zip_path: Path,
    output_dir: Path,
    compression: str = "zlib",
    workers: int = 4,
) -> tuple[bool, list[Path]]:
    """Extract, compress, and verify every TIFF in a zip archive.

    Non-TIFF files are extracted and copied to the output directory unchanged,
    preserving their relative paths within the zip.

    Each TIFF is handled end-to-end (extract → compress → verify → cleanup)
    by a single worker process.  At most *workers* uncompressed files exist
    on disk at any time.

    Parameters:
        zip_path: Path to the zip file.
        output_dir: Root directory for compressed output files.
        compression: TIFF compression codec.
        workers: Number of parallel worker processes.

    Returns:
        Tuple of (success, list_of_output_paths).
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    logger.info("Processing: %s", zip_path)

    # Pre-scan member list (cheap — only reads the central directory)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_members = [m for m in zf.namelist() if _is_real_member(m)]
    except zipfile.BadZipFile:
        logger.error("  %s is not a valid zip file", zip_path.name)
        return False, []
    except Exception as e:
        logger.error("  Error reading %s: %s", zip_path.name, e)
        return False, []

    tiff_members = [m for m in all_members if _is_tiff_member(m)]
    other_members = [m for m in all_members if not _is_tiff_member(m)]

    if not all_members:
        logger.warning("  No files in %s", zip_path.name)
        return False, []

    zip_output_dir = output_dir / zip_path.stem

    logger.info(
        "  Processing %d TIFFs + %d other files ('%s', %d workers) ...",
        len(tiff_members),
        len(other_members),
        compression,
        workers,
    )

    output_paths: list[Path] = []
    failed: list[str] = []

    # Each TIFF worker gets its own temp sub-directory so filenames can't collide.
    with tempfile.TemporaryDirectory(prefix=f"tiffproc_{zip_path.stem}_") as tmp_root:

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures: dict = {}

            # Submit TIFF compression jobs
            tiff_work: dict[str, Path] = {}
            for i, member in enumerate(tiff_members):
                worker_tmp = Path(tmp_root) / str(i)
                worker_tmp.mkdir()
                rel = Path(member)
                out_file = zip_output_dir / rel.with_suffix(".tif")
                args = (str(zip_path), member, str(worker_tmp), str(out_file), compression)
                tiff_work[member] = out_file
                futures[pool.submit(_process_single_tiff, *args)] = (member, "tiff")

            # Submit non-TIFF extraction jobs
            for member in other_members:
                out_file = zip_output_dir / member
                futures[pool.submit(
                    _extract_member, str(zip_path), member, str(out_file)
                )] = (member, "other")

            for fut in as_completed(futures):
                member, kind = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logger.error("    %s: %s", member, e)
                    failed.append(member)
                    continue

                if not res["success"]:
                    logger.error(
                        "    %s: %s", Path(member).name, res.get("error", "unknown")
                    )
                    failed.append(member)
                    continue

                if kind == "tiff":
                    out_file = tiff_work[member]
                    output_paths.append(out_file)
                    pct = (
                        (1 - res["output_size"] / max(res["input_size"], 1)) * 100
                    )
                    logger.info(
                        "    %s: %.1f MB -> %.1f MB (%.0f%% reduction)",
                        Path(member).name,
                        res["input_size"] / 1_048_576,
                        res["output_size"] / 1_048_576,
                        pct,
                    )
                else:
                    output_paths.append(zip_output_dir / member)
                    logger.info("    %s: copied", Path(member).name)

    if failed:
        logger.warning("  %d file(s) failed: %s", len(failed), failed)
        return False, []

    if len(output_paths) != len(all_members):
        logger.warning(
            "  Count mismatch: expected %d, got %d",
            len(all_members),
            len(output_paths),
        )
        return False, []

    logger.info("  OK — %d files processed", len(output_paths))
    return True, output_paths


def _compress_and_verify_tiff(
    input_path: str,
    output_path: str,
    compression: str,
) -> dict:
    """Compress and verify a single loose TIFF (for ``ProcessPoolExecutor``).

    All arguments are plain strings for pickle-friendliness.
    """
    import tifffile as _tifffile

    input_p = Path(input_path)
    output_p = Path(output_path)
    result: dict = {"input": input_path, "success": False, "input_size": 0, "output_size": 0}

    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        res = compress_tiff(input_p, output_p, compression)
        result.update(res)

        if result["success"]:
            with _tifffile.TiffFile(output_p) as tif:
                _ = tif.pages[0].shape

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    return result


def process_files(
    files: list[Path],
    output_dir: Path,
    base_dir: Path | None = None,
    compression: str = "zlib",
    workers: int = 4,
) -> tuple[bool, list[Path]]:
    """Compress TIFFs and copy non-TIFF files to the output directory.

    Parameters:
        files: Paths to individual files (TIFF and non-TIFF).
        output_dir: Root directory for output files.
        base_dir: If given, output paths preserve the relative structure
            under *base_dir*.  Otherwise files are written flat into
            *output_dir*.
        compression: TIFF compression codec.
        workers: Number of parallel worker processes.

    Returns:
        Tuple of (all_succeeded, list_of_output_paths).
    """
    if not files:
        return True, []

    tiff_files = [f for f in files if _is_tiff_path(f)]
    other_files = [f for f in files if f.is_file() and not _is_tiff_path(f)]

    logger.info(
        "Processing %d TIFFs + %d other files ...",
        len(tiff_files),
        len(other_files),
    )

    output_paths: list[Path] = []
    failed: list[str] = []

    def _rel(f: Path) -> Path:
        if base_dir is not None:
            try:
                return f.relative_to(base_dir)
            except ValueError:
                pass
        return Path(f.name)

    # --- Copy non-TIFF files (I/O bound, threads are fine) ---------------
    for f in other_files:
        out_file = output_dir / _rel(f)
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out_file)
            output_paths.append(out_file)
            logger.info("    %s: copied", f.name)
        except Exception as e:
            logger.error("    %s: copy failed: %s", f.name, e)
            failed.append(str(f))

    # --- Compress TIFFs (CPU bound, use processes) -----------------------
    tiff_work: dict[str, Path] = {}
    for tf in tiff_files:
        out_file = output_dir / _rel(tf).with_suffix(".tif")
        tiff_work[str(tf)] = out_file

    if tiff_work:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _compress_and_verify_tiff, inp, str(out), compression
                ): inp
                for inp, out in tiff_work.items()
            }
            for fut in as_completed(futures):
                inp = futures[fut]
                out = tiff_work[inp]
                try:
                    res = fut.result()
                except Exception as e:
                    logger.error("    %s: %s", inp, e)
                    failed.append(inp)
                    continue

                if res["success"]:
                    output_paths.append(out)
                    pct = (1 - res["output_size"] / max(res["input_size"], 1)) * 100
                    logger.info(
                        "    %s: %.1f MB -> %.1f MB (%.0f%% reduction)",
                        Path(inp).name,
                        res["input_size"] / 1_048_576,
                        res["output_size"] / 1_048_576,
                        pct,
                    )
                else:
                    logger.error(
                        "    %s: %s", Path(inp).name, res.get("error", "unknown")
                    )
                    failed.append(inp)

    if failed:
        logger.warning("  %d file(s) failed", len(failed))
        return False, output_paths  # return partial results

    logger.info("  OK — %d files processed", len(output_paths))
    return True, output_paths


# Backward-compatible alias
process_tiffs = process_files


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def compress_tiffs(
    inputs: list[Path],
    output_dir: Path = Path("compressed_tiffs"),
    compression: str = "zlib",
    delete_originals: bool = False,
    workers: int = 4,
    max_parallel_zips: int = 1,
    dry_run: bool = False,
) -> dict:
    """Compress TIFFs and copy other files from zip archives, directories, and loose files.

    TIFFs are recompressed with the specified codec, verified, and written to
    the output directory.  All other files are copied through unchanged,
    preserving relative paths.

    Parameters:
        inputs: Paths to zip files, directories, or individual files.
        output_dir: Root output directory.
        compression: TIFF compression codec ('zlib', 'lzw', 'lzma', or 'zstd').
        delete_originals: Remove original zips after successful verification
            (loose TIFFs and directories are never deleted).
        workers: Parallel worker processes.
        max_parallel_zips: How many zip files to process concurrently.
            Keep at 1 (default) to bound temporary disk usage; raise when you
            have sufficient headroom.
        dry_run: Preview mode – process without deleting originals.

    Returns:
        Summary dict with keys ``processed``, ``succeeded``, ``failed``,
        ``deleted``.
    """
    if compression not in VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown compression '{compression}'. "
            f"Choose from {VALID_COMPRESSIONS}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Sort inputs into categories ------------------------------------
    zips: list[Path] = []
    loose_files: list[Path] = []  # TIFFs and non-TIFFs
    dir_files: dict[Path, list[Path]] = {}  # base_dir -> all files

    for p in inputs:
        p = Path(p)
        if not p.exists():
            logger.warning("Skipping non-existent path: %s", p)
            continue
        if p.is_dir():
            found = sorted(f for f in p.rglob("*") if f.is_file())
            if found:
                dir_files[p] = found
            else:
                logger.warning("No files found in directory: %s", p)
        elif p.suffix.lower() == ".zip":
            zips.append(p)
        elif p.is_file():
            loose_files.append(p)
        else:
            logger.warning("Skipping unsupported path: %s", p)

    total_items = len(zips) + len(dir_files) + (1 if loose_files else 0)
    summary = {"processed": total_items, "succeeded": 0, "failed": 0, "deleted": 0}

    if total_items == 0:
        logger.error("No valid inputs to process")
        summary["failed"] = -1
        return summary

    failed_zips: list[Path] = []

    # --- Process zip files -----------------------------------------------
    def _handle_zip_result(zp: Path, ok: bool) -> None:
        if ok:
            summary["succeeded"] += 1
            if delete_originals and not dry_run:
                try:
                    zp.unlink()
                    logger.info("  Deleted original: %s", zp)
                    summary["deleted"] += 1
                except OSError as e:
                    logger.error("  Could not delete %s: %s", zp, e)
            elif delete_originals:
                logger.info("  [DRY-RUN] Would delete: %s", zp)
            else:
                logger.info("  Kept original: %s", zp)
        else:
            logger.error("  FAILED: %s (original preserved)", zp)
            failed_zips.append(zp)
            summary["failed"] += 1

    if max_parallel_zips <= 1:
        for zp in zips:
            ok, _ = process_zip(
                zp, output_dir, compression=compression, workers=workers
            )
            _handle_zip_result(zp, ok)
    elif zips:
        with ThreadPoolExecutor(max_workers=max_parallel_zips) as pool:
            future_to_zip = {
                pool.submit(
                    process_zip, zp, output_dir, compression=compression, workers=workers
                ): zp
                for zp in zips
            }
            for fut in as_completed(future_to_zip):
                zp = future_to_zip[fut]
                try:
                    ok, _ = fut.result()
                except Exception as e:
                    logger.error("  Unexpected error for %s: %s", zp.name, e)
                    ok = False
                _handle_zip_result(zp, ok)

    # --- Process directories (all files) ---------------------------------
    for base_dir, files in dir_files.items():
        logger.info("Processing directory: %s", base_dir)
        out_subdir = output_dir / base_dir.name
        ok, _ = process_files(
            files, out_subdir, base_dir=base_dir,
            compression=compression, workers=workers,
        )
        if ok:
            summary["succeeded"] += 1
        else:
            summary["failed"] += 1

    # --- Process loose files (TIFFs compressed, others copied) -----------
    if loose_files:
        ok, _ = process_files(
            loose_files, output_dir, compression=compression, workers=workers,
        )
        if ok:
            summary["succeeded"] += 1
        else:
            summary["failed"] += 1

    # --- Summary ---------------------------------------------------------
    logger.info("=" * 50)
    logger.info(
        "Done - %d processed, %d succeeded, %d failed",
        summary["processed"],
        summary["succeeded"],
        summary["failed"],
    )
    if delete_originals:
        logger.info("Deleted: %d original zips", summary["deleted"])
    if failed_zips:
        logger.warning("Failed zips: %s", [z.name for z in failed_zips])

    return summary


# Keep backward-compatible alias
compress_tiffs_from_zips = compress_tiffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for TIFF compression."""
    parser = argparse.ArgumentParser(
        description="Compress TIFFs (and copy other files) from zip archives, "
        "directories, or loose files.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Zip files, directories, or individual files to process. "
        "TIFFs are recompressed; all other files are copied through unchanged.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("compressed_tiffs"),
        help="Output directory (default: ./compressed_tiffs)",
    )
    parser.add_argument(
        "-c",
        "--compression",
        default="zlib",
        choices=VALID_COMPRESSIONS,
        help="TIFF compression codec (default: zlib)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete original zip files after successful verification "
        "(loose TIFFs and directories are never deleted)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "-Z",
        "--max-parallel-zips",
        type=int,
        default=1,
        help="Number of zip files to process concurrently (default: 1). "
        "Raising this trades temporary disk space for throughput.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process without deleting originals (preview mode)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = compress_tiffs(
        inputs=args.inputs,
        output_dir=args.output_dir,
        compression=args.compression,
        delete_originals=args.delete,
        workers=args.workers,
        max_parallel_zips=args.max_parallel_zips,
        dry_run=args.dry_run,
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())