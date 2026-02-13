"""Utilities for compressing TIFF files extracted from zip archives.

Processes zip files sequentially (one temp extraction at a time) to bound disk
usage, while compressing individual TIFFs in parallel threads within each zip
for throughput.  Thread-based parallelism is effective here because the
underlying imagecodecs library releases the GIL during (de)compression.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tifffile

logger = logging.getLogger(__name__)

VALID_COMPRESSIONS = ("zlib", "lzw", "lzma", "zstd")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def compress_tiff(
    input_path: Path,
    output_path: Path,
    compression: str = "zlib",
) -> dict:
    """Read a TIFF and rewrite it with the specified compression.

    Preserves ImageJ metadata, resolution, and photometric interpretation
    when present in the source file.

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

            # Preserve ImageJ metadata
            if tif.is_imagej and tif.imagej_metadata:
                kwargs["imagej"] = True
                kwargs["metadata"] = tif.imagej_metadata

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


# ---------------------------------------------------------------------------
# Zip-level processing
# ---------------------------------------------------------------------------


def process_zip(
    zip_path: Path,
    output_dir: Path,
    compression: str = "zlib",
    workers: int = 4,
) -> tuple[bool, list[Path]]:
    """Extract TIFFs from a zip, compress them, verify, and clean up.

    A temporary directory is used for extraction and is automatically removed
    after processing, regardless of success or failure.

    Parameters:
        zip_path: Path to the zip file.
        output_dir: Root directory for compressed output files.
        compression: TIFF compression codec.
        workers: Number of parallel compression threads.

    Returns:
        Tuple of (success, list_of_output_paths).
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    logger.info("Processing: %s", zip_path)

    with tempfile.TemporaryDirectory(prefix=f"tiffproc_{zip_path.stem}_") as temp_dir:
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()

        # --- Extract -----------------------------------------------------
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tiff_members = [m for m in zf.namelist() if _is_tiff_member(m)]
                if not tiff_members:
                    logger.warning("  No TIFF files in %s", zip_path.name)
                    return False, []
                logger.info("  Extracting %d TIFF files ...", len(tiff_members))
                zf.extractall(extract_dir, members=tiff_members)
        except zipfile.BadZipFile:
            logger.error("  %s is not a valid zip file", zip_path.name)
            return False, []
        except Exception as e:
            logger.error("  Extraction error for %s: %s", zip_path.name, e)
            return False, []

        extracted_tiffs = sorted(extract_dir.rglob("*.[tT][iI][fF]*"))

        # --- Compress ----------------------------------------------------
        zip_output_dir = output_dir / zip_path.stem
        zip_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "  Compressing %d files ('%s', %d workers) ...",
            len(extracted_tiffs),
            compression,
            workers,
        )

        compressed: list[Path] = []
        failed: list[str] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map: dict = {}
            for tiff_file in extracted_tiffs:
                rel = tiff_file.relative_to(extract_dir)
                out_file = zip_output_dir / rel.with_suffix(".tif")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                fut = pool.submit(compress_tiff, tiff_file, out_file, compression)
                future_map[fut] = (tiff_file, out_file)

            for fut in as_completed(future_map):
                src, dst = future_map[fut]
                try:
                    res = fut.result()
                    if res["success"]:
                        compressed.append(dst)
                        pct = (
                            (1 - res["output_size"] / max(res["input_size"], 1)) * 100
                        )
                        logger.info(
                            "    %s: %.1f MB -> %.1f MB (%.0f%% reduction)",
                            src.name,
                            res["input_size"] / 1_048_576,
                            res["output_size"] / 1_048_576,
                            pct,
                        )
                    else:
                        failed.append(src.name)
                except Exception as e:
                    logger.error("    %s: %s", src.name, e)
                    failed.append(src.name)

        # --- Verify ------------------------------------------------------
        if failed:
            logger.warning("  %d file(s) failed: %s", len(failed), failed)
            return False, []

        if len(compressed) != len(extracted_tiffs):
            logger.warning(
                "  Count mismatch: expected %d, got %d",
                len(extracted_tiffs),
                len(compressed),
            )
            return False, []

        logger.info("  Verifying %d output files ...", len(compressed))
        for path in compressed:
            try:
                with tifffile.TiffFile(path) as tif:
                    _ = tif.pages[0].shape
            except Exception as e:
                logger.error("  Corrupt output %s: %s", path.name, e)
                return False, []

        logger.info("  OK - %d files verified", len(compressed))
        return True, compressed


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def compress_tiffs_from_zips(
    zip_files: list[Path],
    output_dir: Path = Path("compressed_tiffs"),
    compression: str = "zlib",
    delete_originals: bool = False,
    workers: int = 4,
    dry_run: bool = False,
) -> dict:
    """Compress TIFFs inside multiple zip archives.

    Zips are processed sequentially (one temp extraction at a time) to bound
    disk usage, while TIFF compression within each zip runs in parallel
    threads.

    Parameters:
        zip_files: Zip file paths to process.
        output_dir: Root output directory.
        compression: TIFF compression codec ('zlib', 'lzw', 'lzma', or 'zstd').
        delete_originals: Remove original zips after successful verification.
        workers: Parallel compression threads per zip.
        dry_run: Preview mode - process without deleting originals.

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

    valid = [
        Path(z)
        for z in zip_files
        if Path(z).exists() and Path(z).suffix.lower() == ".zip"
    ]
    skipped = {str(z) for z in zip_files} - {str(z) for z in valid}
    for s in skipped:
        logger.warning("Skipping invalid path: %s", s)

    summary = {"processed": len(valid), "succeeded": 0, "failed": 0, "deleted": 0}
    if not valid:
        logger.error("No valid zip files to process")
        summary["failed"] = -1  # sentinel: no inputs
        return summary

    failed_zips: list[Path] = []

    for zp in valid:
        ok, _ = process_zip(
            zp, output_dir, compression=compression, workers=workers
        )
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
        logger.warning("Failed: %s", [z.name for z in failed_zips])

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for TIFF compression from zip archives."""
    parser = argparse.ArgumentParser(
        description="Extract TIFFs from zip files, recompress, and verify.",
    )
    parser.add_argument(
        "zip_files", nargs="+", type=Path, help="Zip files to process"
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
        help="Delete original zip files after successful verification",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Parallel compression threads (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process without deleting originals (preview mode)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = compress_tiffs_from_zips(
        zip_files=args.zip_files,
        output_dir=args.output_dir,
        compression=args.compression,
        delete_originals=args.delete,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
