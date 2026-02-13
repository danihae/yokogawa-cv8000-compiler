"""Tests for yokogawa_cv8000_compiler.tiff_compression."""

import io
import zipfile

import numpy as np
import pytest
import tifffile

from yokogawa_cv8000_compiler.tiff_compression import (
    VALID_COMPRESSIONS,
    _is_tiff_member,
    compress_tiff,
    compress_tiffs_from_zips,
    main,
    process_zip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_test_tiff(path, shape=(64, 64), dtype=np.uint16, **kwargs):
    """Write a minimal TIFF for testing."""
    data = np.random.randint(0, 1000, shape, dtype=dtype)
    tifffile.imwrite(str(path), data, **kwargs)
    return data


def _make_zip_with_tiffs(zip_path, num_tiffs=3, shape=(64, 64), nested=False):
    """Create a zip file containing uncompressed TIFF files."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(num_tiffs):
            buf = io.BytesIO()
            data = np.random.randint(0, 1000, shape, dtype=np.uint16)
            tifffile.imwrite(buf, data)
            prefix = "subdir/" if nested else ""
            zf.writestr(f"{prefix}image_{i:03d}.tif", buf.getvalue())


# ---------------------------------------------------------------------------
# _is_tiff_member
# ---------------------------------------------------------------------------


class TestIsTiffMember:
    def test_tif_extension(self):
        assert _is_tiff_member("image.tif") is True

    def test_tiff_extension(self):
        assert _is_tiff_member("image.tiff") is True

    def test_uppercase(self):
        assert _is_tiff_member("IMAGE.TIF") is True

    def test_non_tiff(self):
        assert _is_tiff_member("readme.txt") is False

    def test_directory_entry(self):
        assert _is_tiff_member("folder/") is False

    def test_macosx_artifact(self):
        assert _is_tiff_member("__MACOSX/._image.tif") is False

    def test_nested_path(self):
        assert _is_tiff_member("sub/dir/image.tif") is True


# ---------------------------------------------------------------------------
# compress_tiff
# ---------------------------------------------------------------------------


class TestCompressTiff:
    def test_basic_compression(self, tmp_path):
        src = tmp_path / "input.tif"
        dst = tmp_path / "output.tif"
        orig_data = _write_test_tiff(src)

        result = compress_tiff(src, dst, compression="zlib")

        assert result["success"] is True
        assert result["input_size"] > 0
        assert result["output_size"] > 0
        assert dst.exists()

        # Data roundtrip check
        roundtripped = tifffile.imread(str(dst))
        np.testing.assert_array_equal(roundtripped, orig_data)

    def test_preserves_imagej_metadata(self, tmp_path):
        src = tmp_path / "input.tif"
        dst = tmp_path / "output.tif"
        data = np.random.randint(0, 500, (3, 64, 64), dtype=np.uint16)
        tifffile.imwrite(
            str(src),
            data,
            imagej=True,
            resolution=(1 / 0.325, 1 / 0.325),
            metadata={"unit": "um", "axes": "ZYX"},
        )

        result = compress_tiff(src, dst, compression="zlib")
        assert result["success"] is True

        with tifffile.TiffFile(str(dst)) as tif:
            assert tif.is_imagej
            assert tif.imagej_metadata is not None

    def test_invalid_input_returns_error(self, tmp_path):
        src = tmp_path / "bad.tif"
        dst = tmp_path / "output.tif"
        src.write_text("not a tiff")

        result = compress_tiff(src, dst, compression="zlib")
        assert result["success"] is False
        assert "error" in result

    def test_all_valid_compressions(self, tmp_path):
        for comp in VALID_COMPRESSIONS:
            src = tmp_path / f"input_{comp}.tif"
            dst = tmp_path / f"output_{comp}.tif"
            _write_test_tiff(src)
            result = compress_tiff(src, dst, compression=comp)
            if not result["success"] and "imagecodecs" in result.get("error", ""):
                pytest.skip(f"{comp} requires the imagecodecs package")
            assert result["success"] is True, f"Failed for compression={comp}"


# ---------------------------------------------------------------------------
# process_zip
# ---------------------------------------------------------------------------


class TestProcessZip:
    def test_basic_zip_processing(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        out_dir = tmp_path / "output"
        _make_zip_with_tiffs(zip_path, num_tiffs=3)

        success, outputs = process_zip(zip_path, out_dir, workers=2)

        assert success is True
        assert len(outputs) == 3
        for f in outputs:
            assert f.exists()
            # Verify readable
            with tifffile.TiffFile(str(f)) as tif:
                assert tif.pages[0].shape == (64, 64)

    def test_nested_tiffs_in_zip(self, tmp_path):
        zip_path = tmp_path / "nested.zip"
        out_dir = tmp_path / "output"
        _make_zip_with_tiffs(zip_path, num_tiffs=2, nested=True)

        success, outputs = process_zip(zip_path, out_dir, workers=1)
        assert success is True
        assert len(outputs) == 2

    def test_empty_zip(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        out_dir = tmp_path / "output"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no tiffs here")

        success, outputs = process_zip(zip_path, out_dir)
        assert success is False
        assert outputs == []

    def test_bad_zip_file(self, tmp_path):
        zip_path = tmp_path / "bad.zip"
        zip_path.write_text("not a zip")
        out_dir = tmp_path / "output"

        success, outputs = process_zip(zip_path, out_dir)
        assert success is False
        assert outputs == []

    def test_output_dir_mirrors_zip_stem(self, tmp_path):
        zip_path = tmp_path / "experiment_001.zip"
        out_dir = tmp_path / "output"
        _make_zip_with_tiffs(zip_path, num_tiffs=1)

        success, outputs = process_zip(zip_path, out_dir)
        assert success is True
        # Output should be under output/experiment_001/
        assert outputs[0].parent.name == "experiment_001"

    def test_macosx_artifacts_skipped(self, tmp_path):
        zip_path = tmp_path / "macos.zip"
        out_dir = tmp_path / "output"
        buf = io.BytesIO()
        tifffile.imwrite(buf, np.zeros((32, 32), dtype=np.uint16))
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("image.tif", buf.getvalue())
            zf.writestr("__MACOSX/._image.tif", b"resource fork data")

        success, outputs = process_zip(zip_path, out_dir)
        assert success is True
        assert len(outputs) == 1


# ---------------------------------------------------------------------------
# compress_tiffs_from_zips
# ---------------------------------------------------------------------------


class TestCompressTiffsFromZips:
    def test_multiple_zips(self, tmp_path):
        out_dir = tmp_path / "output"
        zips = []
        for i in range(3):
            zp = tmp_path / f"batch_{i}.zip"
            _make_zip_with_tiffs(zp, num_tiffs=2)
            zips.append(zp)

        result = compress_tiffs_from_zips(zips, out_dir, workers=2)

        assert result["processed"] == 3
        assert result["succeeded"] == 3
        assert result["failed"] == 0

    def test_delete_originals(self, tmp_path):
        out_dir = tmp_path / "output"
        zp = tmp_path / "deleteme.zip"
        _make_zip_with_tiffs(zp, num_tiffs=1)

        result = compress_tiffs_from_zips(
            [zp], out_dir, delete_originals=True, workers=1
        )

        assert result["deleted"] == 1
        assert not zp.exists()

    def test_dry_run_preserves_originals(self, tmp_path):
        out_dir = tmp_path / "output"
        zp = tmp_path / "keepme.zip"
        _make_zip_with_tiffs(zp, num_tiffs=1)

        result = compress_tiffs_from_zips(
            [zp], out_dir, delete_originals=True, dry_run=True, workers=1
        )

        assert result["deleted"] == 0
        assert zp.exists()

    def test_invalid_compression_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown compression"):
            compress_tiffs_from_zips([], tmp_path, compression="gzip")

    def test_skips_nonexistent_files(self, tmp_path):
        out_dir = tmp_path / "output"
        result = compress_tiffs_from_zips(
            [tmp_path / "missing.zip"], out_dir, workers=1
        )
        assert result["processed"] == 0

    def test_skips_non_zip_extension(self, tmp_path):
        out_dir = tmp_path / "output"
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        result = compress_tiffs_from_zips([txt], out_dir, workers=1)
        assert result["processed"] == 0


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_basic_cli(self, tmp_path):
        zp = tmp_path / "cli_test.zip"
        out_dir = tmp_path / "cli_output"
        _make_zip_with_tiffs(zp, num_tiffs=2)

        exit_code = main([str(zp), "-o", str(out_dir), "-j", "1"])
        assert exit_code == 0

        outputs = list(out_dir.rglob("*.tif"))
        assert len(outputs) == 2

    def test_cli_delete_flag(self, tmp_path):
        zp = tmp_path / "cli_del.zip"
        out_dir = tmp_path / "cli_output"
        _make_zip_with_tiffs(zp, num_tiffs=1)

        exit_code = main([str(zp), "-o", str(out_dir), "--delete", "-j", "1"])
        assert exit_code == 0
        assert not zp.exists()

    def test_cli_dry_run(self, tmp_path):
        zp = tmp_path / "cli_dry.zip"
        out_dir = tmp_path / "cli_output"
        _make_zip_with_tiffs(zp, num_tiffs=1)

        exit_code = main(
            [str(zp), "-o", str(out_dir), "--delete", "--dry-run", "-j", "1"]
        )
        assert exit_code == 0
        assert zp.exists()  # Not deleted in dry-run

    def test_cli_nonexistent_input(self, tmp_path):
        exit_code = main([str(tmp_path / "nope.zip"), "-o", str(tmp_path / "out")])
        assert exit_code == 1
