"""Tests for yokogawa_cv8000_compiler.functions using synthetic data."""

import os
from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd
import pytest

from yokogawa_cv8000_compiler.compiling import compile_field
from yokogawa_cv8000_compiler.parsing import data_summary, parse_measurement_data
from yokogawa_cv8000_compiler.utils import filter_df, get_well_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small synthetic DataFrame mimicking parse_measurement_data output."""
    begin = pd.Timestamp("2025-01-01 10:00:00")
    rows = []
    for tp in range(1, 3):          # 2 timepoints
        for z in range(1, 4):       # 3 z-slices
            for ch in [1, 2]:       # 2 channels
                rows.append({
                    "filename": f"/data/img_tp{tp}_z{z}_ch{ch}_W01_T0001.tif",
                    "title": "Exp1",
                    "plate": "Exp1",
                    "diff": pd.Timedelta(minutes=tp),
                    "condition": "ctrl",
                    "timestamp": begin + pd.Timedelta(minutes=tp, seconds=z),
                    "begin": begin,
                    "end": begin + pd.Timedelta(hours=1),
                    "column": 1,
                    "row": 1,
                    "well": "W01",
                    "timepoint": tp,
                    "field": 1,
                    "z_idx": z,
                    "timeline_idx": 1,
                    "action_idx": 1,
                    "action": "3D",
                    "channel": ch,
                    "color": ch + 10,
                    "fixed": False,
                    "pixelsize": 0.325,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_well_name
# ---------------------------------------------------------------------------

class TestGetWellName:
    def test_basic(self):
        assert get_well_name("03_02") == "B3"

    def test_first_row(self):
        assert get_well_name("01_01") == "A1"

    def test_double_digit_column(self):
        assert get_well_name("12_03") == "C12"

    def test_row_boundaries(self):
        # Row 8 → H
        assert get_well_name("05_08") == "H5"


# ---------------------------------------------------------------------------
# filter_df
# ---------------------------------------------------------------------------

class TestFilterDf:
    def test_filter_single_column(self, sample_df):
        result = filter_df(sample_df, channel=1)
        assert (result["channel"] == 1).all()
        assert len(result) == len(sample_df) // 2

    def test_filter_multiple_columns(self, sample_df):
        result = filter_df(sample_df, channel=2, timepoint=1)
        assert len(result) == 3  # 3 z-slices
        assert (result["channel"] == 2).all()
        assert (result["timepoint"] == 1).all()

    def test_filter_no_match(self, sample_df):
        result = filter_df(sample_df, well="NONEXISTENT")
        assert len(result) == 0

    def test_filter_result_sorted_by_timestamp(self, sample_df):
        result = filter_df(sample_df, channel=1)
        timestamps = result["timestamp"].values
        assert (timestamps[:-1] <= timestamps[1:]).all()

    def test_filter_preserves_dtypes(self, sample_df):
        result = filter_df(sample_df, channel=1)
        assert result["channel"].dtype == sample_df["channel"].dtype
        assert result["pixelsize"].dtype == sample_df["pixelsize"].dtype


# ---------------------------------------------------------------------------
# data_summary
# ---------------------------------------------------------------------------

class TestDataSummary:
    def test_returns_unique_values(self, sample_df):
        plates, wells, fields = data_summary(sample_df, print_summary=False)
        assert list(plates) == ["Exp1"]
        assert list(wells) == ["W01"]
        assert list(fields) == [1]

    def test_prints_summary(self, sample_df, capsys):
        data_summary(sample_df, print_summary=True)
        captured = capsys.readouterr()
        assert "1 plates" in captured.out
        assert "1 wells" in captured.out
        assert "2 timepoints" in captured.out

    def test_no_print_when_disabled(self, sample_df, capsys):
        data_summary(sample_df, print_summary=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# compile_field
# ---------------------------------------------------------------------------

class TestCompileField:
    def test_invalid_proj_mode_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown proj_mode"):
            compile_field(sample_df, "Exp1", "W01", 1, "/tmp", proj_mode="invalid")

    def test_empty_df_returns_none(self, sample_df):
        # No data for well W99 → should return without error
        result = compile_field(sample_df, "Exp1", "W99", 1, "/tmp")
        assert result is None

    def test_mip_projection(self, sample_df, tmp_path):
        """MIP should produce the max across z-slices."""
        fake_images = {
            f: np.random.randint(0, 1000, (64, 64), dtype=np.uint16)
            for f in sample_df["filename"].unique()
        }

        def mock_imread(path):
            return fake_images[path]

        with patch("yokogawa_cv8000_compiler.compiling.tifffile.imread", side_effect=mock_imread):
            with patch("yokogawa_cv8000_compiler.compiling.tifffile.imwrite") as mock_write:
                compile_field(sample_df, "Exp1", "W01", 1, str(tmp_path), proj_mode="mip")

        # 2 channels → 2 files written
        assert mock_write.call_count == 2

        # Check shape: 2 timepoints, each projected to (64, 64)
        for call in mock_write.call_args_list:
            data = call[0][1]  # positional arg: data array
            assert data.shape == (2, 64, 64)

    def test_map_projection(self, sample_df, tmp_path):
        """MAP should pick the z-slice with highest mean intensity."""
        # Make z=3 always brightest so we know which slice is picked
        def mock_imread(path):
            if "_z3_" in path:
                return np.full((64, 64), 1000, dtype=np.uint16)
            return np.full((64, 64), 100, dtype=np.uint16)

        with patch("yokogawa_cv8000_compiler.compiling.tifffile.imread", side_effect=mock_imread):
            with patch("yokogawa_cv8000_compiler.compiling.tifffile.imwrite") as mock_write:
                compile_field(sample_df, "Exp1", "W01", 1, str(tmp_path), proj_mode="map")

        assert mock_write.call_count == 2
        for call in mock_write.call_args_list:
            data = call[0][1]
            # Every projected frame should be the bright slice (all 1000s)
            assert (data == 1000).all()

    def test_output_filename_pattern(self, sample_df, tmp_path):
        """Output filenames should contain plate, well, field, channel, color, action."""
        def mock_imread(path):
            return np.zeros((64, 64), dtype=np.uint16)

        with patch("yokogawa_cv8000_compiler.compiling.tifffile.imread", side_effect=mock_imread):
            with patch("yokogawa_cv8000_compiler.compiling.tifffile.imwrite") as mock_write:
                compile_field(sample_df, "Exp1", "W01", 1, str(tmp_path))

        written_paths = [call[0][0] for call in mock_write.call_args_list]
        for path in written_paths:
            basename = os.path.basename(path)
            assert basename.startswith("plateExp1_wellW01_field1_channel")
            assert "_action3D.tif" in basename

    def test_pixelsize_in_resolution(self, sample_df, tmp_path):
        """Resolution metadata should reflect the pixel size."""
        def mock_imread(path):
            return np.zeros((64, 64), dtype=np.uint16)

        with patch("yokogawa_cv8000_compiler.compiling.tifffile.imread", side_effect=mock_imread):
            with patch("yokogawa_cv8000_compiler.compiling.tifffile.imwrite") as mock_write:
                compile_field(sample_df, "Exp1", "W01", 1, str(tmp_path))

        for call in mock_write.call_args_list:
            resolution = call[1]["resolution"]
            expected = 1 / 0.325
            assert abs(resolution[0] - expected) < 1e-6
            assert abs(resolution[1] - expected) < 1e-6


# ---------------------------------------------------------------------------
# parse_measurement_data (with synthetic XML files)
# ---------------------------------------------------------------------------

# Minimal XML content for MeasurementData.mlf
MLF_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<bts:MeasurementData xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">
  <bts:MeasurementRecord bts:Type="IMG" bts:Time="2025-01-01T10:01:00"
    bts:Column="1" bts:Row="1" bts:TimePoint="1" bts:FieldIndex="1"
    bts:ZIndex="1" bts:TimelineIndex="1" bts:ActionIndex="1"
    bts:Action="3D" bts:Ch="1">img_W01_T0001.tif</bts:MeasurementRecord>
  <bts:MeasurementRecord bts:Type="ERR" bts:Time="2025-01-01T10:01:01"
    bts:Column="1" bts:Row="1" bts:TimePoint="1" bts:FieldIndex="1"
    bts:ZIndex="1" bts:TimelineIndex="1" bts:ActionIndex="1"
    bts:Action="3D" bts:Ch="1">error_record.tif</bts:MeasurementRecord>
</bts:MeasurementData>
"""

# Minimal XML content for MeasurementDetail.mrf
MRF_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<bts:MeasurementDetail xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">
  <bts:Title>TestPlate</bts:Title>
  <bts:BeginTime>2025-01-01T10:00:00</bts:BeginTime>
  <bts:EndTime>2025-01-01T11:00:00</bts:EndTime>
  <bts:MeasurementChannel bts:Ch="1" bts:FilterWheelPosition="2"
    bts:HorizontalPixelDimension="0.325" />
</bts:MeasurementDetail>
"""


class TestParseMeasurementData:
    def test_parses_mlf_and_mrf(self, tmp_path):
        """End-to-end parse with synthetic XML — ERR records should be skipped."""
        # Create directory structure: tmp_path/plate/measurement/data/
        data_dir = tmp_path / "plate" / "measurement" / "data"
        data_dir.mkdir(parents=True)

        mlf_file = data_dir / "MeasurementData.mlf"
        mlf_file.write_text(MLF_XML)

        mrf_file = data_dir / "MeasurementDetail.mrf"
        mrf_file.write_text(MRF_XML)

        df_imgs, plates, wells, fields = parse_measurement_data([str(mlf_file)])

        # ERR record should be skipped, so only 1 row
        assert len(df_imgs) == 1
        assert df_imgs["action"].iloc[0] == "3D"
        assert df_imgs["channel"].iloc[0] == 1
        assert df_imgs["well"].iloc[0] == "W01"
        assert df_imgs["plate"].iloc[0] == "TestPlate"
        assert df_imgs["color"].iloc[0] == 2  # FilterWheelPosition
        assert df_imgs["pixelsize"].iloc[0] == pytest.approx(0.325)

    def test_reads_wells_txt_conditions(self, tmp_path):
        """Conditions from Wells.txt should be attached to rows."""
        data_dir = tmp_path / "plate" / "measurement" / "data"
        data_dir.mkdir(parents=True)

        mlf_file = data_dir / "MeasurementData.mlf"
        mlf_file.write_text(MLF_XML)

        mrf_file = data_dir / "MeasurementDetail.mrf"
        mrf_file.write_text(MRF_XML)

        # Wells.txt lives one level above the data dir
        wells_txt = tmp_path / "plate" / "measurement" / "Wells.txt"
        wells_txt.write_text("W1 - treated\nW2 - control\n")

        df_imgs, _, _, _ = parse_measurement_data([str(mlf_file)])
        # W01 → strip zeros → W1 → "treated"
        assert df_imgs["condition"].iloc[0] == "treated"

    def test_no_wells_txt(self, tmp_path):
        """Missing Wells.txt should result in None condition."""
        data_dir = tmp_path / "plate" / "measurement" / "data"
        data_dir.mkdir(parents=True)

        mlf_file = data_dir / "MeasurementData.mlf"
        mlf_file.write_text(MLF_XML)

        mrf_file = data_dir / "MeasurementDetail.mrf"
        mrf_file.write_text(MRF_XML)

        df_imgs, _, _, _ = parse_measurement_data([str(mlf_file)])
        assert df_imgs["condition"].iloc[0] is None

    def test_column_count_matches(self, tmp_path):
        """DataFrame should have exactly 21 columns."""
        data_dir = tmp_path / "plate" / "measurement" / "data"
        data_dir.mkdir(parents=True)

        (data_dir / "MeasurementData.mlf").write_text(MLF_XML)
        (data_dir / "MeasurementDetail.mrf").write_text(MRF_XML)

        df_imgs, _, _, _ = parse_measurement_data([str(data_dir / "MeasurementData.mlf")])
        assert len(df_imgs.columns) == 21

    def test_types_preserved(self, tmp_path):
        """Numeric and datetime columns should not be strings."""
        data_dir = tmp_path / "plate" / "measurement" / "data"
        data_dir.mkdir(parents=True)

        (data_dir / "MeasurementData.mlf").write_text(MLF_XML)
        (data_dir / "MeasurementDetail.mrf").write_text(MRF_XML)

        df_imgs, _, _, _ = parse_measurement_data([str(data_dir / "MeasurementData.mlf")])
        assert df_imgs["channel"].dtype in (np.int64, int)
        assert df_imgs["field"].dtype in (np.int64, int)
        assert pd.api.types.is_datetime64_any_dtype(df_imgs["timestamp"])
        assert df_imgs["fixed"].dtype == bool
