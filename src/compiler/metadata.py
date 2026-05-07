from pathlib import Path
from typing import Annotated, Literal, Optional, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_pascal

from .discovery import path_exists


class Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="allow",
        populate_by_name=True,
    )


class MeasurementRecordBase(Base):
    type: str
    time: str
    column: int
    row: int
    time_point: int
    timeline_index: int
    x: float
    y: float
    value: str | None = None
    field_index: int | None = None
    partial_tile_index: int | None = None


class ImageMeasurementRecord(MeasurementRecordBase):
    type: Literal["IMG"]
    tile_x_index: int | None = None
    tile_y_index: int | None = None
    z_index: int
    z_image_processing: str | None = None
    z_top: float | None = None
    z_bottom: float | None = None
    action_index: int
    action: str
    z: float
    ch: int


class SoftFocusMeasurementRecord(MeasurementRecordBase):
    """Records of in-line soft-focus measurements (Type=\"SF\").

    The body holds the measured focus value rather than a TIF path.
    """
    type: Literal["SF"]
    z_index: int | None = None
    action_index: int | None = None
    action: str | None = None
    z: float | None = None
    ch: int | None = None


class ErrorMeasurementRecord(MeasurementRecordBase):
    type: Literal["ERR"]


class MeasurementData(Base):
    xmlns: Annotated[Optional[dict], Field(alias="xmlns", default=None)]
    version: Optional[str] = None
    measurement_record: list[
        ImageMeasurementRecord | SoftFocusMeasurementRecord | ErrorMeasurementRecord
    ] | None = None

    @field_validator("measurement_record", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class MeasurementSamplePlate(Base):
    name: str
    well_plate_file_name: str
    well_plate_product_file_name: str | None = None


class MeasurementChannel(Base):
    ch: int
    horizontal_pixel_dimension: float
    vertical_pixel_dimension: float
    camera_number: int
    input_bit_depth: int
    input_level: int
    horizontal_pixels: int
    vertical_pixels: int
    filter_wheel_position: int | None = None
    filter_position: int | None = None
    shading_correction_source: str | None = None
    objective_magnification_ratio: float | None = None
    original_horizontal_pixels: int | None = None
    original_vertical_pixels: int | None = None


class MeasurementDetail(Base):
    xmlns: Annotated[Optional[dict], Field(alias="xmlns", default=None)]
    version: Optional[str] = None
    operator_name: str | None = None
    title: str | None = None
    application: str | None = None
    begin_time: str
    end_time: str | None = None
    measurement_setting_file_name: str
    column_count: int
    row_count: int
    time_point_count: int
    field_count: int
    z_count: int
    target_system: str | None = None
    release_number: str | None = None
    status: str | None = None
    measurement_sample_plate: MeasurementSamplePlate
    measurement_channel: list[MeasurementChannel]

    @field_validator("measurement_channel", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class WellPlate(Base):
    xmlns: Annotated[Optional[dict], Field(alias="xmlns", default=None)]
    version: Optional[str] = None
    name: str
    product_i_d: str | None = None
    usage: str | None = None
    density_unit: str | None = None
    columns: int
    rows: int
    description: str | None = None


class TargetWell(Base):
    column: float
    row: float
    value: bool


class WellSequence(Base):
    is_selected: bool = Field(alias='IsSelected')
    target_well: Optional[list[TargetWell]] = Field(default=None, alias='TargetWell')

    @field_validator('target_well', mode='before')
    @classmethod
    def _ensure_list(cls, v: Any):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise ValueError(f'Expected dict, list, or None, got {type(v)}')


class Point(Base):
    x: float = Field(alias="X")
    y: float = Field(alias="Y")


class FixedPosition(Base):
    is_proportional: bool
    point: list[Point] = Field(alias="Point")

    @field_validator("point", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class TiledArea(Base):
    start_point_x: float = Field(alias="StartPointX")
    start_point_y: float = Field(alias="StartPointY")
    end_point_x: float = Field(alias="EndPointX")
    end_point_y: float = Field(alias="EndPointY")


class PartialTiledPosition(Base):
    overlapping_pixels: int = Field(alias="OverlappingPixels")
    scan_method: Literal["Raster", "Tile"] = Field(alias="ScanMethod")
    fill: str = Field(alias="Fill")
    tiled_area: list[TiledArea] = Field(alias="TiledArea")

    @field_validator("tiled_area", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class PointSequence(Base):
    method: Literal["FixedPosition", "PartialTiledPosition"] = Field(alias="Method")
    fixed_position: FixedPosition | None = None
    partial_tiled_position: PartialTiledPosition | None = None


class LiveOption(Base):
    period: str = Field(alias="Period")
    interval: str = Field(alias="Interval")
    kind: str = Field(alias="Kind")
    perform_af: str = Field(alias="PerformAF")


class _ActionAcquireBase(Base):
    """Fields shared by all action-acquire structures."""
    x_offset: str = Field(alias="XOffset")
    y_offset: str = Field(alias="YOffset")


class ActionAcquire3D(_ActionAcquireBase):
    af_shift_base: str = Field(alias="AFShiftBase")
    top_distance: str = Field(alias="TopDistance")
    bottom_distance: str = Field(alias="BottomDistance")
    slice_length: str = Field(alias="SliceLength")
    use_soft_focus: str = Field(alias="UseSoftFocus")
    ch: str | list[str] = Field(alias="Ch")
    image_processing: Optional[str] = Field(alias="ImageProcessing", default=None)

    @field_validator("ch", mode="before")
    @classmethod
    def _ensure_list_or_str(cls, v):
        if isinstance(v, (str, list)):
            return v
        raise TypeError(f"Expected string or list, got {type(v).__name__}")


class ActionAcquireBF3D(_ActionAcquireBase):
    af_shift_base: str = Field(alias="AFShiftBase")
    top_distance: str = Field(alias="TopDistance")
    bottom_distance: str = Field(alias="BottomDistance")
    slice_length: str = Field(alias="SliceLength")
    ch: str = Field(alias="Ch")


class ActionAcquireBF(_ActionAcquireBase):
    z_offset: Optional[str] = Field(alias="ZOffset", default=None)
    live_option: Optional[LiveOption] = Field(alias="LiveOption", default=None)
    ch: str = Field(alias="Ch")


class ActionAcquire(_ActionAcquireBase):
    z_offset: Optional[str] = Field(alias="ZOffset", default=None)
    ignore_soft_focus: Optional[str] = Field(alias="IgnoreSoftFocus", default=None)
    connected_action: Optional[str] = Field(alias="ConnectedAction", default=None)
    live_option: Optional[LiveOption] = Field(alias="LiveOption", default=None)
    ch: str = Field(alias="Ch")


class ActionSoftFocus(_ActionAcquireBase):
    """Soft-focus probe action — produces SF-type measurement records but no TIFs."""
    af_shift_base: Optional[str] = Field(alias="AFShiftBase", default=None)
    top_distance: Optional[str] = Field(alias="TopDistance", default=None)
    bottom_distance: Optional[str] = Field(alias="BottomDistance", default=None)
    slice_length: Optional[str] = Field(alias="SliceLength", default=None)
    ch: Optional[str | list[str]] = Field(alias="Ch", default=None)


class ActionList(Base):
    run_mode: str = Field(alias="RunMode")
    a_f_search: Optional[str] = Field(alias="AFSearch", default=None)

    action_acquire: Optional[list[ActionAcquire]] = Field(default=None, alias="ActionAcquire")
    action_acquire_3_d: Optional[list[ActionAcquire3D]] = Field(default=None, alias="ActionAcquire3D")
    action_acquire_bf: Optional[list[ActionAcquireBF]] = Field(default=None, alias="ActionAcquireBF")
    action_acquire_bf_3_d: Optional[list[ActionAcquireBF3D]] = Field(default=None, alias="ActionAcquireBF3D")
    action_soft_focus: Optional[list[ActionSoftFocus]] = Field(default=None, alias="ActionSoftFocus")

    @field_validator(
        "action_acquire",
        "action_acquire_3_d",
        "action_acquire_bf",
        "action_acquire_bf_3_d",
        "action_soft_focus",
        mode="before",
    )
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class Timeline(Base):
    name: str
    initial_time: int
    period: int
    interval: int
    expected_time: int
    color: str | None = None
    override_expected_time: bool | None = None
    well_sequence: WellSequence
    point_sequence: PointSequence
    action_list: ActionList


class Timelapse(Base):
    timeline: list[Timeline]

    @field_validator('timeline', mode='before')
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise ValueError(f'Expected dict or list, got {type(v)}')


class LightSource(Base):
    name: str
    type: str
    wave_length: int
    power: int


class LightSourceList(Base):
    use_calibrated_laser_power: bool | None = None
    light_source: list[LightSource]

    @field_validator("light_source", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class Channel(Base):
    ch: int
    target: str | None = None
    objective_i_d: str | None = None
    objective: str | None = None
    magnification: int | None = None
    method_i_d: int | None = None
    method: str | None = None
    filter_i_d: int | None = None
    acquisition: str | None = None
    exposure_time: int | None = None
    binning: int | None = None
    color: str | None = None
    min_level: float | None = None
    max_level: float | None = None
    c_s_u_i_d: int | None = None
    pinhole_diameter: int | None = None
    andor_parameter_i_d: int | None = None
    andor_parameter: str | None = None
    kind: str | None = None
    camera_type: str | None = None
    input_level: int | None = None
    fluorophore: str | None = None
    light_source_name: str | list[str] | None = None


class ChannelList(Base):
    channel: list[Channel]

    @field_validator("channel", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class MeasurementSetting(Base):
    xmlns: Annotated[Optional[dict], Field(alias="xmlns", default=None)]
    version: Optional[str] = None
    product_i_d: str | None = None
    application: str | None = None
    columns: int
    rows: int
    timelapse: Timelapse
    light_source_list: LightSourceList | None = None
    channel_list: ChannelList


class CellVoyagerAcquisition(Base):
    parent: Path = Field(alias="parent")
    well_plate: WellPlate
    measurement_data: MeasurementData
    measurement_detail: MeasurementDetail
    measurement_setting: MeasurementSetting

    @field_validator("parent")
    @classmethod
    def validate_parent(cls, v):
        if not path_exists(v):
            raise ValueError(f"Provided path does not exist: {v}")
        return v

    def get_image_measurement_records(self) -> list[ImageMeasurementRecord]:
        if self.measurement_data.measurement_record:
            return [
                record
                for record in self.measurement_data.measurement_record
                if isinstance(record, ImageMeasurementRecord)
            ]
        raise ValueError("No measurement records found in dataset.")

    def is_tiled(self) -> bool:
        """True if any image record carries tile indices."""
        if not self.measurement_data.measurement_record:
            return False
        for r in self.measurement_data.measurement_record:
            if isinstance(r, ImageMeasurementRecord) and r.tile_x_index is not None:
                return True
        return False

    def get_partial_tiled_position(
        self, timeline_index: int = 1
    ) -> PartialTiledPosition | None:
        """Look up the PartialTiledPosition for a given 1-based timeline index, if any."""
        timelines = self.measurement_setting.timelapse.timeline
        if timeline_index < 1 or timeline_index > len(timelines):
            return None
        return timelines[timeline_index - 1].point_sequence.partial_tiled_position

    def get_tile_overlap(self, timeline_index: int = 1) -> int:
        """Return OverlappingPixels for the given timeline, or 0 if not tiled."""
        ptp = self.get_partial_tiled_position(timeline_index)
        return ptp.overlapping_pixels if ptp is not None else 0
