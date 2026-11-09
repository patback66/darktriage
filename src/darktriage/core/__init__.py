"""DarkTriage scoring and triage core package."""

from darktriage.core.exif_reader import (
    Assessment,
    BurstGroup,
    ImagePair,
    decode_capture_time,
    detect_bursts,
    discover_pairs,
    read_capture_time,
    select_burst_winners,
)
from darktriage.core.metrics import (
    ExposureReport,
    Score,
    check_exposure,
    compute_composition_score,
    compute_tenengrad,
    prepare_proxy,
    score_image,
)
from darktriage.core.xmp_writer import (
    RatingDecision,
    WriteResult,
    XmpWriter,
    derive_rating_and_label,
    xmp_sidecar_path,
)

__all__ = [
    "Assessment",
    "BurstGroup",
    "ImagePair",
    "decode_capture_time",
    "detect_bursts",
    "discover_pairs",
    "read_capture_time",
    "select_burst_winners",
    "ExposureReport",
    "Score",
    "check_exposure",
    "compute_composition_score",
    "compute_tenengrad",
    "prepare_proxy",
    "score_image",
    "RatingDecision",
    "WriteResult",
    "XmpWriter",
    "derive_rating_and_label",
    "xmp_sidecar_path",
]