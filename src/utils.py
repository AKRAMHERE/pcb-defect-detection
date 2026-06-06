"""
utils.py
--------
Shared utilities, data structures, and helper functions for the
PCB Defect Detection pipeline.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Structured representation of a single defect detection.

    Attributes:
        class_id:   Integer class index from model output.
        class_name: Human-readable defect label.
        confidence: Model confidence score in [0, 1].
        bbox:       Bounding box as (x1, y1, x2, y2) pixel coordinates.
        frame_id:   Frame number (video) or image filename.
        timestamp:  ISO-8601 timestamp of detection.
    """
    class_id:   int
    class_name: str
    confidence: float
    bbox:       tuple       # (x1, y1, x2, y2)
    frame_id:   str = ""
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def bbox_width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def bbox_height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    def to_csv_row(self) -> dict:
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return {
            "frame_id":   self.frame_id,
            "class_id":   self.class_id,
            "defect_type": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bbox_width":  self.bbox_width,
            "bbox_height": self.bbox_height,
            "bbox_area":   round(self.bbox_area, 1),
            "timestamp":   self.timestamp,
        }


@dataclass
class InspectionSummary:
    """Aggregated statistics for a completed inspection run."""
    source:              str
    total_frames:        int = 0
    total_defects:       int = 0
    defect_counts:       dict = field(default_factory=dict)
    avg_confidence:      float = 0.0
    avg_inference_ms:    float = 0.0
    avg_fps:             float = 0.0
    p95_inference_ms:    float = 0.0
    start_time:          str = field(default_factory=lambda: datetime.now().isoformat())
    end_time:            Optional[str] = None

    def finalize(self):
        self.end_time = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "source":           self.source,
            "total_frames":     self.total_frames,
            "total_defects":    self.total_defects,
            "defect_counts":    self.defect_counts,
            "avg_confidence":   round(self.avg_confidence, 4),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_fps":          round(self.avg_fps, 2),
            "p95_inference_ms": round(self.p95_inference_ms, 2),
            "start_time":       self.start_time,
            "end_time":         self.end_time,
        }


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ------------------------------------------------------------------
# Path and file utilities
# ------------------------------------------------------------------

def validate_model_path(path: Union[str, Path]) -> bool:
    """Return True if model weights file exists and is non-empty."""
    p = Path(path)
    return p.exists() and p.is_file() and p.suffix == ".pt" and p.stat().st_size > 0


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_filename(stem: str, suffix: str) -> str:
    """Generate a timestamped filename, e.g. 'report_20240615_143022.csv'."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}{suffix}"


def resolve_output_path(
    base_dir: Union[str, Path],
    source_path: Union[str, Path],
    suffix: str,
) -> Path:
    """
    Construct output path mirroring source filename under base_dir.

    Example:
        source: /data/pcb_001.jpg
        base_dir: outputs/images
        suffix: _annotated.jpg
        → outputs/images/pcb_001_annotated.jpg
    """
    base = ensure_dir(base_dir)
    stem = Path(source_path).stem
    return base / f"{stem}{suffix}"


# ------------------------------------------------------------------
# Image validation
# ------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def is_image_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_video_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def collect_image_paths(directory: Union[str, Path]) -> list[Path]:
    """Recursively collect all supported image files from a directory."""
    return sorted(
        p for p in Path(directory).rglob("*")
        if p.is_file() and is_image_file(p)
    )


# ------------------------------------------------------------------
# FPS tracker
# ------------------------------------------------------------------

class FPSTracker:
    """
    Rolling-window FPS tracker for real-time inference pipelines.

    Uses an exponential moving average for smooth display.
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps: float = 0.0
        self._last_tick: Optional[float] = None

    def tick(self) -> float:
        """Call once per processed frame. Returns smoothed FPS."""
        import time
        now = time.perf_counter()
        if self._last_tick is not None:
            instant_fps = 1.0 / max(now - self._last_tick, 1e-9)
            self._fps = self._alpha * instant_fps + (1 - self._alpha) * self._fps
        else:
            self._fps = 0.0
        self._last_tick = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        self._fps = 0.0
        self._last_tick = None
