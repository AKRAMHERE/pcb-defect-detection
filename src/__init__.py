"""
PCB Defect Detection and Automated Inspection System
-----------------------------------------------------
YOLOv8n-based manufacturing quality control pipeline.

Modules:
    detector          - Core inference engine
    trainer           - Training and evaluation orchestration
    image_processor   - Single/batch image inspection
    video_processor   - Video file and webcam inspection
    report_generator  - CSV/JSON inspection reporting
    utils             - Shared data structures and utilities
"""

from .detector        import PCBDefectDetector
from .trainer         import PCBTrainer
from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .report_generator import ReportGenerator
from .utils import (
    DetectionResult,
    InspectionSummary,
    FPSTracker,
    setup_logger,
)

__version__ = "1.0.0"
__author__  = "PCB Inspection System"

__all__ = [
    "PCBDefectDetector",
    "PCBTrainer",
    "ImageProcessor",
    "VideoProcessor",
    "ReportGenerator",
    "DetectionResult",
    "InspectionSummary",
    "FPSTracker",
    "setup_logger",
]
