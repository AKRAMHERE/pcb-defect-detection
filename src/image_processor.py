"""
image_processor.py
------------------
Single-image and batch-image PCB inspection pipeline.

Handles image loading, defect detection, annotated output saving,
and per-image result collection for downstream reporting.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .detector import PCBDefectDetector
from .utils import (
    DetectionResult,
    InspectionSummary,
    collect_image_paths,
    ensure_dir,
    is_image_file,
    resolve_output_path,
    setup_logger,
    timestamp_filename,
)

logger = setup_logger(__name__)


class ImageProcessor:
    """
    Batch and single-image inspection engine for PCB defect detection.

    Wraps PCBDefectDetector with image I/O, output persistence,
    and structured result aggregation.

    Args:
        detector:    Initialized PCBDefectDetector instance.
        output_dir:  Root directory for saving annotated outputs.
        save_output: Whether to persist annotated images to disk.
    """

    def __init__(
        self,
        detector: PCBDefectDetector,
        output_dir: Union[str, Path] = "outputs/images",
        save_output: bool = True,
    ):
        self.detector   = detector
        self.output_dir = ensure_dir(output_dir)
        self.save_output = save_output

    # ------------------------------------------------------------------
    # Single image inspection
    # ------------------------------------------------------------------

    def inspect_image(
        self,
        image_path: Union[str, Path],
        display: bool = False,
    ) -> tuple[list[DetectionResult], Optional[str]]:
        """
        Run defect detection on a single PCB image.

        Args:
            image_path: Path to input image file.
            display:    Show annotated result in an OpenCV window (debug use).

        Returns:
            Tuple of (detections list, saved_output_path or None).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not is_image_file(image_path):
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"OpenCV failed to decode image: {image_path}")

        detections, annotated, inference_ms = self.detector.detect(frame)

        # Tag detections with source filename
        for det in detections:
            det.frame_id = image_path.name

        logger.info(
            f"[{image_path.name}] Defects: {len(detections)} | "
            f"Inference: {inference_ms:.1f} ms"
        )

        saved_path = None
        if self.save_output and annotated is not None:
            saved_path = self._save_annotated(annotated, image_path)

        if display and annotated is not None:
            self._display_frame(annotated, title=f"PCB Inspection: {image_path.name}")

        return detections, saved_path

    def inspect_from_array(
        self,
        frame: np.ndarray,
        frame_id: str = "array_input",
    ) -> tuple[list[DetectionResult], np.ndarray, float]:
        """
        Run detection on a pre-loaded NumPy BGR array.

        Useful for integration with video pipelines or external capture systems.
        """
        detections, annotated, inference_ms = self.detector.detect(frame)
        for det in detections:
            det.frame_id = frame_id
        return detections, annotated, inference_ms

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def inspect_directory(
        self,
        input_dir: Union[str, Path],
        recursive: bool = True,
    ) -> tuple[list[DetectionResult], InspectionSummary]:
        """
        Recursively inspect all images in a directory.

        Returns aggregated detection list and inspection summary statistics.

        Args:
            input_dir: Directory containing PCB images.
            recursive: Recurse into subdirectories.

        Returns:
            (all_detections, InspectionSummary)
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        image_paths = collect_image_paths(input_dir) if recursive else [
            p for p in sorted(input_dir.iterdir()) if is_image_file(p)
        ]

        if not image_paths:
            logger.warning(f"No supported images found in: {input_dir}")
            return [], InspectionSummary(source=str(input_dir))

        logger.info(f"Batch inspection: {len(image_paths)} images from {input_dir}")

        summary = InspectionSummary(source=str(input_dir))
        all_detections: list[DetectionResult] = []
        confidence_sum = 0.0

        for i, path in enumerate(image_paths, 1):
            try:
                detections, _ = self.inspect_image(path)
                all_detections.extend(detections)

                for det in detections:
                    confidence_sum += det.confidence
                    summary.defect_counts[det.class_name] = (
                        summary.defect_counts.get(det.class_name, 0) + 1
                    )

                if i % 10 == 0 or i == len(image_paths):
                    logger.info(f"  Progress: {i}/{len(image_paths)} images processed")

            except Exception as e:
                logger.error(f"  Failed to process {path.name}: {e}")
                continue

        summary.total_frames  = len(image_paths)
        summary.total_defects = len(all_detections)
        summary.avg_confidence = (
            confidence_sum / len(all_detections) if all_detections else 0.0
        )

        perf = self.detector.get_performance_stats()
        summary.avg_inference_ms = perf.get("avg_inference_ms", 0.0)
        summary.avg_fps          = perf.get("avg_fps", 0.0)
        summary.p95_inference_ms = perf.get("p95_inference_ms", 0.0)
        summary.finalize()

        self._log_summary(summary)
        return all_detections, summary

    # ------------------------------------------------------------------
    # I/O utilities
    # ------------------------------------------------------------------

    def _save_annotated(
        self,
        frame: np.ndarray,
        source_path: Path,
    ) -> str:
        """Save annotated image to output directory."""
        out_path = resolve_output_path(self.output_dir, source_path, "_annotated.jpg")
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.debug(f"Annotated image saved: {out_path}")
        return str(out_path)

    def save_frame(
        self,
        frame: np.ndarray,
        filename: Optional[str] = None,
    ) -> str:
        """Save arbitrary frame to output directory with optional custom filename."""
        fname = filename or timestamp_filename("inspection", ".jpg")
        out_path = self.output_dir / fname
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(out_path)

    @staticmethod
    def _display_frame(frame: np.ndarray, title: str = "PCB Inspection"):
        """Display frame in OpenCV window. Press any key to continue."""
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @staticmethod
    def _log_summary(summary: InspectionSummary):
        logger.info("=" * 50)
        logger.info("Batch Inspection Summary")
        logger.info(f"  Source:          {summary.source}")
        logger.info(f"  Total images:    {summary.total_frames}")
        logger.info(f"  Total defects:   {summary.total_defects}")
        logger.info(f"  Avg confidence:  {summary.avg_confidence:.3f}")
        logger.info(f"  Avg inference:   {summary.avg_inference_ms:.1f} ms")
        logger.info(f"  Avg FPS:         {summary.avg_fps:.1f}")
        if summary.defect_counts:
            logger.info("  Defects by type:")
            for cls, count in sorted(summary.defect_counts.items()):
                logger.info(f"    {cls:<20} {count}")
        logger.info("=" * 50)
