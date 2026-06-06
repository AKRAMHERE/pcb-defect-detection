"""
video_processor.py
------------------
Real-time video and webcam PCB defect inspection pipeline.

Implements frame-by-frame inference with:
- FPS-throttled inference (skip frames for CPU performance)
- Live HUD overlay (FPS, inference time, defect count)
- Video recording of annotated output
- Webcam hot-swap support
- Graceful interruption handling
"""

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .detector import PCBDefectDetector
from .utils import (
    DetectionResult,
    FPSTracker,
    InspectionSummary,
    ensure_dir,
    setup_logger,
    timestamp_filename,
)

logger = setup_logger(__name__)

# OpenCV fourcc codes for video encoding
FOURCC_MP4  = cv2.VideoWriter_fourcc(*"mp4v")
FOURCC_XVID = cv2.VideoWriter_fourcc(*"XVID")


class VideoProcessor:
    """
    Frame-by-frame defect detection on video files and live webcam streams.

    CPU-Optimized inference strategy:
    - Configurable inference_every_n_frames to skip frames between detections
    - Last known bounding boxes rendered on skipped frames (ghost overlay)
    - Adaptive frame dropping when CPU cannot sustain target throughput

    Args:
        detector:           Initialized PCBDefectDetector instance.
        output_dir:         Directory for saving annotated video output.
        inference_every_n:  Run model inference every N frames (default=1 for accuracy,
                            increase to 2-3 for CPU-limited environments).
        target_display_fps: Cap display loop at this FPS (0 = unlimited).
    """

    def __init__(
        self,
        detector: PCBDefectDetector,
        output_dir: Union[str, Path] = "outputs/videos",
        inference_every_n: int = 1,
        target_display_fps: int = 30,
    ):
        self.detector          = detector
        self.output_dir        = ensure_dir(output_dir)
        self.inference_every_n = max(1, inference_every_n)
        self.target_display_fps = target_display_fps
        self._fps_tracker      = FPSTracker()

    # ------------------------------------------------------------------
    # Video file processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: Union[str, Path],
        save_output: bool = True,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> tuple[list[DetectionResult], InspectionSummary]:
        """
        Run defect detection on a video file.

        Args:
            video_path:  Input video file (.mp4, .avi, etc.).
            save_output: Write annotated video to output_dir.
            display:     Show live inference window (headless environments: False).
            max_frames:  Process only first N frames (None = full video).

        Returns:
            (all_detections, InspectionSummary)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Processing video: {video_path.name} | "
            f"{width}x{height} @ {src_fps:.1f} FPS | "
            f"{total_frames} frames"
        )

        writer = None
        if save_output:
            out_name = timestamp_filename(f"{video_path.stem}_annotated", ".mp4")
            out_path = self.output_dir / out_name
            writer = cv2.VideoWriter(str(out_path), FOURCC_MP4, src_fps, (width, height))

        all_detections, summary = self._run_capture_loop(
            cap=cap,
            source_name=video_path.name,
            writer=writer,
            display=display,
            max_frames=max_frames,
            is_webcam=False,
        )

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Annotated video saved: {out_path}")

        summary.finalize()
        return all_detections, summary

    # ------------------------------------------------------------------
    # Webcam / live stream
    # ------------------------------------------------------------------

    def process_webcam(
        self,
        camera_index: int = 0,
        save_output: bool = False,
        resolution: tuple[int, int] = (1280, 720),
        max_frames: Optional[int] = None,
    ) -> tuple[list[DetectionResult], InspectionSummary]:
        """
        Run live PCB defect detection from webcam stream.

        Args:
            camera_index: OpenCV camera device index (0 = default webcam).
            save_output:  Record annotated stream to video file.
            resolution:   Requested capture resolution (W, H).
            max_frames:   Stop after N frames (None = run until 'q' pressed).

        Returns:
            (all_detections, InspectionSummary)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {camera_index}. "
                "Check camera connection or try a different index."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize latency

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Webcam opened: index={camera_index} | {actual_w}x{actual_h}")
        logger.info("Press 'q' to stop | 's' to save snapshot | '+'/'-' to adjust threshold")

        writer = None
        if save_output:
            out_name = timestamp_filename("webcam_inspection", ".mp4")
            out_path = self.output_dir / out_name
            writer = cv2.VideoWriter(
                str(out_path), FOURCC_MP4, 20.0, (actual_w, actual_h)
            )

        all_detections, summary = self._run_capture_loop(
            cap=cap,
            source_name=f"webcam_{camera_index}",
            writer=writer,
            display=True,
            max_frames=max_frames,
            is_webcam=True,
        )

        cap.release()
        if writer:
            writer.release()

        summary.finalize()
        return all_detections, summary

    # ------------------------------------------------------------------
    # Core capture/inference loop
    # ------------------------------------------------------------------

    def _run_capture_loop(
        self,
        cap: cv2.VideoCapture,
        source_name: str,
        writer: Optional[cv2.VideoWriter],
        display: bool,
        max_frames: Optional[int],
        is_webcam: bool,
    ) -> tuple[list[DetectionResult], InspectionSummary]:
        """
        Unified frame capture + inference loop for video and webcam.

        Frame inference strategy:
        - Full inference on every Nth frame (configurable)
        - Ghost overlay (last known boxes) on skipped frames
        - Keyboard interaction (q=quit, s=snapshot, +/-=threshold)
        """
        summary = InspectionSummary(source=source_name)
        all_detections: list[DetectionResult] = []
        last_detections: list[DetectionResult] = []
        frame_idx  = 0
        conf_sum   = 0.0
        self._fps_tracker.reset()

        min_frame_ms = (1000.0 / self.target_display_fps) if self.target_display_fps > 0 else 0

        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    logger.warning("Webcam frame read failed. Retrying...")
                    time.sleep(0.05)
                    continue
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Inference on every Nth frame
            if frame_idx % self.inference_every_n == 0:
                detections, annotated, inference_ms = self.detector.detect(frame)
                for det in detections:
                    det.frame_id = f"{source_name}_f{frame_idx:06d}"
                last_detections = detections
            else:
                # Ghost overlay using last known detections
                annotated = self.detector._annotate_frame(frame.copy(), last_detections)
                detections = []
                inference_ms = 0.0

            all_detections.extend(detections)
            for det in detections:
                conf_sum += det.confidence
                summary.defect_counts[det.class_name] = (
                    summary.defect_counts.get(det.class_name, 0) + 1
                )

            fps = self._fps_tracker.tick()
            if annotated is not None:
                annotated = self.detector.overlay_stats(
                    annotated, fps, len(last_detections), inference_ms
                )

            if writer and annotated is not None:
                writer.write(annotated)

            if display and annotated is not None:
                cv2.imshow("PCB Defect Inspection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Inspection stopped by user.")
                    break
                elif key == ord("s"):
                    snap_path = self.output_dir / timestamp_filename("snapshot", ".jpg")
                    cv2.imwrite(str(snap_path), annotated)
                    logger.info(f"Snapshot saved: {snap_path}")
                elif key == ord("+") or key == ord("="):
                    new_conf = min(0.95, self.detector.confidence_threshold + 0.05)
                    self.detector.set_confidence_threshold(new_conf)
                elif key == ord("-"):
                    new_conf = max(0.05, self.detector.confidence_threshold - 0.05)
                    self.detector.set_confidence_threshold(new_conf)

            # FPS cap (reduce CPU load in display loop)
            elapsed_ms = (time.perf_counter() - loop_start) * 1000
            wait_ms = min_frame_ms - elapsed_ms
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)

            if frame_idx % 100 == 0:
                logger.info(
                    f"  Frame {frame_idx} | FPS: {fps:.1f} | "
                    f"Defects: {len(all_detections)}"
                )

        if display:
            cv2.destroyAllWindows()

        perf = self.detector.get_performance_stats()
        summary.total_frames   = frame_idx
        summary.total_defects  = len(all_detections)
        summary.avg_confidence = conf_sum / len(all_detections) if all_detections else 0.0
        summary.avg_inference_ms = perf.get("avg_inference_ms", 0.0)
        summary.avg_fps          = perf.get("avg_fps", 0.0)
        summary.p95_inference_ms = perf.get("p95_inference_ms", 0.0)

        logger.info(
            f"Video processing complete | Frames: {frame_idx} | "
            f"Defects: {len(all_detections)} | Avg FPS: {summary.avg_fps:.1f}"
        )
        return all_detections, summary
