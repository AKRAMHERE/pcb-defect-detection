"""
main.py
-------
CLI entrypoint for the PCB Defect Detection and Automated Inspection System.

Usage modes:
    python main.py train   --dataset data/pcb_dataset
    python main.py image   --input data/test_pcb.jpg  --model models/best.pt
    python main.py batch   --input data/test_images/  --model models/best.pt
    python main.py video   --input recordings/pcb.mp4 --model models/best.pt
    python main.py webcam  --model models/best.pt --camera 0
    python main.py eval    --dataset data/pcb_dataset --model models/best.pt
    python main.py demo    --model models/best.pt
"""

import argparse
import sys
from pathlib import Path

# ── allow running as `python main.py` from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    PCBDefectDetector,
    PCBTrainer,
    ImageProcessor,
    VideoProcessor,
    ReportGenerator,
    setup_logger,
)
from configs.config import load_config

logger = setup_logger("main")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCB Defect Detection and Automated Inspection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train  --dataset data/pcb_dataset --epochs 50
  python main.py image  --input data/test.jpg --model models/best.pt
  python main.py batch  --input data/test_images/ --model models/best.pt
  python main.py video  --input recording.mp4 --model models/best.pt
  python main.py webcam --model models/best.pt
  python main.py eval   --dataset data/pcb_dataset --model models/best.pt
        """,
    )
    parser.add_argument(
        "mode",
        choices=["train", "image", "batch", "video", "webcam", "eval", "demo"],
        help="Operation mode",
    )
    parser.add_argument("--model",     type=str, default="models/pcb_defect_v1_best.pt",
                        help="Path to YOLOv8 .pt weights")
    parser.add_argument("--dataset",   type=str, default="data/pcb_dataset",
                        help="Root directory of YOLO-format dataset")
    parser.add_argument("--input",     type=str, default=None,
                        help="Input image/video file or directory")
    parser.add_argument("--output",    type=str, default="outputs",
                        help="Root output directory")
    parser.add_argument("--conf",      type=float, default=0.25,
                        help="Confidence threshold [0.0-1.0]")
    parser.add_argument("--iou",       type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--epochs",    type=int,   default=50,
                        help="Training epochs")
    parser.add_argument("--batch",     type=int,   default=8,
                        help="Training batch size")
    parser.add_argument("--imgsz",     type=int,   default=640,
                        help="Training image size")
    parser.add_argument("--camera",    type=int,   default=0,
                        help="Webcam device index")
    parser.add_argument("--no-save",   action="store_true",
                        help="Do not save annotated outputs")
    parser.add_argument("--display",   action="store_true",
                        help="Display inference window (requires display)")
    parser.add_argument("--fail-threshold", type=int, default=0,
                        help="Defect count threshold for PASS/FAIL verdict")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Inference every N frames (video mode, for CPU perf)")
    parser.add_argument("--run-name",  type=str, default="pcb_defect_v1",
                        help="Training run name")
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_train(args):
    """Train YOLOv8n on PCB defect dataset."""
    trainer = PCBTrainer(
        dataset_dir=args.dataset,
        output_dir=Path(args.output) / "models",
        config={
            "epochs": args.epochs,
            "batch":  args.batch,
            "imgsz":  args.imgsz,
        },
    )
    best_weights = trainer.train(run_name=args.run_name)
    logger.info(f"Training complete. Best weights: {best_weights}")

    # Auto-evaluate after training
    if best_weights.exists():
        logger.info("Running post-training evaluation on validation split...")
        metrics = trainer.evaluate(best_weights, split="val")
        logger.info(f"mAP@50: {metrics['mAP50']:.4f} | mAP@50-95: {metrics['mAP50_95']:.4f}")


def handle_image(args):
    """Inspect a single PCB image."""
    _require_input(args)
    detector = _build_detector(args)
    processor = ImageProcessor(
        detector=detector,
        output_dir=Path(args.output) / "images",
        save_output=not args.no_save,
    )
    reporter = ReportGenerator(
        output_dir=Path(args.output) / "reports",
        fail_threshold=args.fail_threshold,
    )

    detections, saved_path = processor.inspect_image(
        image_path=args.input,
        display=args.display,
    )

    from src.utils import InspectionSummary
    summary = InspectionSummary(source=args.input)
    summary.total_frames  = 1
    summary.total_defects = len(detections)
    for det in detections:
        summary.defect_counts[det.class_name] = (
            summary.defect_counts.get(det.class_name, 0) + 1
        )
    if detections:
        summary.avg_confidence = sum(d.confidence for d in detections) / len(detections)
    perf = detector.get_performance_stats()
    summary.avg_inference_ms = perf.get("avg_inference_ms", 0.0)
    summary.avg_fps          = perf.get("avg_fps", 0.0)
    summary.finalize()

    reporter.generate_full_report(detections, summary, report_name="image_inspection")
    if saved_path:
        logger.info(f"Annotated output: {saved_path}")


def handle_batch(args):
    """Inspect all PCB images in a directory."""
    _require_input(args)
    detector  = _build_detector(args)
    processor = ImageProcessor(
        detector=detector,
        output_dir=Path(args.output) / "images",
        save_output=not args.no_save,
    )
    reporter = ReportGenerator(
        output_dir=Path(args.output) / "reports",
        fail_threshold=args.fail_threshold,
    )

    detections, summary = processor.inspect_directory(args.input)
    reporter.generate_full_report(detections, summary, report_name="batch_inspection")


def handle_video(args):
    """Process a recorded video file."""
    _require_input(args)
    detector = _build_detector(args)
    vprocessor = VideoProcessor(
        detector=detector,
        output_dir=Path(args.output) / "videos",
        inference_every_n=args.skip_frames,
    )
    reporter = ReportGenerator(
        output_dir=Path(args.output) / "reports",
        fail_threshold=args.fail_threshold,
    )

    detections, summary = vprocessor.process_video(
        video_path=args.input,
        save_output=not args.no_save,
        display=args.display,
    )
    reporter.generate_full_report(detections, summary, report_name="video_inspection")


def handle_webcam(args):
    """Run live webcam inspection."""
    detector = _build_detector(args)
    vprocessor = VideoProcessor(
        detector=detector,
        output_dir=Path(args.output) / "videos",
        inference_every_n=args.skip_frames,
    )
    reporter = ReportGenerator(
        output_dir=Path(args.output) / "reports",
        fail_threshold=args.fail_threshold,
    )

    detections, summary = vprocessor.process_webcam(
        camera_index=args.camera,
        save_output=not args.no_save,
    )
    reporter.generate_full_report(detections, summary, report_name="webcam_inspection")


def handle_eval(args):
    """Evaluate trained model on dataset val/test split."""
    trainer = PCBTrainer(dataset_dir=args.dataset)
    metrics = trainer.evaluate(weights_path=args.model, split="val")
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")


def handle_demo(args):
    """
    Run demo: generate synthetic PCB-like test image and run inference.
    Useful for validating installation without dataset.
    """
    import numpy as np
    import cv2

    logger.info("Running demo mode with synthetic test image...")

    # Generate synthetic PCB-like image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 40, 10)  # Dark green PCB background

    # Simulate PCB traces
    for y in range(0, 480, 40):
        cv2.line(frame, (0, y), (640, y), (0, 80, 0), 1)
    for x in range(0, 640, 40):
        cv2.line(frame, (x, 0), (x, 480), (0, 80, 0), 1)

    # Add simulated component pads
    pad_positions = [(120, 100), (250, 200), (400, 150), (500, 300)]
    for px, py in pad_positions:
        cv2.rectangle(frame, (px-15, py-10), (px+15, py+10), (0, 200, 200), -1)
        cv2.rectangle(frame, (px-15, py-10), (px+15, py+10), (0, 255, 255), 1)

    detector = _build_detector(args)
    processor = ImageProcessor(
        detector=detector,
        output_dir=Path(args.output) / "images",
        save_output=True,
    )

    detections, annotated, inference_ms = processor.inspect_from_array(
        frame, frame_id="demo_synthetic"
    )
    saved = processor.save_frame(annotated, "demo_result.jpg")

    logger.info(f"Demo inference time: {inference_ms:.1f} ms")
    logger.info(f"Demo result saved: {saved}")
    logger.info("Demo complete. System is operational.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_detector(args) -> PCBDefectDetector:
    return PCBDefectDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device="cpu",
    )


def _require_input(args):
    if not args.input:
        logger.error("--input is required for this mode.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    mode_dispatch = {
        "train":  handle_train,
        "image":  handle_image,
        "batch":  handle_batch,
        "video":  handle_video,
        "webcam": handle_webcam,
        "eval":   handle_eval,
        "demo":   handle_demo,
    }

    logger.info(f"PCB Defect Detection System | Mode: {args.mode.upper()}")
    try:
        mode_dispatch[args.mode](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled error in {args.mode} mode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
