"""
report_generator.py
-------------------
Inspection reporting engine for the PCB Defect Detection system.

Generates:
- CSV inspection logs (per-detection records)
- JSON summary reports (aggregated statistics)
- Console-formatted summary tables
- Defect frequency analysis
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

from .utils import (
    DetectionResult,
    InspectionSummary,
    ensure_dir,
    setup_logger,
    timestamp_filename,
)

logger = setup_logger(__name__)

# CSV schema definition
DETECTION_CSV_FIELDS = [
    "frame_id",
    "class_id",
    "defect_type",
    "confidence",
    "x1", "y1", "x2", "y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "timestamp",
]


class ReportGenerator:
    """
    Generates structured inspection reports from detection results.

    Supports:
    - Per-detection CSV logs for integration with MES/QC systems
    - Aggregated JSON summary for dashboards and archival
    - Defect frequency analysis with pass/fail thresholds

    Args:
        output_dir:       Directory for report output files.
        fail_threshold:   Max allowable defects before board is flagged as FAIL.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/reports",
        fail_threshold: int = 0,
    ):
        self.output_dir    = ensure_dir(output_dir)
        self.fail_threshold = fail_threshold

    # ------------------------------------------------------------------
    # CSV report
    # ------------------------------------------------------------------

    def save_csv_report(
        self,
        detections: list[DetectionResult],
        report_name: str = "inspection",
    ) -> Path:
        """
        Write per-detection CSV report.

        Columns: frame_id, class_id, defect_type, confidence,
                 x1, y1, x2, y2, bbox_width, bbox_height, bbox_area, timestamp

        Args:
            detections:  List of DetectionResult objects from inspection run.
            report_name: Base filename prefix.

        Returns:
            Path to written CSV file.
        """
        filename = timestamp_filename(report_name, ".csv")
        csv_path = self.output_dir / filename

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_FIELDS)
            writer.writeheader()
            for det in detections:
                writer.writerow(det.to_csv_row())

        logger.info(f"CSV report saved: {csv_path} ({len(detections)} records)")
        return csv_path

    # ------------------------------------------------------------------
    # JSON summary report
    # ------------------------------------------------------------------

    def save_json_summary(
        self,
        summary: InspectionSummary,
        detections: list[DetectionResult],
        report_name: str = "summary",
    ) -> Path:
        """
        Write aggregated JSON inspection summary.

        Includes:
        - Source metadata
        - Defect counts per category
        - Performance metrics
        - Pass/fail verdict
        - Per-class confidence breakdown

        Returns:
            Path to written JSON file.
        """
        filename = timestamp_filename(report_name, ".json")
        json_path = self.output_dir / filename

        # Per-class confidence aggregation
        class_confidences: dict[str, list[float]] = {}
        for det in detections:
            class_confidences.setdefault(det.class_name, []).append(det.confidence)

        class_stats = {
            cls: {
                "count":       len(confs),
                "avg_conf":    round(sum(confs) / len(confs), 4),
                "max_conf":    round(max(confs), 4),
                "min_conf":    round(min(confs), 4),
            }
            for cls, confs in class_confidences.items()
        }

        verdict = self._compute_verdict(summary.total_defects)

        report = {
            **summary.to_dict(),
            "verdict":         verdict,
            "fail_threshold":  self.fail_threshold,
            "class_statistics": class_stats,
            "report_generated": datetime.now().isoformat(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON summary saved: {json_path}")
        return json_path

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(
        self,
        summary: InspectionSummary,
        detections: list[DetectionResult],
    ):
        """
        Print formatted inspection summary to stdout.

        Suitable for CI/CD pipeline output and operator terminal display.
        """
        verdict = self._compute_verdict(summary.total_defects)
        verdict_color = "\033[91m" if verdict == "FAIL" else "\033[92m"
        reset = "\033[0m"
        bold  = "\033[1m"

        print(f"\n{bold}{'='*60}{reset}")
        print(f"{bold}PCB Inspection Report{reset}")
        print(f"{'='*60}")
        print(f"  Source:           {summary.source}")
        print(f"  Frames/Images:    {summary.total_frames}")
        print(f"  Total Defects:    {summary.total_defects}")
        print(f"  Avg Confidence:   {summary.avg_confidence:.3f}")
        print(f"  Avg Inference:    {summary.avg_inference_ms:.1f} ms")
        print(f"  P95 Inference:    {summary.p95_inference_ms:.1f} ms")
        print(f"  Avg FPS:          {summary.avg_fps:.1f}")
        print(f"  Start:            {summary.start_time}")
        print(f"  End:              {summary.end_time or 'N/A'}")
        print()

        if summary.defect_counts:
            print(f"  {'Defect Type':<25} {'Count':>8}")
            print(f"  {'-'*35}")
            for cls, count in sorted(
                summary.defect_counts.items(), key=lambda x: -x[1]
            ):
                bar = "█" * min(count, 30)
                print(f"  {cls:<25} {count:>8}  {bar}")
        else:
            print("  No defects detected.")

        print()
        print(
            f"  Verdict: {verdict_color}{bold}{verdict}{reset}  "
            f"(threshold: {self.fail_threshold} defects)"
        )
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def defect_frequency_analysis(
        self,
        detections: list[DetectionResult],
    ) -> dict:
        """
        Compute defect frequency distribution and confidence statistics.

        Returns dict suitable for dashboard integration or further analysis.
        """
        if not detections:
            return {"total": 0, "classes": {}}

        class_data: dict[str, dict] = {}
        for det in detections:
            cls = det.class_name
            if cls not in class_data:
                class_data[cls] = {"count": 0, "confidences": []}
            class_data[cls]["count"] += 1
            class_data[cls]["confidences"].append(det.confidence)

        total = len(detections)
        result = {}
        for cls, data in class_data.items():
            confs = data["confidences"]
            result[cls] = {
                "count":      data["count"],
                "percentage": round(data["count"] / total * 100, 1),
                "avg_conf":   round(sum(confs) / len(confs), 4),
                "max_conf":   round(max(confs), 4),
            }

        return {
            "total": total,
            "classes": dict(sorted(result.items(), key=lambda x: -x[1]["count"])),
        }

    def generate_full_report(
        self,
        detections: list[DetectionResult],
        summary: InspectionSummary,
        report_name: str = "inspection",
    ) -> dict[str, Path]:
        """
        Generate CSV + JSON reports and print console summary in one call.

        Returns:
            Dict with keys 'csv' and 'json' pointing to output files.
        """
        csv_path  = self.save_csv_report(detections, report_name)
        json_path = self.save_json_summary(summary, detections, report_name)
        self.print_summary(summary, detections)
        return {"csv": csv_path, "json": json_path}

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    def _compute_verdict(self, defect_count: int) -> str:
        """Return 'PASS' if defect count within threshold, else 'FAIL'."""
        if self.fail_threshold < 0:
            return "PASS"
        return "PASS" if defect_count <= self.fail_threshold else "FAIL"
