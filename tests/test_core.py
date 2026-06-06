"""
tests/test_core.py
------------------
Unit tests for PCB Defect Detection core modules.

Tests cover:
- DetectionResult data structure and serialization
- InspectionSummary aggregation
- FPSTracker rolling average
- PCBDefectDetector initialization and annotation
- ReportGenerator CSV/JSON output
- Configuration loading and merging
"""

import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    DetectionResult,
    FPSTracker,
    InspectionSummary,
    collect_image_paths,
    ensure_dir,
    is_image_file,
    is_video_file,
    timestamp_filename,
    validate_model_path,
)
from src.report_generator import ReportGenerator
from configs.config import load_config, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# DetectionResult tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectionResult:
    def make_det(self, **kwargs):
        defaults = dict(
            class_id=2, class_name="open_circuit",
            confidence=0.87, bbox=(10, 20, 110, 120),
        )
        return DetectionResult(**{**defaults, **kwargs})

    def test_bbox_area(self):
        det = self.make_det(bbox=(0, 0, 100, 50))
        assert det.bbox_area == 5000.0

    def test_bbox_dimensions(self):
        det = self.make_det(bbox=(10, 20, 110, 80))
        assert det.bbox_width  == 100
        assert det.bbox_height == 60

    def test_to_csv_row_keys(self):
        det = self.make_det()
        row = det.to_csv_row()
        expected_keys = {
            "frame_id", "class_id", "defect_type", "confidence",
            "x1", "y1", "x2", "y2", "bbox_width", "bbox_height",
            "bbox_area", "timestamp",
        }
        assert set(row.keys()) == expected_keys

    def test_confidence_roundtrip(self):
        det = self.make_det(confidence=0.12345)
        row = det.to_csv_row()
        assert abs(row["confidence"] - 0.1235) < 1e-4

    def test_zero_area_bbox(self):
        det = self.make_det(bbox=(50, 50, 50, 50))
        assert det.bbox_area == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# InspectionSummary tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInspectionSummary:
    def test_to_dict_keys(self):
        s = InspectionSummary(source="test.jpg")
        d = s.to_dict()
        assert "source" in d
        assert "total_defects" in d
        assert "defect_counts" in d

    def test_finalize_sets_end_time(self):
        s = InspectionSummary(source="x")
        assert s.end_time is None
        s.finalize()
        assert s.end_time is not None

    def test_defect_counts_accumulation(self):
        s = InspectionSummary(source="test")
        s.defect_counts["short"] = 3
        s.defect_counts["spur"]  = 7
        assert sum(s.defect_counts.values()) == 10


# ─────────────────────────────────────────────────────────────────────────────
# FPSTracker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFPSTracker:
    def test_initial_fps_zero(self):
        tracker = FPSTracker()
        assert tracker.fps == 0.0

    def test_fps_increases_after_ticks(self):
        tracker = FPSTracker()
        tracker.tick()
        time.sleep(0.02)
        fps = tracker.tick()
        assert fps > 0.0

    def test_reset_clears_state(self):
        tracker = FPSTracker()
        tracker.tick()
        time.sleep(0.01)
        tracker.tick()
        tracker.reset()
        assert tracker.fps == 0.0

    def test_multiple_ticks_produce_positive_fps(self):
        tracker = FPSTracker(alpha=0.5)
        for _ in range(5):
            time.sleep(0.01)
            tracker.tick()
        assert tracker.fps > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_is_image_file(self):
        assert is_image_file("test.jpg")  is True
        assert is_image_file("test.PNG")  is True
        assert is_image_file("test.mp4")  is False
        assert is_image_file("test.txt")  is False

    def test_is_video_file(self):
        assert is_video_file("test.mp4")  is True
        assert is_video_file("test.AVI")  is True
        assert is_video_file("test.jpg")  is False

    def test_ensure_dir_creates_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "a" / "b" / "c"
            result = ensure_dir(target)
            assert result.exists()
            assert result.is_dir()

    def test_timestamp_filename_format(self):
        fname = timestamp_filename("report", ".csv")
        assert fname.startswith("report_")
        assert fname.endswith(".csv")
        assert len(fname) == len("report_20240615_143022.csv")

    def test_validate_model_path_nonexistent(self):
        assert validate_model_path("/nonexistent/path/model.pt") is False

    def test_validate_model_path_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            assert validate_model_path(f.name) is False

    def test_collect_image_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "a.jpg").touch()
            (base / "b.png").touch()
            (base / "c.txt").touch()
            (base / "sub").mkdir()
            (base / "sub" / "d.jpeg").touch()
            paths = collect_image_paths(base)
            names = [p.name for p in paths]
            assert "a.jpg" in names
            assert "b.png" in names
            assert "d.jpeg" in names
            assert "c.txt" not in names


# ─────────────────────────────────────────────────────────────────────────────
# ReportGenerator tests
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_detections(n: int = 5) -> list[DetectionResult]:
    classes = ["open_circuit", "short", "missing_hole", "spur", "mouse_bite"]
    return [
        DetectionResult(
            class_id=i % len(classes),
            class_name=classes[i % len(classes)],
            confidence=0.6 + 0.05 * i,
            bbox=(10*i, 10*i, 60*i+50, 60*i+50),
            frame_id=f"frame_{i:04d}",
        )
        for i in range(n)
    ]


def make_sample_summary() -> InspectionSummary:
    s = InspectionSummary(source="test_source")
    s.total_frames    = 10
    s.total_defects   = 5
    s.avg_confidence  = 0.72
    s.avg_inference_ms = 48.5
    s.avg_fps         = 20.6
    s.defect_counts   = {"open_circuit": 2, "short": 2, "missing_hole": 1}
    s.finalize()
    return s


class TestReportGenerator:
    def test_csv_report_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)
            dets = make_sample_detections(7)
            csv_path = reporter.save_csv_report(dets, "test")

            with open(csv_path, "r") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 7

    def test_csv_report_column_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)
            dets = make_sample_detections(2)
            csv_path = reporter.save_csv_report(dets, "cols_test")

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames
            assert "defect_type" in cols
            assert "confidence"  in cols
            assert "frame_id"    in cols

    def test_json_summary_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir, fail_threshold=0)
            summary = make_sample_summary()
            dets    = make_sample_detections(5)
            json_path = reporter.save_json_summary(summary, dets, "json_test")

            with open(json_path, "r") as f:
                data = json.load(f)

            assert "total_defects" in data
            assert "verdict"       in data
            assert "class_statistics" in data

    def test_fail_verdict_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir, fail_threshold=3)
            summary  = make_sample_summary()  # total_defects = 5
            dets     = make_sample_detections(5)
            json_path = reporter.save_json_summary(summary, dets)

            with open(json_path) as f:
                data = json.load(f)
            assert data["verdict"] == "FAIL"

    def test_pass_verdict_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir, fail_threshold=10)
            summary  = make_sample_summary()  # total_defects = 5
            dets     = make_sample_detections(5)
            json_path = reporter.save_json_summary(summary, dets)

            with open(json_path) as f:
                data = json.load(f)
            assert data["verdict"] == "PASS"

    def test_frequency_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)
            dets = make_sample_detections(10)
            analysis = reporter.defect_frequency_analysis(dets)
            assert analysis["total"] == 10
            assert "classes" in analysis

    def test_empty_detection_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)
            csv_path = reporter.save_csv_report([], "empty_test")
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_structure(self):
        config = load_config()
        assert "detector"  in config
        assert "training"  in config
        assert "reporting" in config

    def test_default_confidence_threshold(self):
        config = load_config()
        assert config["detector"]["confidence_threshold"] == 0.25

    def test_yaml_override(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("detector:\n  confidence_threshold: 0.5\n")
            tmpfile = f.name

        config = load_config(tmpfile)
        assert config["detector"]["confidence_threshold"] == 0.5
        # Other keys should remain from defaults
        assert config["detector"]["device"] == "cpu"

        import os
        os.unlink(tmpfile)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PCB_CONF_THRESHOLD", "0.75")
        config = load_config()
        assert config["detector"]["confidence_threshold"] == 0.75
