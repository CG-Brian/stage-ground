import pandas as pd

from stageground.evaluation.plots import (
    plot_accuracy_vs_unsupported,
    plot_coverage_vs_supported_accuracy,
    plot_error_breakdown,
)
from stageground.evaluation.records import build_record

TABLE = pd.DataFrame([
    {"arm": "C_constrained", "n_evaluable": 10, "accuracy": 0.9, "supported_accuracy": 0.2,
     "unsupported_rate": 0.6, "abstention_rate": 0.01, "coverage": 0.99},
    {"arm": "D_grounded", "n_evaluable": 10, "accuracy": 0.7, "supported_accuracy": 0.65,
     "unsupported_rate": 0.0, "abstention_rate": 0.3, "coverage": 0.7},
])


def test_plot_accuracy_vs_unsupported_writes_file(tmp_path):
    outpath = tmp_path / "fig1.png"
    plot_accuracy_vs_unsupported(TABLE, outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_plot_coverage_vs_supported_accuracy_writes_file(tmp_path):
    outpath = tmp_path / "fig2.png"
    plot_coverage_vs_supported_accuracy(TABLE, outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_plot_error_breakdown_writes_file(tmp_path):
    records = [
        build_record(
            case_id="c1", arm="C_constrained", target="M", ground_truth="M0",
            raw_output={"value": "M1", "evidence": None, "confidence": "low", "reason": None},
            report_text="no M descriptor here",
        ),
        build_record(
            case_id="c2", arm="D_grounded", target="M", ground_truth="M0",
            raw_output={"value": "unknown", "evidence": None, "confidence": "low",
                        "reason": "no basis"},
            report_text="no M descriptor here",
        ),
    ]
    outpath = tmp_path / "fig3.png"
    plot_error_breakdown(records, outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_plot_error_breakdown_filters_by_target(tmp_path):
    records = [
        build_record(
            case_id="c1", arm="C_constrained", target="M", ground_truth="M0",
            raw_output={"value": "M1", "evidence": None, "confidence": "low", "reason": None},
            report_text="no M descriptor here",
        ),
    ]
    outpath = tmp_path / "fig3_m.png"
    plot_error_breakdown(records, outpath, target="M")
    assert outpath.exists()
    assert outpath.stat().st_size > 0
