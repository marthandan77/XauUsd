from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_streamlit_app_starts_with_apply_and_scan_controls():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Apply Settings" in labels
    assert "Scan Market" in labels


def test_scan_button_runs_all_modules_without_exceptions():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    _button_by_label(app, "Scan Market").click().run()

    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert "Data source" in metric_labels
    assert "Final action" in metric_labels
    assert "Regime" in metric_labels
