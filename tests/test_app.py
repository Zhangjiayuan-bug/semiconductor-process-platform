from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_app_mode_starts_without_exception():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    assert not app.exception


def test_process_template_mode_starts_without_exception():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=60)
    app.radio[0].set_value("工艺质量演示模板").run(timeout=30)
    assert not app.exception
