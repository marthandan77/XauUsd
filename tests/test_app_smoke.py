from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
