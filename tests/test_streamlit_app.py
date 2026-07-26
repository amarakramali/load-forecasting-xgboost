from streamlit.testing.v1 import AppTest


def test_demo_renders_bundled_forecast_without_errors() -> None:
    app = AppTest.from_file("streamlit_app.py").run(timeout=10)

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Max Forecast (MW)",
        "Min Forecast (MW)",
        "Ø Forecast (MW)",
    ]
