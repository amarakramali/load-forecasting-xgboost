from importlib.metadata import version

import pytest

from aep_load_forecasting import __version__
from aep_load_forecasting.baseline_eval import parse_args as parse_baseline_args
from aep_load_forecasting.demo_pipeline import parse_args as parse_demo_args
from aep_load_forecasting.forecast_24h import parse_args as parse_forecast_args
from aep_load_forecasting.make_features import parse_args as parse_feature_args
from aep_load_forecasting.sample_data import parse_args as parse_sample_args
from aep_load_forecasting.xgb_eval import parse_args as parse_xgb_args

CLI_PARSERS = (
    parse_demo_args,
    parse_sample_args,
    parse_feature_args,
    parse_baseline_args,
    parse_xgb_args,
    parse_forecast_args,
)


def test_package_version_matches_distribution_metadata() -> None:
    assert __version__ == version("aep-load-forecasting")


@pytest.mark.parametrize("parse_args", CLI_PARSERS)
def test_cli_version_option(parse_args, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == (
        f"aep-load-forecasting {__version__}"
    )
