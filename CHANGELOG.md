# Changelog

All notable changes to this project are documented in this file.

## [0.1.1] - 2026-07-28

### Fixed

- Install the importable code under the conventional
  `aep_load_forecasting` package instead of the ambiguous top-level `src`
  package.
- Verify the built wheel in CI by installing it and smoke-testing every
  packaged console command.

### Added

- Expose the installed package version through `aep_load_forecasting.__version__`
  and the `--version` option on all six `aep-*` commands.

## [0.1.0] - 2026-07-28

### Added

- Provide an installable, reproducible AEP load-forecasting workflow with
  leakage-safe baseline and XGBoost evaluation.
- Provide a deterministic synthetic-data demo, next-24-hour forecasting,
  Streamlit visualization, machine-readable run manifests, automated tests,
  and six command-line entry points.

[0.1.1]: https://github.com/amarakramali/load-forecasting-xgboost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/amarakramali/load-forecasting-xgboost/releases/tag/v0.1.0
