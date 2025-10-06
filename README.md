# MEF Dispatch 2024 – Module Overview

This repository now separates the Track C dispatcher into focused modules so that
core logic, data loading and reporting remain maintainable for the Master thesis.

## Module Split

- `modules/io_utils.py` – CSV/time-series parsing, fleet and flow loaders, time
  handling utilities (`parse_ts`, `force_hourly`, etc.). Resampling now uses
  lowercase `'h'` to avoid pandas deprecation warnings.
- `modules/mustrun.py` – Must-run heuristics and cost-based profiles
  (`compute_fossil_min_profiles_*`, low-price profiles, efficiency constants).
- `modules/plots.py` – All plotting helpers and BBH colour scheme, enhanced
  diagnostic charts, validation visualisations and reservoir QA plots.
- `modules/validation.py` – Correlation filtering, baseline validation pipeline
  (`validate_run`, `_filtered_corr_and_offenders`, multi-stage enhanced
  validation) and writers for validation reports.

`mef_dispatch_2024_Final_Version.py` now imports from these modules and keeps the
execution flow, CLI parsing and hydro logic in one place.

## Usage Notes

- Existing CLI commands continue to work. Ensure `PYTHONPATH` includes the
  repository root so the `modules` package can be resolved.
- Validation outputs and plots are still written below `out/<run>/analysis/`.
- The new structure should be referenced in the thesis methodology chapter under
  software implementation/documentation.
