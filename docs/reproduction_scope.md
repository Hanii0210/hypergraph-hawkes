# Reproduction scope

## Study implementation

The numerical implementation, fixed scientific configurations, seed rules,
convergence criteria, and saved figure/table inputs are those used for the
reported study. The record verifier and documentation do not refit models or
replace numerical results. `docs/code_manifest.json` retains the study code
identifier `HTH_JSCS_v2`; release file checksums are in `FILELIST.sha256`.

The studies use free nonnegative group coefficients. The legacy CP tensor API
remains in the shared code but is not the fitted model used in the reported
experiments.

## Completion-window arithmetic

Completion windows have left endpoint `c - delta`. The implementation uses
ordinary binary64 subtraction and comparison. When acquisition timestamps are
exactly `delta` apart in decimal units, rounding can affect endpoint inclusion;
for example, `0.20 - 0.05` is slightly greater than `0.15` in binary64.

An outward-rounded alternative changes completion counts in the ret-1
window-04 training interval and the PVC-3 window-02 test interval. It can also
change likelihoods and KKT residuals evaluated at saved parameters. This archive
retains the arithmetic used to produce the reported fits. Results under another
boundary convention require a separately identified implementation and fits.

## Environments and workflow behavior

The run metadata record Python 3.14.7, NumPy 2.5.2, and SciPy 1.18.0 on Windows 11.
`requirements.txt` provides dependency lower bounds, not a locked copy of that
environment. The NumPy lower bound is 2.0 because the summary code uses
`numpy.trapezoid`. Numerical endpoints and elapsed times can depend on the
platform and numerical libraries.

The retained runners have the following behavior:

- Scientific completion is determined from planned counts and qualification
  fields in `summary.json`, not the top-level invocation status alone.
- Neural continuation is supported at the five-window stage. An unqualified
  first-window run can prevent that stage from passing its readiness check.
- New experiments write resumable records under `experiments/results`.
  The compact publication records under `records` are not a resumable copy
  of those working directories.

## Release checks

Release checks used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib
3.10.8 on Linux. All 14 implementation-check groups passed. The table script
generated nine LaTeX tables; the figure script generated four main figures
and three supplementary figures in PDF and PNG formats. Times New Roman was
unavailable in that environment, so those local rendering checks used a
fallback font. The distributed plotting code and paper inputs are unchanged.

The formal plan checks returned 1100 synthetic paths, 4200 fits, and 12600
starts, and 15 neural windows, 135 fits, and 405 starts. The saved-record
verifier passed against the packaged records. These checks do not rerun the
full experiments. Details are recorded in `docs/release_validation.json`.

## What the saved records establish

`paper/data` provides the full-precision numerical inputs for the published
figures and tables. `records` adds endpoint records, study plans, environment
metadata, and derived start summaries. The verifier checks consistency among
these files, including start selection and recorded qualification criteria.

Complete synthetic and neural optimization histories are not included. Their
derived summaries retain history lengths, final increments, minimum objective
increments, and endpoint diagnostics. These summaries are not substitutes for
event-level likelihood or gradient calculations. See [record schema](run_records.md).

Raw CRCNS recordings and extracted spike arrays are not redistributed. Their
official sources and exact required files are listed in [data access](data_access.md).
