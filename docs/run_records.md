# Saved records

The archive supports inspection of the reported runs without loading raw data
or rerunning the experiments. `paper/data` contains the inputs used by the
figure and table scripts. `records` adds study metadata, individual fit
endpoints, timing repetitions, and compact summaries of optimization starts.

## Contents

| Location | Contents |
| --- | --- |
| `records/synthetic/study/formal` | Formal configuration, plan, environment, readiness, execution, and aggregate metadata |
| `records/synthetic/syn13_overlapping_hyperedges/formal` | 600 dataset records, 2200 model-fit records, and 600 comparison records |
| `records/synthetic/syn14_time_varying_baseline/formal` | 500 dataset records, 2000 model-fit records, and 500 comparison records |
| `records/benchmark` | 340 JSON records, including the 270 measured timing repetitions and excluded warm-ups |
| `records/realdata` | Configuration, first-window/full-study metadata, three recording definitions, 15 window dataset records, and 15 candidate records |
| `paper/data/realdata_fits.json` | The 15 neural window results, containing all 135 selected model fits |
| `records/derived/synthetic_starts.jsonl` | 12600 current synthetic start summaries |
| `records/derived/realdata_starts.jsonl` | 405 current neural start summaries |
| `records/derived/realdata_archived_starts.jsonl` | Nine pre-continuation neural start summaries |

The 200 constant-background control paths are reused within the 1100-path
synthetic design; they are not additional datasets. The nine archived neural
starts document earlier endpoints and do not add to the 405 planned starts.
There are 6793 copied original JSON records in total; the three derived JSONL
files are additional, separately identified summaries.

## Original endpoint records

Each synthetic dataset is located at
`records/synthetic/<experiment>/formal/<scenario>/rep_<NNN>/` and contains:

- `dataset.json`: task, seeds, generating parameters, intervals, event counts,
  and generation diagnostics;
- `models/<model>/fit.json`: selected start, fitted parameters, training/test
  scores, qualification diagnostics, and estimation errors;
- `comparisons.json`: paired test-score differences for the prespecified
  model comparisons.

The source archive and member path for each endpoint appear in
`records/provenance.jsonl`. For records that refer to a study machine, the
same entry lists the fields where a local path was replaced.

Paths such as `start_1.json` inside `fit.json` and `code_snapshot` paths in
environment metadata refer to the original experiment directories. Complete
start files and repeated source snapshots are not included in this compact
archive; these fields preserve run provenance, not a list of dependencies
present under `records`. Use the matching entries in `records/derived` for
the start summaries. The same distinction applies to source-archive references.

## Derived start summaries

Each JSONL file contains one independent JSON object per line. A row identifies
its dataset, model, start, and seed and retains the recorded EM and final
qualification statuses, iteration/sweep counts, training objective and score,
KKT residual, selected-start flag, and summaries of the saved histories.
Seeds are decimal strings so that readers do not truncate 64-bit integers in
software that represents all JSON numbers as floating-point values.

History summaries include lengths, minimum objective increments, final
relative increments, and the EM-to-coordinate objective connection. They are
calculated from the saved records, not from newly fitted models. They do not
contain every objective value, every parameter iterate, or the full parameter
arrays of unselected starts.

The `original_record` object gives the source archive and member path. Its `included_in_package` field is false for full start records
represented only by a derived row. See [field definitions](record_schema.md).

## Verification

Run `python scripts/verify_saved_records.py` from the project root. The verifier
checks record counts, plan/seed correspondence, start selection, recorded
stopping criteria, paper-input correspondence, and benchmark aggregates.
It does not read raw recordings or refit models. Checking a saved KKT residual
against its threshold is not a new gradient calculation from the event data.

## Excluded working files

Complete synthetic/neural optimization histories, synthetic event arrays,
raw CRCNS recordings, extracted spike arrays, empty console logs, and redundant
source snapshots are not included. Benchmark records retain the diagnostics
originally stored for each timing repetition; they do not record every parameter
iterate.

New runs create a different, resumable tree under `experiments/results`.
Do not use the compact `records` tree as a resume directory or as input to
`scripts/export_paper_results.py`. Rebuilding the reported figures and tables
uses `paper/data` directly.
