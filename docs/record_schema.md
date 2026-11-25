# Compact record schema

All paths in this document are relative to the archive root. Original endpoint
records keep their study schemas. The derived JSONL files described below use
schema version 1 and contain one JSON object per line.

## Files and row counts

| File | Rows | Role |
| --- | ---: | --- |
| `records/derived/synthetic_starts.jsonl` | 12600 | Current synthetic starts |
| `records/derived/realdata_starts.jsonl` | 405 | Current neural starts |
| `records/derived/realdata_archived_starts.jsonl` | 9 | Neural starts before continuation |

The key `(dataset_id, model, start_index)` identifies a current start. An
archived start can have the same key as its continued current record; the
files and `record_role` distinguish them. Archived rows are not additional
planned fits or starts.

## Identity and selection fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `record_kind` | `derived_start_endpoint_summary` |
| `study` | `synthetic` or `realdata` |
| `record_role` | `current` or `archived_precontinuation` |
| `dataset_id` | Synthetic task identifier, or neural task path including the window |
| `recording_id` | Neural recording identifier; null for synthetic rows |
| `model` | Model label in the experiment plan |
| `start_index` | One-based start number, 1 through 3 |
| `seed` | Original uint64 seed as a decimal string |
| `selected_for_current_fit` | True for the selected current start; always false for archived rows |
| `original_package_version` | Version identifier recorded in the source start file |

String encoding preserves seed values in JSON readers that otherwise convert
integers to binary64. Use `int(row["seed"])` in Python for exact integer access.
Selection uses the largest qualified training objective; test scores do not
select a start.

## Endpoint and stopping fields

| Field | Meaning |
| --- | --- |
| `em_status`, `em_converged` | Qualification and convergence flag at the EM endpoint |
| `final_status`, `final_converged` | Qualification and convergence after the recorded coordinate work |
| `em_iterations` | Number of saved EM iterations |
| `coordinate_sweeps` | Recorded coordinate sweep count, including continuation where applicable |
| `training_objective` | Final objective recorded by the optimizer |
| `training_loglik` | Training log likelihood from the saved score components |
| `kkt_residual` | Final recorded scaled KKT residual |
| `final_relative_increment` | Last recorded coordinate relative objective increment |
| `last_five_relative_increments` | The final five recorded coordinate relative increments |

EM qualification and final qualification are distinct: coordinate refinement
can qualify a start whose EM stage reached its iteration budget. The configured
final criteria include a KKT residual at most `1e-4` and five consecutive
relative increments below `1e-7`. Original objectives and component-based
log likelihoods can differ in their last floating-point digits because their
summation order differs.

## History summary fields

The `history` object contains the following values calculated from saved
history arrays. Those complete arrays are not included in the compact rows.

| Field within `history` | Meaning |
| --- | --- |
| `em_length` | Length of the saved EM objective array |
| `coordinate_length` | Length of the saved coordinate objective array |
| `combined_length` | Length of the combined objective history |
| `combined_history_source` | `top_level_history` or `em_result_plus_refinement`, identifying how the combined history was obtained |
| `em_min_adjacent_increment` | Minimum difference between adjacent saved EM objectives; excludes the unsaved objective before the first iteration |
| `coordinate_min_increment_including_initial` | Minimum coordinate objective increment, including the change from its initial objective |
| `em_to_coordinate_objective_gap` | Coordinate initial objective minus the saved EM endpoint objective |
| `objective_minus_training_loglik` | Saved final optimizer objective minus saved training log likelihood |
| `final_objective_history_gap` | Saved final objective minus the last combined-history entry |

These values summarize recorded trajectories. They do not constitute an
independent recomputation of event-level objectives or gradients.

## Source records

Each derived row identifies its full start record by `original_record.source_archive`
and `original_record.path`. The field `included_in_package` is false for these
full start records.

`records/provenance.jsonl` maps each packaged endpoint or metadata file to its
source archive and member. For records with local machine paths, the entry also
lists the affected JSON fields and replacement paths.

`records/manifest.json` lists the record files, row counts, source archives,
and derived summaries.
