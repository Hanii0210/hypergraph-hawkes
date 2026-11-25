# Computational benchmark

`configs/syn07_scalability.json` specifies five event-count settings and four
node-count settings. Each uses one generated path and one fixed initial
state. Feature construction, one cached EM update, and a complete fit are
timed separately, with one excluded warm-up and ten measurements per phase.
Full fits include coordinate refinement and the configured stopping checks.
Simulation, imports, loading, and serialization are outside the measured
phases.

`records/benchmark` contains 340 JSON records, including all 270 measured
repetitions, the excluded warm-ups, and the configuration, environment, and
summary metadata. `paper/data/timing_summary.json` and
`paper/data/timing_analysis.json` provide the figure/table inputs.

The summary reports medians, quartiles, qualification, and event-count slopes.
Slopes describe the measured grid and implementation. Timing repetitions do
not represent independent event paths. Peak memory includes the worker
process and is recorded only where the implemented operating-system API is
available.

A new benchmark run writes to
`experiments/results/synthetic/syn07_scalability`. Its per-setting directories
contain `features`, `iteration`, and `fit` records. Each measured repetition
retains endpoint diagnostics and trajectory summaries, not every parameter
iterate. Hardware timings should be interpreted for the recorded machine.
