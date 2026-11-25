# Release notes

## 2026-11-12

- Package the study implementation, fixed configurations, and saved paper
  inputs without changes to the numerical methods or reported results.
- Include synthetic endpoint records, neural window metadata, benchmark
  records, and compact summaries of individual optimization starts.
- Add a verifier for record consistency and release checksums.
- Document the record schema, experiment workflows, recorded environments,
  and floating-point completion-window convention.
- Provide official CRCNS dataset links and required recording filenames;
  exclude raw neural recordings and extracted spike arrays.
- Remove private absolute machine paths from added record copies and record
  these metadata transformations in the provenance ledger.
- Require NumPy 2.0 or later for `numpy.trapezoid` and add article citation
  metadata.
