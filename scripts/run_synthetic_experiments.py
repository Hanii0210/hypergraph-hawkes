"""Run the prespecified Syn13/Syn14 pilot and package all records."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.synthetic_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
