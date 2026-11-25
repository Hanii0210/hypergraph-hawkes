"""Prespecified six-cell design for overlapping three-node completion effects."""
EXPERIMENT = "syn13_overlapping_hyperedges"


def scenarios():
    out = []
    for horizon in (500, 1000):
        for label, weight in (("null", 0.0), ("moderate", 0.2), ("strong", 0.4)):
            out.append({"experiment": EXPERIMENT, "experiment_id": 13,
                        "scenario_id": len(out)+1, "name": f"{label}_T{horizon}",
                        "T": horizon, "triple_weight": weight,
                        "background": "constant", "amplitude": 0.0,
                        "models": ["M0", "C-M1", "C-M2"],
                        "warm_from": {"C-M1": "M0", "C-M2": "C-M1"},
                        "formal_repetitions": 100})
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.synthetic_pipeline import main
    raise SystemExit(main(experiment="syn13"))
