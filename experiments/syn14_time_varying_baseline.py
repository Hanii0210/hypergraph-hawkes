"""Prespecified baseline-confounding controls; constant paths reuse Syn13."""
EXPERIMENT = "syn14_time_varying_baseline"


def scenarios():
    out = []
    for amplitude in (0.3, 0.6):
        for label, weight in (("null", 0.0), ("strong", 0.4)):
            out.append({"experiment": EXPERIMENT, "experiment_id": 14,
                        "scenario_id": len(out)+1,
                        "name": f"sinusoidal_a{amplitude:g}_{label}", "T": 1000,
                        "triple_weight": weight, "background": "sinusoidal",
                        "amplitude": amplitude, "formal_repetitions": 100})
    for label, weight in (("null", 0.0), ("moderate", 0.2)):
        out.append({"experiment": EXPERIMENT, "experiment_id": 14,
                    "scenario_id": len(out)+1, "name": f"linear_{label}", "T": 1000,
                    "triple_weight": weight, "background": "linear",
                    "amplitude": 0.6, "formal_repetitions": 50})
    for spec in out:
        spec["models"] = ["C-M1", "C-M2", "V-M1", "V-M2"]
        spec["warm_from"] = {"C-M2": "C-M1", "V-M1": "C-M1", "V-M2": "V-M1"}
    return out


def include_constant_controls(cases):
    """Add only the two variable-background fits to the same stored Syn13 paths."""
    for spec in cases:
        if spec["experiment_id"] == 13 and spec["T"] == 1000 and spec["triple_weight"] in (0.0, 0.4):
            spec["models"] += ["V-M1", "V-M2"]
            spec["warm_from"].update({"V-M1": "C-M1", "V-M2": "V-M1"})
            spec["shared_with"] = EXPERIMENT


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.synthetic_pipeline import main
    raise SystemExit(main(experiment="syn14"))
