"""Create supplementary tables from saved results; no simulation or fitting.

All CSV values retain Python's round-trip float precision. LaTeX rounding is
for display only. Sources are local, immutable copies in the data directory.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


SYN = load("synthetic_replicates.json")
REAL = load("realdata_analysis.json")
FITS = load("realdata_fits.json")
TIMING = load("timing_analysis.json")
GROUPS = defaultdict(dict)
for row in SYN:
    key = (row["scenario"], row["model"])
    assert row["dataset_id"] not in GROUPS[key]
    GROUPS[key][row["dataset_id"]] = row

SCENARIOS = [f"{signal}_T{T}" for T in [500, 1000]
             for signal in ["null", "moderate", "strong"]]
SCENARIOS += [f"sinusoidal_a{a}_{signal}" for a in [0.3, 0.6]
              for signal in ["null", "strong"]]
SCENARIOS += ["linear_null", "linear_moderate"]
MODELS = ["M0", "C-M1", "C-M2", "V-M1", "V-M2"]
TRIPLES = [(1, 2, 3), (3, 4, 5), (1, 2, 4), (1, 3, 4), (2, 3, 5), (3, 5, 6)]
RECORD_ORDER = {"ret-1": 0, "PVC-3": 1, "PVC-11": 2}


def stats(values):
    x = np.asarray(values, dtype=float)
    assert len(x) > 1 and np.isfinite(x).all()
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def rows_for(scenario, model):
    return list(GROUPS[(scenario, model)].values())


def paired(scenario, left, right):
    a, b = GROUPS[(scenario, left)], GROUPS[(scenario, right)]
    assert a.keys() == b.keys() and a
    return np.array([r["test_loglik"] - b[k]["test_loglik"] for k, r in a.items()])


def scenario_info(scenario):
    row = next(r for r in SYN if r["scenario"] == scenario)
    background = ("Periodic " + scenario.split("_a")[1].split("_")[0]
                  if scenario.startswith("sinusoidal") else row["background"].capitalize())
    return background, row["T"], row["truth"]


def scenario_title(scenario):
    background, T, alpha = scenario_info(scenario)
    return rf"{background}, $T={T}$, $\alpha={alpha:g}$"


def number(value, digits=4):
    if value is None:
        return "---"
    rendered = f"{value:.{digits}f}"
    # Avoid a negative-zero glyph after display rounding; CSV keeps the value.
    return rendered[1:] if rendered.startswith("-") and float(rendered) == 0 else rendered


def edge(nodes):
    return r"$\{" + ",".join(map(str, nodes)) + r"\}$"


def csv_write(name, rows):
    with (DATA / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def longtable(name, caption, label, columns, headers, body, note=""):
    ncol = len(headers)
    if name.startswith("synthetic_"):
        # A scenario and its model/coefficient rows remain together on one page.
        body = [line + "*" if line.endswith(r"\\") and i + 1 < len(body)
                and not body[i + 1].startswith(r"\addlinespace") else line
                for i, line in enumerate(body)]
    header = " & ".join(headers) + r" \\\midrule"
    result = [r"\begingroup", r"\fontsize{8}{10}\selectfont",
              r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.05}",
              r"\begin{longtable}{@{}" + columns + r"@{}}",
              r"\caption{" + caption + r"}\label{" + label + r"}\\",
              r"\toprule", header, r"\endfirsthead",
              r"\multicolumn{" + str(ncol) + r"}{l}{\tablename\ \thetable\ (continued)}\\",
              r"\toprule", header, r"\endhead", r"\midrule",
              r"\multicolumn{" + str(ncol) + r"}{r}{Continued on next page}\\",
              r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    result.extend(body)
    result.extend([r"\end{longtable}"])
    if note:
        result.extend([r"\noindent " + note + r"\par"])
    result.extend([r"\endgroup", ""])
    (TABLES / (name + ".tex")).write_text("\n".join(result), encoding="utf-8")


def group(title, ncol):
    return r"\multicolumn{" + str(ncol) + r"}{l}{\textit{" + title + r"}}\\*"


def latex_row(cells):
    return " & ".join(map(str, cells)) + r" \\"


def synthetic_prediction():
    records, body = [], []
    for scenario in SCENARIOS:
        comparisons = [(m, m[:-1] + "1") for m in ["C-M2", "V-M2"]
                       if (scenario, m) in GROUPS]
        if (scenario, "V-M2") in GROUPS:
            comparisons.append(("V-M2", "C-M2"))
        body.append(group(scenario_title(scenario), 6))
        for left, right in comparisons:
            values = paired(scenario, left, right)
            mean, mcse = stats(values)
            source = rows_for(scenario, left)
            per_time = values / np.array([r["test_duration"] for r in source])
            time_mean, time_mcse = stats(per_time)
            background, T, alpha = scenario_info(scenario)
            records.append(dict(scenario=scenario, background=background, T=T, alpha=alpha,
                                n=len(values), left=left, right=right, mean=mean, mcse=mcse,
                                mean_per_time=time_mean, mcse_per_time=time_mcse))
            body.append(latex_row([f"{left} $-$ {right}", len(values), number(mean),
                                   number(mcse), number(time_mean, 6), number(time_mcse, 6)]))
        body.append(r"\addlinespace[3pt]")
    assert len(records) == 28
    csv_write("synthetic_prediction", records)
    longtable("synthetic_prediction", "Paired full test log-likelihood differences in all simulation settings.",
              "tab:syn-prediction", "lrrrrr", ["Comparison", "$n$", "Mean", "MCSE", "Mean/time", "MCSE/time"], body,
              r"The first two comparisons in each applicable setting add the six triple candidates to the two-node completion model. The V-M2 minus C-M2 comparison changes the background of the complete model. Test duration is $T/3$. The constant-background V fits reuse the corresponding C paths. MCSE is computed across independent paths; $\alpha$ is the weight of each designated triple.")
    return len(records)


def synthetic_recovery():
    records, body = [], []
    for scenario in SCENARIOS:
        for model in ["C-M2", "V-M2"]:
            if (scenario, model) not in GROUPS:
                continue
            data = rows_for(scenario, model)
            body.append(group(scenario_title(scenario) + f", {model}, $n={len(data)}$", 6))
            for index, nodes in enumerate(TRIPLES):
                truth = data[0]["truth"] if index < 2 else 0.0
                values = np.array([(r["triple_estimates"] + r["decoy_estimates"])[index] for r in data])
                mean, mcse = stats(values)
                errors = values - truth
                bias, rmse = float(errors.mean()), float(np.sqrt(np.mean(errors**2)))
                records.append(dict(scenario=scenario, model=model, n=len(data),
                                    edge="-".join(map(str, nodes)), truth=truth, mean=mean,
                                    mcse_mean=mcse, bias=bias, rmse=rmse))
                body.append(latex_row([edge(nodes), number(truth, 1), number(mean), number(mcse),
                                       number(bias), number(rmse)]))
            body.append(r"\addlinespace[3pt]")
    assert len(records) == 120
    csv_write("synthetic_recovery", records)
    longtable("synthetic_recovery", "Recovery of all six triple coefficients in every M2 setting.",
              "tab:syn-recovery", "lrrrrr", ["Triple", "Truth", "Mean", "MCSE", "Bias", "RMSE"], body,
              r"Nodes use the one-based numbering in the main text. Each row summarises one coefficient across independent paths. MCSE describes the mean estimate, and also the bias because the truth is fixed. RMSE is the square root of the mean squared estimation error. The four additional triples have zero generating weight.")
    return len(records)


def synthetic_nuisance():
    records, body = [], []
    quantities = ["pairwise_mse", "baseline_train_mse", "baseline_test_mse", "pair_completion_mse"]
    for scenario in SCENARIOS:
        body.append(group(scenario_title(scenario), 6))
        for model in MODELS:
            if (scenario, model) not in GROUPS:
                continue
            data = rows_for(scenario, model)
            record = dict(scenario=scenario, model=model, n=len(data))
            cells = [model, len(data)]
            for quantity in quantities:
                values = [r[quantity] for r in data]
                if all(v is None for v in values):
                    mean = mcse = None
                    cells.append("---")
                else:
                    assert all(v is not None for v in values)
                    mean, mcse = stats(values)
                    cells.append(f"{mean * 1000:.3f} ({mcse * 1000:.3f})")
                record[quantity + "_mean"] = mean
                record[quantity + "_mcse"] = mcse
            records.append(record)
            body.append(latex_row(cells))
        body.append(r"\addlinespace[3pt]")
    assert len(records) == 46
    csv_write("synthetic_nuisance", records)
    longtable("synthetic_nuisance", r"Mean squared errors with MCSEs in parentheses. All displayed errors and MCSEs are multiplied by $10^3$.",
              "tab:syn-nuisance", "lrrrrr", ["Model", "$n$", "Ordinary", "Background: train", "Background: test", "Two-node"], body,
              r"The ordinary error averages squared errors over the 30 allowed off-diagonal coefficients. The two-node error averages over all 15 two-node completion coefficients; it is undefined for M0. For interval $I$, the background error is $(6|I|)^{-1}\sum_{n=1}^6\int_I\{\widehat\mu_n(t)-\mu_n(t)\}^2\,\mathrm dt$. Saved background errors use a 1001-point trapezoidal grid; a 2001-point refinement and an independent analytic-integral check confirmed numerical agreement. MCSEs are calculated across paths after these within-path averages. The CSV file retains unscaled values.")
    return len(records)


def real_windows():
    return sorted(REAL["windows"], key=lambda w: (RECORD_ORDER[w["short_name"]], w["window_index"]))


def real_fit_map():
    return {(r["task"]["dataset_id"], r["task"]["window_index"]): r for r in FITS}


def neural_scores():
    records, body, fit_map = [], [], real_fit_map()
    names = ["C-P", "C-H2-top3", "V-P", "V-H2-top3"]
    for w in real_windows():
        model = fit_map[(w["dataset"], w["window_index"])]["models"]
        scores = [model[name]["test_score"]["loglik"] for name in names]
        n = model[names[0]]["test_score"]["n_events"]
        duration = model[names[0]]["test_score"]["duration"]
        assert all(model[name]["test_score"]["n_events"] == n for name in names)
        dc, dv = scores[1] - scores[0], scores[3] - scores[2]
        assert abs(dc - w["primary_comparison"]["constant"]["delta_loglik"]) < 2e-9
        assert abs(dv - w["primary_comparison"]["linear"]["delta_loglik"]) < 2e-9
        record = dict(record=w["short_name"], window=w["window_index"], test_events=n,
                      test_start=w["intervals"]["test"]["start"], test_end=w["intervals"]["test"]["end"],
                      test_duration=duration, C_P=scores[0], C_H=scores[1], V_P=scores[2], V_H=scores[3],
                      delta_C=dc, delta_V=dv, V_H_minus_C_H=scores[3] - scores[1],
                      delta_C_per_time=dc / duration, delta_V_per_time=dv / duration,
                      C_all_group_weights_zero=all(v == 0 for v in model[names[1]]["parameters"]["alpha_hyper"]),
                      V_all_group_weights_zero=all(v == 0 for v in model[names[3]]["parameters"]["alpha_hyper"]))
        records.append(record)
        body.append(latex_row([w["short_name"], w["window_index"], n] + [number(v, 3) for v in scores] + [number(dc, 3), number(dv, 3)]))
    assert len(records) == 15
    csv_write("neural_scores", records)
    longtable("neural_scores", "Full unpenalised test log-likelihoods and completion increments for all neural windows.",
              "tab:neural-scores", "lrrrrrrrr", ["Record", "Window", "$m_{test}$", "C-P", "C-H", "V-P", "V-H", r"$\Delta_C$", r"$\Delta_V$"], body,
              r"H uses the top-three candidates. $\Delta_C=\ell_{\mathrm{C-H}}-\ell_{\mathrm{C-P}}$, and $\Delta_V=\ell_{\mathrm{V-H}}-\ell_{\mathrm{V-P}}$. Test durations are $20/3$ seconds for ret-1 and PVC-3, and $40/3$ seconds for PVC-11. In ret-1 window 2 all added group weights are zero under both backgrounds; its small score differences reflect separate approximate optimisation endpoints. Differences are calculated before display rounding. Windows from one record are ordered dependent observations.")
    return len(records)


def neural_candidates():
    records, body, fit_map = [], [], real_fit_map()
    for w in real_windows():
        edges = [[v + 1 for v in e] for e in w["candidate_prefixes"]["3"]["edges"]]
        assert len(edges) == 3
        counts = [w["intervals"][k]["n_events"] for k in ["selection", "estimation", "test"]]
        record = dict(record=w["short_name"], window=w["window_index"],
                      eligible_candidates=fit_map[(w["dataset"], w["window_index"])]["candidates"]["n_eligible"],
                      candidate_events=counts[0], estimation_events=counts[1], test_events=counts[2],
                      first="-".join(map(str, edges[0])), second="-".join(map(str, edges[1])), third="-".join(map(str, edges[2])))
        for part in ["selection", "estimation", "test"]:
            record[part + "_start"] = w["intervals"][part]["start"]
            record[part + "_end"] = w["intervals"][part]["end"]
        records.append(record)
        body.append(latex_row([w["short_name"], w["window_index"]] + counts + [edge(e) for e in edges]))
    assert len(records) == 15
    csv_write("neural_candidates", records)
    longtable("neural_candidates", "Ordered completion candidates and event counts in the three analysis blocks.",
              "tab:neural-candidates", "lrrrrlll", ["Record", "Window", "Candidate", "Estimation", "Test", "First pair", "Second pair", "Third pair"], body,
              r"Candidate, Estimation and Test are event counts. Pair labels are one-based local node numbers from Table~\ref{tab:neural-mapping}. Ranking uses only the first block; the top-one and top-two analyses use the corresponding prefixes. Every ret-1 window has 21 eligible pairs, and every PVC-3 or PVC-11 window has 45. All retained candidates have positive estimation exposure. Block endpoints are supplied without rounding in the accompanying CSV.")
    return len(records)


def neural_mapping():
    records, body = [], []
    for name in RECORD_ORDER:
        windows = [w for w in real_windows() if w["short_name"] == name]
        mapping = windows[0]["selected_mapping"]
        assert all(w["selected_mapping"] == mapping for w in windows)
        for m in mapping:
            if "original_file" in m:
                identity = m["original_file"]
            elif "original_cell" in m:
                identity = "cell " + str(m["original_cell"])
            else:
                identity = "index " + str(m["original_index"])
            record = dict(record=name, local_node=m["new_node"] + 1, source_index=m["source_index"],
                          source_identifier=identity, screen_events=m["count_in_screening_span"],
                          screen_start=m["screening_span_start"], screen_end=m["screening_span_end"])
            records.append(record)
            body.append(latex_row([name, record["local_node"], record["source_index"], identity, record["screen_events"]]))
    assert len(records) == 27
    csv_write("neural_mapping", records)
    longtable("neural_mapping", "Fixed node subsets and their source identifiers.",
              "tab:neural-mapping", "lrrlr", ["Record", "Local node", "Source index", "Source identifier", "Screen events"], body,
              r"Local nodes are one-based; source indices retain their zero-based values. PVC-3 identifiers are literal filenames. Screening uses $[0,20)$ seconds for ret-1 and PVC-3 and $[0,40)$ for PVC-11. The same nodes are used in all five windows.")
    return len(records)


def neural_sensitivity():
    records, body = [], []
    for w in real_windows():
        record = dict(record=w["short_name"], window=w["window_index"])
        cells = [w["short_name"], w["window_index"]]
        for k in [1, 2, 3]:
            for bg, code in [("constant", "C"), ("linear", "V")]:
                value = w["all_comparisons"][f"top{k}"][bg]["delta_loglik"]
                record[f"top{k}_{code}"] = value
                cells.append(number(value))
        records.append(record)
        body.append(latex_row(cells))
    assert len(records) == 15
    csv_write("neural_sensitivity", records)
    longtable("neural_sensitivity", "Completion increments for nested prefixes of the same candidate ranking.",
              "tab:neural-sensitivity", "lrrrrrrr", ["Record", "Window", "Top 1: C", "Top 1: V", "Top 2: C", "Top 2: V", "Top 3: C", "Top 3: V"], body,
              r"Each entry is the full test log-likelihood of H minus the separately fitted P model under the same background. Differences are calculated before display rounding; the CSV retains full precision. C and V share each prefix. All windows and both negative and zero-weight outcomes are retained. Test scores do not select the prefix; the main analysis fixes the top-three prefix.")
    return len(records)


def timing():
    records, body = [], []
    phases = ["features", "iteration", "fit"]
    for setting in TIMING["settings"]:
        row = dict(sweep=setting["sweep"], n_nodes=setting["n_nodes"], T=setting["T"], n_events=setting["n_events"])
        cells = ["Events" if row["sweep"] == "event_count" else "Nodes", row["n_nodes"], row["T"], row["n_events"]]
        for phase in phases:
            value = next(r for r in TIMING["rows"] if r["path"] == setting["path"] and r["phase"] == phase)
            assert value["n_measured"] == 10
            for field in ["median_seconds", "q25_seconds", "q75_seconds"]:
                row[phase + "_" + field] = value[field]
            cells.append(f'{value["median_seconds"] * 1000:.3f} [{value["q25_seconds"] * 1000:.3f}, {value["q75_seconds"] * 1000:.3f}]')
        records.append(row)
        body.append(latex_row(cells))
    assert len(records) == 9
    csv_write("timing", records)
    longtable("timing", "Runtime in milliseconds: median [first quartile, third quartile] over ten timing repetitions.",
              "tab:timing", "lrrrlll", ["Sweep", "$N$", "$T$", "Events", "Features", "One EM update", "Complete fit"], body,
              r"Each setting uses one fixed generated path and one fixed positive initial state. The three phases have separate warm-ups, which are excluded. Feature construction builds the history features and exact exposures. One EM update uses cached features and includes an objective evaluation; feature preparation and adaptive stopping checks are excluded. Complete fit includes parameter initialisation, both feature constructions, EM and coordinate refinement with their stopping checks. All phases exclude simulation, imports, event loading and result serialisation. Timing repetitions measure runtime variation. The benchmark has one two-node candidate and $\eta=0.001$; it uses the free-coefficient solver.")
    return len(records)


def timing_slopes():
    records, body = [], []
    labels = {"features": "Feature construction", "iteration": "One cached EM update", "fit": "Complete fit"}
    for phase, label in labels.items():
        value = TIMING["slopes"][phase]
        x, y = np.log(value["event_counts"]), np.log(value["median_seconds"])
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - float(np.sum((y - (slope * x + intercept))**2) / np.sum((y - y.mean())**2))
        last = float(np.polyfit(x[-3:], y[-3:], 1)[0])
        assert abs(slope - value["main_slope"]) < 1e-12
        assert abs(r2 - value["r_squared_log_scale"]) < 1e-12
        assert abs(last - value["last_three_sensitivity_slope"]) < 1e-12
        record = dict(phase=phase, all_five_slope=float(slope), all_five_r_squared=r2, last_three_slope=last)
        records.append(record)
        body.append(latex_row([label, number(slope), number(r2), number(last)]))
    csv_write("timing_slopes", records)
    longtable("timing_slopes", "Descriptive log--log slopes for the event-count sweep.",
              "tab:timing-slopes", "lrrr", ["Phase", "All five: slope", "$R^2$", "Last three: slope"], body,
              r"Ordinary least squares regresses log median seconds on log observed event count. The five counts are 135, 270, 523, 1094 and 2247 at $N=4$. The final column uses only the last three counts. These finite-grid summaries describe the measured implementation; they are not asymptotic complexity exponents.")
    return len(records)


if __name__ == "__main__":
    assert len(SYN) == 4200 and len({r["dataset_id"] for r in SYN}) == 1100
    assert len(GROUPS) == 46 and len(FITS) == 15
    assert len(real_fit_map()) == 15 and len(REAL["windows"]) == 15
    counts = {name: globals()[name]() for name in ["synthetic_prediction", "synthetic_recovery",
              "synthetic_nuisance", "neural_scores", "neural_candidates", "neural_mapping",
              "neural_sensitivity", "timing", "timing_slopes"]}
    print(json.dumps(counts, indent=2))
