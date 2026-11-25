"""Reproduce manuscript figures from saved results; no simulation or fitting."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import argparse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figure"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"
SYN = json.loads((DATA / "synthetic_replicates.json").read_text(encoding="utf-8"))
REAL = json.loads((DATA / "realdata_analysis.json").read_text(encoding="utf-8"))
TIMING = json.loads((DATA / "timing_summary.json").read_text(encoding="utf-8"))
BLACK, GREY = "#161616", "#707070"
BLUE, ORANGE, GREEN = BLACK, GREY, "#404040"
# Sizes are set at the manuscript's 34-pica text width, not at slide size.
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
    "mathtext.fontset": "stix", "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.linewidth": .55, "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "lines.linewidth": .8, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white"})
RNG = np.random.default_rng(20261006)  # display jitter only
NUMBERS = []

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=300)
    plt.close(fig)

def stats(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

def select(scenario, model):
    return sorted([r for r in SYN if r["scenario"] == scenario and r["model"] == model], key=lambda r:r["repeat"])

def contrast(scenario, left, right):
    a, b = select(scenario, left), select(scenario, right)
    assert len(a) == len(b) > 0
    assert [r["dataset_id"] for r in a] == [r["dataset_id"] for r in b]
    return np.array([x["test_loglik"]-y["test_loglik"] for x,y in zip(a,b)])

def record(figure, scenario, quantity, values):
    mean, se = stats(values)
    NUMBERS.append(dict(figure=figure, scenario=scenario, quantity=quantity, n=len(values), mean=mean, mcse=se))
    return mean, se

def panel(ax, letter):
    ax.text(.025, .94, f"({letter})", transform=ax.transAxes,
            ha="left", va="top", fontsize=8)


def mechanism():
    fig, (raster, curve) = plt.subplots(2, 1, figsize=(5.65, 2.75), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.105, right=.98, bottom=.15, top=.97, hspace=.1)
    events = {1: [1, 1.6], 2: [1.2], 3: [1.4], 4: [1.8]}
    for n, ts in events.items():
        raster.vlines(ts, n-.19, n+.19, color=BLACK, lw=1)
    raster.set(yticks=[1, 2, 3, 4], ylim=(4.45, -.4), ylabel="Node", xlim=(.55, 2.9))
    # The bracket denotes the witness window for the first completion.
    raster.plot([.65, .65, 1.4, 1.4], [.84, .64, .64, .84], color=BLACK, lw=.55)
    raster.text(1.025, .6, r"$\Delta$", ha="center", va="bottom", fontsize=8)
    for ax in (raster, curve):
        for t in [1.4, 1.6]:
            ax.axvline(t, color=".65", ls=(0, (2, 2)), lw=.5, zorder=0)
    for start, end, anchor in [(.55, 1.4, None), (1.4, 1.6, 1.4), (1.6, 2.9, 1.6)]:
        ts = np.linspace(start, end, 180)
        hs = np.zeros_like(ts) if anchor is None else np.exp(-(ts-anchor))
        curve.plot(ts, hs, color=BLACK, lw=.85)
    for t, old in [(1.4, 0), (1.6, np.exp(-.2))]:
        curve.plot(t, old, "o", color=BLACK, ms=3, zorder=5)
        curve.plot(t, 1, "o", mfc="white", mec=BLACK, mew=.7, ms=3, zorder=5)
    curve.axvline(2.35, color=".65", lw=.5, ls=(0, (1, 3)))
    curve.text(2.35, 1.04, r"$c_2+\Delta$", ha="center", va="bottom", fontsize=7)
    curve.text(1.4, -.11, r"$c_1$", ha="center", va="top", fontsize=7)
    curve.text(1.6, -.11, r"$c_2$", ha="center", va="top", fontsize=7)
    curve.set(xlabel="Time", ylabel=r"$H_e(t)$", ylim=(-.24, 1.22),
              yticks=[0, .5, 1], xticks=[.6, 1, 1.4, 1.8, 2.2, 2.6])
    panel(raster, "a"); panel(curve, "b")
    raster.texts[-1].set_position((.945, .94))
    save(fig, "hth_combo_data")

def overlap():
    fig, axs = plt.subplots(2, 3, figsize=(5.65, 3.55), sharex=True)
    fig.subplots_adjust(left=.09, right=.945, bottom=.12, top=.90, wspace=.36, hspace=.17)
    titles = [r"$e_1=\{1,2,3\}$", r"$e_2=\{3,4,5\}$", "C-M2 minus C-M1"]
    for ax, title in zip(axs[0], titles): ax.set_title(title, pad=6)
    for row, T in enumerate([500, 1000]):
        means, ses = [], []
        for a, label in [(0, "null"), (.2, "moderate"), (.4, "strong")]:
            s = f"{label}_T{T}"
            for col in [0, 1]:
                ax = axs[row, col]
                v = np.array([r["triple_estimates"][col] for r in select(s, "C-M2")])
                ax.scatter(a+RNG.uniform(-.023, .023, len(v)), v, s=2,
                           color=".68", linewidths=0, zorder=1)
                ax.boxplot([v], positions=[a], widths=.066, showfliers=False,
                    manage_ticks=False, boxprops={"color": BLACK, "lw": .65},
                    medianprops={"color": BLACK, "lw": .8},
                    whiskerprops={"color": BLACK, "lw": .65},
                    capprops={"color": BLACK, "lw": .65})
                ax.plot([a-.055, a+.055], [a, a], color=BLACK, ls=(0, (3, 2)), lw=.65)
                ax.set(ylim=(-.025, .8), yticks=[0, .2, .4, .6, .8])
            m, se = record("2", s, "C-M2 minus C-M1", contrast(s, "C-M2", "C-M1"))
            means.append(m); ses.append(se)
        axs[row, 2].errorbar([0, .2, .4], means, yerr=2*np.array(ses),
            fmt="o-", color=BLACK, ms=3, lw=.75, elinewidth=.65, capsize=2, capthick=.65)
        axs[row, 2].axhline(0, color=".5", ls=(0, (3, 3)), lw=.55)
        axs[row, 2].set(ylim=(-.65, 5.6), yticks=[0, 2, 4], ylabel=r"$\Delta\ell$")
        axs[row, 0].set_ylabel(r"$\widehat\alpha_e$")
        axs[row, 1].tick_params(labelleft=False)
        axs[row, 2].text(1.07, .5, f"$T={T}$", transform=axs[row, 2].transAxes,
                         rotation=90, ha="center", va="center", fontsize=8)
        for col in range(3):
            ax = axs[row, col]
            ax.set(xlim=(-.075, .475), xticks=[0, .2, .4], xticklabels=["0", "0.2", "0.4"])
            panel(ax, "abcdef"[row*3+col])
            if row == 1: ax.set_xlabel(r"$\alpha$")
    save(fig, "overlapping_hyperedges")

def background():
    cells = [[("null_T1000", 0), ("sinusoidal_a0.3_null", 0),
              ("sinusoidal_a0.6_null", 0), ("linear_null", 0)],
             [("strong_T1000", .4), ("sinusoidal_a0.3_strong", .4),
              ("sinusoidal_a0.6_strong", .4), ("linear_moderate", .2)]]
    fig, axs = plt.subplots(2, 3, figsize=(5.65, 3.7), sharex=True, sharey="col")
    fig.subplots_adjust(left=.085, right=.947, bottom=.16, top=.82, wspace=.37, hspace=.18)
    for ax, title in zip(axs[0], ["Total triple weight", "M2 minus M1", "V-M2 minus C-M2"]):
        ax.set_title(title, pad=5)
    for row, settings in enumerate(cells):
        for x, (s, truth) in enumerate(settings):
            axs[row, 0].plot([x-.3, x+.3], [2*truth, 2*truth],
                            color=".45", ls=(0, (3, 2)), lw=.6)
            for model, color, marker, off, fill in [("C", BLACK, "o", -.13, BLACK),
                                                     ("V", GREY, "s", .13, "white")]:
                quantities = [(0, model+" sum of all six triple weights", [r["triple_sum"] for r in select(s, model+"-M2")]),
                              (1, model+"-M2 minus "+model+"-M1", contrast(s, model+"-M2", model+"-M1"))]
                for col, quantity, values in quantities:
                    m, se = record("3", s, quantity, values)
                    axs[row, col].errorbar(x+off, m, yerr=2*se, fmt=marker,
                        color=color, mfc=fill, mew=.65, ms=3.2, elinewidth=.65, capsize=2, capthick=.6)
            m, se = record("3", s, "V-M2 minus C-M2", contrast(s, "V-M2", "C-M2"))
            axs[row, 2].errorbar(x, m, yerr=2*se, fmt="o", color=BLACK,
                ms=3, elinewidth=.65, capsize=2, capthick=.6)
        for col in range(3):
            ax = axs[row, col]
            ax.set(xlim=(-.55, 3.55), xticks=range(4), xticklabels=["Const.", "Per. 0.3", "Per. 0.6", "Linear"])
            ax.tick_params(axis="x", labelrotation=35)
            if col > 0: ax.axhline(0, color=".5", ls=(0, (3, 3)), lw=.55)
            panel(ax, "abcdef"[row*3+col])
        axs[row, 0].set_ylabel(r"$\sum_e\widehat\alpha_e$")
        for col in [1, 2]: axs[row, col].set_ylabel(r"$\Delta\ell$")
        axs[row, 2].text(1.07, .5, [r"$\alpha=0$", r"$\alpha>0$"][row],
            transform=axs[row, 2].transAxes, rotation=90, ha="center", va="center", fontsize=8)
    axs[0, 0].set(ylim=(-.06, 1.02), yticks=[0, .4, .8])
    axs[0, 1].set(ylim=(-1.1, 6.25), yticks=[0, 2, 4, 6])
    axs[0, 2].set(ylim=(-7, 74), yticks=[0, 20, 40, 60])
    fig.legend(handles=[Line2D([], [], marker="o", color=BLACK, ls="", ms=3.2, label="C: constant fit"),
        Line2D([], [], marker="s", color=GREY, mfc="white", ls="", ms=3.2, label="V: varying fit"),
        Line2D([], [], color=".45", ls=(0, (3, 2)), lw=.6, label="True total")],
        loc="upper center", bbox_to_anchor=(.51, .997), ncol=3, handlelength=1.5, columnspacing=1.6)
    save(fig, "time_varying_background")

def neural():
    fig, axs = plt.subplots(2, 3, figsize=(5.65, 3.45), sharex=True, sharey="row")
    fig.subplots_adjust(left=.09, right=.985, bottom=.13, top=.81, wspace=.13, hspace=.18)
    for col, name in enumerate(["ret-1", "PVC-3", "PVC-11"]):
        ws = sorted([w for w in REAL["windows"] if w["short_name"] == name], key=lambda w:w["window_index"])
        xs = [w["window_index"] for w in ws]
        axs[0, col].set_title(name, pad=5)
        for bg, color, marker, style, fill in [("constant", BLACK, "o", "-", BLACK),
                                               ("linear", GREY, "s", "--", "white")]:
            axs[0, col].plot(xs, [w["primary_comparison"][bg]["delta_loglik"] for w in ws],
                marker=marker, ls=style, color=color, mfc=fill, mew=.65, ms=3, lw=.75)
        axs[1, col].plot(xs, [w["V_H2_top3_minus_C_H2_top3"] for w in ws],
                        "o-", color=BLACK, ms=3, lw=.75)
        for row in range(2):
            ax = axs[row, col]
            ax.axhline(0, color=".55", ls=(0, (3, 3)), lw=.55)
            ax.set(xticks=range(1, 6), xlim=(.65, 5.35))
            panel(ax, "abcdef"[row*3+col])
            if row == 1 and col == 2:
                ax.texts[-1].set_position((.91, .94))
        axs[1, col].set_xlabel("Window")
    axs[0, 0].set(ylim=(-2.8, 12.6), yticks=[0, 5, 10], ylabel=r"$\ell_H-\ell_P$")
    axs[1, 0].set(ylim=(-30, 7), yticks=[-30, -20, -10, 0], ylabel=r"$\ell_{\mathrm{V-H}}-\ell_{\mathrm{C-H}}$")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLACK, ms=3, lw=.75, label="C: constant"),
        Line2D([], [], marker="s", color=GREY, mfc="white", ls="--", ms=3, lw=.75, label="V: linear")],
        loc="upper center", bbox_to_anchor=(.52, .998), ncol=2, handlelength=2.5, columnspacing=2.5)
    save(fig, "real04_summary_panel")

def decoys():
    fig,axs=plt.subplots(2,2,figsize=(5.65,4.3),sharex=True,sharey=True,layout="constrained")
    cells=[f"{label}_T{T}" for T in [500,1000] for label in ["null","moderate","strong"]]
    for j,ax in enumerate(axs.flat):
        for x,s in enumerate(cells):
            v=[r["decoy_estimates"][j] for r in select(s,"C-M2")];c=BLUE if x<3 else ORANGE
            ax.scatter(x+RNG.uniform(-.14,.14,len(v)),v,s=4,color=c,alpha=.33,rasterized=True)
            ax.boxplot([v],positions=[x],widths=.32,showfliers=False,medianprops={"color":"black"},boxprops={"color":c},whiskerprops={"color":c},capprops={"color":c})
        ax.axhline(0,color="black",ls="--",lw=.8);ax.axvline(2.5,color="#cccccc",lw=.6)
        ax.set(xticks=range(6),xticklabels=["0\n500","0.2\n500","0.4\n500","0\n1000","0.2\n1000","0.4\n1000"],ylabel="Estimated weight")
        if j >= 2:
            ax.set_xlabel(r"Signal $\alpha$ (top); duration $T$ (bottom)")
        ax.set_title([r"(a) $\{1,2,4\}$",r"(b) $\{1,3,4\}$",r"(c) $\{2,3,5\}$",r"(d) $\{3,5,6\}$"][j],loc="left")
    save(fig,"decoy_weights")

def sensitivity():
    fig,axs=plt.subplots(2,3,figsize=(5.65,3.8),sharex=True,layout="constrained")
    for j,name in enumerate(["ret-1","PVC-3","PVC-11"]):
        ws=[w for w in REAL["windows"] if w["short_name"]==name]
        for i,bg in enumerate(["constant","linear"]):
            ax=axs[i,j]
            for top,c,marker in [(1,BLUE,"o"),(2,ORANGE,"s"),(3,GREEN,"^")]:
                ax.plot([w["window_index"] for w in ws],[w["all_comparisons"][f"top{top}"][bg]["delta_loglik"] for w in ws],color=c,marker=marker,ms=3,lw=.8,label=f"Top {top}")
            ax.axhline(0,color="#888888",lw=.6);ax.set_title(f'{name}: {"C" if i==0 else "V"}',loc="left");ax.set(xticks=range(1,6),xlabel="Window")
            if j==0:ax.set_ylabel("H minus P: test log-likelihood")
    axs[0,0].legend(frameon=False,fontsize=7)
    save(fig,"candidate_sensitivity")

def timing():
    fig,axs=plt.subplots(1,2,figsize=(5.65,2.7),layout="constrained")
    for ax,sweep,xcol,title in zip(axs,["event_count","node_count"],["n_events","n_nodes"],["(a) Event-count sweep, N = 4","(b) Node-count sweep, T = 200"]):
        for phase,c,marker,label in [("features",BLUE,"o","Feature construction"),("iteration",ORANGE,"s","One cached EM update"),("fit",GREEN,"^","Complete fit")]:
            rows=sorted([r for r in TIMING["rows"] if r["sweep"]==sweep and r["phase"]==phase],key=lambda r:r[xcol])
            x=np.array([r[xcol] for r in rows]);m=np.array([r["median_seconds"] for r in rows]);lo=np.array([r["q25_seconds"] for r in rows]);hi=np.array([r["q75_seconds"] for r in rows])
            ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt=marker+"-",color=c,lw=.8,ms=3,capsize=2,label=label)
        ax.set(xscale="log",yscale="log",ylabel="Seconds (median and IQR)",xlabel="Observed events" if sweep=="event_count" else "Nodes");ax.set_title(title,loc="left")
        ticks = [135,270,523,1094,2247] if sweep=="event_count" else [3,5,8,12]
        ax.set_xticks(ticks, labels=[str(v) for v in ticks])
        ax.minorticks_off()
    fig.legend(*axs[0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False,fontsize=7.5)
    save(fig,"syn07_scalability")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", action="store_true", help="Only generate the four main-text figures.")
    args = parser.parse_args()
    mechanism();overlap();background();neural()
    if not args.main: decoys();sensitivity();timing()
    with (DATA / "figure_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(NUMBERS[0]));writer.writeheader();writer.writerows(NUMBERS)
    print("Generated four main-text figures from saved results." if args.main else
          "Generated four main-text and three supplementary figures from saved results.")
