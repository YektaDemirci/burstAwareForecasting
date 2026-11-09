"""1-step-ahead MSE table for the paper: vanilla Informer vs the proposed model, overall and on burst points.
Reads final_study/results/runs_fin.csv (written by evaluate_final.py, or pass --csv) and writes mse_table_h1.{md,tex,png}
next to it. Cells are mean +- std over seeds; the change is between the means. Run from the repo root.
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "final_study/results/mse_table_h1"
NAMES = {"paper": "Paper self-similar", "ss2": "Self-similar 2", "ss3": "Self-similar 3", "ss4": "Self-similar 4"}

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="final_study/results/runs_fin.csv")
d = pd.read_csv(ap.parse_args().csv)
d = d[d.H == 1]
rows, tex = [], []
for key, name in NAMES.items():
    x = d[d.dataset == key]
    m, s = (lambda c: x[c].mean()), (lambda c: x[c].std())
    oc = (m("O_proposed") / m("O_vanilla") - 1) * 100
    bc = (m("B_proposed") / m("B_vanilla") - 1) * 100
    rows.append([name, f"{m('O_vanilla'):.4f} ± {s('O_vanilla'):.4f}", f"{m('O_proposed'):.4f} ± {s('O_proposed'):.4f}", f"{oc:+.1f}%",
                 f"{m('B_vanilla'):.3f} ± {s('B_vanilla'):.3f}", f"{m('B_proposed'):.3f} ± {s('B_proposed'):.3f}", f"{bc:+.1f}%"])
    tex.append(f"{name} & ${m('O_vanilla'):.4f} \\pm {s('O_vanilla'):.4f}$ & ${m('O_proposed'):.4f} \\pm {s('O_proposed'):.4f}$ & ${oc:+.1f}$ & "
               f"${m('B_vanilla'):.3f} \\pm {s('B_vanilla'):.3f}$ & $\\mathbf{{{m('B_proposed'):.3f}}} \\pm {s('B_proposed'):.3f}$ & ${bc:+.1f}$ \\\\")
    print(f"{name}: {len(x)} seeds")

md = ["| Dataset | Overall MSE: Informer | Overall MSE: Proposed | Δ | Burst MSE: Informer | Burst MSE: Proposed | Δ |",
      "|---|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
open(OUT + ".md", "w").write("\n".join(md) + "\n")

T = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Overall MSE} & \multicolumn{3}{c}{Burst MSE} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Dataset & Informer & Proposed & $\Delta$\% & Informer & Proposed & $\Delta$\% \\", r"\midrule"] + tex + [r"\bottomrule", r"\end{tabular}"]
open(OUT + ".tex", "w").write("\n".join(T) + "\n")

fig, ax = plt.subplots(figsize=(11.5, 2.4))
ax.axis("off")
t = ax.table(cellText=rows, colLabels=["Dataset", "Informer", "Proposed", "Δ", "Informer", "Proposed", "Δ"], loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(10)
t.scale(1, 1.6)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#999999")
    if r == 0:
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(weight="bold")
ax.text(0.385, 0.93, "Overall MSE", ha="center", transform=ax.transAxes, weight="bold")
ax.text(0.80, 0.93, "Burst MSE", ha="center", transform=ax.transAxes, weight="bold")
ax.set_title("1-step-ahead test MSE (mean ± std over 4 seeds)", pad=18)
plt.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("\n".join(md))
