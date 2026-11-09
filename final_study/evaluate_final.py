"""Scores final_study/run_final.sh on the test regions (the paper series and self-similar 2-4). Run from the repo root at any time; it scores finished runs and caches forecasts.

All MSEs are in normalized units. Burst points are past-only events (utils.peakfeatures), with thresholds from the training region.
O: all points. B: burst points. B seq: every step of forecasts whose horizon holds a burst, the same as B at H=1.
B listed: burst points from the dataset's peak file, the paper's labels.
Advantage > 0: lower burst MSE than vanilla raised by the constant that gives vanilla the proposed model's O. The constant is set
on the test data, which favours vanilla. Its CI resamples burst clusters within one run.
AP: average precision for burst points, scoring the proposed gate and the vanilla forecast.
Summaries give each dataset equal weight. Their CI (t interval) is across datasets. A dataset with several seeds also gets its own summary,
whose CI is across seeds, so training noise only.
Optional: --datasets and --horizons limit what is scored. --tag writes the table to runs_<prefix>_<tag>.csv, leaving runs_<prefix>.csv as it is.
"""
import argparse
import glob
import os
import re
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
from data.data_loader import Dataset_Pred
from models.model import Informer
from utils.peakfeatures import causal_burst_events

SEQ, LABEL = 128, 64
CLUSTER_GAP = 128
DATASETS = {"paper": ("data/ETT/", "1.04_750_10ms.csv", "bursts177.csv", "10L")}
for i in (2, 3, 4):  # self_similar_2/3/4 are simulate.py seeds 11, 12 and 18
    DATASETS[f"ss{i}"] = ("data/fresh/", f"self_similar_{i}.csv", f"self_similar_{i}_peaks.csv", "10L")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset(ds, h, feature):
    root, data, peaks, freq = DATASETS[ds]
    return Dataset_Pred(root_path=root, flag="pred", size=[SEQ, LABEL, h], features="S", data_path=data, target="OT",
                        timeenc=1, freq=freq, peak_path=peaks, peak_feature=feature)


def forecast(path, ds, h, feature, heads):
    """Forecast and gate logit as [window, step]."""
    freq = DATASETS[ds][3]
    model = Informer(1, 1, 1, SEQ, LABEL, h, 5, 512, 8, 2, 1, 2048, 0.05, "prob", "timeF", freq[-1], freq, "gelu",
                     False, True, True, device, heads).float().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    torch.manual_seed(0)  # ProbSparse attention samples randomly, also at eval time.
    pred, logit = [], []
    with torch.no_grad():
        for bx, by, mx, my, px, py, ix, iy in DataLoader(dataset(ds, h, feature), batch_size=64):
            dec = torch.cat([by[:, :LABEL].float(), torch.zeros(len(by), h, 1)], 1).to(device)
            out = model(bx.float().to(device), mx.float().to(device), dec, my.float().to(device),
                        batch_x_peak=px.float().to(device), batch_y_peak=py.float().to(device),
                        batch_x_idx=ix.to(device), batch_y_idx=iy.to(device))
            pred.append(out[0].cpu().numpy()[..., 0])
            logit.append(out[1].cpu().numpy()[..., 0])
    del model
    torch.cuda.empty_cache()
    return np.concatenate(pred), np.concatenate(logit)


def targets(ds, h):
    """Targets, past-only and listed burst flags, and burst clusters, as [window, step]. Same span and split as the data loader."""
    root, data, peaks, _ = DATASETS[ds]
    d = dataset(ds, h, "none")
    listed = pd.read_csv(root + peaks, header=None)[0].values
    x = pd.read_csv(root + data).OT.values[listed[0]:listed[-1] + 1].astype(float)
    events = causal_burst_events(x, int(len(x) * 0.7))[-len(d.peaks):]
    pos = np.flatnonzero(events)
    cluster = np.full(len(events), -1)
    cluster[pos] = np.r_[0, np.cumsum(np.diff(pos) > CLUSTER_GAP)]
    idx = np.arange(len(d))[:, None] + SEQ + np.arange(h)
    return d.data_y[idx, 0], events[idx], d.peaks[idx] == 1, cluster[idx]


def cluster_ci(values, clusters, rng, n=4000):
    u, inv = np.unique(clusters, return_inverse=True)
    s, k = np.bincount(inv, values), np.bincount(inv)
    draws = rng.randint(0, len(u), (n, len(u)))
    return np.percentile(s[draws].sum(1) / k[draws].sum(1), [2.5, 97.5])


def shift_to_o(err, target_o):
    """Upward constant that brings a forecast's overall MSE to target_o, or the one that minimizes O if none does."""
    extra = target_o - (err ** 2).mean()
    return err.mean() + np.sqrt(max(err.mean() ** 2 + extra, 0.))


def average_precision(score, y):
    y = y[np.argsort(-score)]
    return (np.cumsum(y) / np.arange(1, len(y) + 1))[y].mean()


def load_run(a, ds, m, seed, h):
    run = f"{a.prefix}_{ds}_{m}_s{seed}"
    f = os.path.join(a.preds, f"{run}_h{h}.npz")
    if not os.path.exists(f):
        ckpt = os.path.join(a.ckpt_root, f"{run}_128_{h}_{DATASETS[ds][3]}", "checkpoint.pth")
        pred, logit = forecast(ckpt, ds, h, *(("causal", True) if m == "proposed" else ("none", False)))
        np.savez(f, pred=pred.astype(np.float32), logit=logit.astype(np.float32))
    return np.load(f)


def score(a, ds, seed, h, data, rng):
    T, Y, L, C = data
    van, pro = load_run(a, ds, "baseline", seed, h), load_run(a, ds, "proposed", seed, h)
    ev, ep = T - van["pred"], T - pro["pred"]
    W = Y.any(1)
    gap = ((ev - shift_to_o(ev, (ep ** 2).mean())) ** 2 - ep ** 2)[Y]
    lo, hi = cluster_ci(gap, C[Y], rng)
    out = dict(dataset=ds, seed=seed, H=h, bursts=int(Y.sum()), O_vanilla=(ev ** 2).mean(), O_proposed=(ep ** 2).mean(),
               B_vanilla=(ev[Y] ** 2).mean(), B_proposed=(ep[Y] ** 2).mean(), Bseq_vanilla=(ev[W] ** 2).mean(),
               Bseq_proposed=(ep[W] ** 2).mean(), Blisted_vanilla=(ev[L] ** 2).mean(), Blisted_proposed=(ep[L] ** 2).mean(),
               advantage=gap.mean(), adv_lo=lo, adv_hi=hi, AP_gate=average_precision(pro["logit"].ravel(), Y.ravel()),
               AP_vanilla=average_precision(van["pred"].ravel(), Y.ravel()))
    out["O_change"] = 100 * (out["O_proposed"] / out["O_vanilla"] - 1)
    out["B_change"] = 100 * (out["B_proposed"] / out["B_vanilla"] - 1)
    out["Bseq_change"] = 100 * (out["Bseq_proposed"] / out["Bseq_vanilla"] - 1)
    out["Blisted_change"] = 100 * (out["Blisted_proposed"] / out["Blisted_vanilla"] - 1)
    return out


def interval(v):
    if len(v) < 2:
        return "-"
    half = stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
    return f"[{np.mean(v) - half:+.3f}, {np.mean(v) + half:+.3f}]"


ROW = "{:9s}{:>5s}{:>7s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}{:>9s}{:>8s}{:>8s}   {}"
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="fin")
ap.add_argument("--ckpt_root", default="./checkpoints/")
ap.add_argument("--logs", default="final_study/logs")
ap.add_argument("--preds", default="final_study/results/preds")
ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
ap.add_argument("--horizons", nargs="*", type=int, default=None)
ap.add_argument("--tag", default="")
a = ap.parse_args()
os.makedirs(a.preds, exist_ok=True)
rng = np.random.RandomState(0)
rows = []
for h in sorted({int(re.search(r"_h(\d+)\.done$", f).group(1)) for f in glob.glob(f"{a.logs}/{a.prefix}_*.done")}):
    if a.horizons and h not in a.horizons:
        continue
    print(f"\n=== H={h} ===")
    print(ROW.format("dataset", "seed", "bursts", "O van", "O prop", "O chg%", "B van", "B prop", "B chg%", "Bseq%", "Blist%", "AP gate", "AP van", "advantage [CI]"))
    for ds in a.datasets:
        seeds = []
        for f in glob.glob(f"{a.logs}/{a.prefix}_{ds}_baseline_s*_h{h}.done"):
            s = int(re.search(r"_s(\d+)_h", f).group(1))
            if os.path.exists(f"{a.logs}/{a.prefix}_{ds}_proposed_s{s}_h{h}.done"):
                seeds.append(s)
        if not seeds:
            continue
        data = targets(ds, h)
        for s in sorted(seeds):
            r = score(a, ds, s, h, data, rng)
            rows.append(r)
            print(ROW.format(ds, str(s), str(r["bursts"]), f"{r['O_vanilla']:.4f}", f"{r['O_proposed']:.4f}", f"{r['O_change']:+.1f}",
                             f"{r['B_vanilla']:.3f}", f"{r['B_proposed']:.3f}", f"{r['B_change']:+.1f}", f"{r['Bseq_change']:+.1f}", f"{r['Blisted_change']:+.1f}",
                             f"{r['AP_gate']:.3f}", f"{r['AP_vanilla']:.3f}", f"{r['advantage']:+.3f} [{r['adv_lo']:+.3f}, {r['adv_hi']:+.3f}]"), flush=True)
    frame = pd.DataFrame([r for r in rows if r["H"] == h])
    if frame.empty:
        continue
    per = frame.groupby("dataset").mean(numeric_only=True)
    print(f"\nSummary at H={h}: mean [95% CI]")
    groups = [("Self-similar, all", per, "datasets"), ("Self-similar, unseen", per[per.index != "paper"], "datasets")]
    groups += [(ds, g, "seeds") for ds, g in frame.groupby("dataset", sort=False) if len(g) > 1]
    for title, g, unit in groups:
        if g.empty:
            continue
        print(f"  {title} ({len(g)} {unit}):")
        for col, label in (("O_change", "O change %"), ("B_change", "B change %"), ("Bseq_change", "B seq change %"), ("advantage", "advantage")):
            print(f"    {label:16s}{g[col].mean():+8.3f}  {interval(g[col].values)}")
        print(f"    {'AP gate / vanilla':16s}{g['AP_gate'].mean():8.3f} / {g['AP_vanilla'].mean():.3f}", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(f"final_study/results/runs_{a.prefix}{'_' + a.tag if a.tag else ''}.csv", index=False)
