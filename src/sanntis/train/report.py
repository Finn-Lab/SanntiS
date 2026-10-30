#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation report: markdown + PNG figures + TSV, from the
pooled out-of-fold predictions of the ``mc`` CV splits on the training data.

Needs the ``[train]`` extra (matplotlib). Colours are the EMBL-EBI Visual Framework palette (vf-core design tokens):
SanntiS = EMBL green; two-series metric figures use blue (BGC-level) / dark orange (ORF-level). Recessive grids,
text in ink colours, a legend for every multi-series figure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sanntis.config import Recipe, load_recipe
from sanntis.model.npc import pathway_calls, remap
from sanntis.train import metrics as M
from sanntis.train.npc_train import CvPreds, cross_split_thresholds, top1
from sanntis.train.tune import load_oof, scored_at, subset

log = logging.getLogger(__name__)

# EMBL-EBI Visual Framework: vf-color--blue, --orange--dark, --green; neutrals 100/200
BLUE, ORANGE = "#3B6FB6", "#B65417"
SANNTIS = "#18974C"
INK, INK2, GRID, BAND, SURFACE = "#1a1c1a", "#54585A", "#E4E4E4", "#F3F3F3", "#FFFFFF"
LINEAGES = ("prokaryotes", "fungi", "plants", "other eukaryotes")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE})
    return plt


# ── strata ───────────────────────────────────────────────────────────────────


def annotate_strata(s: M.Scored, table: pd.DataFrame) -> M.Scored:
    """Add lineage to the GT rows."""
    lineage = table["lineage"] if "lineage" in table else pd.Series([None] * len(table))
    gt = s.gt.copy()
    gt["lineage"] = [lineage.iloc[int(b)] for b in gt["bgc"]]
    return M.Scored(gt, s.pred)


def radar_table(s: M.Scored) -> pd.DataFrame:
    """Recall_T and BGC recall per class and lineage stratum (one row per axis; empty strata left out)."""
    a = M.aggregate(s)
    rows = [{"sector": "class", "axis": c, "n": int(round(a[f"n_gt[{c}]"])), "recall_t": a[f"recall_t[{c}]"],
             "bgc_recall": a[f"bgc_recall[{c}]"]} for c in M.CLASSES if a.get(f"n_gt[{c}]", 0) > 0]
    for v in LINEAGES if "lineage" in s.gt else ():
        m = (s.gt["lineage"] == v).to_numpy()
        if m.any():
            b = M.aggregate(s, gt_mask=m)
            rows.append({"sector": "lineage", "axis": v, "n": int(m.sum()), "recall_t": b["recall_t"],
                         "bgc_recall": b["bgc_recall"]})
    return pd.DataFrame(rows, columns=["sector", "axis", "n", "recall_t", "bgc_recall"])


def strata_table(s: M.Scored) -> pd.DataFrame:
    rows = []
    for v in LINEAGES:
        m = (s.gt["lineage"] == v).to_numpy()
        if not m.any():
            continue
        a = M.aggregate(s, gt_mask=m)
        rows.append({"axis": "lineage", "stratum": v, "n": int(m.sum()), "bgc_macro_recall": a["bgc_macro_recall"],
                     "recall_macro_t": a["recall_macro_t"], "bgc_recall": a["bgc_recall"],
                     "recall_t": a["recall_t"]})
    return pd.DataFrame(rows)


# ── NPC validation ───────────────────────────────────────────────────────────


def npc_validation(npc_preds: Path, targets=(0.9, 0.8)) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score


    z = np.load(npc_preds)
    n = len({k.split("_")[0] for k in z.files})
    parts = [CvPreds(*(z[f"s{i}_{f}"] for f in ("pp", "ps", "yp", "ys", "m")), np.zeros(0), []) for i in range(n)]
    rows = []
    if len(parts) < 2:
        log.warning("NPC validation needs predictions from >= 2 splits; got %d", len(parts))
        return pd.DataFrame()
    for i, d in enumerate(parts):
        th = cross_split_thresholds(parts, i, targets)
        P = remap(d.pp, th.pathway)
        ap = [average_precision_score(d.yp[:, j], P[:, j]) for j in range(d.yp.shape[1]) if d.yp[:, j].any()]
        calls = pathway_calls(P)
        f1 = []
        for j in range(d.yp.shape[1]):
            y = d.yp[:, j] > 0.5
            if y.any():
                f1.append(2 * (y & calls[:, j]).sum() / max(y.sum() + calls[:, j].sum(), 1))
        c, q = top1(d.yp, P)
        m = d.m > 0.5
        sc_c, sc_q = top1(d.ys[m], d.ps[m])
        top3 = np.mean([d.ys[m][r, np.argsort(-d.ps[m][r])[:3]].max() > 0.5 for r in range(int(m.sum()))]) \
            if m.any() else np.nan
        row = {"split": i, "n": len(P), "pathway_macro_ap": float(np.mean(ap)), "pathway_macro_f1": float(np.mean(f1)),
               "pathway_top1": float(c.mean()), "superclass_top1": float(sc_c.mean()) if m.any() else np.nan,
               "superclass_top3": float(top3)}
        for x in targets:
            ans = sc_q >= th.abstention["sc"][x]
            row[f"coverage@{x}"] = float(ans.mean()) if m.any() else np.nan
            row[f"precision@{x}"] = float(sc_c[ans].mean()) if ans.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    mean = df.drop(columns="split").mean().to_dict()
    mean.update({"split": "mean", "n": int(df["n"].sum())})
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


# ── figures ──────────────────────────────────────────────────────────────────


def key_domains_md(kd: dict | None) -> str:
    """Section on the NPC pathway key-domain flag (``tune.json`` ``npc.key_domains``)."""
    if not kd:
        return ""
    st = kd["setting"]

    def pct(v):
        return "–" if v is None else f"{100 * v:.1f} %"

    def ratio(v):
        return "–" if v is None else f"{v:.2f}"

    types = f", InterPro {' / '.join(st['entry_types'])} entries only" if st.get("entry_types") else ""
    bg = (f", in <= {100 * st['background_max']:g} % of the non-BGC background entities"
          if st.get("background_max") is not None else "")
    rows = "".join(f"| {name} | {r['calls']} | {pct(r['false_alarm'])} | {pct(r['catch'])} | {pct(r['err_flagged'])} "
                   f"| {pct(r['err_unflagged'])} | {ratio(r['err_ratio'])} | {pct(r['na_rate'])} |\n"
                   for name, r in (("All calls", kd["all"]), ("Confident tier", kd["confident"])))
    cats = "".join(f"| {c['category']} | {c['n_true_regions']} | {pct(c['oof_coverage'])} | "
                   f"{'gated (na)' if c['gated'] else 'flagged'} | {c['calls']} | {pct(c['false_alarm'])} | "
                   f"{pct(c['catch'])} |\n" for c in kd["per_category"])
    return ("\n## NPC pathway key-domain support (`mc`, cross-split thresholds)\n\n"
            f"Rules per pathway (domains and domain pairs): Wilson lower bound of the precision >= {st['tau']}, "
            f"support >= {st['min_support']}{types}{bg}, set-cover short list, fitted on each split's training "
            "loci. A pathway call is flagged (`npc_pathway_key_domains_unsupported`) when none of its rules is in "
            f"the region; pathways with OOF rule coverage < {st['min_coverage']} are gated (`na`, not flagged). "
            "False alarm = flagged correct calls / correct calls; catch = flagged wrong calls / wrong calls.\n\n"
            "| Calls | n | False alarm | Catch | Error rate, flagged | Error rate, not flagged | Ratio | na |\n"
            "|---|---|---|---|---|---|---|---|\n" + rows +
            "\n| Pathway | True regions | OOF coverage | Flag | Calls | False alarm | Catch |\n"
            "|---|---|---|---|---|---|---|\n" + cats)


def fig_bars(val: dict, ci: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(M.METRICS))
    v = np.array([val[m] for m in M.METRICS])
    lo = v - np.array([ci[m][0] for m in M.METRICS])
    hi = np.array([ci[m][1] for m in M.METRICS]) - v
    ax.bar(x, v, width=0.6, color=BLUE, zorder=2)
    ax.errorbar(x, v, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for xi, vi, top in zip(x, v, v + hi, strict=True):
        ax.text(xi, min(top + 0.02, 1.1), f"{vi:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [M.METRIC_LABELS[m] for m in M.METRICS], rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("value")
    ax.set_title("SanntiS, validation (pooled OOF, 95% bootstrap CI)", fontsize=9, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shade_degenerate(ax, s: pd.DataFrame, ratio: float | None) -> None:
    if ratio is None:
        return
    deg = s.loc[M.degenerate(s, ratio).to_numpy(), "threshold"]
    if len(deg):
        ax.axvspan(deg.min() - 0.025, deg.max() + 0.025, color=BAND, zorder=0, lw=0)


def fig_operating_point(sweep: pd.DataFrame, tune: dict, path: Path) -> None:
    plt = _plt()
    s = sweep[sweep["stratum"] == "all"]
    metric, floor = tune["floor_metric"], tune["floor"]
    t_star = tune["t_star"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _shade_degenerate(ax, s, tune.get("min_pred_ratio"))
    ax.plot(s["threshold"], s["f2_macro"], color=BLUE, lw=2, marker="o", ms=4, label="F2_macro")
    ax.plot(s["threshold"], s[metric], color=ORANGE, lw=2, marker="o", ms=4, label=M.METRIC_LABELS[metric])
    if metric != "precision_t":
        ax.plot(s["threshold"], s["precision_t"], color=INK2, lw=1.2, ls="--", label="Precision_T")
    ax.axhline(floor, color=INK2, lw=1, ls="--")
    ax.text(s["threshold"].max(), floor - 0.06, f"{M.METRIC_LABELS[metric]} floor {floor:.2f}", fontsize=7,
            color=INK2, ha="right")
    ax.axvline(t_star, color=INK2, lw=1, ls=":")
    ax.text(t_star + 0.01, 0.02, f"t* = {t_star:.2f}", fontsize=8, color=INK)
    ax.set_xlabel("detector threshold t (low = t - 0.15)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


SECTOR_TITLES = {"class": "BGC class", "lineage": "Lineage"}


def fig_radar_sectors(ref: pd.DataFrame, metric: str, label: str, title: str, path: Path) -> None:
    """One radar, two sectors (class, lineage): ``metric`` per stratum (``radar_table``); a gap and a shaded band
    separate the sectors."""
    plt = _plt()
    sectors = [x for x in SECTOR_TITLES if (ref["sector"] == x).any()]
    slots, ang_of = [], {}
    for sec in sectors:
        slots += [(sec, a) for a in ref.loc[ref["sector"] == sec, "axis"]] + [None]  # None = gap
    step = 2 * np.pi / len(slots)
    for i, sl in enumerate(slots):
        if sl:
            ang_of[sl] = (i + 0.5) * step
    fig = plt.figure(figsize=(6.4, 7.0))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise from the top
    i0 = 0
    for k, sec in enumerate(sectors):
        n = int((ref["sector"] == sec).sum())
        lo, hi = i0 * step, (i0 + n) * step
        if k % 2 == 0:
            ax.fill_between(np.linspace(lo, hi, 50), 0, 1, color=BAND, zorder=0)
        ax.text((lo + hi) / 2, 1.36, SECTOR_TITLES[sec], ha="center", va="center", fontsize=9, fontweight="bold",
                color=INK)
        i0 += n + 1
    keys = [sl for sl in slots if sl]
    angs = [ang_of[k] for k in keys]
    val = ref.set_index(["sector", "axis"])[metric]
    v = [float(val.get(k, np.nan)) for k in keys]
    ax.plot(angs + angs[:1], v + v[:1], color=SANNTIS, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, zorder=3)
    ax.fill(angs + angs[:1], v + v[:1], color=SANNTIS, alpha=0.10, zorder=2)
    n_of = ref.set_index(["sector", "axis"])["n"]
    ax.set_xticks(angs, [f"{a}\n(n={int(n_of[(s_, a)])})" for s_, a in keys], fontsize=7.5, color=INK)
    ax.tick_params(axis="x", pad=10)
    ax.set_rlabel_position(np.degrees((len(slots) - 0.5) * step))  # in the last gap
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1"], fontsize=7, color=INK2)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.set_title(f"{title}: {label}", fontsize=9, loc="left", color=INK, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def radar_figs(table: pd.DataFrame, title: str, key: str, fig_dir: Path) -> str:
    """Recall_T and BGC recall sector radars; returns their markdown."""
    md = []
    for metric, label in (("recall_t", "Recall_T"), ("bgc_recall", "BGC recall")):
        f = f"{key}_radar_{metric}.png"
        fig_radar_sectors(table, metric, label, title, fig_dir / f)
        md.append(f"![{title} {label} radar](figures/{f})")
    return " ".join(md)


def class_curves(oof: list, table: pd.DataFrame, sweep_all: pd.DataFrame) -> pd.DataFrame:
    """Per class and threshold, pooled OOF: BGC recall (sweep) and BGC precision of the kept regions the RF
    post-filter labels with that class (share overlapping >= 1 true-BGC ORF; class-matched: a true BGC of that
    class)."""
    cls_of = table["classes"].to_numpy()
    rows = []
    for o in oof:
        pos = np.asarray(o.label) == 1
        cum_any = np.concatenate([[0], np.cumsum(pos)])
        cum_c = {c: np.concatenate([[0], np.cumsum(pos & np.array([c in cls_of[e] for e in o.entity]))])
                 for c in M.CLASSES}
        for r in o.regions.itertuples():
            rows.append({"threshold": round(float(r.threshold), 2), "bgc_class": r.bgc_class,
                         "hit": cum_any[r.end] > cum_any[r.start],
                         "hit_class": cum_c[r.bgc_class][r.end] > cum_c[r.bgc_class][r.start]
                         if r.bgc_class in cum_c else False})
    pr = pd.DataFrame(rows, columns=["threshold", "bgc_class", "hit", "hit_class"])
    sw = sweep_all.assign(threshold=sweep_all["threshold"].round(2)).set_index("threshold")
    chk = pr.groupby("threshold")["hit"].mean()
    if len(chk) and not np.allclose(chk.values, sw.loc[chk.index, "bgc_precision"].values):
        log.warning("per-class curves: pooled BGC precision differs from the sweep")
    g = pr.groupby(["bgc_class", "threshold"]).agg(p=("hit", "mean"), pc=("hit_class", "mean"), n=("hit", "size"))
    out = []
    for c in (c for c in M.CLASSES if f"bgc_recall[{c}]" in sw.columns):
        for t in sw.index:
            k = (c, t)
            out.append({"class": c, "threshold": t,
                        "bgc_precision": g["p"].get(k, np.nan), "bgc_precision_class_matched": g["pc"].get(k, np.nan),
                        "n_pred": int(g["n"].get(k, 0)), "bgc_recall": sw.at[t, f"bgc_recall[{c}]"],
                        "n_gt": int(sw.at[t, f"n_gt[{c}]"])})
    return pd.DataFrame(out)


def fig_class_curves(tab: pd.DataFrame, sweep_all: pd.DataFrame, tune: dict, path: Path) -> None:
    """One panel per class (+ all): BGC recall, BGC precision and class-matched BGC precision vs threshold."""
    plt = _plt()
    t_star = tune["t_star"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    near = lambda x: np.isclose(np.asarray(x, float), t_star)  # noqa: E731
    panels = [*(c for c in M.CLASSES if c in set(tab["class"])), "all"]
    for ax in axes.flat[len(panels):-1]:
        ax.axis("off")
    for ax, c in zip(axes.flat, panels, strict=False):
        _shade_degenerate(ax, sweep_all, tune.get("min_pred_ratio"))
        if c == "all":
            x, p, pc, r = (sweep_all["threshold"], sweep_all["bgc_precision"], None,
                           sweep_all["bgc_macro_recall"])
            n_gt, n_pred = int(sweep_all["n_gt"].iloc[0]), int(sweep_all.loc[near(sweep_all["threshold"]),
                                                                             "n_pred"].iloc[0])
            title, rlab = "All", "BGC recall (All: macro)"
        else:
            sub = tab[tab["class"] == c]
            x, p, pc, r = sub["threshold"], sub["bgc_precision"], sub["bgc_precision_class_matched"], sub["bgc_recall"]
            n_gt, n_pred = int(sub["n_gt"].iloc[0]), int(sub.loc[near(sub["threshold"]), "n_pred"].iloc[0])
            title, rlab = c, "BGC recall (All: macro)"
        ax.plot(x, r, color=BLUE, lw=2, marker="o", ms=3.5, label=rlab)
        ax.plot(x, p, color=ORANGE, lw=2, marker="o", ms=3.5, label="BGC precision")
        if pc is not None:
            ax.plot(x, pc, color=ORANGE, lw=1.3, ls="--", label="BGC precision, class-matched")
        ax.axvline(t_star, color=INK2, lw=1, ls=":")
        ax.set_title(f"{title}   (BGCs {n_gt:,}; regions at t* {n_pred:,})", fontsize=9, loc="left", color=INK)
        ax.set_ylim(0, 1.02)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("detector threshold t", fontsize=8)
    axes[0, 0].text(t_star + 0.01, 0.03, f"t* = {t_star:.2f}", fontsize=8, color=INK)
    leg = axes.flat[-1]
    leg.axis("off")
    h, lab = axes[0, 0].get_legend_handles_labels()
    leg.legend(h, lab, frameon=False, fontsize=9, loc="center left")
    leg.text(0, 0.1, "Pooled out-of-fold validation.\nRegion class = RF post-filter class.\nShaded: degenerate "
             "thresholds\n(excluded from the t* choice).", fontsize=8, color=INK2, transform=leg.transAxes)
    fig.suptitle("Validation: BGC precision and recall per class vs operating point", fontsize=11, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────────────


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def _fmt(v: float, ci: tuple[float, float] | None = None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")


def write_report(run: Path, table: pd.DataFrame, *, metadata: dict, recipe: Recipe | None = None) -> Path:
    recipe = recipe or load_recipe()
    run = Path(run)
    out = run / "report"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in fig_dir.glob("*.png"):  # figures of an earlier layout must not linger in the bundle
        f.unlink()
    tune = json.loads((run / "tune" / "tune.json").read_text())
    sweep = pd.read_csv(run / "tune" / "sweep.tsv", sep="\t")
    sweep.to_csv(out / "sweep.tsv", sep="\t", index=False)
    oof = load_oof(run / "cv", table, recipe.splits.n_splits)
    s = annotate_strata(scored_at(oof, tune["t_star"], table), table)

    # metrics table
    rows = []
    strata = {"All": s, **{k.capitalize(): subset(s, k) for k in ("proka", "euk") if (s.gt["kingdom"] == k).any()}}
    vals, cis = {}, {}
    for name, sub in strata.items():
        vals[name] = M.aggregate(sub)
        cis[name] = {k: tuple(v) for k, v in tune["validation_ci"][name.lower()].items()}
        rows.append({"dataset": "validation", "stratum": name, "n_bgc": int(vals[name]["n_gt"]),
                     **{m: vals[name][m] for m in M.METRICS},
                     **{f"{m}_ci": f"{cis[name][m][0]:.4f}-{cis[name][m][1]:.4f}" for m in M.METRICS}})
    pd.DataFrame(rows).to_csv(out / "validation.tsv", sep="\t", index=False)
    st = strata_table(s)
    st.to_csv(out / "validation_strata.tsv", sep="\t", index=False)

    fig_bars(vals["All"], cis["All"], fig_dir / "validation_bars.png")
    fig_operating_point(sweep, tune, fig_dir / "operating_point.png")
    cc = class_curves(oof, table, sweep[sweep["stratum"] == "all"])
    cc.to_csv(out / "validation_class_curves.tsv", sep="\t", index=False)
    fig_class_curves(cc, sweep[sweep["stratum"] == "all"], tune, fig_dir / "validation_class_curves.png")
    radars_md = radar_figs(radar_table(s), "Validation", "validation", fig_dir)

    npc_md = ""
    npc_file = run / "tune" / "npc_cv_preds.npz"
    if npc_file.exists():
        nv = npc_validation(npc_file)
        nv.to_csv(out / "npc_validation.tsv", sep="\t", index=False)
    if npc_file.exists() and len(nv):
        r = nv.iloc[-1]
        npc_md = ("\n## NP classification, validation regions (`mc`, cross-split thresholds)\n\n"
                  "| n | Pathway macro AP | Pathway macro F1 | Pathway top-1 | Superclass top-1 | Superclass top-3 | "
                  "Coverage @0.9 (prec.) | Coverage @0.8 (prec.) |\n|---|---|---|---|---|---|---|---|\n"
                  f"| {int(r['n'])} | {_fmt(r['pathway_macro_ap'])} | {_fmt(r['pathway_macro_f1'])} | "
                  f"{_fmt(r['pathway_top1'])} | {_fmt(r['superclass_top1'])} | {_fmt(r['superclass_top3'])} | "
                  f"{_fmt(r['coverage@0.9'])} ({_fmt(r['precision@0.9'])}) | "
                  f"{_fmt(r['coverage@0.8'])} ({_fmt(r['precision@0.8'])}) |\n")

    npc_tsv = ", `npc_validation.tsv`" if npc_md else ""
    kd_md = key_domains_md((tune.get("npc") or {}).get("key_domains"))
    kd_file = run / "tune" / "npc_key_domains_oof.tsv"
    if kd_md and kd_file.exists():
        (out / kd_file.name).write_bytes(kd_file.read_bytes())
        npc_tsv += f", `{kd_file.name}`"
    best_epochs = ", ".join(map(str, tune["best_epochs"]))
    fm, fv = tune["floor_metric"], tune["floor"]
    op_point = (f"t* = {tune['t_star']:.2f}, low = {tune['threshold_low']:.2f}: highest F2_macro with "
                f"{M.METRIC_LABELS[fm]} >= {fv:.2f} ({'met' if tune['floor_met'] else 'NOT met'})"
                + (f", thresholds with fewer regions than {tune['min_pred_ratio']:g} x true BGCs excluded"
                   if tune.get("min_pred_ratio") is not None else "")
                + f"; F2_macro {vals['All']['f2_macro']:.3f}, {M.METRIC_LABELS[fm]} {vals['All'][fm]:.3f}, "
                f"Precision_T {vals['All']['precision_t']:.3f} at t*")
    meta_rows = "\n".join(f"| {k} | {v} |" for k, v in metadata.items())
    hdr = "| Metric | " + " | ".join(strata) + " |\n|---|" + "---|" * len(strata) + "\n"
    body = "".join(f"| {M.METRIC_LABELS[m]} | " + " | ".join(_fmt(vals[n][m], cis[n][m]) for n in strata) + " |\n"
                   for m in M.METRICS)
    body += "| n (BGCs) | " + " | ".join(str(int(vals[n]["n_gt"])) for n in strata) + " |\n"
    strata_md = md_table(st.round(3)) if len(st) else ""
    md = f"""# SanntiS training run report

## Run metadata

| Field | Value |
|---|---|
{meta_rows}
| Epochs (median CV best epoch; per run) | {tune['epochs']} ({best_epochs}) |
| Operating point | {op_point} |

## Detection, validation (pooled out-of-fold, `mc` splits)

{hdr}{body}
![metric set](figures/validation_bars.png)

![operating point](figures/operating_point.png)

Per class (BGC precision of the regions the RF post-filter assigns to the class; class-matched: the region overlaps
a true BGC of that class):

![per-class operating curves](figures/validation_class_curves.png)

{radars_md}

Strata (lineage):

{strata_md}
{npc_md}{kd_md}
Tables: `validation.tsv`, `validation_strata.tsv`, `validation_class_curves.tsv`, `sweep.tsv`{npc_tsv}.
"""
    (out / "report.md").write_text(md)
    return out
