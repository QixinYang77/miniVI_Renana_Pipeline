from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas.errors import EmptyDataError
from scipy import stats


DEFAULT_STYLE_OPTS: dict[str, Any] = {
    "font_family": "Arial",
    "font_size": 6.0,
    "axes_labelsize": 6.0,
    "axes_titlesize": 6.0,
    "tick_labelsize": 5.0,
    "legend_fontsize": 5.0,
    "axes_linewidth": 0.5,
    "remove_top_right_spines": True,
    "show_p_after_sig": True,
    "show_only_significant": True,
}

DEFAULT_STATS_OPTS: dict[str, Any] = {
    "multiple_comparison_panel": "ss_cs_rate_merged",
    "correction_method": "bh_fdr",
}

DEFAULT_ROW_LABELS: dict[str, str] = {
    "primary": "Inside Primary PF",
    "secondary": "Inside Secondary PF",
    "combined": "Inside Combined PF",
    "outside_combined": "Outside Combined PF",
    "all_bins": "All bins",
}

DEFAULT_GROUP_SPECS: list[dict[str, Any]] = [
    {
        "id": "csplus_eb_tuned_pass100",
        "title": "CS+ PF split statistics (EB tuned pass100)",
        "category_in": ["CSplus"],
        "is_place_cell": True,
        "selected_in_pass_any_folder": True,
        "regions": ["primary", "secondary", "combined", "outside_combined", "all_bins"],
    },
    {
        "id": "csplus_all",
        "title": "CS+ PF split statistics (all place cells)",
        "category_in": ["CSplus"],
        "is_place_cell": True,
        "regions": ["primary", "secondary", "combined", "outside_combined", "all_bins"],
    },
    {
        "id": "csminus_all",
        "title": "CS- PF split statistics (all place cells)",
        "category_in": ["CSminus"],
        "is_place_cell": True,
        "regions": ["primary", "secondary", "combined", "outside_combined", "all_bins"],
        "mrl_panel_mode": "ss_only",
    },
    {
        "id": "nonplc_eb_tuned_pass100",
        "title": "Non-place cells PF split statistics (EB tuned)",
        "category_in": ["all-nonPLC"],
        "is_place_cell": False,
        "selected_in_pass_any_folder": True,
        "regions": ["all_bins"],
    },
]


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.copy()
    as_str = series.astype(str).str.strip().str.lower()
    return as_str.isin({"1", "true", "t", "yes", "y"})


def _paired_test_auto(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 3:
        return np.nan, np.nan, "n<3", n, np.nan
    d = x - y
    if np.allclose(d, 0.0, atol=1e-12, rtol=0.0):
        return 1.0, 0.0, "all_equal", n, np.nan
    shapiro_p = np.nan
    try:
        if 3 <= n <= 5000:
            shapiro_p = float(stats.shapiro(d).pvalue)
    except Exception:
        shapiro_p = np.nan
    if np.isfinite(shapiro_p) and shapiro_p >= 0.05:
        res = stats.ttest_rel(x, y, nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "paired t-test", n, float(shapiro_p)
    try:
        res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue), float(res.statistic), "wilcoxon", n, float(shapiro_p)
    except Exception:
        res = stats.ttest_rel(x, y, nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "paired t-test(fallback)", n, float(shapiro_p)


def _onesample_test_auto(x, mu=0.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return np.nan, np.nan, "n<3", n, np.nan
    d = x - float(mu)
    if np.allclose(d, 0.0, atol=1e-12, rtol=0.0):
        return 1.0, 0.0, "all_equal", n, np.nan
    shapiro_p = np.nan
    try:
        if 3 <= n <= 5000:
            shapiro_p = float(stats.shapiro(d).pvalue)
    except Exception:
        shapiro_p = np.nan
    if np.isfinite(shapiro_p) and shapiro_p >= 0.05:
        res = stats.ttest_1samp(x, popmean=float(mu), nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "one-sample t-test", n, float(shapiro_p)
    try:
        res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue), float(res.statistic), "wilcoxon(1-sample)", n, float(shapiro_p)
    except Exception:
        res = stats.ttest_1samp(x, popmean=float(mu), nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "one-sample t-test(fallback)", n, float(shapiro_p)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    m = int(pv.size)
    if m <= 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_ord = ranked * float(m) / ranks
    q_ord = np.minimum.accumulate(q_ord[::-1])[::-1]
    q_ord = np.clip(q_ord, 0.0, 1.0)
    qv = np.empty(m, dtype=float)
    qv[order] = q_ord
    q[valid] = qv
    return q


def _sig_label(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "p=nan"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.3g}"


def _sig_text(sig: str, p_raw: float, *, show_p: bool) -> str:
    if not bool(show_p):
        return str(sig)
    return f"{sig} ({_format_p_value(float(p_raw))})"


def _draw_bracket(ax, x1, x2, y, h, text, tick_labelsize: float):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.5, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=tick_labelsize, color="black")


def _nonpaired_test_auto(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n = int(min(x.size, y.size))
    if x.size < 3 or y.size < 3:
        return np.nan, np.nan, "n<3", n, np.nan
    shapiro_px = np.nan
    shapiro_py = np.nan
    try:
        if 3 <= x.size <= 5000:
            shapiro_px = float(stats.shapiro(x).pvalue)
    except Exception:
        shapiro_px = np.nan
    try:
        if 3 <= y.size <= 5000:
            shapiro_py = float(stats.shapiro(y).pvalue)
    except Exception:
        shapiro_py = np.nan
    if np.isfinite(shapiro_px) and np.isfinite(shapiro_py) and shapiro_px >= 0.05 and shapiro_py >= 0.05:
        res = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "welch t-test", int(n), float(min(shapiro_px, shapiro_py))
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue), float(res.statistic), "mann-whitney", int(n), float(min(shapiro_px, shapiro_py))
    except Exception:
        res = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(res.pvalue), float(res.statistic), "welch t-test(fallback)", int(n), float(min(shapiro_px, shapiro_py))


def _metric_pairs(region_df: pd.DataFrame, metric: str, id_cols: list[str]):
    sub = region_df.loc[
        region_df["metric"].astype(str) == str(metric),
        id_cols + ["preferred_mean", "nonpreferred_mean"],
    ].copy()
    sub = sub.drop_duplicates(subset=id_cols).sort_values(id_cols).reset_index(drop=True)
    return sub, sub["preferred_mean"].to_numpy(dtype=float), sub["nonpreferred_mean"].to_numpy(dtype=float)


def _metric_vals(df: pd.DataFrame, metric: str, side: str, id_cols: list[str]) -> np.ndarray:
    key = "preferred_mean" if str(side) == "pref" else "nonpreferred_mean"
    sub = df.loc[df["metric"].astype(str) == str(metric), id_cols + [key]].copy()
    sub = sub.drop_duplicates(subset=id_cols).sort_values(id_cols).reset_index(drop=True)
    vals = pd.to_numeric(sub[key], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _metric_vals_first_available(df: pd.DataFrame, metrics: list[str], side: str, id_cols: list[str]) -> np.ndarray:
    for metric in list(metrics):
        vals = _metric_vals(df, str(metric), side, id_cols)
        if int(vals.size) > 0:
            return vals
    return np.array([], dtype=float)


def _build_merged_4groups(region_df: pd.DataFrame, metric_ss: str, metric_cs: str, id_cols: list[str]) -> pd.DataFrame:
    ss_sub = region_df.loc[
        region_df["metric"].astype(str) == str(metric_ss),
        id_cols + ["preferred_mean", "nonpreferred_mean"],
    ].copy()
    cs_sub = region_df.loc[
        region_df["metric"].astype(str) == str(metric_cs),
        id_cols + ["preferred_mean", "nonpreferred_mean"],
    ].copy()
    ss_sub = ss_sub.drop_duplicates(subset=id_cols).rename(columns={"preferred_mean": "ss_pref", "nonpreferred_mean": "ss_nonpref"})
    cs_sub = cs_sub.drop_duplicates(subset=id_cols).rename(columns={"preferred_mean": "cs_pref", "nonpreferred_mean": "cs_nonpref"})
    merged = ss_sub.merge(cs_sub, on=id_cols, how="inner")
    finite_all4 = (
        np.isfinite(merged["ss_pref"].to_numpy(dtype=float))
        & np.isfinite(merged["ss_nonpref"].to_numpy(dtype=float))
        & np.isfinite(merged["cs_pref"].to_numpy(dtype=float))
        & np.isfinite(merged["cs_nonpref"].to_numpy(dtype=float))
    )
    return merged.loc[finite_all4].copy()


def _finalize_axis(ax, style: dict[str, Any]):
    ax.tick_params(labelsize=float(style["tick_labelsize"]), length=1.75, direction="in")
    ax.grid(False)
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_linewidth(float(style["axes_linewidth"]))
    if bool(style["remove_top_right_spines"]):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _draw_boxplot(ax, groups, positions, colors):
    clean = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))] for g in groups]
    if not any(int(g.size) > 0 for g in clean):
        return clean
    bp = ax.boxplot(
        clean,
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.6},
        whiskerprops={"color": "0.4", "linewidth": 0.5},
        capprops={"color": "0.4", "linewidth": 0.5},
        boxprops={"linewidth": 0.5, "color": "0.4"},
    )
    for patch, col in zip(bp["boxes"], list(colors)):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.28)
    return clean


def _overlay_pair_lines_points(ax, y1, y2, x1, x2, c1, c2):
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    good = np.isfinite(y1) & np.isfinite(y2)
    y1 = y1[good]
    y2 = y2[good]
    n = int(y1.size)
    if n <= 0:
        return 0
    for i in range(n):
        ax.plot([x1, x2], [y1[i], y2[i]], color="0.78", lw=0.45, alpha=0.85, zorder=2)
    ax.scatter(np.full(n, x1), y1, s=10, c=c1, edgecolors="none", alpha=0.95, zorder=3)
    ax.scatter(np.full(n, x2), y2, s=10, c=c2, edgecolors="none", alpha=0.95, zorder=3)
    return n


def _plot_region_row(
    region_df: pd.DataFrame,
    axes_row: np.ndarray,
    tests: list[dict[str, Any]],
    panel_vals: dict[tuple[str, str], np.ndarray],
    row_region: str,
    row_label: str,
    id_cols: list[str],
    style: dict[str, Any],
    show_titles: bool,
    mrl_panel_mode: str,
):
    x_pos = {"ss_pref": 0.0, "ss_nonpref": 1.0, "cs_pref": 2.0, "cs_nonpref": 3.0}

    def _maybe_title(ax, txt):
        if bool(show_titles):
            ax.set_title(txt, fontsize=float(style["axes_titlesize"]))

    ax = axes_row[0]
    _, pref, nonpref = _metric_pairs(region_df, "all", id_cols)
    valid = np.isfinite(pref) & np.isfinite(nonpref)
    pref_v = pref[valid]
    nonpref_v = nonpref[valid]
    n = int(pref_v.size)
    _draw_boxplot(ax, [pref_v, nonpref_v], [0, 1], ["#1F77B4", "#D62728"])
    _overlay_pair_lines_points(ax, pref_v, nonpref_v, 0.0, 1.0, "#1F77B4", "#D62728")
    panel_vals[(row_region, "all")] = np.concatenate([pref_v, nonpref_v]) if n > 0 else np.array([], dtype=float)
    p, stat, test_name, n_pair, shapiro_p = _paired_test_auto(pref_v, nonpref_v)
    tests.append({"region": row_region, "panel": "all", "comparison": "all_pref_vs_nonpref", "n": int(n_pair), "p_raw": p, "statistic": stat, "test": test_name, "shapiro_p": shapiro_p, "x1": 0.0, "x2": 1.0, "level": 0})
    _maybe_title(ax, f"All rate (n={n})")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["PD", "NPD"], fontsize=float(style["tick_labelsize"]))
    ax.set_ylabel("Rate, all (Hz)", fontsize=float(style["axes_labelsize"]))
    _finalize_axis(ax, style)

    ax = axes_row[1]
    merged_rate = _build_merged_4groups(region_df, "ss", "cs", id_cols)
    vals_rate = {k: merged_rate[k].to_numpy(dtype=float) if k in merged_rate.columns else np.array([], dtype=float) for k in x_pos.keys()}
    n_merge_rate = int(len(merged_rate))
    _draw_boxplot(ax, [vals_rate["ss_pref"], vals_rate["ss_nonpref"], vals_rate["cs_pref"], vals_rate["cs_nonpref"]], [x_pos["ss_pref"], x_pos["ss_nonpref"], x_pos["cs_pref"], x_pos["cs_nonpref"]], ["#026C80", "#026C80", "#EE9B00", "#EE9B00"])
    if n_merge_rate > 0:
        _overlay_pair_lines_points(ax, vals_rate["ss_pref"], vals_rate["ss_nonpref"], x_pos["ss_pref"], x_pos["ss_nonpref"], "#026C80", "#026C80")
        _overlay_pair_lines_points(ax, vals_rate["cs_pref"], vals_rate["cs_nonpref"], x_pos["cs_pref"], x_pos["cs_nonpref"], "#EE9B00", "#EE9B00")
    panel_vals[(row_region, "ss_cs_rate_merged")] = np.concatenate([vals_rate[k] for k in ("ss_pref", "ss_nonpref", "cs_pref", "cs_nonpref")]) if n_merge_rate > 0 else np.array([], dtype=float)
    for cmp_name, left_key, right_key, lvl in [("ss_pref_vs_ss_nonpref", "ss_pref", "ss_nonpref", 0), ("cs_pref_vs_cs_nonpref", "cs_pref", "cs_nonpref", 0), ("ss_pref_vs_cs_pref", "ss_pref", "cs_pref", 1), ("ss_nonpref_vs_cs_nonpref", "ss_nonpref", "cs_nonpref", 2)]:
        p, stat, test_name, n_pair, shapiro_p = _paired_test_auto(vals_rate[left_key], vals_rate[right_key])
        tests.append({"region": row_region, "panel": "ss_cs_rate_merged", "comparison": cmp_name, "n": int(n_pair), "p_raw": p, "statistic": stat, "test": test_name, "shapiro_p": shapiro_p, "x1": float(x_pos[left_key]), "x2": float(x_pos[right_key]), "level": int(lvl)})
    _maybe_title(ax, f"SS+CS rate merged (n={n_merge_rate})")
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["PD", "NPD", "PD", "NPD"], fontsize=float(style["tick_labelsize"]))
    if bool(show_titles):
        handles = [Patch(facecolor="#026C80", edgecolor="#026C80", alpha=0.35, label="SS"), Patch(facecolor="#EE9B00", edgecolor="#EE9B00", alpha=0.35, label="CS")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.92, 1.0), frameon=False, fontsize=float(style["legend_fontsize"]), handlelength=1.0, borderaxespad=0.0)
    ax.set_ylabel("Rate, SS/CS (Hz)", fontsize=float(style["axes_labelsize"]))
    _finalize_axis(ax, style)

    for panel_idx, metric, ylabel in [(2, "theta", r"$\theta$ amp. (spkh.)"), (3, "slow", "Slow Vm (spkh.)"), (7, "occupancy_time", "Occupancy (s)"), (8, "speed", "Speed (cm/s)")]:
        ax = axes_row[panel_idx]
        _, pref, nonpref = _metric_pairs(region_df, metric, id_cols)
        valid = np.isfinite(pref) & np.isfinite(nonpref)
        pref_v = pref[valid]; nonpref_v = nonpref[valid]
        n = int(pref_v.size)
        _draw_boxplot(ax, [pref_v, nonpref_v], [0, 1], ["#1F77B4", "#D62728"])
        _overlay_pair_lines_points(ax, pref_v, nonpref_v, 0.0, 1.0, "#1F77B4", "#D62728")
        panel_key = "theta" if metric == "theta" else ("slow" if metric == "slow" else ("occupancy" if metric == "occupancy_time" else "speed"))
        panel_vals[(row_region, panel_key)] = np.concatenate([pref_v, nonpref_v]) if n > 0 else np.array([], dtype=float)
        p, stat, test_name, n_pair, shapiro_p = _paired_test_auto(pref_v, nonpref_v)
        tests.append({"region": row_region, "panel": panel_key, "comparison": f"{panel_key}_pref_vs_nonpref", "n": int(n_pair), "p_raw": p, "statistic": stat, "test": test_name, "shapiro_p": shapiro_p, "x1": 0.0, "x2": 1.0, "level": 0})
        title_map = {"theta": "Theta amp", "slow": "Slow Vm", "occupancy_time": "Occupancy time", "speed": "Speed"}
        _maybe_title(ax, f"{title_map[metric]} (n={n})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PD", "NPD"], fontsize=float(style["tick_labelsize"]))
        ax.set_ylabel(ylabel, fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)

    ax = axes_row[4]
    if str(mrl_panel_mode).strip().lower() == "ss_only":
        ss_mrl_df = (
            region_df.loc[region_df["metric"].astype(str) == "ss_mrl_overall", id_cols + ["preferred_mean"]]
            .copy()
            .drop_duplicates(subset=id_cols)
            .rename(columns={"preferred_mean": "ss_mrl"})
        )
        ss_vals = ss_mrl_df["ss_mrl"].to_numpy(dtype=float) if "ss_mrl" in ss_mrl_df.columns else np.array([], dtype=float)
        ss_vals = ss_vals[np.isfinite(ss_vals)]
        n_ss = int(ss_vals.size)
        _draw_boxplot(ax, [ss_vals], [0], ["#026C80"])
        if n_ss > 0:
            jitter = np.linspace(-0.08, 0.08, n_ss) if n_ss > 1 else np.array([0.0], dtype=float)
            ax.scatter(jitter, ss_vals, s=10, c="#026C80", edgecolors="none", alpha=0.95, zorder=3)
        panel_vals[(row_region, "ss_cs_mrl")] = np.asarray(ss_vals, dtype=float)
        _maybe_title(ax, f"SS MRL (n={n_ss})")
        ax.set_xticks([0]); ax.set_xticklabels(["SS"], fontsize=float(style["tick_labelsize"]))
    else:
        ss_mrl_df = region_df.loc[region_df["metric"].astype(str) == "ss_mrl_overall", id_cols + ["preferred_mean"]].copy().drop_duplicates(subset=id_cols).rename(columns={"preferred_mean": "ss_mrl"})
        cs_mrl_df = region_df.loc[region_df["metric"].astype(str) == "cs_mrl_overall", id_cols + ["preferred_mean"]].copy().drop_duplicates(subset=id_cols).rename(columns={"preferred_mean": "cs_mrl"})
        merged_mrl = ss_mrl_df.merge(cs_mrl_df, on=id_cols, how="inner")
        ss_vals = merged_mrl["ss_mrl"].to_numpy(dtype=float) if "ss_mrl" in merged_mrl.columns else np.array([], dtype=float)
        cs_vals = merged_mrl["cs_mrl"].to_numpy(dtype=float) if "cs_mrl" in merged_mrl.columns else np.array([], dtype=float)
        valid = np.isfinite(ss_vals) & np.isfinite(cs_vals)
        ss_vals = ss_vals[valid]; cs_vals = cs_vals[valid]
        n_merge_mrl = int(ss_vals.size)
        _draw_boxplot(ax, [ss_vals, cs_vals], [0, 1], ["#026C80", "#EE9B00"])
        _overlay_pair_lines_points(ax, ss_vals, cs_vals, 0.0, 1.0, "#026C80", "#EE9B00")
        panel_vals[(row_region, "ss_cs_mrl")] = np.concatenate([ss_vals, cs_vals]) if n_merge_mrl > 0 else np.array([], dtype=float)
        p, stat, test_name, n_pair, shapiro_p = _paired_test_auto(ss_vals, cs_vals)
        tests.append({"region": row_region, "panel": "ss_cs_mrl", "comparison": "ss_mrl_overall_vs_cs_mrl_overall", "n": int(n_pair), "p_raw": p, "statistic": stat, "test": test_name, "shapiro_p": shapiro_p, "x1": 0.0, "x2": 1.0, "level": 0})
        _maybe_title(ax, f"SS vs CS MRL (n={n_merge_mrl})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["SS", "CS"], fontsize=float(style["tick_labelsize"]))
    ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
    _finalize_axis(ax, style)

    for panel_idx, metric_name, panel_key, panel_title in [
        (5, "theta_mrl_overall", "theta_mrl", "Theta MRL"),
        (6, "slow_mrl_overall", "slow_mrl", "Slow Vm MRL"),
    ]:
        ax = axes_row[panel_idx]
        vals = _metric_vals(region_df, metric_name, "pref", id_cols)
        vals = vals[np.isfinite(vals)]
        n_vals = int(vals.size)
        _draw_boxplot(ax, [vals], [0], ["#7A7A7A"])
        if n_vals > 0:
            jitter = np.linspace(-0.08, 0.08, n_vals) if n_vals > 1 else np.array([0.0], dtype=float)
            ax.scatter(jitter, vals, s=10, c="#7A7A7A", edgecolors="none", alpha=0.95, zorder=3)
        panel_vals[(row_region, panel_key)] = np.asarray(vals, dtype=float)
        p, stat, test_name, n_one, shapiro_p = _onesample_test_auto(vals, mu=0.0)
        tests.append({
            "region": row_region,
            "panel": panel_key,
            "comparison": f"{metric_name}_vs_0",
            "n": int(n_one),
            "p_raw": p,
            "statistic": stat,
            "test": test_name,
            "shapiro_p": shapiro_p,
            "x1": -0.12,
            "x2": 0.12,
            "level": 0,
        })
        _maybe_title(ax, f"{panel_title} (n={n_vals})")
        ax.set_xticks([0]); ax.set_xticklabels(["All bins"], fontsize=float(style["tick_labelsize"]))
        ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)

    # Panel 8: HDP-HD corr (no stats requested)
    ax = axes_row[9]
    corr_sub = region_df.loc[region_df["metric"].astype(str) == "hd_pref_neural_vs_behavior", id_cols + ["preferred_mean"]].copy()
    corr_sub = corr_sub.drop_duplicates(subset=id_cols).sort_values(id_cols).reset_index(drop=True)
    corr_vals = pd.to_numeric(corr_sub["preferred_mean"], errors="coerce").to_numpy(dtype=float)
    corr_vals = corr_vals[np.isfinite(corr_vals)]
    n_corr = int(corr_vals.size)
    mean_corr = float(np.nanmean(corr_vals)) if n_corr > 0 else np.nan
    sem_corr = float(np.nanstd(corr_vals, ddof=1) / np.sqrt(n_corr)) if n_corr > 1 else 0.0
    if np.isfinite(mean_corr):
        ax.bar([0], [mean_corr], width=0.42, color="#2A9D8F", edgecolor="#2A9D8F", alpha=0.5, zorder=2, linewidth=0.5)
        ax.errorbar([0], [mean_corr], yerr=[sem_corr], fmt="none", ecolor="black", elinewidth=0.6, capsize=2, capthick=0.6, zorder=4)
    if n_corr > 0:
        jitter = np.linspace(-0.06, 0.06, n_corr) if n_corr > 1 else np.array([0.0], dtype=float)
        ax.scatter(jitter, corr_vals, s=10, c="#2A9D8F", edgecolors="none", alpha=0.95, zorder=3)
    ax.axhline(0.0, color="0.5", lw=0.5, ls="--", alpha=0.8)
    panel_vals[(row_region, "hdp_hd_corr")] = np.asarray(corr_vals, dtype=float)
    _maybe_title(ax, f"HDP-HD corr (n={n_corr})")
    ax.set_xticks([0]); ax.set_xticklabels(["All bins"], fontsize=float(style["tick_labelsize"]))
    ax.set_ylabel("corr", fontsize=float(style["axes_labelsize"]))
    _finalize_axis(ax, style)

    for panel_idx, metric_name, panel_key, panel_title, xlab in [
        (10, "eb_empfit_corr_all", "eb_empfit_corr_all", "EB emp-fit corr (All)", "All bins"),
        (11, "eb_empfit_corr_ss", "eb_empfit_corr_ss", "EB emp-fit corr (SS)", "SS bins"),
        (12, "eb_empfit_corr_cs", "eb_empfit_corr_cs", "EB emp-fit corr (CS)", "CS bins"),
    ]:
        ax = axes_row[panel_idx]
        vals = _metric_vals(region_df, metric_name, "pref", id_cols)
        vals = vals[np.isfinite(vals)]
        n_vals = int(vals.size)
        _draw_boxplot(ax, [vals], [0], ["#6A3D9A"])
        if n_vals > 0:
            jitter = np.linspace(-0.08, 0.08, n_vals) if n_vals > 1 else np.array([0.0], dtype=float)
            ax.scatter(jitter, vals, s=10, c="#6A3D9A", edgecolors="none", alpha=0.95, zorder=3)
        panel_vals[(row_region, panel_key)] = np.asarray(vals, dtype=float)
        p, stat, test_name, n_one, shapiro_p = _onesample_test_auto(vals, mu=0.0)
        tests.append({
            "region": row_region,
            "panel": panel_key,
            "comparison": f"{metric_name}_vs_0",
            "n": int(n_one),
            "p_raw": p,
            "statistic": stat,
            "test": test_name,
            "shapiro_p": shapiro_p,
            "x1": -0.12,
            "x2": 0.12,
            "level": 0,
        })
        _maybe_title(ax, f"{panel_title} (n={n_vals})")
        ax.set_xticks([0]); ax.set_xticklabels([xlab], fontsize=float(style["tick_labelsize"]))
        ax.set_ylabel("corr", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)

    axes_row[0].text(-0.60, 0.5, str(row_label), transform=axes_row[0].transAxes, rotation=90, va="center", ha="center", fontsize=float(style["axes_labelsize"]), fontweight="bold")


def _apply_style(style: dict[str, Any]):
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update({
        "font.family": str(style["font_family"]),
        "font.size": float(style["font_size"]),
        "axes.labelsize": float(style["axes_labelsize"]),
        "axes.titlesize": float(style["axes_titlesize"]),
        "xtick.labelsize": float(style["tick_labelsize"]),
        "ytick.labelsize": float(style["tick_labelsize"]),
        "xtick.major.size": 1.75,
        "ytick.major.size": 1.75,
        "xtick.minor.size": 1.0,
        "ytick.minor.size": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": float(style["legend_fontsize"]),
        "axes.linewidth": float(style["axes_linewidth"]),
    })


def plot_pf_split_combined_csplus_vs_csminus_row3(
    *,
    pf_split_csv: str | Path,
    out_dir: str | Path,
    any_pass_threshold: int,
    any_pass_suffix: str,
    region: str = "combined",
    regions: list[str] | None = None,
    csplus_group: str = "eb_tuned",
    style_opts: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    csplus_group_norm = str(csplus_group).strip().lower()
    if csplus_group_norm in {"both", "all_and_eb_tuned", "eb_tuned_and_all"}:
        out: dict[str, Any] = {"results": {}}
        for mode in ("eb_tuned", "all"):
            out["results"][mode] = plot_pf_split_combined_csplus_vs_csminus_row3(
                pf_split_csv=pf_split_csv,
                out_dir=out_dir,
                any_pass_threshold=any_pass_threshold,
                any_pass_suffix=any_pass_suffix,
                region=region,
                regions=regions,
                csplus_group=mode,
                style_opts=style_opts,
                show=show,
            )
        return out
    if csplus_group_norm not in {"eb_tuned", "all"}:
        raise ValueError("csplus_group must be one of {'eb_tuned', 'all', 'both'}")

    style = dict(DEFAULT_STYLE_OPTS)
    user_style = style_opts if isinstance(style_opts, dict) else {}
    if user_style:
        style.update(user_style)
    if "show_p_after_sig" not in user_style:
        style["show_p_after_sig"] = False
    if "show_only_significant" not in user_style:
        style["show_only_significant"] = True
    _apply_style(style)

    pf_split_csv = Path(pf_split_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pf_split_csv.exists():
        raise FileNotFoundError(f"Missing PF split stats CSV: {pf_split_csv}")
    try:
        pf_df = pd.read_csv(pf_split_csv)
    except EmptyDataError as exc:
        raise RuntimeError(f"PF split stats CSV is empty: {pf_split_csv}") from exc
    if pf_df.empty:
        raise RuntimeError(f"PF split stats CSV has no rows: {pf_split_csv}")

    required_cols = [
        "animal_id", "cell_idx", "cell_num", "category", "region", "metric",
        "preferred_mean", "nonpreferred_mean", "is_place_cell", "selected_in_pass_any_folder",
    ]
    missing = [c for c in required_cols if c not in pf_df.columns]
    if missing:
        raise KeyError(f"Missing required PF split stats columns: {missing}")
    for c in ["cell_idx", "cell_num", "preferred_mean", "nonpreferred_mean"]:
        pf_df[c] = pd.to_numeric(pf_df[c], errors="coerce")
    pf_df["is_place_cell"] = _coerce_bool_series(pf_df["is_place_cell"])
    pf_df["selected_in_pass_any_folder"] = _coerce_bool_series(pf_df["selected_in_pass_any_folder"])

    id_cols = ["animal_id", "cell_idx", "cell_num"]
    region_keys = [str(r).strip() for r in (regions if isinstance(regions, list) and len(regions) > 0 else [region])]
    region_keys = [r for r in region_keys if len(r) > 0]
    if len(region_keys) <= 0:
        region_keys = ["combined"]

    row_payloads: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    csplus_requires_selected = bool(csplus_group_norm == "eb_tuned")
    for region_key in region_keys:
        csplus_mask = (
            (pf_df["category"].astype(str) == "CSplus")
            & (pf_df["is_place_cell"] == True)
            & (pf_df["region"].astype(str) == region_key)
        )
        if csplus_requires_selected:
            csplus_mask = csplus_mask & (pf_df["selected_in_pass_any_folder"] == True)
        csplus_df = pf_df.loc[csplus_mask].copy()
        csminus_df = pf_df.loc[
            (pf_df["category"].astype(str) == "CSminus")
            & (pf_df["is_place_cell"] == True)
            & (pf_df["region"].astype(str) == region_key)
        ].copy()
        if csplus_df.empty and csminus_df.empty:
            continue
        row_payloads.append((region_key, csplus_df, csminus_df))

    if len(row_payloads) <= 0:
        raise RuntimeError(f"No rows available for requested regions: {region_keys}")

    left_bg = "#FFF3E0"
    right_bg = "#E3F2FD"
    csplus_color = "#EE9B00"
    csminus_color = "#026C80"
    pref_color = "#1F77B4"
    nonpref_color = "#D62728"
    ss_color = "#026C80"
    cs_color = "#EE9B00"
    mrl_all_color = "#7A7A7A"

    n_rows = len(row_payloads)
    # Error-proof sizing: subplot width is determined by number of plotted groups.
    # Example: 4 groups * 0.2 in = 0.8 in panel width.
    group_width_in = float(style.get("group_width_in", 0.2))
    if (not np.isfinite(group_width_in)) or group_width_in <= 0:
        group_width_in = 0.2
    col_group_counts = [4, 6, 4, 4, 4, 2, 2, 2, 2, 2]
    width_ratios = [float(v) for v in col_group_counts]
    fig_w = float(group_width_in * float(np.sum(width_ratios)))
    fig_h = float(1.0 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        10,
        figsize=(fig_w, fig_h),
        dpi=180,
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    tests: list[dict[str, Any]] = []

    for ridx, (region_key, csplus_df, csminus_df) in enumerate(row_payloads):
        row_axes = axes[ridx, :]
        def _shade_split_background(
            ax: Any,
            left_xmin: float,
            left_xmax: float,
            right_xmin: float,
            right_xmax: float,
        ) -> None:
            ax.set_xlim(float(left_xmin), float(right_xmax))
            ax.axvspan(float(left_xmin), float(left_xmax), facecolor=left_bg, edgecolor="none", alpha=0.8, zorder=0)
            ax.axvspan(float(right_xmin), float(right_xmax), facecolor=right_bg, edgecolor="none", alpha=0.8, zorder=0)

        def _plot_pref_nonpref_column(ax, metric: str, title: str, ylabel: str):
            _, lpref, lnon = _metric_pairs(csplus_df, metric, id_cols)
            _, rpref, rnon = _metric_pairs(csminus_df, metric, id_cols)
            lmask = np.isfinite(lpref) & np.isfinite(lnon)
            rmask = np.isfinite(rpref) & np.isfinite(rnon)
            lpref = lpref[lmask]; lnon = lnon[lmask]
            rpref = rpref[rmask]; rnon = rnon[rmask]

            _draw_boxplot(ax, [lpref, lnon], [0.0, 1.0], [pref_color, nonpref_color])
            _draw_boxplot(ax, [rpref, rnon], [2.0, 3.0], [pref_color, nonpref_color])
            _overlay_pair_lines_points(ax, lpref, lnon, 0.0, 1.0, pref_color, nonpref_color)
            _overlay_pair_lines_points(ax, rpref, rnon, 2.0, 3.0, pref_color, nonpref_color)

            pL, sL, tL, nL, shpL = _paired_test_auto(lpref, lnon)
            tests.append({"region": region_key, "panel": metric, "side": "csplus", "comparison": f"{metric}_pref_vs_nonpref", "p_raw": pL, "test": tL, "statistic": sL, "n": int(nL), "x1": 0.0, "x2": 1.0, "level": 0, "shapiro_p": shpL})
            pR, sR, tR, nR, shpR = _paired_test_auto(rpref, rnon)
            tests.append({"region": region_key, "panel": metric, "side": "csminus", "comparison": f"{metric}_pref_vs_nonpref", "p_raw": pR, "test": tR, "statistic": sR, "n": int(nR), "x1": 2.0, "x2": 3.0, "level": 0, "shapiro_p": shpR})
            pPP, sPP, tPP, nPP, shpPP = _nonpaired_test_auto(lpref, rpref)
            tests.append({"region": region_key, "panel": metric, "side": "cross_group", "comparison": f"{metric}_pd_csplus_vs_pd_csminus", "p_raw": pPP, "test": tPP, "statistic": sPP, "n": int(nPP), "x1": 0.0, "x2": 2.0, "level": 1, "shapiro_p": shpPP})
            pNN, sNN, tNN, nNN, shpNN = _nonpaired_test_auto(lnon, rnon)
            tests.append({"region": region_key, "panel": metric, "side": "cross_group", "comparison": f"{metric}_npd_csplus_vs_npd_csminus", "p_raw": pNN, "test": tNN, "statistic": sNN, "n": int(nNN), "x1": 1.0, "x2": 3.0, "level": 2, "shapiro_p": shpNN})

            ax.set_xticks([0.0, 1.0, 2.0, 3.0])
            ax.set_xticklabels(["PD", "NPD", "PD", "NPD"], fontsize=float(style["tick_labelsize"]))
            ax.set_ylabel(ylabel, fontsize=float(style["axes_labelsize"]))
            if ridx == 0:
                ax.set_title(title, fontsize=float(style["axes_titlesize"]))
            _finalize_axis(ax, style)
            _shade_split_background(ax, -0.5, 1.5, 1.5, 3.5)

        _plot_pref_nonpref_column(row_axes[0], "all", "All rate", "Rate, all (Hz)")
        _plot_pref_nonpref_column(row_axes[2], "theta", r"$\theta$ amp.", r"$\theta$ amp. (spkh.)")
        _plot_pref_nonpref_column(row_axes[3], "slow", "Slow Vm", "Slow Vm (spkh.)")

        # Col 2: SS/CS merged (4 comparisons per side)
        ax = row_axes[1]
        left_merged = _build_merged_4groups(csplus_df, "ss", "cs", id_cols)
        lx = {"ss_pref": 0.0, "ss_nonpref": 1.0, "cs_pref": 2.0, "cs_nonpref": 3.0}
        rx = {"ss_pref": 4.0, "ss_nonpref": 5.0}
        lvals = {k: left_merged[k].to_numpy(dtype=float) if k in left_merged.columns else np.array([], dtype=float) for k in lx}
        _, r_ss_pref, r_ss_nonpref = _metric_pairs(csminus_df, "ss", id_cols)
        r_ss_mask = np.isfinite(r_ss_pref) & np.isfinite(r_ss_nonpref)
        rvals = {
            "ss_pref": np.asarray(r_ss_pref[r_ss_mask], dtype=float),
            "ss_nonpref": np.asarray(r_ss_nonpref[r_ss_mask], dtype=float),
        }

        _draw_boxplot(ax, [lvals["ss_pref"], lvals["ss_nonpref"], lvals["cs_pref"], lvals["cs_nonpref"]], [lx["ss_pref"], lx["ss_nonpref"], lx["cs_pref"], lx["cs_nonpref"]], [ss_color, ss_color, cs_color, cs_color])
        _draw_boxplot(ax, [rvals["ss_pref"], rvals["ss_nonpref"]], [rx["ss_pref"], rx["ss_nonpref"]], [ss_color, ss_color])
        _overlay_pair_lines_points(ax, lvals["ss_pref"], lvals["ss_nonpref"], lx["ss_pref"], lx["ss_nonpref"], ss_color, ss_color)
        _overlay_pair_lines_points(ax, lvals["cs_pref"], lvals["cs_nonpref"], lx["cs_pref"], lx["cs_nonpref"], cs_color, cs_color)
        _overlay_pair_lines_points(ax, rvals["ss_pref"], rvals["ss_nonpref"], rx["ss_pref"], rx["ss_nonpref"], ss_color, ss_color)

        cmp_specs = [
            ("ss_pref_vs_ss_nonpref", "ss_pref", "ss_nonpref", 0),
            ("cs_pref_vs_cs_nonpref", "cs_pref", "cs_nonpref", 0),
            ("ss_pref_vs_cs_pref", "ss_pref", "cs_pref", 1),
            ("ss_nonpref_vs_cs_nonpref", "ss_nonpref", "cs_nonpref", 2),
        ]
        for cmp_name, lk, rk, lvl in cmp_specs:
            p, stat, tname, n, shp = _paired_test_auto(lvals[lk], lvals[rk])
            tests.append({"region": region_key, "panel": "ss_cs_rate_merged", "side": "csplus", "comparison": cmp_name, "p_raw": p, "test": tname, "statistic": stat, "n": int(n), "x1": float(lx[lk]), "x2": float(lx[rk]), "level": int(lvl), "shapiro_p": shp})
        p, stat, tname, n, shp = _paired_test_auto(rvals["ss_pref"], rvals["ss_nonpref"])
        tests.append({"region": region_key, "panel": "ss_cs_rate_merged", "side": "csminus", "comparison": "ss_pref_vs_ss_nonpref", "p_raw": p, "test": tname, "statistic": stat, "n": int(n), "x1": float(rx["ss_pref"]), "x2": float(rx["ss_nonpref"]), "level": 0, "shapiro_p": shp})

        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["PD", "NPD", "PD", "NPD", "PD", "NPD"], fontsize=float(style["tick_labelsize"]))
        if ridx == 0:
            ax.set_title("SS/CS rate merged", fontsize=float(style["axes_titlesize"]))
            legend_handles = [
                Patch(facecolor=ss_color, edgecolor="none", alpha=0.5, label="SS"),
                Patch(facecolor=cs_color, edgecolor="none", alpha=0.5, label="CS"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=2,
                frameon=False,
                handlelength=1.0,
                columnspacing=0.9,
                fontsize=float(style["legend_fontsize"]),
            )
        ax.set_ylabel("Rate, SS/CS (Hz)", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)
        _shade_split_background(ax, -0.5, 3.5, 3.5, 5.5)

        # Col 5: MRL
        ax = row_axes[4]
        l_all = _metric_vals_first_available(csplus_df, ["all_mrl_overall", "ss_mrl_overall"], "pref", id_cols)
        csplus_ss_mrl_df = (
            csplus_df.loc[csplus_df["metric"].astype(str) == "ss_mrl_overall", id_cols + ["preferred_mean"]]
            .copy()
            .drop_duplicates(subset=id_cols)
            .rename(columns={"preferred_mean": "ss_mrl"})
        )
        csplus_cs_mrl_df = (
            csplus_df.loc[csplus_df["metric"].astype(str) == "cs_mrl_overall", id_cols + ["preferred_mean"]]
            .copy()
            .drop_duplicates(subset=id_cols)
            .rename(columns={"preferred_mean": "cs_mrl"})
        )
        csplus_mrl_join = csplus_ss_mrl_df.merge(csplus_cs_mrl_df, on=id_cols, how="inner")
        l_ss = pd.to_numeric(csplus_mrl_join["ss_mrl"], errors="coerce").to_numpy(dtype=float)
        l_cs = pd.to_numeric(csplus_mrl_join["cs_mrl"], errors="coerce").to_numpy(dtype=float)
        lmask = np.isfinite(l_ss) & np.isfinite(l_cs)
        l_ss = l_ss[lmask]
        l_cs = l_cs[lmask]
        r_ss = _metric_vals(csminus_df, "ss_mrl_overall", "pref", id_cols)

        _draw_boxplot(ax, [l_all, l_ss, l_cs], [0.0, 1.0, 2.0], [mrl_all_color, ss_color, cs_color])
        if int(l_all.size) > 0:
            jitter = np.linspace(-0.08, 0.08, int(l_all.size)) if int(l_all.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(0.0 + jitter, l_all, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)
        _overlay_pair_lines_points(ax, l_ss, l_cs, 1.0, 2.0, ss_color, cs_color)
        _draw_boxplot(ax, [r_ss], [3.0], [mrl_all_color])
        if int(r_ss.size) > 0:
            jitter = np.linspace(-0.08, 0.08, int(r_ss.size)) if int(r_ss.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(3.0 + jitter, r_ss, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)

        p, stat, tname, n, shp = _paired_test_auto(l_ss, l_cs)
        tests.append({"region": region_key, "panel": "mrl", "side": "csplus", "comparison": "ss_vs_cs", "p_raw": p, "test": tname, "statistic": stat, "n": int(n), "x1": 1.0, "x2": 2.0, "level": 0, "shapiro_p": shp})
        p, stat, tname, n, shp = _nonpaired_test_auto(l_all, r_ss)
        tests.append({"region": region_key, "panel": "mrl", "side": "cross_group", "comparison": "all_csplus_vs_allss_csminus", "p_raw": p, "test": tname, "statistic": stat, "n": int(n), "x1": 0.0, "x2": 3.0, "level": 1, "shapiro_p": shp})

        ax.set_xticks([0.0, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["All", "SS", "CS", "All/SS"], fontsize=float(style["tick_labelsize"]))
        if ridx == 0:
            ax.set_title("MRL", fontsize=float(style["axes_titlesize"]))
        ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)
        _shade_split_background(ax, -0.5, 2.5, 2.5, 3.5)

        # Col 6/7: Theta/Slow MRL overall (CS+ vs CS- nonpaired)
        for col_idx, metric_name, title_text in [
            (5, "theta_mrl_overall", "Theta MRL"),
            (6, "slow_mrl_overall", "Slow Vm MRL"),
        ]:
            ax = row_axes[col_idx]
            l_vals = _metric_vals(csplus_df, metric_name, "pref", id_cols)
            r_vals = _metric_vals(csminus_df, metric_name, "pref", id_cols)
            l_vals = l_vals[np.isfinite(l_vals)]
            r_vals = r_vals[np.isfinite(r_vals)]
            _draw_boxplot(ax, [l_vals, r_vals], [0.0, 1.0], [csplus_color, csminus_color])
            if int(l_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(l_vals.size)) if int(l_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(0.0 + j, l_vals, s=10, c=csplus_color, edgecolors="none", alpha=0.95, zorder=3)
            if int(r_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(r_vals.size)) if int(r_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(1.0 + j, r_vals, s=10, c=csminus_color, edgecolors="none", alpha=0.95, zorder=3)
            p, stat, tname, n, shp = _nonpaired_test_auto(l_vals, r_vals)
            tests.append({
                "region": region_key,
                "panel": metric_name,
                "side": "cross_group",
                "comparison": f"{metric_name}_csplus_vs_csminus",
                "p_raw": p,
                "test": tname,
                "statistic": stat,
                "n": int(n),
                "x1": 0.0,
                "x2": 1.0,
                "level": 0,
                "shapiro_p": shp,
            })
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(["CS+", "CS-"], fontsize=float(style["tick_labelsize"]))
            if ridx == 0:
                ax.set_title(title_text, fontsize=float(style["axes_titlesize"]))
            ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
            _finalize_axis(ax, style)
            _shade_split_background(ax, -0.5, 0.5, 0.5, 1.5)

        # Col 8/9/10: EB empirical-vs-fitted similarity (CS+ vs CS- nonpaired)
        for col_idx, metric_name, title_text in [
            (7, "eb_empfit_corr_all", "EB emp-fit corr (All)"),
            (8, "eb_empfit_corr_ss", "EB emp-fit corr (SS)"),
            (9, "eb_empfit_corr_cs", "EB emp-fit corr (CS)"),
        ]:
            ax = row_axes[col_idx]
            l_vals = _metric_vals(csplus_df, metric_name, "pref", id_cols)
            r_vals = _metric_vals(csminus_df, metric_name, "pref", id_cols)
            l_vals = l_vals[np.isfinite(l_vals)]
            r_vals = r_vals[np.isfinite(r_vals)]
            _draw_boxplot(ax, [l_vals, r_vals], [0.0, 1.0], [csplus_color, csminus_color])
            if int(l_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(l_vals.size)) if int(l_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(0.0 + j, l_vals, s=10, c=csplus_color, edgecolors="none", alpha=0.95, zorder=3)
            if int(r_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(r_vals.size)) if int(r_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(1.0 + j, r_vals, s=10, c=csminus_color, edgecolors="none", alpha=0.95, zorder=3)
            p, stat, tname, n, shp = _nonpaired_test_auto(l_vals, r_vals)
            tests.append({
                "region": region_key,
                "panel": metric_name,
                "side": "cross_group",
                "comparison": f"{metric_name}_csplus_vs_csminus",
                "p_raw": p,
                "test": tname,
                "statistic": stat,
                "n": int(n),
                "x1": 0.0,
                "x2": 1.0,
                "level": 0,
                "shapiro_p": shp,
            })
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(["CS+", "CS-"], fontsize=float(style["tick_labelsize"]))
            if ridx == 0:
                ax.set_title(title_text, fontsize=float(style["axes_titlesize"]))
            ax.set_ylabel("corr", fontsize=float(style["axes_labelsize"]))
            _finalize_axis(ax, style)
            _shade_split_background(ax, -0.5, 0.5, 0.5, 1.5)

        row_label = DEFAULT_ROW_LABELS.get(region_key, region_key)
        row_axes[0].text(
            -0.40,
            0.5,
            str(row_label),
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=float(style["axes_labelsize"]),
            fontweight="bold",
        )

        # Draw brackets per row
        show_p = bool(style.get("show_p_after_sig", False))
        show_only_sig = bool(style.get("show_only_significant", True))
        for ax, panel in zip(
            row_axes,
            [
                "all",
                "ss_cs_rate_merged",
                "theta",
                "slow",
                "mrl",
                "theta_mrl_overall",
                "slow_mrl_overall",
                "eb_empfit_corr_all",
                "eb_empfit_corr_ss",
                "eb_empfit_corr_cs",
            ],
        ):
            panel_tests = [t for t in tests if str(t["panel"]) == panel and str(t["region"]) == region_key]
            if show_only_sig:
                panel_tests = [t for t in panel_tests if np.isfinite(t["p_raw"]) and float(t["p_raw"]) < 0.05]
            if len(panel_tests) <= 0:
                continue
            vals = []
            for c in ax.collections:
                off = c.get_offsets()
                if hasattr(off, "shape") and off.shape[0] > 0:
                    vals.extend(list(np.asarray(off[:, 1], dtype=float)))
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size <= 0:
                continue
            y_max = float(np.nanmax(vals))
            y_min = float(np.nanmin(vals))
            y_rng = max(1e-9, y_max - y_min)
            y_scale_abs = max(1.0, abs(y_max), abs(y_min))
            base = y_max + float(style.get("bracket_base_factor", 0.14)) * y_rng
            step = max(
                float(style.get("bracket_step_factor", 0.20)) * y_rng,
                float(style.get("bracket_step_min_abs", 0.06)) * y_scale_abs,
            )
            h = max(
                float(style.get("bracket_height_factor", 0.05)) * y_rng,
                float(style.get("bracket_height_min_abs", 0.02)) * y_scale_abs,
            )
            if panel == "ss_cs_rate_merged":
                base = y_max + float(style.get("bracket_base_factor_merged", 0.18)) * y_rng
                step = max(
                    float(style.get("bracket_step_factor_merged", 0.24)) * y_rng,
                    float(style.get("bracket_step_min_abs_merged", 0.08)) * y_scale_abs,
                )
                h = max(
                    float(style.get("bracket_height_factor_merged", 0.06)) * y_rng,
                    float(style.get("bracket_height_min_abs_merged", 0.025)) * y_scale_abs,
                )
            max_lvl = 0
            for t in sorted(panel_tests, key=lambda z: (int(z["level"]), float(z["x1"]), float(z["x2"]))):
                lvl = int(t["level"])
                max_lvl = max(max_lvl, lvl)
                yy = base + lvl * step
                sig = _sig_label(float(t["p_raw"])) if np.isfinite(float(t["p_raw"])) else "n.s."
                txt = _sig_text(sig, float(t["p_raw"]), show_p=show_p)
                _draw_bracket(ax, float(t["x1"]), float(t["x2"]), yy, h, txt, float(style["tick_labelsize"]))
            ax.set_ylim(top=base + (max_lvl + 1.8) * step)

    region_tag = "_".join([str(x) for x, _, _ in row_payloads])
    if csplus_group_norm == "eb_tuned":
        csplus_mode_title = "CS+ EB tuned"
        csplus_mode_tag = "csplus_eb_tuned"
    else:
        csplus_mode_title = "CS+ all"
        csplus_mode_tag = "csplus_all"
    fig.suptitle(
        f"{csplus_mode_title} vs CS- all | threshold={int(any_pass_threshold)}",
        fontsize=float(style["axes_titlesize"]) + 1.0,
        y=1.03,
    )
    fig_path = out_dir / f"pf_split_{csplus_mode_tag}_vs_csminus_{region_tag}_first10_{any_pass_suffix}.svg"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    stats_df = pd.DataFrame(tests, columns=["region", "panel", "side", "comparison", "n", "test", "statistic", "shapiro_p", "p_raw", "x1", "x2", "level"])
    stats_path = out_dir / f"pf_split_{csplus_mode_tag}_vs_csminus_{region_tag}_first10_stats_{any_pass_suffix}.csv"
    stats_df.to_csv(stats_path, index=False)
    return {"figure": fig_path, "stats_csv": stats_path, "csplus_group": csplus_group_norm}


def plot_pf_split_direction_selectivity_csplus_vs_csminus_first4(
    *,
    pf_split_csv: str | Path,
    out_dir: str | Path,
    any_pass_threshold: int,
    any_pass_suffix: str,
    region: str = "combined",
    regions: list[str] | None = None,
    csplus_group: str = "eb_tuned",
    style_opts: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    def _compute_ds(pref_vals: np.ndarray, nonpref_vals: np.ndarray) -> np.ndarray:
        pref_vals = np.asarray(pref_vals, dtype=float)
        nonpref_vals = np.asarray(nonpref_vals, dtype=float)
        den = pref_vals + nonpref_vals
        out = np.full(pref_vals.shape, np.nan, dtype=float)
        ok = np.isfinite(pref_vals) & np.isfinite(nonpref_vals) & np.isfinite(den) & (np.abs(den) > 1e-9)
        if np.any(ok):
            out[ok] = (pref_vals[ok] - nonpref_vals[ok]) / den[ok]
        return out

    def _metric_ds_frame(df: pd.DataFrame, metric_name: str, id_cols: list[str]) -> pd.DataFrame:
        sub, pref, nonpref = _metric_pairs(df, metric_name, id_cols)
        # For slow Vm DS, use magnitude as requested.
        if str(metric_name).strip().lower() == "slow":
            pref = np.abs(np.asarray(pref, dtype=float))
            nonpref = np.abs(np.asarray(nonpref, dtype=float))
        ds = _compute_ds(pref, nonpref)
        out = sub[id_cols].copy()
        out["ds"] = ds
        out = out.loc[np.isfinite(out["ds"].to_numpy(dtype=float))].copy()
        return out

    csplus_group_norm = str(csplus_group).strip().lower()
    if csplus_group_norm in {"both", "all_and_eb_tuned", "eb_tuned_and_all"}:
        out: dict[str, Any] = {"results": {}}
        for mode in ("eb_tuned", "all"):
            out["results"][mode] = plot_pf_split_direction_selectivity_csplus_vs_csminus_first4(
                pf_split_csv=pf_split_csv,
                out_dir=out_dir,
                any_pass_threshold=any_pass_threshold,
                any_pass_suffix=any_pass_suffix,
                region=region,
                regions=regions,
                csplus_group=mode,
                style_opts=style_opts,
                show=show,
            )
        return out
    if csplus_group_norm not in {"eb_tuned", "all"}:
        raise ValueError("csplus_group must be one of {'eb_tuned', 'all', 'both'}")

    style = dict(DEFAULT_STYLE_OPTS)
    user_style = style_opts if isinstance(style_opts, dict) else {}
    if user_style:
        style.update(user_style)
    if "show_p_after_sig" not in user_style:
        style["show_p_after_sig"] = False
    if "show_only_significant" not in user_style:
        style["show_only_significant"] = True
    _apply_style(style)

    pf_split_csv = Path(pf_split_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pf_split_csv.exists():
        raise FileNotFoundError(f"Missing PF split stats CSV: {pf_split_csv}")
    try:
        pf_df = pd.read_csv(pf_split_csv)
    except EmptyDataError as exc:
        raise RuntimeError(f"PF split stats CSV is empty: {pf_split_csv}") from exc
    if pf_df.empty:
        raise RuntimeError(f"PF split stats CSV has no rows: {pf_split_csv}")

    required_cols = [
        "animal_id", "cell_idx", "cell_num", "category", "region", "metric",
        "preferred_mean", "nonpreferred_mean", "is_place_cell", "selected_in_pass_any_folder",
    ]
    missing = [c for c in required_cols if c not in pf_df.columns]
    if missing:
        raise KeyError(f"Missing required PF split stats columns: {missing}")
    for c in ["cell_idx", "cell_num", "preferred_mean", "nonpreferred_mean"]:
        pf_df[c] = pd.to_numeric(pf_df[c], errors="coerce")
    pf_df["is_place_cell"] = _coerce_bool_series(pf_df["is_place_cell"])
    pf_df["selected_in_pass_any_folder"] = _coerce_bool_series(pf_df["selected_in_pass_any_folder"])

    id_cols = ["animal_id", "cell_idx", "cell_num"]
    region_keys = [str(r).strip() for r in (regions if isinstance(regions, list) and len(regions) > 0 else [region])]
    region_keys = [r for r in region_keys if len(r) > 0]
    if len(region_keys) <= 0:
        region_keys = ["combined"]

    row_payloads: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    csplus_requires_selected = bool(csplus_group_norm == "eb_tuned")
    for region_key in region_keys:
        csplus_mask = (
            (pf_df["category"].astype(str) == "CSplus")
            & (pf_df["is_place_cell"] == True)
            & (pf_df["region"].astype(str) == region_key)
        )
        if csplus_requires_selected:
            csplus_mask = csplus_mask & (pf_df["selected_in_pass_any_folder"] == True)
        csplus_df = pf_df.loc[csplus_mask].copy()
        csminus_df = pf_df.loc[
            (pf_df["category"].astype(str) == "CSminus")
            & (pf_df["is_place_cell"] == True)
            & (pf_df["region"].astype(str) == region_key)
        ].copy()
        if csplus_df.empty and csminus_df.empty:
            continue
        row_payloads.append((region_key, csplus_df, csminus_df))
    if len(row_payloads) <= 0:
        raise RuntimeError(f"No rows available for requested regions: {region_keys}")

    left_bg = "#FFF3E0"
    right_bg = "#E3F2FD"
    csplus_color = "#EE9B00"
    csminus_color = "#026C80"
    ss_color = "#026C80"
    cs_color = "#EE9B00"

    mrl_all_color = "#7A7A7A"
    n_rows = len(row_payloads)
    # Error-proof sizing: subplot width is determined by number of plotted groups.
    # Example: 4 groups * 0.2 in = 0.8 in panel width.
    group_width_in = float(style.get("group_width_in", 0.2))
    if (not np.isfinite(group_width_in)) or group_width_in <= 0:
        group_width_in = 0.2
    col_group_counts = [4, 2, 2, 4, 2, 2, 2, 2, 2]
    width_ratios = [float(v) for v in col_group_counts]
    fig_w = float(group_width_in * float(np.sum(width_ratios)))
    fig_h = float(1.0 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        9,
        figsize=(fig_w, fig_h),
        dpi=180,
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    tests: list[dict[str, Any]] = []

    def _shade(ax: Any, left_xmin: float, left_xmax: float, right_xmin: float, right_xmax: float) -> None:
        ax.set_xlim(float(left_xmin), float(right_xmax))
        ax.axvspan(float(left_xmin), float(left_xmax), facecolor=left_bg, edgecolor="none", alpha=0.8, zorder=0)
        ax.axvspan(float(right_xmin), float(right_xmax), facecolor=right_bg, edgecolor="none", alpha=0.8, zorder=0)

    for ridx, (region_key, csplus_df, csminus_df) in enumerate(row_payloads):
        row_axes = axes[ridx, :]

        def _plot_two_group_ds(ax: Any, metric_name: str, title_text: str, ylabel_text: str) -> None:
            l_df = _metric_ds_frame(csplus_df, metric_name, id_cols)
            r_df = _metric_ds_frame(csminus_df, metric_name, id_cols)
            l_vals = pd.to_numeric(l_df["ds"], errors="coerce").to_numpy(dtype=float)
            r_vals = pd.to_numeric(r_df["ds"], errors="coerce").to_numpy(dtype=float)
            l_vals = l_vals[np.isfinite(l_vals)]
            r_vals = r_vals[np.isfinite(r_vals)]
            _draw_boxplot(ax, [l_vals, r_vals], [0.0, 1.0], [csplus_color, csminus_color])
            if int(l_vals.size) > 0:
                jl = np.linspace(-0.08, 0.08, int(l_vals.size)) if int(l_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(0.0 + jl, l_vals, s=10, c=csplus_color, edgecolors="none", alpha=0.95, zorder=3)
            if int(r_vals.size) > 0:
                jr = np.linspace(-0.08, 0.08, int(r_vals.size)) if int(r_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(1.0 + jr, r_vals, s=10, c=csminus_color, edgecolors="none", alpha=0.95, zorder=3)
            p, stat, tname, n, shp = _nonpaired_test_auto(l_vals, r_vals)
            tests.append({
                "region": region_key,
                "panel": f"{metric_name}_ds",
                "side": "cross_group",
                "comparison": f"{metric_name}_ds_csplus_vs_csminus",
                "p_raw": p,
                "test": tname,
                "statistic": stat,
                "n": int(n),
                "x1": 0.0,
                "x2": 1.0,
                "level": 0,
                "shapiro_p": shp,
            })
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(["CS+", "CS-"], fontsize=float(style["tick_labelsize"]))
            if ridx == 0:
                ax.set_title(title_text, fontsize=float(style["axes_titlesize"]))
            ax.set_ylabel(ylabel_text, fontsize=float(style["axes_labelsize"]))
            _finalize_axis(ax, style)
            _shade(ax, -0.5, 0.5, 0.5, 1.5)

        # Col 1: merged DS panel (All/SS/CS for CS+ + All/SS for CS-)
        ax = row_axes[0]
        csplus_all_ds_df = _metric_ds_frame(csplus_df, "all", id_cols)
        l_all = pd.to_numeric(csplus_all_ds_df["ds"], errors="coerce").to_numpy(dtype=float)
        l_all = l_all[np.isfinite(l_all)]
        csplus_ss_ds_df = _metric_ds_frame(csplus_df, "ss", id_cols).rename(columns={"ds": "ss_ds"})
        csplus_cs_ds_df = _metric_ds_frame(csplus_df, "cs", id_cols).rename(columns={"ds": "cs_ds"})
        csplus_join = csplus_ss_ds_df.merge(csplus_cs_ds_df, on=id_cols, how="inner")
        l_ss = pd.to_numeric(csplus_join["ss_ds"], errors="coerce").to_numpy(dtype=float)
        l_cs = pd.to_numeric(csplus_join["cs_ds"], errors="coerce").to_numpy(dtype=float)
        l_mask = np.isfinite(l_ss) & np.isfinite(l_cs)
        l_ss = l_ss[l_mask]
        l_cs = l_cs[l_mask]

        csminus_ss_ds_df = _metric_ds_frame(csminus_df, "ss", id_cols)
        r_ss = pd.to_numeric(csminus_ss_ds_df["ds"], errors="coerce").to_numpy(dtype=float)
        r_ss = r_ss[np.isfinite(r_ss)]

        _draw_boxplot(ax, [l_all, l_ss, l_cs, r_ss], [0.0, 1.0, 2.0, 3.0], [mrl_all_color, ss_color, cs_color, mrl_all_color])
        if int(l_all.size) > 0:
            jl = np.linspace(-0.08, 0.08, int(l_all.size)) if int(l_all.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(0.0 + jl, l_all, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)
        _overlay_pair_lines_points(ax, l_ss, l_cs, 1.0, 2.0, ss_color, cs_color)
        if int(r_ss.size) > 0:
            jr = np.linspace(-0.08, 0.08, int(r_ss.size)) if int(r_ss.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(3.0 + jr, r_ss, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)

        p, stat, tname, n, shp = _paired_test_auto(l_ss, l_cs)
        tests.append({
            "region": region_key,
            "panel": "ds_rate_merged",
            "side": "csplus",
            "comparison": "ss_ds_vs_cs_ds",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 1.0,
            "x2": 2.0,
            "level": 0,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_all, r_ss)
        tests.append({
            "region": region_key,
            "panel": "ds_rate_merged",
            "side": "cross_group",
            "comparison": "all_ds_csplus_vs_allss_ds_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 0.0,
            "x2": 3.0,
            "level": 1,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_ss, r_ss)
        tests.append({
            "region": region_key,
            "panel": "ds_rate_merged",
            "side": "cross_group",
            "comparison": "ss_ds_csplus_vs_allss_ds_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 1.0,
            "x2": 3.0,
            "level": 2,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_cs, r_ss)
        tests.append({
            "region": region_key,
            "panel": "ds_rate_merged",
            "side": "cross_group",
            "comparison": "cs_ds_csplus_vs_allss_ds_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 2.0,
            "x2": 3.0,
            "level": 3,
            "shapiro_p": shp,
        })

        ax.set_xticks([0.0, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["All", "SS", "CS", "All/SS"], fontsize=float(style["tick_labelsize"]))
        if ridx == 0:
            ax.set_title("DS, all/SS/CS", fontsize=float(style["axes_titlesize"]))
            legend_handles = [
                Patch(facecolor=ss_color, edgecolor="none", alpha=0.5, label="SS"),
                Patch(facecolor=cs_color, edgecolor="none", alpha=0.5, label="CS"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=2,
                frameon=False,
                handlelength=1.0,
                columnspacing=0.9,
                fontsize=float(style["legend_fontsize"]),
            )
        ax.set_ylabel("DS", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)
        _shade(ax, -0.5, 2.5, 2.5, 3.5)

        # Col 2/3: two-group DS panels
        _plot_two_group_ds(row_axes[1], "theta", r"DS, $\theta$ amp.", "DS")
        _plot_two_group_ds(row_axes[2], "slow", "DS, Slow Vm", "DS")

        # Col 5: MRL (same layout as combined figure)
        ax = row_axes[3]
        l_all = _metric_vals_first_available(csplus_df, ["all_mrl_overall", "ss_mrl_overall"], "pref", id_cols)
        csplus_ss_mrl_df = (
            csplus_df.loc[csplus_df["metric"].astype(str) == "ss_mrl_overall", id_cols + ["preferred_mean"]]
            .copy()
            .drop_duplicates(subset=id_cols)
            .rename(columns={"preferred_mean": "ss_mrl"})
        )
        csplus_cs_mrl_df = (
            csplus_df.loc[csplus_df["metric"].astype(str) == "cs_mrl_overall", id_cols + ["preferred_mean"]]
            .copy()
            .drop_duplicates(subset=id_cols)
            .rename(columns={"preferred_mean": "cs_mrl"})
        )
        csplus_mrl_join = csplus_ss_mrl_df.merge(csplus_cs_mrl_df, on=id_cols, how="inner")
        l_ss = pd.to_numeric(csplus_mrl_join["ss_mrl"], errors="coerce").to_numpy(dtype=float)
        l_cs = pd.to_numeric(csplus_mrl_join["cs_mrl"], errors="coerce").to_numpy(dtype=float)
        lmask = np.isfinite(l_ss) & np.isfinite(l_cs)
        l_ss = l_ss[lmask]
        l_cs = l_cs[lmask]
        r_ss = _metric_vals(csminus_df, "ss_mrl_overall", "pref", id_cols)

        _draw_boxplot(ax, [l_all, l_ss, l_cs], [0.0, 1.0, 2.0], [mrl_all_color, ss_color, cs_color])
        if int(l_all.size) > 0:
            jitter = np.linspace(-0.08, 0.08, int(l_all.size)) if int(l_all.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(0.0 + jitter, l_all, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)
        _overlay_pair_lines_points(ax, l_ss, l_cs, 1.0, 2.0, ss_color, cs_color)
        _draw_boxplot(ax, [r_ss], [3.0], [mrl_all_color])
        if int(r_ss.size) > 0:
            jitter = np.linspace(-0.08, 0.08, int(r_ss.size)) if int(r_ss.size) > 1 else np.array([0.0], dtype=float)
            ax.scatter(3.0 + jitter, r_ss, s=10, c=mrl_all_color, edgecolors="none", alpha=0.95, zorder=3)

        p, stat, tname, n, shp = _paired_test_auto(l_ss, l_cs)
        tests.append({
            "region": region_key,
            "panel": "mrl",
            "side": "csplus",
            "comparison": "ss_vs_cs",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 1.0,
            "x2": 2.0,
            "level": 0,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_all, r_ss)
        tests.append({
            "region": region_key,
            "panel": "mrl",
            "side": "cross_group",
            "comparison": "all_csplus_vs_allss_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 0.0,
            "x2": 3.0,
            "level": 1,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_ss, r_ss)
        tests.append({
            "region": region_key,
            "panel": "mrl",
            "side": "cross_group",
            "comparison": "ss_csplus_vs_allss_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 1.0,
            "x2": 3.0,
            "level": 2,
            "shapiro_p": shp,
        })
        p, stat, tname, n, shp = _nonpaired_test_auto(l_cs, r_ss)
        tests.append({
            "region": region_key,
            "panel": "mrl",
            "side": "cross_group",
            "comparison": "cs_csplus_vs_allss_csminus",
            "p_raw": p,
            "test": tname,
            "statistic": stat,
            "n": int(n),
            "x1": 2.0,
            "x2": 3.0,
            "level": 3,
            "shapiro_p": shp,
        })
        ax.set_xticks([0.0, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["All", "SS", "CS", "All/SS"], fontsize=float(style["tick_labelsize"]))
        if ridx == 0:
            ax.set_title("MRL", fontsize=float(style["axes_titlesize"]))
        ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
        _finalize_axis(ax, style)
        _shade(ax, -0.5, 2.5, 2.5, 3.5)

        # Col 6/7: Theta/Slow MRL overall (CS+ vs CS- nonpaired)
        for col_idx, metric_name, title_text in [
            (4, "theta_mrl_overall", "Theta MRL"),
            (5, "slow_mrl_overall", "Slow Vm MRL"),
        ]:
            ax = row_axes[col_idx]
            l_vals = _metric_vals(csplus_df, metric_name, "pref", id_cols)
            r_vals = _metric_vals(csminus_df, metric_name, "pref", id_cols)
            l_vals = l_vals[np.isfinite(l_vals)]
            r_vals = r_vals[np.isfinite(r_vals)]
            _draw_boxplot(ax, [l_vals, r_vals], [0.0, 1.0], [csplus_color, csminus_color])
            if int(l_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(l_vals.size)) if int(l_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(0.0 + j, l_vals, s=10, c=csplus_color, edgecolors="none", alpha=0.95, zorder=3)
            if int(r_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(r_vals.size)) if int(r_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(1.0 + j, r_vals, s=10, c=csminus_color, edgecolors="none", alpha=0.95, zorder=3)
            p, stat, tname, n, shp = _nonpaired_test_auto(l_vals, r_vals)
            tests.append({
                "region": region_key,
                "panel": metric_name,
                "side": "cross_group",
                "comparison": f"{metric_name}_csplus_vs_csminus",
                "p_raw": p,
                "test": tname,
                "statistic": stat,
                "n": int(n),
                "x1": 0.0,
                "x2": 1.0,
                "level": 0,
                "shapiro_p": shp,
            })
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(["CS+", "CS-"], fontsize=float(style["tick_labelsize"]))
            if ridx == 0:
                ax.set_title(title_text, fontsize=float(style["axes_titlesize"]))
            ax.set_ylabel("MRL", fontsize=float(style["axes_labelsize"]))
            _finalize_axis(ax, style)
            _shade(ax, -0.5, 0.5, 0.5, 1.5)

        # Col 8/9/10: EB empirical-vs-fitted similarity (CS+ vs CS- nonpaired)
        for col_idx, metric_name, title_text in [
            (6, "eb_empfit_corr_all", "EB emp-fit corr (All)"),
            (7, "eb_empfit_corr_ss", "EB emp-fit corr (SS)"),
            (8, "eb_empfit_corr_cs", "EB emp-fit corr (CS)"),
        ]:
            ax = row_axes[col_idx]
            l_vals = _metric_vals(csplus_df, metric_name, "pref", id_cols)
            r_vals = _metric_vals(csminus_df, metric_name, "pref", id_cols)
            l_vals = l_vals[np.isfinite(l_vals)]
            r_vals = r_vals[np.isfinite(r_vals)]
            _draw_boxplot(ax, [l_vals, r_vals], [0.0, 1.0], [csplus_color, csminus_color])
            if int(l_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(l_vals.size)) if int(l_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(0.0 + j, l_vals, s=10, c=csplus_color, edgecolors="none", alpha=0.95, zorder=3)
            if int(r_vals.size) > 0:
                j = np.linspace(-0.08, 0.08, int(r_vals.size)) if int(r_vals.size) > 1 else np.array([0.0], dtype=float)
                ax.scatter(1.0 + j, r_vals, s=10, c=csminus_color, edgecolors="none", alpha=0.95, zorder=3)
            p, stat, tname, n, shp = _nonpaired_test_auto(l_vals, r_vals)
            tests.append({
                "region": region_key,
                "panel": metric_name,
                "side": "cross_group",
                "comparison": f"{metric_name}_csplus_vs_csminus",
                "p_raw": p,
                "test": tname,
                "statistic": stat,
                "n": int(n),
                "x1": 0.0,
                "x2": 1.0,
                "level": 0,
                "shapiro_p": shp,
            })
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(["CS+", "CS-"], fontsize=float(style["tick_labelsize"]))
            if ridx == 0:
                ax.set_title(title_text, fontsize=float(style["axes_titlesize"]))
            ax.set_ylabel("corr", fontsize=float(style["axes_labelsize"]))
            _finalize_axis(ax, style)
            _shade(ax, -0.5, 0.5, 0.5, 1.5)

        row_label = DEFAULT_ROW_LABELS.get(region_key, region_key)
        row_axes[0].text(
            -0.40,
            0.5,
            str(row_label),
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=float(style["axes_labelsize"]),
            fontweight="bold",
        )

        show_p = bool(style.get("show_p_after_sig", False))
        show_only_sig = bool(style.get("show_only_significant", True))
        for ax, panel in zip(
            row_axes,
            [
                "ds_rate_merged",
                "theta_ds",
                "slow_ds",
                "mrl",
                "theta_mrl_overall",
                "slow_mrl_overall",
                "eb_empfit_corr_all",
                "eb_empfit_corr_ss",
                "eb_empfit_corr_cs",
            ],
        ):
            panel_tests = [t for t in tests if str(t["panel"]) == panel and str(t["region"]) == region_key]
            if show_only_sig:
                panel_tests = [t for t in panel_tests if np.isfinite(t["p_raw"]) and float(t["p_raw"]) < 0.05]
            if len(panel_tests) <= 0:
                continue
            vals = []
            for c in ax.collections:
                off = c.get_offsets()
                if hasattr(off, "shape") and off.shape[0] > 0:
                    vals.extend(list(np.asarray(off[:, 1], dtype=float)))
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size <= 0:
                continue
            y_max = float(np.nanmax(vals))
            y_min = float(np.nanmin(vals))
            y_rng = max(1e-9, y_max - y_min)
            y_scale_abs = max(1.0, abs(y_max), abs(y_min))
            base = y_max + float(style.get("bracket_base_factor", 0.14)) * y_rng
            step = max(
                float(style.get("bracket_step_factor", 0.20)) * y_rng,
                float(style.get("bracket_step_min_abs", 0.06)) * y_scale_abs,
            )
            h = max(
                float(style.get("bracket_height_factor", 0.05)) * y_rng,
                float(style.get("bracket_height_min_abs", 0.02)) * y_scale_abs,
            )
            max_lvl = 0
            for t in sorted(panel_tests, key=lambda z: (int(z["level"]), float(z["x1"]), float(z["x2"]))):
                lvl = int(t["level"])
                max_lvl = max(max_lvl, lvl)
                yy = base + lvl * step
                sig = _sig_label(float(t["p_raw"])) if np.isfinite(float(t["p_raw"])) else "n.s."
                txt = _sig_text(sig, float(t["p_raw"]), show_p=show_p)
                _draw_bracket(ax, float(t["x1"]), float(t["x2"]), yy, h, txt, float(style["tick_labelsize"]))
            ax.set_ylim(top=base + (max_lvl + 1.8) * step)

    region_tag = "_".join([str(x) for x, _, _ in row_payloads])
    if csplus_group_norm == "eb_tuned":
        csplus_mode_title = "CS+ EB tuned"
        csplus_mode_tag = "csplus_eb_tuned"
    else:
        csplus_mode_title = "CS+ all"
        csplus_mode_tag = "csplus_all"
    fig.suptitle(
        f"Direction selectivity: {csplus_mode_title} vs CS- all | threshold={int(any_pass_threshold)}",
        fontsize=float(style["axes_titlesize"]) + 1.0,
        y=1.03,
    )
    fig_path = out_dir / f"pf_split_ds_{csplus_mode_tag}_vs_csminus_{region_tag}_first9_{any_pass_suffix}.svg"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    stats_df = pd.DataFrame(
        tests,
        columns=["region", "panel", "side", "comparison", "n", "test", "statistic", "shapiro_p", "p_raw", "x1", "x2", "level"],
    )
    stats_path = out_dir / f"pf_split_ds_{csplus_mode_tag}_vs_csminus_{region_tag}_first9_stats_{any_pass_suffix}.csv"
    stats_df.to_csv(stats_path, index=False)
    return {"figure": fig_path, "stats_csv": stats_path, "csplus_group": csplus_group_norm}


def plot_pf_split_group_figures(
    *,
    pf_split_csv: str | Path,
    out_dir: str | Path,
    any_pass_threshold: int,
    any_pass_suffix: str,
    group_specs: list[dict[str, Any]] | None = None,
    style_opts: dict[str, Any] | None = None,
    stats_opts: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, list[Path]]:
    style = dict(DEFAULT_STYLE_OPTS)
    if isinstance(style_opts, dict):
        style.update(style_opts)
    stats_cfg = dict(DEFAULT_STATS_OPTS)
    if isinstance(stats_opts, dict):
        stats_cfg.update(stats_opts)
    _apply_style(style)

    pf_split_csv = Path(pf_split_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pf_split_csv.exists():
        raise FileNotFoundError(f"Missing PF split stats CSV: {pf_split_csv}")
    try:
        pf_df = pd.read_csv(pf_split_csv)
    except EmptyDataError as exc:
        raise RuntimeError(f"PF split stats CSV is empty: {pf_split_csv}") from exc
    if pf_df.empty:
        raise RuntimeError(f"PF split stats CSV has no rows: {pf_split_csv}")

    required_cols = [
        "animal_id",
        "cell_idx",
        "cell_num",
        "category",
        "region",
        "metric",
        "preferred_mean",
        "nonpreferred_mean",
        "is_place_cell",
        "pass_100_any",
        "selected_in_pass_any_folder",
    ]
    missing = [c for c in required_cols if c not in pf_df.columns]
    if missing:
        raise KeyError(f"Missing required PF split stats columns: {missing}")
    for c in ["cell_idx", "cell_num", "preferred_mean", "nonpreferred_mean"]:
        pf_df[c] = pd.to_numeric(pf_df[c], errors="coerce")
    pf_df["is_place_cell"] = _coerce_bool_series(pf_df["is_place_cell"])
    pf_df["pass_100_any"] = _coerce_bool_series(pf_df["pass_100_any"])
    pf_df["selected_in_pass_any_folder"] = _coerce_bool_series(pf_df["selected_in_pass_any_folder"])

    id_cols = ["animal_id", "cell_idx", "cell_num"]
    use_specs = list(group_specs) if isinstance(group_specs, list) and len(group_specs) > 0 else list(DEFAULT_GROUP_SPECS)
    saved_figs: list[Path] = []
    saved_stats: list[Path] = []

    for spec in use_specs:
        spec_id = str(spec.get("id", "group")).strip() or "group"
        title = str(spec.get("title", spec_id))
        dfg = pf_df.copy()
        cats = spec.get("category_in")
        if isinstance(cats, (list, tuple)) and len(cats) > 0:
            dfg = dfg.loc[dfg["category"].astype(str).isin([str(x) for x in cats])].copy()
        if "is_place_cell" in spec:
            dfg = dfg.loc[dfg["is_place_cell"] == bool(spec["is_place_cell"])].copy()
        if "pass_100_any" in spec:
            dfg = dfg.loc[dfg["pass_100_any"] == bool(spec["pass_100_any"])].copy()
        if "selected_in_pass_any_folder" in spec:
            dfg = dfg.loc[dfg["selected_in_pass_any_folder"] == bool(spec["selected_in_pass_any_folder"])].copy()
        if dfg.empty:
            continue
        region_order = list(spec.get("regions", ["primary", "secondary", "combined", "outside_combined", "all_bins"]))
        mrl_panel_mode = str(spec.get("mrl_panel_mode", "ss_cs")).strip().lower()
        if mrl_panel_mode not in {"ss_cs", "ss_only"}:
            mrl_panel_mode = "ss_cs"
        row_labels = dict(DEFAULT_ROW_LABELS)
        if isinstance(spec.get("row_labels"), dict):
            row_labels.update({str(k): str(v) for k, v in spec["row_labels"].items()})
        present_rows = [(r, row_labels.get(r, r)) for r in region_order if not dfg.loc[dfg["region"].astype(str) == str(r)].empty]
        if not present_rows:
            continue

        n_rows = len(present_rows)
        width_ratios = [1, 2, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
        fig_w = float(sum(width_ratios))
        fig_h = float(1.2 * n_rows)
        fig, axes = plt.subplots(n_rows, 13, figsize=(fig_w, fig_h), dpi=180, constrained_layout=True, gridspec_kw={"width_ratios": width_ratios})
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        tests: list[dict[str, Any]] = []
        panel_vals: dict[tuple[str, str], np.ndarray] = {}
        for ridx, (row_region, row_label) in enumerate(present_rows):
            row_df = dfg.loc[dfg["region"].astype(str) == str(row_region)].copy()
            _plot_region_row(
                row_df,
                axes[ridx, :],
                tests,
                panel_vals,
                row_region,
                row_label,
                id_cols,
                style,
                show_titles=True,
                mrl_panel_mode=mrl_panel_mode,
            )

        test_df = pd.DataFrame(tests)
        if not test_df.empty:
            test_df["q_bh_fdr"] = test_df["p_raw"].astype(float)
            if str(stats_cfg.get("correction_method", "bh_fdr")).strip().lower() == "bh_fdr":
                panel_for_corr = str(stats_cfg.get("multiple_comparison_panel", "ss_cs_rate_merged"))
                for row_region, _ in present_rows:
                    mask = (test_df["panel"].astype(str) == panel_for_corr) & (test_df["region"].astype(str) == str(row_region))
                    if int(np.sum(mask)) > 0:
                        test_df.loc[mask, "q_bh_fdr"] = _bh_fdr(test_df.loc[mask, "p_raw"].to_numpy(dtype=float))
            test_df["sig_q"] = [_sig_label(v) for v in test_df["q_bh_fdr"].to_numpy(dtype=float)]
            panel_order = [
                "all",
                "ss_cs_rate_merged",
                "theta",
                "slow",
                "ss_cs_mrl",
                "theta_mrl",
                "slow_mrl",
                "occupancy",
                "speed",
                None,
                "eb_empfit_corr_all",
                "eb_empfit_corr_ss",
                "eb_empfit_corr_cs",
            ]
            for ridx, (row_region, _row_label) in enumerate(present_rows):
                for cidx, panel in enumerate(panel_order):
                    if panel is None:
                        continue
                    ax = axes[ridx, cidx]
                    panel_tests = test_df.loc[(test_df["panel"] == panel) & (test_df["region"].astype(str) == str(row_region))].copy().sort_values(["level", "x1", "x2"])
                    if panel_tests.empty:
                        continue
                    y_values = np.asarray(panel_vals.get((row_region, panel), np.array([], dtype=float)), dtype=float)
                    y_values = y_values[np.isfinite(y_values)]
                    if y_values.size == 0:
                        continue
                    y_max = float(np.nanmax(y_values))
                    y_min = float(np.nanmin(y_values))
                    y_rng = max(1e-9, y_max - y_min)
                    base = y_max + 0.08 * y_rng
                    step = 0.09 * y_rng
                    h = 0.03 * y_rng
                    if str(panel) == "ss_cs_rate_merged":
                        base = y_max + 0.12 * y_rng
                        step = 0.15 * y_rng
                        h = 0.04 * y_rng
                    max_level = 0
                    for _, r in panel_tests.iterrows():
                        level = int(r["level"]) if np.isfinite(r["level"]) else 0
                        max_level = max(max_level, level)
                        y = base + level * step
                        txt = _sig_text(r["sig_q"], r["p_raw"], show_p=bool(style.get("show_p_after_sig", True)))
                        _draw_bracket(ax, float(r["x1"]), float(r["x2"]), y, h, txt, float(style["tick_labelsize"]))
                    ax.set_ylim(top=base + (max_level + 1.8) * step)

        fig.suptitle(f"{title} | threshold={int(any_pass_threshold)}", fontsize=float(style["axes_titlesize"]) + 1.0, y=1.02)
        fig_path = out_dir / f"pf_split_{spec_id}_preferred_nonpreferred_{any_pass_suffix}.svg"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        if bool(show):
            plt.show()
        else:
            plt.close(fig)
        saved_figs.append(fig_path)

        stats_path = out_dir / f"pf_split_{spec_id}_stats_{any_pass_suffix}.csv"
        if test_df.empty:
            pd.DataFrame(columns=["region", "panel", "comparison", "n", "test", "statistic", "shapiro_p", "p_raw", "q_bh_fdr", "sig_q"]).to_csv(stats_path, index=False)
        else:
            test_df = test_df[["region", "panel", "comparison", "n", "test", "statistic", "shapiro_p", "p_raw", "q_bh_fdr", "sig_q"]]
            test_df.to_csv(stats_path, index=False)
        saved_stats.append(stats_path)

    return {"figures": saved_figs, "stats_csvs": saved_stats}


def plot_any_pass_category_contribution(
    *,
    pf_split_csv: str | Path,
    out_dir: str | Path,
    any_pass_threshold: int,
    any_pass_suffix: str,
    style_opts: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    style = dict(DEFAULT_STYLE_OPTS)
    if isinstance(style_opts, dict):
        style.update(style_opts)
    _apply_style(style)

    pf_split_csv = Path(pf_split_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pf_split_csv.exists():
        raise FileNotFoundError(f"Missing PF split stats CSV: {pf_split_csv}")
    try:
        pf_df = pd.read_csv(pf_split_csv)
    except EmptyDataError as exc:
        raise RuntimeError(f"PF split stats CSV is empty: {pf_split_csv}") from exc
    if pf_df.empty:
        raise RuntimeError(f"PF split stats CSV has no rows: {pf_split_csv}")

    required_cols = [
        "animal_id",
        "cell_idx",
        "cell_num",
        "category",
        "is_place_cell",
        "selected_in_pass_any_folder",
    ]
    missing = [c for c in required_cols if c not in pf_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    pass_col_candidates = [
        f"pass_{int(any_pass_threshold)}_any",
        f"pass{int(any_pass_threshold)}_all",  # fallback from older manifests if present
        "pass_100_any",
    ]
    pass_col = next((c for c in pass_col_candidates if c in pf_df.columns), None)
    if pass_col is None:
        raise KeyError(
            "Could not find any-pass column in PF split CSV. "
            f"Tried: {pass_col_candidates}"
        )

    for c in ["cell_idx", "cell_num"]:
        pf_df[c] = pd.to_numeric(pf_df[c], errors="coerce")
    pf_df["is_place_cell"] = _coerce_bool_series(pf_df["is_place_cell"])
    pf_df["selected_in_pass_any_folder"] = _coerce_bool_series(pf_df["selected_in_pass_any_folder"])
    pf_df[pass_col] = _coerce_bool_series(pf_df[pass_col])

    id_cols = ["animal_id", "cell_idx", "cell_num"]
    per_cell = (
        pf_df[id_cols + ["category", "is_place_cell", "selected_in_pass_any_folder", pass_col]]
        .drop_duplicates(subset=id_cols)
        .copy()
    )
    if per_cell.empty:
        raise RuntimeError("No cells found after per-cell deduplication.")

    def _group_label(row: pd.Series) -> str:
        cat = str(row.get("category", ""))
        is_pc = bool(row.get("is_place_cell", False))
        if cat == "CSplus" and is_pc:
            return "CS+ PLCs"
        if cat == "CSminus" and is_pc:
            return "CS- PLCs"
        return "Non-PLCs"

    per_cell["group_label"] = per_cell.apply(_group_label, axis=1)
    group_order = ["CS+ PLCs", "CS- PLCs", "Non-PLCs"]
    tuned_col = "selected_in_pass_any_folder"

    rows = []
    total_cells_all = int(len(per_cell))
    total_any_pass = int(np.sum(per_cell[pass_col] == True))
    for g in group_order:
        sub = per_cell.loc[per_cell["group_label"] == g].copy()
        tuned_n = int(np.sum(sub[tuned_col] == True))
        total_n = int(len(sub))
        nontuned_n = int(max(0, total_n - tuned_n))
        pct_tuned_within = (100.0 * tuned_n / total_n) if total_n > 0 else np.nan
        pct_nontuned_within = (100.0 * nontuned_n / total_n) if total_n > 0 else np.nan
        pct_tuned_all = (100.0 * tuned_n / total_cells_all) if total_cells_all > 0 else np.nan
        pct_nontuned_all = (100.0 * nontuned_n / total_cells_all) if total_cells_all > 0 else np.nan
        pct_total_all = (100.0 * total_n / total_cells_all) if total_cells_all > 0 else np.nan
        rows.append(
            {
                "group": g,
                "tuned_n": tuned_n,
                "nontuned_n": nontuned_n,
                "total_n": total_n,
                "pct_tuned_within_group": pct_tuned_within,
                "pct_nontuned_within_group": pct_nontuned_within,
                "pct_tuned_of_all_cells": pct_tuned_all,
                "pct_nontuned_of_all_cells": pct_nontuned_all,
                "pct_total_of_all_cells": pct_total_all,
                "all_cells_total_n": total_cells_all,
                "any_pass_total_n": total_any_pass,
                "any_pass_col": str(pass_col),
            }
        )
    out_df = pd.DataFrame(rows)

    x = np.arange(len(group_order), dtype=float)
    tuned_pct = out_df["pct_tuned_within_group"].to_numpy(dtype=float) / 100.0
    nontuned_pct = out_df["pct_nontuned_within_group"].to_numpy(dtype=float) / 100.0
    tuned_n = out_df["tuned_n"].to_numpy(dtype=int)
    nontuned_n = out_df["nontuned_n"].to_numpy(dtype=int)
    total_n = out_df["total_n"].to_numpy(dtype=int)

    tuned_color = "#8E00D0"
    nontuned_color = "#8C8C8C"
    fig, ax = plt.subplots(figsize=(1.4, 1.0), dpi=180, constrained_layout=True)
    bars_tuned = ax.bar(
        x,
        tuned_pct,
        width=0.66,
        color=tuned_color,
        edgecolor=tuned_color,
        linewidth=0.4,
        label="EB tuned",
        zorder=2,
    )
    bars_nontuned = ax.bar(
        x,
        nontuned_pct,
        width=0.66,
        bottom=tuned_pct,
        color=nontuned_color,
        edgecolor=nontuned_color,
        linewidth=0.4,
        label="Non-tuned",
        zorder=2,
    )

    for i in range(len(group_order)):
        if tuned_n[i] > 0 and np.isfinite(tuned_pct[i]) and tuned_pct[i] > 0:
            ax.text(
                float(x[i]),
                float(tuned_pct[i]) * 0.5,
                f"{int(tuned_n[i])}",
                ha="center",
                va="center",
                fontsize=float(style["tick_labelsize"]),
                color="white",
                zorder=3,
            )
        if nontuned_n[i] > 0 and np.isfinite(nontuned_pct[i]) and nontuned_pct[i] > 0:
            ax.text(
                float(x[i]),
                float(tuned_pct[i] + nontuned_pct[i] * 0.5),
                f"{int(nontuned_n[i])}",
                ha="center",
                va="center",
                fontsize=float(style["tick_labelsize"]),
                color="black",
                zorder=3,
            )
        # Requested: remove per-bar "n=..." text.

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=float(style["tick_labelsize"]))
    ax.set_ylabel("proportion", fontsize=float(style["axes_labelsize"]))
    ax.set_ylim(0.0, 1.08)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=float(style["legend_fontsize"]),
        handlelength=1.2,
        columnspacing=0.9,
    )
    _finalize_axis(ax, style)

    fig_path = out_dir / f"any_pass_category_contribution_{any_pass_suffix}.svg"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    stats_path = out_dir / f"any_pass_category_contribution_{any_pass_suffix}.csv"
    out_df.to_csv(stats_path, index=False)
    return {"figure": fig_path, "stats_csv": stats_path, "table": out_df}
