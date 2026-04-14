"""Core pooled-figure plotting functions (direct, notebook-independent)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.spatial_heatmaps import (
    compute_cb_in_pf_counts,
    get_deleted_cells_with_fallback,
    is_csplus_place_cell,
)


VIOLIN_MEDIAN_COLOR = '#1F77B4'
VIOLIN_MEDIAN_LINEWIDTH = 1.0

ALL_COLOR = 'black'
SS_COLOR = '#026C80'
CS_COLOR = '#EE9B00'
CS_PLC_BG = '#FFF3E0'
NON_CS_PLC_BG = '#E3F2FD'


@dataclass
class PooledStatsData:
    df_all_raw: pd.DataFrame
    df_all: pd.DataFrame
    df_pc: pd.DataFrame
    df_cs_plc: pd.DataFrame
    df_non_cs_plc: pd.DataFrame
    deleted_cells: set[tuple[str, int]]
    cb_in_pf_counts: dict[tuple[str, int], int]


def _style_violin_medians(parts, color: str = VIOLIN_MEDIAN_COLOR, linewidth: float = VIOLIN_MEDIAN_LINEWIDTH) -> None:
    if 'cmedians' in parts:
        parts['cmedians'].set_color(color)
        parts['cmedians'].set_linewidth(linewidth)


def _global_ylim(arrays, pad_frac: float = 0.05, pct_low: float = 1, pct_high: float = 100):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vals.append(finite)
    if not vals:
        return None
    merged = np.concatenate(vals)
    y_min = np.percentile(merged, pct_low)
    y_max = np.percentile(merged, pct_high)
    y_span = y_max - y_min if y_max != y_min else 1.0
    pad = pad_frac * y_span
    return y_min - pad, y_max + pad


def _sig_label(p_val: float) -> str:
    if not np.isfinite(p_val):
        return ''
    if p_val < 0.001:
        return '***'
    if p_val < 0.01:
        return '**'
    if p_val < 0.05:
        return '*'
    return 'n.s.'


def _sig_stars(p_val: float) -> str:
    if not np.isfinite(p_val):
        return ''
    if p_val < 0.001:
        return '***'
    if p_val < 0.01:
        return '**'
    if p_val < 0.05:
        return '*'
    return ''


def _paired_test(vals_in, vals_out):
    vals_in = np.asarray(vals_in, dtype=float)
    vals_out = np.asarray(vals_out, dtype=float)
    n_pairs = min(len(vals_in), len(vals_out))
    if n_pairs == 0:
        return np.nan, np.nan, 'n/a', 0, np.nan

    paired_in = vals_in[:n_pairs]
    paired_out = vals_out[:n_pairs]
    valid = np.isfinite(paired_in) & np.isfinite(paired_out)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        return np.nan, np.nan, 'n/a', n_valid, np.nan

    diffs = paired_in[valid] - paired_out[valid]
    try:
        shapiro_p = scipy_stats.shapiro(diffs).pvalue
    except ValueError:
        shapiro_p = np.nan

    if n_valid <= 6:
        try:
            result = scipy_stats.wilcoxon(paired_in[valid], paired_out[valid])
            return result.pvalue, result.statistic, 'wilcoxon', n_valid, shapiro_p
        except ValueError:
            return np.nan, np.nan, 'wilcoxon', n_valid, shapiro_p

    if np.isfinite(shapiro_p) and shapiro_p >= 0.05:
        result = scipy_stats.ttest_rel(paired_in[valid], paired_out[valid], nan_policy='omit')
        return result.pvalue, result.statistic, 'paired t-test', n_valid, shapiro_p

    try:
        result = scipy_stats.wilcoxon(paired_in[valid], paired_out[valid])
        return result.pvalue, result.statistic, 'wilcoxon', n_valid, shapiro_p
    except ValueError:
        return np.nan, np.nan, 'wilcoxon', n_valid, shapiro_p


def _unpaired_test_auto(data1, data2):
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    data1 = data1[np.isfinite(data1)]
    data2 = data2[np.isfinite(data2)]

    if len(data1) < 3 or len(data2) < 3:
        return np.nan, 'n/a', np.nan, np.nan

    try:
        shapiro_p1 = scipy_stats.shapiro(data1).pvalue
    except ValueError:
        shapiro_p1 = 0.0
    try:
        shapiro_p2 = scipy_stats.shapiro(data2).pvalue
    except ValueError:
        shapiro_p2 = 0.0

    if min(len(data1), len(data2)) <= 6:
        _, p_val = scipy_stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = 'Mann-Whitney'
    elif shapiro_p1 >= 0.05 and shapiro_p2 >= 0.05:
        _, p_val = scipy_stats.ttest_ind(data1, data2)
        test_name = 't-test'
    else:
        _, p_val = scipy_stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = 'Mann-Whitney'

    return p_val, test_name, shapiro_p1, shapiro_p2


def _add_bracket(ax, pos1, pos2, y, h, sig_text, color='black'):
    drop = h
    ax.plot([pos1, pos1], [y, y + drop], color=color, linewidth=0.8)
    ax.plot([pos2, pos2], [y, y + drop], color=color, linewidth=0.8)
    ax.plot([pos1, pos2], [y + drop, y + drop], color=color, linewidth=0.8)
    ax.text(
        (pos1 + pos2) / 2,
        y + drop + 0.02,
        sig_text,
        ha='center',
        va='bottom',
        fontsize=5,
        fontname='Arial',
        color=color,
    )


def prepare_pooled_stats_tables(
    data_folder: str,
    folders: list[str],
    cb_num_threshold: int = 10,
    cs_peak_rate_threshold: float = 0.5,
    snr_threshold: float = 3.5,
    bad_frac_threshold: float = 0.9,
) -> PooledStatsData:
    deleted_cells = get_deleted_cells_with_fallback(
        data_folder,
        folders,
        snr_threshold=snr_threshold,
        bad_frac_threshold=bad_frac_threshold,
    )
    cb_in_pf_counts = compute_cb_in_pf_counts(data_folder, folders)

    all_dfs = []
    for folder in folders:
        pkl_path = os.path.join(data_folder, folder, 'pooled_stats.pkl')
        if os.path.exists(pkl_path):
            df = pd.read_pickle(pkl_path)
            df['session'] = folder
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError('No pooled_stats.pkl files were found for the requested folders.')

    df_all_raw = pd.concat(all_dfs, ignore_index=True)
    valid_mask = ~df_all_raw.apply(lambda row: (row['session'], row['cell_idx']) in deleted_cells, axis=1)
    df_all = df_all_raw[valid_mask].copy().reset_index(drop=True)
    df_pc = df_all[df_all['is_place_cell'] == True].copy()

    n_cb_in_pf = df_pc.apply(
        lambda row: cb_in_pf_counts.get((row['session'], row['cell_idx']), 0),
        axis=1,
    ).values
    cs_peak_rates = pd.to_numeric(df_pc.get('peak_rate_cs', np.nan), errors='coerce').to_numpy(dtype=float)
    is_cs_plc = np.asarray(
        [
            is_csplus_place_cell(
                is_place_cell=True,
                n_cb_in_pf=int(n_cb),
                cs_peak_rate=float(cs_peak),
                cb_num_threshold=int(cb_num_threshold),
                cs_peak_rate_threshold=float(cs_peak_rate_threshold),
            )
            for n_cb, cs_peak in zip(n_cb_in_pf, cs_peak_rates)
        ],
        dtype=bool,
    )

    df_pc['is_cs_plc'] = is_cs_plc
    df_pc['n_cb_in_pf'] = n_cb_in_pf

    df_all.loc[df_all['is_place_cell'] == True, 'is_cs_plc'] = False
    df_all.loc[df_all['is_place_cell'] == True, 'n_cb_in_pf'] = 0
    df_all.loc[df_pc.index, 'is_cs_plc'] = is_cs_plc
    df_all.loc[df_pc.index, 'n_cb_in_pf'] = n_cb_in_pf

    # All-spike PF reference in-field firing-rate metrics (Hz).
    # These are directly sourced from in/out stats computed using the all-spike PF mask.
    df_all["fr_in_allpf_all"] = pd.to_numeric(df_all.get("all_inout_loco_in", np.nan), errors="coerce")
    df_all["fr_in_allpf_ss"] = pd.to_numeric(df_all.get("ss_inout_loco_in", np.nan), errors="coerce")
    df_all["fr_in_allpf_cs"] = pd.to_numeric(df_all.get("cs_inout_loco_in", np.nan), errors="coerce")

    df_cs_plc = df_pc[df_pc['is_cs_plc'] == True].copy()
    df_non_cs_plc = df_pc[df_pc['is_cs_plc'] == False].copy()

    return PooledStatsData(
        df_all_raw=df_all_raw,
        df_all=df_all,
        df_pc=df_pc,
        df_cs_plc=df_cs_plc,
        df_non_cs_plc=df_non_cs_plc,
        deleted_cells=deleted_cells,
        cb_in_pf_counts=cb_in_pf_counts,
    )


def _get_peak_rate_data(df_subset: pd.DataFrame):
    all_data = df_subset['peak_rate_all'].values
    ss_data = df_subset['peak_rate_ss'].values
    cs_data = df_subset['peak_rate_cs'].values

    all_valid = all_data[np.isfinite(all_data)]
    ss_valid = ss_data[np.isfinite(ss_data)]
    cs_valid = cs_data[np.isfinite(cs_data)]

    paired_mask = np.isfinite(ss_data) & np.isfinite(cs_data)
    ss_paired = ss_data[paired_mask]
    cs_paired = cs_data[paired_mask]

    return all_valid, ss_valid, cs_valid, ss_paired, cs_paired


def _half_violin_panel(ax, positions, data_list, colors_list, paired_specs, rng_seed=42):
    paired_left = {spec[0] for spec in paired_specs}
    paired_right = {spec[1] for spec in paired_specs}

    for i, (pos, data, color) in enumerate(zip(positions, data_list, colors_list)):
        if len(data) < 3:
            continue
        parts = ax.violinplot([data], positions=[pos], showmedians=True, showextrema=False)
        _style_violin_medians(parts)
        body = parts['bodies'][0]
        body.set_facecolor(color)
        body.set_edgecolor('none')
        body.set_alpha(0.5)

        verts = body.get_paths()[0].vertices
        segs = parts['cmedians'].get_segments()
        if i in paired_left:
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
            segs[0][:, 0] = np.clip(segs[0][:, 0], -np.inf, pos)
        elif i in paired_right:
            verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
            segs[0][:, 0] = np.clip(segs[0][:, 0], pos, np.inf)
        parts['cmedians'].set_segments(segs)

    for left_idx, right_idx in paired_specs:
        left_pos = positions[left_idx]
        right_pos = positions[right_idx]
        left_data = data_list[left_idx]
        right_data = data_list[right_idx]
        n_pairs = min(len(left_data), len(right_data))
        for i in range(n_pairs):
            if np.isfinite(left_data[i]) and np.isfinite(right_data[i]):
                ax.plot(
                    [left_pos + 0.08, right_pos - 0.08],
                    [left_data[i], right_data[i]],
                    color='black',
                    alpha=0.2,
                    linewidth=0.5,
                )

    def _beeswarm(vals, base_x, direction):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return np.array([])
        y_range = np.ptp(vals) if len(vals) > 1 else 1.0
        y_spacing = y_range * 0.04
        x_step = 0.06
        sorted_idx = np.argsort(vals)
        xs = np.zeros(len(vals))
        for si, oi in enumerate(sorted_idx):
            layer = 0
            while True:
                candidate = x_step * layer
                conflict = False
                for sj in range(si):
                    oj = sorted_idx[sj]
                    if abs(vals[oi] - vals[oj]) < y_spacing:
                        if abs(candidate - xs[oj]) < x_step * 0.9:
                            conflict = True
                            break
                if not conflict:
                    xs[oi] = candidate
                    break
                layer += 1
        return base_x + direction * (0.02 + xs)

    for i, (pos, data) in enumerate(zip(positions, data_list)):
        if len(data) < 1:
            continue
        if i in paired_left:
            xs = _beeswarm(data, pos, direction=1)
        elif i in paired_right:
            xs = _beeswarm(data, pos, direction=-1)
        else:
            xs = _beeswarm(data, pos, direction=1)
        ax.scatter(xs, data, s=4, color='black', alpha=0.5, linewidths=0)


def _get_numeric_triplet_data(
    df_subset: pd.DataFrame,
    all_col: str,
    ss_col: str,
    cs_col: str,
):
    def _series(col_name: str) -> np.ndarray:
        series = df_subset.get(col_name, pd.Series(np.nan, index=df_subset.index))
        return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)

    all_vals = _series(all_col)
    ss_vals = _series(ss_col)
    cs_vals = _series(cs_col)

    all_valid = all_vals[np.isfinite(all_vals)]
    ss_valid = ss_vals[np.isfinite(ss_vals)]
    cs_valid = cs_vals[np.isfinite(cs_vals)]

    paired_mask = np.isfinite(ss_vals) & np.isfinite(cs_vals)
    ss_paired = ss_vals[paired_mask]
    cs_paired = cs_vals[paired_mask]
    return all_valid, ss_valid, cs_valid, ss_paired, cs_paired


def _all_finite_in_unit_interval(data_list) -> bool:
    finite_vals = []
    for arr in data_list:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_vals.append(finite)
    if not finite_vals:
        return False
    merged = np.concatenate(finite_vals)
    return bool(np.all((merged >= 0.0) & (merged <= 1.0)))


def _plot_pf_style_panel(
    ax,
    cs_metric,
    non_metric,
    ylabel: str,
    plot_cs_minus_ss: bool = False,
    fixed_ylim: tuple[float, float] | None = None,
    yticks: list[float] | None = None,
    zero_line: bool = False,
    clamp_unit_interval: bool = False,
):
    positions_5 = [1, 2, 3, 4.5, 5.5]
    cs_all, cs_ss, cs_cs, cs_ss_paired, cs_cs_paired = cs_metric
    non_all, non_ss, _non_cs, _non_ss_paired, _non_cs_paired = non_metric

    ax.axvspan(0.5, 3.5, alpha=0.3, color=CS_PLC_BG, zorder=0)
    ax.axvspan(4.0, 6.0, alpha=0.3, color=NON_CS_PLC_BG, zorder=0)
    ax.text(2, -0.22, 'CS+ PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
    ax.text(5, -0.22, 'CS- PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())

    if plot_cs_minus_ss:
        data_list = [cs_all, cs_ss, cs_cs, non_all, non_ss]
        colors_list = [ALL_COLOR, SS_COLOR, CS_COLOR, ALL_COLOR, SS_COLOR]
        positions = positions_5
    else:
        data_list = [cs_all, cs_ss, cs_cs, non_all]
        colors_list = [ALL_COLOR, SS_COLOR, CS_COLOR, ALL_COLOR]
        positions = [1, 2, 3, 4.5]

    ylim = _global_ylim(data_list)
    _half_violin_panel(ax, positions, data_list, colors_list, paired_specs=[(1, 2)])
    if zero_line:
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    ax.set_xticks(positions)
    ax.set_xticklabels(['All', 'SS', 'CS', 'All', 'SS'][:len(positions)], fontsize=4, fontname='Arial')
    ax.set_xlim(0.3, 6.2 if plot_cs_minus_ss else 5.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=5)
    if yticks is not None:
        ax.set_yticks(yticks)

    if ylim:
        y_range = ylim[1] - ylim[0]
        h = y_range * 0.03
        text_h = y_range * 0.08
        gap = y_range * 0.05
        y_top = y1 = ylim[1] + y_range * 0.05
        if len(cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(cs_ss_paired, cs_cs_paired)
            _add_bracket(ax, 2, 3, y1, h, _sig_label(p))
            y_top = y1 + h + text_h
        y2 = y1 + h + text_h + gap
        if len(cs_all) >= 3 and len(non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_all, non_all)
            _add_bracket(ax, 1, 4.5, y2, h, _sig_label(p), color=ALL_COLOR)
            y_top = y2 + h + text_h
        y3 = y2 + h + text_h + gap
        if plot_cs_minus_ss and len(cs_ss) >= 3 and len(non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_ss, non_ss)
            _add_bracket(ax, 2, 5.5, y3, h, _sig_label(p), color=SS_COLOR)
            y_top = y3 + h + text_h

        if clamp_unit_interval:
            lower = 0.0
            upper = max(1.0, y_top + y_range * 0.05)
        elif fixed_ylim is not None:
            lower = fixed_ylim[0]
            upper = max(fixed_ylim[1], y_top + y_range * 0.05)
        else:
            lower = ylim[0]
            upper = y_top + y_range * 0.05
        ax.set_ylim(lower, upper)
    elif fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif clamp_unit_interval:
        ax.set_ylim(0.0, 1.0)


def plot_combined_cs_plus_minus_4panels(
    df_cs_plc: pd.DataFrame,
    df_non_cs_plc: pd.DataFrame,
    save_path: str,
    plot_cs_minus_ss: bool = False,
):
    panel_h = 1.4
    first_panel_ratio = 1.4
    fig, axes = plt.subplots(
        1,
        7,
        figsize=(9.6, panel_h),
        gridspec_kw={'width_ratios': [first_panel_ratio, 1, 1, 1, 1, 1, first_panel_ratio]},
    )

    positions_6 = [1, 2, 3, 4.5, 5.5, 6.5]
    arena_area_cm2 = 20.0 * 35.5

    # Panel 1: Peak rates.
    ax = axes[0]
    cs_all, cs_ss, cs_cs, cs_ss_paired, cs_cs_paired = _get_peak_rate_data(df_cs_plc)
    non_all, non_ss, non_cs, non_ss_paired, non_cs_paired = _get_peak_rate_data(df_non_cs_plc)

    ax.axvspan(0.5, 3.5, alpha=0.3, color=CS_PLC_BG, zorder=0)
    ax.axvspan(4.0, 7.0, alpha=0.3, color=NON_CS_PLC_BG, zorder=0)
    ax.text(2, -0.22, 'CS+ PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
    ax.text(5.5, -0.22, 'CS- PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())

    data_list = [cs_all, cs_ss, cs_cs, non_all, non_ss, non_cs]
    colors_list = [ALL_COLOR, SS_COLOR, CS_COLOR, ALL_COLOR, SS_COLOR, CS_COLOR]
    ylim = _global_ylim(data_list)
    _half_violin_panel(ax, positions_6, data_list, colors_list, paired_specs=[(1, 2), (4, 5)])

    ax.set_ylabel('Peak rate (Hz)', fontsize=6, fontname='Arial')
    ax.set_xticks(positions_6)
    ax.set_xticklabels(['All', 'SS', 'CS', 'All', 'SS', 'CS'], fontsize=4, fontname='Arial')
    ax.set_xlim(0.3, 7.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=5)

    if ylim:
        y_range = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0], ylim[1] + y_range * 0.65)
        h = y_range * 0.03
        text_h = y_range * 0.08
        gap = y_range * 0.05
        y1 = ylim[1] + y_range * 0.05
        if len(cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(cs_ss_paired, cs_cs_paired)
            _add_bracket(ax, 2, 3, y1, h, _sig_label(p))
        if len(non_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(non_ss_paired, non_cs_paired)
            _add_bracket(ax, 5.5, 6.5, y1, h, _sig_label(p))
        y2 = y1 + h + text_h + gap
        if len(cs_all) >= 3 and len(non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_all, non_all)
            _add_bracket(ax, 1, 4.5, y2, h, _sig_label(p), color=ALL_COLOR)
        y3 = y2 + h + text_h + gap
        if len(cs_ss) >= 3 and len(non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_ss, non_ss)
            _add_bracket(ax, 2, 5.5, y3, h, _sig_label(p), color=SS_COLOR)
        y4 = y3 + h + text_h + gap
        if len(cs_cs) >= 3 and len(non_cs) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_cs, non_cs)
            _add_bracket(ax, 3, 6.5, y4, h, _sig_label(p), color=CS_COLOR)

    def _get_pf_size_sum(df_subset: pd.DataFrame):
        all_sizes, ss_sizes, cs_sizes = [], [], []
        for idx in df_subset.index:
            sizes = df_subset.loc[idx, 'place_field_sizes_cm2']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                all_sizes.append(np.sum(sizes) / arena_area_cm2 * 100)
            else:
                all_sizes.append(np.nan)

            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_ss']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                ss_sizes.append(np.sum(sizes) / arena_area_cm2 * 100)
            else:
                ss_sizes.append(np.nan)

            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_cs']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                cs_sizes.append(np.sum(sizes) / arena_area_cm2 * 100)
            else:
                cs_sizes.append(np.nan)

        all_sizes = np.asarray(all_sizes)
        ss_sizes = np.asarray(ss_sizes)
        cs_sizes = np.asarray(cs_sizes)

        all_valid = all_sizes[np.isfinite(all_sizes)]
        ss_valid = ss_sizes[np.isfinite(ss_sizes)]
        cs_valid = cs_sizes[np.isfinite(cs_sizes)]

        paired_mask = np.isfinite(ss_sizes) & np.isfinite(cs_sizes)
        ss_paired = ss_sizes[paired_mask]
        cs_paired = cs_sizes[paired_mask]
        return all_valid, ss_valid, cs_valid, ss_paired, cs_paired

    pf_cs_all, pf_cs_ss, pf_cs_cs, pf_cs_ss_paired, pf_cs_cs_paired = _get_pf_size_sum(df_cs_plc)
    pf_non_all, pf_non_ss, pf_non_cs, pf_non_ss_paired, pf_non_cs_paired = _get_pf_size_sum(df_non_cs_plc)
    _plot_pf_style_panel(
        axes[1],
        (pf_cs_all, pf_cs_ss, pf_cs_cs, pf_cs_ss_paired, pf_cs_cs_paired),
        (pf_non_all, pf_non_ss, pf_non_cs, pf_non_ss_paired, pf_non_cs_paired),
        ylabel='PF size (% arena)',
        plot_cs_minus_ss=plot_cs_minus_ss,
        fixed_ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
    )

    # Panel 3: Coherence.
    coh_cs = _get_numeric_triplet_data(df_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs')
    coh_non = _get_numeric_triplet_data(df_non_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs')
    _plot_pf_style_panel(
        axes[2],
        coh_cs,
        coh_non,
        ylabel='Coherence (z)',
        plot_cs_minus_ss=plot_cs_minus_ss,
    )

    # Panel 4: Sparsity.
    sparsity_cs = _get_numeric_triplet_data(df_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
    sparsity_non = _get_numeric_triplet_data(df_non_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
    sparsity_data_list = (
        [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0], sparsity_non[1]]
        if plot_cs_minus_ss
        else [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0]]
    )
    _plot_pf_style_panel(
        axes[3],
        sparsity_cs,
        sparsity_non,
        ylabel='Sparsity',
        plot_cs_minus_ss=plot_cs_minus_ss,
        clamp_unit_interval=_all_finite_in_unit_interval(sparsity_data_list),
    )

    # Panel 5: Spatial information.
    si_cs = _get_numeric_triplet_data(df_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs')
    si_non = _get_numeric_triplet_data(df_non_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs')
    _plot_pf_style_panel(
        axes[4],
        si_cs,
        si_non,
        ylabel='SI (bits/spike)',
        plot_cs_minus_ss=plot_cs_minus_ss,
    )

    def _get_selectivity(df_subset: pd.DataFrame):
        all_in = df_subset['all_inout_loco_in'].values
        all_out = df_subset['all_inout_loco_out'].values
        all_sum = all_in + all_out
        with np.errstate(divide='ignore', invalid='ignore'):
            all_sel = np.where((all_sum > 0) & np.isfinite(all_in) & np.isfinite(all_out), (all_in - all_out) / all_sum, np.nan)

        ss_in = df_subset['ss_inout_loco_in'].values
        ss_out = df_subset['ss_inout_loco_out'].values
        ss_sum = ss_in + ss_out
        with np.errstate(divide='ignore', invalid='ignore'):
            ss_sel = np.where((ss_sum > 0) & np.isfinite(ss_in) & np.isfinite(ss_out), (ss_in - ss_out) / ss_sum, np.nan)

        cs_in = df_subset['cs_inout_loco_in'].values
        cs_out = df_subset['cs_inout_loco_out'].values
        cs_sum = cs_in + cs_out
        with np.errstate(divide='ignore', invalid='ignore'):
            cs_sel = np.where((cs_sum > 0) & np.isfinite(cs_in) & np.isfinite(cs_out), (cs_in - cs_out) / cs_sum, np.nan)

        all_valid = all_sel[np.isfinite(all_sel)]
        ss_valid = ss_sel[np.isfinite(ss_sel)]
        cs_valid = cs_sel[np.isfinite(cs_sel)]

        paired_mask = np.isfinite(ss_sel) & np.isfinite(cs_sel)
        ss_paired = ss_sel[paired_mask]
        cs_paired = cs_sel[paired_mask]
        return all_valid, ss_valid, cs_valid, ss_paired, cs_paired

    # Panel 6: Selectivity.
    sel_cs_all, sel_cs_ss, sel_cs_cs, sel_cs_ss_paired, sel_cs_cs_paired = _get_selectivity(df_cs_plc)
    sel_non_all, sel_non_ss, sel_non_cs, sel_non_ss_paired, sel_non_cs_paired = _get_selectivity(df_non_cs_plc)
    _plot_pf_style_panel(
        axes[5],
        (sel_cs_all, sel_cs_ss, sel_cs_cs, sel_cs_ss_paired, sel_cs_cs_paired),
        (sel_non_all, sel_non_ss, sel_non_cs, sel_non_ss_paired, sel_non_cs_paired),
        ylabel='Selectivity',
        plot_cs_minus_ss=plot_cs_minus_ss,
        zero_line=True,
    )

    def _get_fr_in_ref_pf_data(df_subset: pd.DataFrame):
        if "fr_in_allpf_all" in df_subset.columns:
            all_fr = np.asarray(df_subset["fr_in_allpf_all"].values, dtype=float)
            ss_fr = np.asarray(df_subset["fr_in_allpf_ss"].values, dtype=float)
            cs_fr = np.asarray(df_subset["fr_in_allpf_cs"].values, dtype=float)
        else:
            all_fr = np.asarray(df_subset["all_inout_loco_in"].values, dtype=float)
            ss_fr = np.asarray(df_subset["ss_inout_loco_in"].values, dtype=float)
            cs_fr = np.asarray(df_subset["cs_inout_loco_in"].values, dtype=float)

        all_valid = all_fr[np.isfinite(all_fr)]
        ss_valid = ss_fr[np.isfinite(ss_fr)]
        cs_valid = cs_fr[np.isfinite(cs_fr)]

        paired_mask = np.isfinite(ss_fr) & np.isfinite(cs_fr)
        ss_paired = ss_fr[paired_mask]
        cs_paired = cs_fr[paired_mask]
        return all_valid, ss_valid, cs_valid, ss_paired, cs_paired

    # Panel 7: In-field firing rate using all-spike PF reference mask.
    # Match Panel 1 layout/style: All/SS/CS for both CS+ and CS- PLC.
    ax = axes[6]
    fr_cs_all, fr_cs_ss, fr_cs_cs, fr_cs_ss_paired, fr_cs_cs_paired = _get_fr_in_ref_pf_data(df_cs_plc)
    fr_non_all, fr_non_ss, fr_non_cs, fr_non_ss_paired, fr_non_cs_paired = _get_fr_in_ref_pf_data(df_non_cs_plc)

    ax.axvspan(0.5, 3.5, alpha=0.3, color=CS_PLC_BG, zorder=0)
    ax.axvspan(4.0, 7.0, alpha=0.3, color=NON_CS_PLC_BG, zorder=0)
    ax.text(2, -0.22, 'CS+ PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
    ax.text(5.5, -0.22, 'CS- PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())

    data_list = [fr_cs_all, fr_cs_ss, fr_cs_cs, fr_non_all, fr_non_ss, fr_non_cs]
    colors_list = [ALL_COLOR, SS_COLOR, CS_COLOR, ALL_COLOR, SS_COLOR, CS_COLOR]
    ylim = _global_ylim(data_list)
    _half_violin_panel(ax, positions_6, data_list, colors_list, paired_specs=[(1, 2), (4, 5)])
    ax.set_ylabel('In-PF FR (Hz)', fontsize=6, fontname='Arial')
    ax.set_xticks(positions_6)
    ax.set_xticklabels(['All', 'SS', 'CS', 'All', 'SS', 'CS'], fontsize=4, fontname='Arial')
    ax.set_xlim(0.3, 7.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=5)

    if ylim:
        y_range = ylim[1] - ylim[0]
        h = y_range * 0.03
        text_h = y_range * 0.08
        gap = y_range * 0.05
        y_top = y1 = ylim[1] + y_range * 0.05
        if len(fr_cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(fr_cs_ss_paired, fr_cs_cs_paired)
            _add_bracket(ax, 2, 3, y1, h, _sig_label(p))
            y_top = y1 + h + text_h
        if len(fr_non_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(fr_non_ss_paired, fr_non_cs_paired)
            _add_bracket(ax, 5.5, 6.5, y1, h, _sig_label(p))
            y_top = max(y_top, y1 + h + text_h)
        y2 = y1 + h + text_h + gap
        if len(fr_cs_all) >= 3 and len(fr_non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_all, fr_non_all)
            _add_bracket(ax, 1, 4.5, y2, h, _sig_label(p), color=ALL_COLOR)
            y_top = y2 + h + text_h
        y3 = y2 + h + text_h + gap
        if len(fr_cs_ss) >= 3 and len(fr_non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_ss, fr_non_ss)
            _add_bracket(ax, 2, 5.5, y3, h, _sig_label(p), color=SS_COLOR)
            y_top = y3 + h + text_h
        y4 = y3 + h + text_h + gap
        if len(fr_cs_cs) >= 3 and len(fr_non_cs) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_cs, fr_non_cs)
            _add_bracket(ax, 3, 6.5, y4, h, _sig_label(p), color=CS_COLOR)
            y_top = y4 + h + text_h
        ax.set_ylim(ylim[0], y_top + y_range * 0.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def _confidence_interval_corr(x, y, x_pred, confidence: float = 0.95):
    n = len(x)
    if n < 3:
        return np.full_like(x_pred, np.nan), np.full_like(x_pred, np.nan)

    slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
    y_fit = slope * x + intercept
    residuals = y - y_fit
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 2)
    y_pred = slope * x_pred + intercept
    ci = t_val * s_err * np.sqrt(1 / n + (x_pred - x_mean) ** 2 / ss_x)
    return y_pred - ci, y_pred + ci


def _plot_corr_series(ax, x, y, series_label, color, xlabel, ylabel, panel_name, text_y=0.02, linestyle='-'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return False

    ax.scatter(x, y, s=8, color=color, alpha=0.65, linewidths=0)
    lr = scipy_stats.linregress(x, y)
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
    y_line = lr.intercept + lr.slope * x_line
    ax.plot(x_line, y_line, color=color, linewidth=1.0, alpha=0.9, linestyle=linestyle)

    ci_low, ci_high = _confidence_interval_corr(x, y, x_line)
    ax.fill_between(x_line, ci_low, ci_high, color=color, alpha=0.2, edgecolor='none')

    r, p = scipy_stats.pearsonr(x, y)
    stars = _sig_stars(p)
    ax.text(
        0.98,
        float(text_y),
        f"{stars}r={r:.2f}",
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=6,
        fontname='Arial',
        color=color,
    )

    ax.set_xlabel(xlabel, fontsize=6, fontname='Arial')
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    return True


def _pf_sizes_pct_from_col(df, col_name, arena_area_cm2_local):
    if col_name not in df.columns:
        raise KeyError(f"df missing column '{col_name}'")
    out = []
    for sizes in df[col_name].values:
        if isinstance(sizes, (list, tuple, np.ndarray)) and len(sizes) > 0:
            s = np.asarray(sizes, dtype=float)
            s = s[np.isfinite(s)]
            out.append(float(np.nansum(s) / float(arena_area_cm2_local) * 100.0) if s.size else np.nan)
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _selectivity_from_cols(df, in_col, out_col):
    if in_col not in df.columns or out_col not in df.columns:
        raise KeyError(f"df missing '{in_col}' and/or '{out_col}'")
    v_in = np.asarray(df[in_col].values, dtype=float)
    v_out = np.asarray(df[out_col].values, dtype=float)
    v_sum = v_in + v_out
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((v_sum > 0) & np.isfinite(v_in) & np.isfinite(v_out), (v_in - v_out) / v_sum, np.nan)


def plot_cs_plus_plc_correlations_overlay_cs_ss(
    df_cs_plc: pd.DataFrame,
    save_path: str,
    overlay_ss_metrics: bool = True,
):
    cs_color_local = CS_COLOR
    ss_color_local = SS_COLOR
    # In the original pooled notebook this resolves from global `cs_plc_bg`,
    # which is set to light orange.
    cs_plc_bg_local = CS_PLC_BG

    arena_area_cm2_local = 20.0 * 35.5

    if 'peak_rate_cs' not in df_cs_plc.columns:
        raise KeyError("df_cs_plc missing column 'peak_rate_cs'")
    x_peak_cs = np.asarray(df_cs_plc['peak_rate_cs'].values, dtype=float)

    pf_sizes_cs_pct = _pf_sizes_pct_from_col(df_cs_plc, 'place_field_sizes_cm2_cs', arena_area_cm2_local)

    if 'si_bits_per_spike_cs' not in df_cs_plc.columns:
        raise KeyError("df_cs_plc missing column 'si_bits_per_spike_cs'")
    si_cs = np.asarray(df_cs_plc['si_bits_per_spike_cs'].values, dtype=float)

    cs_selectivity = _selectivity_from_cols(df_cs_plc, 'cs_inout_loco_in', 'cs_inout_loco_out')

    fig, axes = plt.subplots(1, 6, figsize=(1.0 * 6, 1.2))
    axes[4].sharey(axes[3])

    x_peak_ss = None
    pf_sizes_ss_pct = None
    si_ss = None
    ss_selectivity = None
    if overlay_ss_metrics:
        if 'peak_rate_ss' not in df_cs_plc.columns:
            raise KeyError("df_cs_plc missing column 'peak_rate_ss'")
        if 'si_bits_per_spike_ss' not in df_cs_plc.columns:
            raise KeyError("df_cs_plc missing column 'si_bits_per_spike_ss'")
        x_peak_ss = np.asarray(df_cs_plc['peak_rate_ss'].values, dtype=float)
        pf_sizes_ss_pct = _pf_sizes_pct_from_col(df_cs_plc, 'place_field_sizes_cm2_ss', arena_area_cm2_local)
        si_ss = np.asarray(df_cs_plc['si_bits_per_spike_ss'].values, dtype=float)
        ss_selectivity = _selectivity_from_cols(df_cs_plc, 'ss_inout_loco_in', 'ss_inout_loco_out')

    panels = [
        (axes[0], x_peak_cs, pf_sizes_cs_pct, x_peak_ss, pf_sizes_ss_pct, 'Peak rate (Hz)', 'PF size (% arena)', 'Peak rate vs PF size'),
        (axes[1], si_cs, x_peak_cs, si_ss, x_peak_ss, 'SI (bits/spike)', 'Peak rate (Hz)', 'SI vs Peak rate'),
        (axes[2], x_peak_cs, cs_selectivity, x_peak_ss, ss_selectivity, 'Peak rate (Hz)', 'Selectivity', 'Peak rate vs Selectivity'),
        (axes[3], si_cs, pf_sizes_cs_pct, si_ss, pf_sizes_ss_pct, 'SI (bits/spike)', 'PF size (% arena)', 'SI vs PF size'),
        (axes[4], cs_selectivity, pf_sizes_cs_pct, ss_selectivity, pf_sizes_ss_pct, 'Selectivity', 'PF size (% arena)', 'Selectivity vs PF size'),
        (axes[5], si_cs, cs_selectivity, si_ss, ss_selectivity, 'SI (bits/spike)', 'Selectivity', 'SI vs Selectivity'),
    ]

    for ax, x_cs, y_cs, x_ss, y_ss, xlabel, ylabel, panel_name in panels:
        ax.set_facecolor(cs_plc_bg_local)
        ax.patch.set_alpha(0.3)
        ax.tick_params(labelsize=5)
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        any_plotted = False
        any_plotted |= _plot_corr_series(
            ax,
            x_cs,
            y_cs,
            series_label='CS',
            color=cs_color_local,
            xlabel=xlabel,
            ylabel=ylabel,
            panel_name=panel_name,
            text_y=0.82,
            linestyle='-',
        )

        if overlay_ss_metrics:
            any_plotted |= _plot_corr_series(
                ax,
                x_ss,
                y_ss,
                series_label='SS',
                color=ss_color_local,
                xlabel=xlabel,
                ylabel=ylabel,
                panel_name=panel_name,
                text_y=0.92,
                linestyle='-',
            )

        if not any_plotted:
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def _build_metric_data_csplus(plcs_csplus, min_bursts_per_condition=3, plateau_threshold_ms=100.0):
    conditions = ['run_in', 'run_out', 'rest_in', 'rest_out']

    metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
        ('burst_rate_hz', 'CB rate (Hz)'),
        ('burst_prob', 'CB prob.'),
        ('plateau_pct', f"%CB > {int(plateau_threshold_ms)} ms"),
    ]

    metric_data = {
        'complex': {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs},
    }
    cond_counts = {
        'complex': {cond: [] for cond in conditions},
    }
    base_metric_keys = {'n_spikes', 'peak_amp', 'duration_ms', 'auc'}

    for cell in plcs_csplus:
        bm = cell.get('burst_metrics')
        complex_by_cond = bm.get('complex') if isinstance(bm, dict) else None
        sbrm = cell.get('spike_burst_rate_metrics') if isinstance(cell, dict) else None

        for cond in conditions:
            bursts = []
            if isinstance(complex_by_cond, dict):
                bursts = complex_by_cond.get(cond, [])
            if not isinstance(bursts, (list, tuple)):
                bursts = []

            cond_counts['complex'][cond].append(int(len(bursts)))

            for metric_key, _ in metric_specs:
                if metric_key in base_metric_keys:
                    vals = []
                    for b in bursts:
                        if not isinstance(b, dict):
                            continue
                        vals.append(float(b.get(metric_key, np.nan)))
                    vals = np.asarray(vals, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size >= int(min_bursts_per_condition):
                        metric_data['complex'][metric_key][cond].append(float(np.nanmean(vals)))
                    else:
                        metric_data['complex'][metric_key][cond].append(np.nan)
                elif metric_key == 'burst_rate_hz':
                    val = np.nan
                    if isinstance(sbrm, dict):
                        br = sbrm.get('burst_rate')
                        if isinstance(br, dict):
                            val = br.get(cond, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    metric_data['complex'][metric_key][cond].append(val if np.isfinite(val) else np.nan)
                elif metric_key == 'burst_prob':
                    val = np.nan
                    if isinstance(sbrm, dict):
                        bp = sbrm.get('burst_prob')
                        if isinstance(bp, dict):
                            val = bp.get(cond, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    metric_data['complex'][metric_key][cond].append(val if np.isfinite(val) else np.nan)
                elif metric_key == 'plateau_pct':
                    if len(bursts) < int(min_bursts_per_condition):
                        metric_data['complex'][metric_key][cond].append(np.nan)
                        continue
                    durations = []
                    for b in bursts:
                        if not isinstance(b, dict):
                            continue
                        durations.append(float(b.get('duration_ms', np.nan)))
                    durations = np.asarray(durations, dtype=float)
                    durations = durations[np.isfinite(durations)]
                    if durations.size == 0:
                        metric_data['complex'][metric_key][cond].append(np.nan)
                        continue
                    pct = 100.0 * float(np.sum(durations > float(plateau_threshold_ms))) / float(durations.size)
                    metric_data['complex'][metric_key][cond].append(pct)
                else:
                    metric_data['complex'][metric_key][cond].append(np.nan)

    return metric_data, cond_counts, conditions


def _unpaired_test_bursts(d1, d2):
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]
    if len(d1) < 3 or len(d2) < 3:
        return np.nan, 'N/A'
    if min(len(d1), len(d2)) <= 6:
        _, p = scipy_stats.mannwhitneyu(d1, d2, alternative='two-sided')
        return p, 'Mann-Whitney U'
    try:
        sp1 = scipy_stats.shapiro(d1[: min(len(d1), 5000)]).pvalue
    except ValueError:
        sp1 = np.nan
    try:
        sp2 = scipy_stats.shapiro(d2[: min(len(d2), 5000)]).pvalue
    except ValueError:
        sp2 = np.nan
    if (np.isfinite(sp1) and sp1 > 0.05) and (np.isfinite(sp2) and sp2 > 0.05):
        _, p = scipy_stats.ttest_ind(d1, d2)
        return p, 'Unpaired t-test'
    _, p = scipy_stats.mannwhitneyu(d1, d2, alternative='two-sided')
    return p, 'Mann-Whitney U'


def plot_complex_burst_metrics_allbursts_csplus(
    plcs_csplus,
    save_path: str,
    state_key: str = 'run',
    min_bursts_per_condition: int = 3,
):
    conditions = ['run_in', 'run_out', 'rest_in', 'rest_out']
    burst_metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
    ]

    burst_data_all = {metric: {cond: [] for cond in conditions} for metric, _ in burst_metric_specs}

    for cell in plcs_csplus:
        bm = cell.get('burst_metrics')
        complex_by_cond = bm.get('complex') if isinstance(bm, dict) else None
        for cond in conditions:
            bursts = []
            if isinstance(complex_by_cond, dict):
                bursts = complex_by_cond.get(cond, [])
            if not isinstance(bursts, (list, tuple)):
                bursts = []
            for b in bursts:
                if not isinstance(b, dict):
                    continue
                for metric_key, _ in burst_metric_specs:
                    val = b.get(metric_key, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    if np.isfinite(val):
                        burst_data_all[metric_key][cond].append(val)

    burst_ylim = {}
    for metric, _ in burst_metric_specs:
        all_vals = [np.asarray(burst_data_all[metric][cond], dtype=float) for cond in conditions]
        burst_ylim[metric] = _global_ylim(all_vals, pct_low=1, pct_high=99)

    panel_w = 0.9
    panel_h = 1.2
    fig, axes = plt.subplots(1, len(burst_metric_specs), figsize=(panel_w * len(burst_metric_specs), panel_h), sharex=True, sharey=False)

    conds = [f'{state_key}_in', f'{state_key}_out']
    condition_labels = ['In PF', 'Out PF']

    for ax, (metric_key, ylabel) in zip(axes, burst_metric_specs):
        ax.axvspan(0.5, 2.5, alpha=0.3, color=CS_PLC_BG, zorder=0)

        vals_in = np.asarray(burst_data_all[metric_key][conds[0]], dtype=float)
        vals_out = np.asarray(burst_data_all[metric_key][conds[1]], dtype=float)

        plot_data = []
        for vals in [vals_in, vals_out]:
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                finite = np.array([np.nan])
            plot_data.append(finite)

        positions = [1, 2]
        colors = ['magenta', 'gray']
        alphas = [0.3, 0.3]
        if any(len(arr) < 2 for arr in plot_data):
            ax.axis('off')
            continue

        half_sides = ['left', 'right']
        rng = np.random.default_rng(0)
        for pos, d, color, alpha, side in zip(positions, plot_data, colors, alphas, half_sides):
            vp = ax.violinplot([d], positions=[pos], showmedians=True, showextrema=False)
            body = vp['bodies'][0]
            verts = body.get_paths()[0].vertices
            if side == 'left':
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
            else:
                verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
            body.set_facecolor(color)
            body.set_edgecolor('none')
            body.set_alpha(alpha)
            if 'cmedians' in vp:
                segs = vp['cmedians'].get_segments()
                if len(segs) > 0:
                    seg = segs[0].copy()
                    if side == 'left':
                        seg[:, 0] = np.clip(seg[:, 0], -np.inf, pos)
                    else:
                        seg[:, 0] = np.clip(seg[:, 0], pos, np.inf)
                    vp['cmedians'].set_segments([seg])
                vp['cmedians'].set_color('#1F77B4')
                vp['cmedians'].set_linewidth(1.0)

            finite_mask = np.isfinite(d)
            n_finite = np.sum(finite_mask)
            if n_finite > 0:
                d_finite = d[finite_mask]
                max_dots = 200
                if n_finite > max_dots:
                    idx_sub = rng.choice(n_finite, max_dots, replace=False)
                    d_plot = d_finite[idx_sub]
                else:
                    d_plot = d_finite
                y_range = np.ptp(d_plot) if len(d_plot) > 1 else 1.0
                y_spacing = y_range * 0.04
                x_step = 0.06
                sorted_idx = np.argsort(d_plot)
                xs_bee = np.zeros(len(d_plot))
                for si, oi in enumerate(sorted_idx):
                    layer = 0
                    while True:
                        candidate = x_step * layer
                        conflict = False
                        for sj in range(si):
                            oj = sorted_idx[sj]
                            if abs(d_plot[oi] - d_plot[oj]) < y_spacing:
                                if abs(candidate - xs_bee[oj]) < x_step * 0.9:
                                    conflict = True
                                    break
                        if not conflict:
                            xs_bee[oi] = candidate
                            break
                        layer += 1
                if side == 'left':
                    xs = pos + 0.05 + xs_bee
                else:
                    xs = pos - 0.05 - xs_bee
                ax.scatter(xs, d_plot, s=4, color='black', alpha=0.4, linewidths=0, zorder=3)

        p_val, _ = _unpaired_test_bursts(vals_in, vals_out)
        sig_label = _sig_label(p_val)
        if sig_label:
            ax.text(1.5, 0.99, sig_label, ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())

        ax.set_xticks([1, 2])
        ax.set_xticklabels(condition_labels, fontsize=5, fontname='Arial')
        ax.set_xlim(0.5, 2.5)
        ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
        ax.tick_params(labelsize=5)
        ylim = burst_ylim.get(metric_key)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for label in list(ax.get_yticklabels()):
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=300)
    return fig


def plot_complex_burst_metrics_csplus(
    plcs_csplus,
    save_path: str,
    state_key: str = 'run',
    min_bursts_per_condition: int = 3,
    plateau_threshold_ms: float = 100.0,
):
    """Per-cell averaged complex-burst metrics (matches pooled notebook cell 58 style)."""
    metric_data, cond_counts, conditions = _build_metric_data_csplus(
        plcs_csplus,
        min_bursts_per_condition=min_bursts_per_condition,
        plateau_threshold_ms=plateau_threshold_ms,
    )

    metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
        ('burst_rate_hz', 'CB rate (Hz)'),
        ('burst_prob', 'CB prob.'),
        ('plateau_pct', f"%CB > {int(plateau_threshold_ms)} ms"),
    ]

    metric_ylim = {}
    for metric, _ in metric_specs:
        all_vals = [np.asarray(metric_data['complex'][metric][cond], dtype=float) for cond in conditions]
        metric_ylim[metric] = _global_ylim(all_vals)

    metric_ylim['burst_prob'] = (0.0, 1.0)
    if metric_ylim.get('plateau_pct') is not None:
        y0, y1 = metric_ylim['plateau_pct']
        y0 = max(0.0, float(y0)) if np.isfinite(y0) else 0.0
        y1 = min(100.0, float(y1)) if np.isfinite(y1) else np.nan
        metric_ylim['plateau_pct'] = (y0, y1) if np.isfinite(y1) and y1 > y0 else (0.0, 100.0)
    else:
        metric_ylim['plateau_pct'] = (0.0, 100.0)

    panel_w = 0.9
    panel_h = 1.2
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(panel_w * len(metric_specs), panel_h), sharex=True, sharey=False)

    conds = [f'{state_key}_in', f'{state_key}_out']
    condition_labels = ['In PF', 'Out PF']

    counts_in_complex = np.asarray(cond_counts['complex'][conds[0]], dtype=float)
    counts_out_complex = np.asarray(cond_counts['complex'][conds[1]], dtype=float)
    pair_mask_complex = (counts_in_complex >= min_bursts_per_condition) & (counts_out_complex >= min_bursts_per_condition)

    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        ax.axvspan(0.5, 2.5, alpha=0.3, color=CS_PLC_BG, zorder=0)

        vals_in_complex = np.asarray(metric_data['complex'][metric_key][conds[0]], dtype=float)
        vals_out_complex = np.asarray(metric_data['complex'][metric_key][conds[1]], dtype=float)
        if pair_mask_complex.size > 0:
            vals_in_complex = vals_in_complex[pair_mask_complex]
            vals_out_complex = vals_out_complex[pair_mask_complex]
        else:
            vals_in_complex = np.array([], dtype=float)
            vals_out_complex = np.array([], dtype=float)

        plot_data = []
        positions = [1, 2]
        colors = ['magenta', 'gray']
        alphas = [0.3, 0.3]
        for vals in [vals_in_complex, vals_out_complex]:
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                finite = np.array([np.nan])
            plot_data.append(finite)

        if any(len(arr) < 2 for arr in plot_data):
            ax.axis('off')
            continue

        half_sides = ['left', 'right']
        for pos, d, color, alpha, side in zip(positions, plot_data, colors, alphas, half_sides):
            vp = ax.violinplot([d], positions=[pos], showmedians=True, showextrema=False)
            body = vp['bodies'][0]
            verts = body.get_paths()[0].vertices
            if side == 'left':
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
            else:
                verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
            body.set_facecolor(color)
            body.set_edgecolor('none')
            body.set_alpha(alpha)
            if 'cmedians' in vp:
                segs = vp['cmedians'].get_segments()
                if len(segs) > 0:
                    seg = segs[0].copy()
                    if side == 'left':
                        seg[:, 0] = np.clip(seg[:, 0], -np.inf, pos)
                    else:
                        seg[:, 0] = np.clip(seg[:, 0], pos, np.inf)
                    vp['cmedians'].set_segments([seg])
                vp['cmedians'].set_color('#1F77B4')
                vp['cmedians'].set_linewidth(1.0)

            finite_mask = np.isfinite(d)
            if np.any(finite_mask):
                d_finite = d[finite_mask]
                y_range = np.ptp(d_finite) if len(d_finite) > 1 else 1.0
                y_spacing = y_range * 0.04
                x_step = 0.06
                sorted_idx = np.argsort(d_finite)
                xs_bee = np.zeros(len(d_finite))
                for si, oi in enumerate(sorted_idx):
                    layer = 0
                    while True:
                        candidate = x_step * layer
                        conflict = False
                        for sj in range(si):
                            oj = sorted_idx[sj]
                            if abs(d_finite[oi] - d_finite[oj]) < y_spacing:
                                if abs(candidate - xs_bee[oj]) < x_step * 0.9:
                                    conflict = True
                                    break
                        if not conflict:
                            xs_bee[oi] = candidate
                            break
                        layer += 1
                if side == 'left':
                    xs = pos + 0.05 + xs_bee
                else:
                    xs = pos - 0.05 - xs_bee
                ax.scatter(xs, d_finite, s=8, color='black', alpha=0.6, linewidths=0, zorder=3)

        n_pairs = min(len(vals_in_complex), len(vals_out_complex))
        for i in range(n_pairs):
            if np.isfinite(vals_in_complex[i]) and np.isfinite(vals_out_complex[i]):
                ax.plot([1.12, 1.88], [vals_in_complex[i], vals_out_complex[i]], color='black', alpha=0.3, linewidth=0.6, zorder=2)

        p_val, _, _, _, _ = _paired_test(vals_in_complex, vals_out_complex)
        sig_label = _sig_label(p_val)
        if sig_label:
            ax.text(
                1.5,
                0.99,
                sig_label,
                ha='center',
                va='top',
                fontsize=6,
                fontname='Arial',
                transform=ax.get_xaxis_transform(),
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(condition_labels, fontsize=5, fontname='Arial')
        ax.set_xlim(0.5, 2.5)
        ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
        ax.tick_params(labelsize=5)
        ylim = metric_ylim.get(metric_key)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for label in list(ax.get_yticklabels()):
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=300)
    return fig


def _confidence_interval(x, y, x_pred, confidence=0.95):
    n = len(x)
    if n < 3:
        return np.full_like(x_pred, np.nan), np.full_like(x_pred, np.nan)
    slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
    y_pred = slope * x_pred + intercept
    y_fit = slope * x + intercept
    residuals = y - y_fit
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 2)
    ci = t_val * s_err * np.sqrt(1 / n + (x_pred - x_mean) ** 2 / ss_x)
    return y_pred - ci, y_pred + ci


def plot_cbrate_correlations_csplus_inout_alt(
    plcs_csplus,
    save_path: str,
    min_bursts_per_condition: int = 3,
):
    metric_data, cond_counts, _ = _build_metric_data_csplus(plcs_csplus, min_bursts_per_condition=min_bursts_per_condition)

    cs_bg_corr = CS_PLC_BG
    in_pf_color = 'magenta'
    out_pf_color = 'gray'

    corr_metrics = [
        ('n_spikes', 'Spks/burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
    ]

    cb_rate_in = np.asarray(metric_data['complex']['burst_rate_hz']['run_in'], dtype=float)
    cb_rate_out = np.asarray(metric_data['complex']['burst_rate_hz']['run_out'], dtype=float)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(3.5, 1.4),
        gridspec_kw={'height_ratios': [2, 3]},
        sharex=False,
    )

    for col_idx in range(4):
        axes[0, col_idx].sharex(axes[1, col_idx])
    for col_idx in range(1, 4):
        axes[1, col_idx].sharey(axes[1, 0])

    conds_run = ['run_in', 'run_out']
    counts_in = np.asarray(cond_counts['complex'][conds_run[0]], dtype=float)
    counts_out = np.asarray(cond_counts['complex'][conds_run[1]], dtype=float)
    pair_mask = (counts_in >= min_bursts_per_condition) & (counts_out >= min_bursts_per_condition)

    for col_idx, (metric_key, xlabel) in enumerate(corr_metrics):
        ax = axes[0, col_idx]

        vals_in = np.asarray(metric_data['complex'][metric_key]['run_in'], dtype=float)
        vals_out = np.asarray(metric_data['complex'][metric_key]['run_out'], dtype=float)
        if pair_mask.size > 0:
            vals_in = vals_in[pair_mask]
            vals_out = vals_out[pair_mask]
        else:
            vals_in = np.array([], dtype=float)
            vals_out = np.array([], dtype=float)

        fin_in = vals_in[np.isfinite(vals_in)]
        fin_out = vals_out[np.isfinite(vals_out)]
        if len(fin_in) < 2 and len(fin_out) < 2:
            ax.axis('off')
            continue

        bp = ax.boxplot(
            [fin_out, fin_in],
            positions=[1, 2],
            vert=False,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='#1F77B4', linewidth=1.0),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            boxprops=dict(linewidth=0.8),
        )
        bp['boxes'][0].set_facecolor(out_pf_color)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(in_pf_color)
        bp['boxes'][1].set_alpha(0.4)

        n_pairs = min(len(fin_in), len(fin_out))
        for i in range(n_pairs):
            if np.isfinite(fin_in[i]) and np.isfinite(fin_out[i]):
                ax.plot([fin_in[i], fin_out[i]], [2, 1], color='black', alpha=0.2, linewidth=0.3, zorder=2)

        ax.scatter(fin_in, np.full(len(fin_in), 2), s=3, color='black', alpha=0.4, linewidths=0, zorder=3)
        ax.scatter(fin_out, np.full(len(fin_out), 1), s=3, color='black', alpha=0.4, linewidths=0, zorder=3)

        p_val, _, _, _, _ = _paired_test(vals_in, vals_out)
        sig = _sig_stars(p_val)
        if sig:
            ax.text(0.98, 0.5, sig, ha='right', va='center', fontsize=6, fontname='Arial', transform=ax.transAxes, rotation=90)

        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Out', 'In'], fontsize=5, fontname='Arial')
        ax.set_ylim(0.3, 2.7)
        ax.tick_params(labelsize=5, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for col_idx, (metric_key, xlabel) in enumerate(corr_metrics):
        ax = axes[1, col_idx]

        metric_in = np.asarray(metric_data['complex'][metric_key]['run_in'], dtype=float)
        metric_out = np.asarray(metric_data['complex'][metric_key]['run_out'], dtype=float)

        valid_in = np.isfinite(cb_rate_in) & np.isfinite(metric_in)
        x_in = metric_in[valid_in]
        y_in = cb_rate_in[valid_in]

        valid_out = np.isfinite(cb_rate_out) & np.isfinite(metric_out)
        x_out = metric_out[valid_out]
        y_out = cb_rate_out[valid_out]

        all_x = np.concatenate([x_in, x_out]) if len(x_in) > 0 and len(x_out) > 0 else (x_in if len(x_in) > 0 else x_out)
        if len(all_x) > 0:
            x_min, x_max = all_x.min(), all_x.max()
            x_pad = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

        ax.axvspan(ax.get_xlim()[0], ax.get_xlim()[1], alpha=0.3, color=cs_bg_corr, zorder=0)
        ax.scatter(x_in, y_in, c=in_pf_color, s=8, alpha=0.6, edgecolors='none', label='In PF')
        ax.scatter(x_out, y_out, c=out_pf_color, s=8, alpha=0.6, edgecolors='none', label='Out PF')

        if len(x_in) >= 3:
            x_line_in = np.linspace(x_in.min(), x_in.max(), 100)
            slope_in, intercept_in, r_in, p_in, _ = scipy_stats.linregress(x_in, y_in)
            y_line_in = slope_in * x_line_in + intercept_in
            ci_low_in, ci_high_in = _confidence_interval(x_in, y_in, x_line_in)
            ax.plot(x_line_in, y_line_in, color=in_pf_color, linewidth=1, linestyle='-')
            ax.fill_between(x_line_in, ci_low_in, ci_high_in, color=in_pf_color, alpha=0.2, edgecolor='none')
            sig_in = _sig_stars(p_in)
        else:
            r_in, sig_in = np.nan, ''

        if len(x_out) >= 3:
            x_line_out = np.linspace(x_out.min(), x_out.max(), 100)
            slope_out, intercept_out, r_out, p_out, _ = scipy_stats.linregress(x_out, y_out)
            y_line_out = slope_out * x_line_out + intercept_out
            ci_low_out, ci_high_out = _confidence_interval(x_out, y_out, x_line_out)
            ax.plot(x_line_out, y_line_out, color=out_pf_color, linewidth=1, linestyle='-')
            ax.fill_between(x_line_out, ci_low_out, ci_high_out, color=out_pf_color, alpha=0.2, edgecolor='none')
            sig_out = _sig_stars(p_out)
        else:
            r_out, sig_out = np.nan, ''

        text_in = f'r={r_in:.2f} {sig_in}' if np.isfinite(r_in) else ''
        text_out = f'r={r_out:.2f} {sig_out}' if np.isfinite(r_out) else ''
        y_text_pos = 0.95
        if text_in:
            ax.text(0.98, y_text_pos, text_in, ha='right', va='top', transform=ax.transAxes, fontsize=5, fontname='Arial', color=in_pf_color)
            y_text_pos -= 0.15
        if text_out:
            ax.text(0.98, y_text_pos, text_out, ha='right', va='top', transform=ax.transAxes, fontsize=5, fontname='Arial', color=out_pf_color)

        ax.set_xlabel(xlabel, fontsize=6, fontname='Arial')
        if col_idx == 0:
            ax.set_ylabel('CB rate (Hz)', fontsize=6, fontname='Arial')
        else:
            ax.tick_params(labelleft=False)
            ax.set_ylabel('')
        ax.tick_params(labelsize=5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    fig.savefig(save_path, dpi=300)
    return fig


def _paired_test_inout(data1, data2):
    d1 = np.asarray(data1, dtype=float)
    d2 = np.asarray(data2, dtype=float)
    mask = np.isfinite(d1) & np.isfinite(d2)
    d1, d2 = d1[mask], d2[mask]
    n = len(d1)
    if n < 3:
        return np.nan, np.nan, 'N/A', n, np.nan
    diff = d1 - d2
    try:
        shapiro_p = scipy_stats.shapiro(diff).pvalue
    except ValueError:
        shapiro_p = np.nan
    if n <= 6:
        stat, p = scipy_stats.wilcoxon(d1, d2)
        test_name = 'Wilcoxon'
    elif np.isfinite(shapiro_p) and shapiro_p > 0.05:
        stat, p = scipy_stats.ttest_rel(d1, d2)
        test_name = 'Paired t-test'
    else:
        stat, p = scipy_stats.wilcoxon(d1, d2)
        test_name = 'Wilcoxon'
    return p, stat, test_name, n, shapiro_p


def _unpaired_test_inout(data1, data2):
    d1 = np.asarray(data1, dtype=float)
    d2 = np.asarray(data2, dtype=float)
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]
    if len(d1) < 3 or len(d2) < 3:
        return np.nan, 'N/A', np.nan, np.nan
    try:
        shapiro_p1 = scipy_stats.shapiro(d1).pvalue
    except ValueError:
        shapiro_p1 = np.nan
    try:
        shapiro_p2 = scipy_stats.shapiro(d2).pvalue
    except ValueError:
        shapiro_p2 = np.nan
    if min(len(d1), len(d2)) <= 6:
        _, p = scipy_stats.mannwhitneyu(d1, d2, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    elif (np.isfinite(shapiro_p1) and shapiro_p1 > 0.05) and (np.isfinite(shapiro_p2) and shapiro_p2 > 0.05):
        _, p = scipy_stats.ttest_ind(d1, d2)
        test_name = 'Unpaired t-test'
    else:
        _, p = scipy_stats.mannwhitneyu(d1, d2, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    return p, test_name, shapiro_p1, shapiro_p2


def _add_bracket_compact(ax, x1, x2, y, h, text, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.003, text, ha='center', va='bottom', fontsize=5, color=color)


def _plot_inout_panel(df_cs_plc, df_non_cs_plc, prefix: str, ylabel: str, title: str, save_path: str):
    in_color = 'magenta'
    out_color = 'gray'

    df_csplus_paired = df_cs_plc[[f'{prefix}_loco_in', f'{prefix}_loco_out']].dropna()
    csplus_in_paired = df_csplus_paired[f'{prefix}_loco_in'].values
    csplus_out_paired = df_csplus_paired[f'{prefix}_loco_out'].values

    df_csminus_paired = df_non_cs_plc[[f'{prefix}_loco_in', f'{prefix}_loco_out']].dropna()
    csminus_in_paired = df_csminus_paired[f'{prefix}_loco_in'].values
    csminus_out_paired = df_csminus_paired[f'{prefix}_loco_out'].values

    fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.5))
    ax.axvspan(0.5, 2.5, alpha=0.3, color=CS_PLC_BG, zorder=0)
    ax.axvspan(3.0, 5.0, alpha=0.3, color=NON_CS_PLC_BG, zorder=0)

    positions = [1, 2, 3.5, 4.5]
    data = [csplus_in_paired, csplus_out_paired, csminus_in_paired, csminus_out_paired]
    violin_colors = [in_color, out_color, in_color, out_color]
    half_sides = ['left', 'right', 'left', 'right']

    for pos, d, vc, side in zip(positions, data, violin_colors, half_sides):
        if len(d) == 0:
            continue
        vp = ax.violinplot([d], positions=[pos], showmedians=True, showextrema=False)
        body = vp['bodies'][0]
        verts = body.get_paths()[0].vertices
        if side == 'left':
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
        else:
            verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
        body.set_facecolor(vc)
        body.set_edgecolor('none')
        body.set_alpha(0.3)
        if 'cmedians' in vp:
            segs = vp['cmedians'].get_segments()
            if len(segs) > 0:
                seg = segs[0].copy()
                if side == 'left':
                    seg[:, 0] = np.clip(seg[:, 0], -np.inf, pos)
                else:
                    seg[:, 0] = np.clip(seg[:, 0], pos, np.inf)
                vp['cmedians'].set_segments([seg])
            vp['cmedians'].set_color('#1F77B4')
            vp['cmedians'].set_linewidth(1.0)

            finite_mask = np.isfinite(d)
            if np.any(finite_mask):
                d_finite = d[finite_mask]
                y_range = np.ptp(d_finite) if len(d_finite) > 1 else 1.0
                y_spacing = y_range * 0.04
                x_step = 0.06
                sorted_idx = np.argsort(d_finite)
                xs_bee = np.zeros(len(d_finite))
                for si, oi in enumerate(sorted_idx):
                    layer = 0
                    while True:
                        candidate = x_step * layer
                        conflict = False
                        for sj in range(si):
                            oj = sorted_idx[sj]
                            if abs(d_finite[oi] - d_finite[oj]) < y_spacing:
                                if abs(candidate - xs_bee[oj]) < x_step * 0.9:
                                    conflict = True
                                    break
                        if not conflict:
                            xs_bee[oi] = candidate
                            break
                        layer += 1
                if side == 'left':
                    xs = pos + 0.05 + xs_bee
                else:
                    xs = pos - 0.05 - xs_bee
                ax.scatter(xs, d_finite, s=8, color='black', alpha=0.6, linewidths=0, zorder=3)

    for i in range(len(csplus_in_paired)):
        if np.isfinite(csplus_in_paired[i]) and np.isfinite(csplus_out_paired[i]):
            ax.plot([1.12, 1.88], [csplus_in_paired[i], csplus_out_paired[i]], color='black', alpha=0.3, linewidth=0.6, zorder=2)
    for i in range(len(csminus_in_paired)):
        if np.isfinite(csminus_in_paired[i]) and np.isfinite(csminus_out_paired[i]):
            ax.plot([3.62, 4.38], [csminus_in_paired[i], csminus_out_paired[i]], color='black', alpha=0.3, linewidth=0.6, zorder=2)

    all_vals = np.concatenate([v for v in data if len(v) > 0])
    if len(all_vals) > 0:
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        yrange = ymax - ymin
        ylim = (ymin - 0.05 * yrange, ymax + 0.85 * yrange)
    else:
        ymin, ymax, yrange = 0, 1, 1
        ylim = (0, 1)

    ax.set_ylim(ylim)
    ax.set_xlim(0.3, 5.2)
    ax.set_xticks([1, 2, 3.5, 4.5])
    ax.set_xticklabels(['In', 'Out', 'In', 'Out'], fontsize=5, fontname='Arial')
    ax.text(1.5, -0.18, 'CS+ PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
    ax.text(4.0, -0.18, 'CS- PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    ax.tick_params(axis='y', labelsize=5)
    ax.set_title(title, fontsize=6, fontname='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    data_ymax = ymax
    bracket_h = 0.05 * yrange
    level_gap = 0.25 * yrange
    y1 = data_ymax + 0.08 * yrange

    if len(csplus_in_paired) >= 3:
        p_csplus, _, _, _, _ = _paired_test_inout(csplus_in_paired, csplus_out_paired)
        _add_bracket_compact(ax, 1, 2, y1, bracket_h, _sig_label(p_csplus))
    if len(csminus_in_paired) >= 3:
        p_csminus, _, _, _, _ = _paired_test_inout(csminus_in_paired, csminus_out_paired)
        _add_bracket_compact(ax, 3.5, 4.5, y1, bracket_h, _sig_label(p_csminus))

    y2 = y1 + level_gap
    if len(csplus_in_paired) >= 3 and len(csminus_in_paired) >= 3:
        p_in, _, _, _ = _unpaired_test_inout(csplus_in_paired, csminus_in_paired)
        _add_bracket_compact(ax, 1, 3.5, y2, bracket_h, _sig_label(p_in), color=in_color)

    y3 = y2 + level_gap
    if len(csplus_out_paired) >= 3 and len(csminus_out_paired) >= 3:
        p_out, _, _, _ = _unpaired_test_inout(csplus_out_paired, csminus_out_paired)
        _add_bracket_compact(ax, 2, 4.5, y3, bracket_h, _sig_label(p_out), color=out_color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_theta_inout_loco_csplus_vs_csminus(df_cs_plc: pd.DataFrame, df_non_cs_plc: pd.DataFrame, save_path: str):
    return _plot_inout_panel(
        df_cs_plc,
        df_non_cs_plc,
        prefix='theta',
        ylabel='Theta amp',
        title='Locomotion',
        save_path=save_path,
    )


def plot_slow_vm_inout_loco_csplus_vs_csminus(df_cs_plc: pd.DataFrame, df_non_cs_plc: pd.DataFrame, save_path: str):
    return _plot_inout_panel(
        df_cs_plc,
        df_non_cs_plc,
        prefix='slow',
        ylabel='Slow Vm',
        title='Locomotion',
        save_path=save_path,
    )
