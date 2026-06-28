"""Core pooled-figure plotting functions (direct, notebook-independent)."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.spatial_heatmaps import (
    CS_PLC_DEFINITION_LEGACY,
    cell_has_cs_place_field,
    compute_cb_in_pf_counts,
    get_deleted_cells_with_fallback,
    is_csplus_place_cell,
    normalize_cs_plc_definition_mode,
)


VIOLIN_MEDIAN_COLOR = '#1F77B4'
VIOLIN_MEDIAN_LINEWIDTH = 1.0

ALL_COLOR = 'black'
SS_COLOR = '#026C80'
CS_COLOR = '#EE9B00'
CB_COLOR = '#C1121F'
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


def _collect_complex_burst_pf_loco_metrics(
    data_folder: str,
    folders: list[str],
) -> dict[tuple[str, int], dict[str, float]]:
    metrics_by_cell: dict[tuple[str, int], dict[str, float]] = {}
    for folder in folders:
        spatial_path = os.path.join(data_folder, folder, 'spatial_analysis_full.pkl')
        if not os.path.exists(spatial_path):
            continue
        with open(spatial_path, 'rb') as f:
            spatial_cells = pickle.load(f)
        for cell in spatial_cells:
            cell_idx = int(cell.get('cell_idx'))
            spike_burst_metrics = cell.get('spike_burst_rate_metrics', {})
            if not isinstance(spike_burst_metrics, dict):
                continue
            burst_rate = spike_burst_metrics.get('burst_rate', {})
            burst_counts = spike_burst_metrics.get('burst_counts', {})
            time_s = spike_burst_metrics.get('time_s', {})
            metrics_by_cell[(folder, cell_idx)] = {
                'complex_burst_event_rate_loco_in_pf': (
                    float(burst_rate.get('run_in', np.nan))
                    if isinstance(burst_rate, dict)
                    else np.nan
                ),
                'complex_burst_event_count_loco_in_pf': (
                    float(burst_counts.get('run_in', np.nan))
                    if isinstance(burst_counts, dict)
                    else np.nan
                ),
                'complex_burst_time_loco_in_pf': (
                    float(time_s.get('run_in', np.nan))
                    if isinstance(time_s, dict)
                    else np.nan
                ),
            }
    return metrics_by_cell


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


def _should_show_sig_bracket(p_val: float, show_only_significant: bool) -> bool:
    if not np.isfinite(p_val):
        return False
    return (float(p_val) < 0.05) if bool(show_only_significant) else True


def _draw_sig_bracket_levels(
    ax,
    bracket_levels,
    ylim,
    show_only_significant: bool,
    sig_marker_offset_frac: float = 0.0001,
) -> float:
    """Draw visible significance brackets with compact vertical spacing."""
    y_range = ylim[1] - ylim[0]
    h = y_range * 0.03
    text_h = y_range * 0.08
    gap = y_range * 0.05
    label_y_offset = y_range * float(sig_marker_offset_frac)
    y = ylim[1] + y_range * 0.05
    y_top = ylim[1]

    for level in bracket_levels:
        visible_specs = [
            spec for spec in level
            if _should_show_sig_bracket(spec['p'], show_only_significant)
        ]
        if not visible_specs:
            continue
        for spec in visible_specs:
            sig_text = _sig_label(spec['p'])
            text_kwargs = {}
            if '*' in sig_text:
                text_kwargs = {
                    'text_y': y + h + label_y_offset,
                    'text_va': 'bottom',
                }
            _add_bracket(
                ax,
                spec['pos1'],
                spec['pos2'],
                y,
                h,
                sig_text,
                color=spec.get('color', 'black'),
                **text_kwargs,
            )
        y_top = y + h + text_h
        y += h + text_h + gap

    return y_top


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


def _add_bracket(ax, pos1, pos2, y, h, sig_text, color='black', text_y=None, text_va='bottom'):
    drop = h
    ax.plot([pos1, pos1], [y, y + drop], color=color, linewidth=0.8)
    ax.plot([pos2, pos2], [y, y + drop], color=color, linewidth=0.8)
    ax.plot([pos1, pos2], [y + drop, y + drop], color=color, linewidth=0.8)
    ax.text(
        (pos1 + pos2) / 2,
        y + drop + 0.02 if text_y is None else text_y,
        sig_text,
        ha='center',
        va=text_va,
        fontsize=5,
        fontname='Arial',
        color=color,
    )


def prepare_pooled_stats_tables(
    data_folder: str,
    folders: list[str],
    cb_num_threshold: int = 10,
    cs_peak_rate_threshold: float = 0.5,
    cs_plc_definition_mode: str = CS_PLC_DEFINITION_LEGACY,
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
    cb_pf_loco_metrics = _collect_complex_burst_pf_loco_metrics(data_folder, folders)

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
    has_cs_place_field = df_pc.apply(lambda row: cell_has_cs_place_field(row.to_dict()), axis=1).to_numpy(dtype=bool)
    cs_plc_mode = normalize_cs_plc_definition_mode(cs_plc_definition_mode)
    is_cs_plc = np.asarray(
        [
            is_csplus_place_cell(
                is_place_cell=True,
                n_cb_in_pf=int(n_cb),
                cs_peak_rate=float(cs_peak),
                cb_num_threshold=int(cb_num_threshold),
                cs_peak_rate_threshold=float(cs_peak_rate_threshold),
                has_cs_place_field=bool(has_cs_pf),
                cs_plc_definition_mode=cs_plc_mode,
            )
            for n_cb, cs_peak, has_cs_pf in zip(
                n_cb_in_pf,
                cs_peak_rates,
                has_cs_place_field,
            )
        ],
        dtype=bool,
    )

    df_pc['is_cs_plc'] = is_cs_plc
    df_pc['n_cb_in_pf'] = n_cb_in_pf
    df_pc['has_cs_place_field'] = has_cs_place_field
    df_pc['cs_plc_definition_mode'] = cs_plc_mode

    df_all.loc[df_all['is_place_cell'] == True, 'is_cs_plc'] = False
    df_all.loc[df_all['is_place_cell'] == True, 'n_cb_in_pf'] = 0
    df_all.loc[df_all['is_place_cell'] == True, 'has_cs_place_field'] = False
    df_all.loc[df_all['is_place_cell'] == True, 'cs_plc_definition_mode'] = cs_plc_mode
    df_all.loc[df_pc.index, 'is_cs_plc'] = is_cs_plc
    df_all.loc[df_pc.index, 'n_cb_in_pf'] = n_cb_in_pf
    df_all.loc[df_pc.index, 'has_cs_place_field'] = has_cs_place_field
    df_all.loc[df_pc.index, 'cs_plc_definition_mode'] = cs_plc_mode

    # All-spike PF reference in-field firing-rate metrics (Hz).
    # These are directly sourced from in/out stats computed using the all-spike PF mask.
    df_all["fr_in_allpf_all"] = pd.to_numeric(df_all.get("all_inout_loco_in", np.nan), errors="coerce")
    df_all["fr_in_allpf_ss"] = pd.to_numeric(df_all.get("ss_inout_loco_in", np.nan), errors="coerce")
    df_all["fr_in_allpf_cs"] = pd.to_numeric(df_all.get("cs_inout_loco_in", np.nan), errors="coerce")
    df_all["cs_rate_loco_in_pf"] = pd.to_numeric(df_all.get("cs_inout_loco_in", np.nan), errors="coerce")
    df_all["complex_burst_event_rate_loco_in_pf"] = df_all.apply(
        lambda row: cb_pf_loco_metrics.get(
            (row["session"], int(row["cell_idx"])),
            {},
        ).get("complex_burst_event_rate_loco_in_pf", np.nan),
        axis=1,
    )
    df_all["complex_burst_event_count_loco_in_pf"] = df_all.apply(
        lambda row: cb_pf_loco_metrics.get(
            (row["session"], int(row["cell_idx"])),
            {},
        ).get(
            "complex_burst_event_count_loco_in_pf",
            cb_in_pf_counts.get((row["session"], int(row["cell_idx"])), np.nan),
        ),
        axis=1,
    )
    df_all["complex_burst_time_loco_in_pf"] = df_all.apply(
        lambda row: cb_pf_loco_metrics.get(
            (row["session"], int(row["cell_idx"])),
            {},
        ).get("complex_burst_time_loco_in_pf", np.nan),
        axis=1,
    )

    df_pc = df_all[df_all['is_place_cell'] == True].copy()
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

    def _centered_beeswarm(vals, base_x):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return np.array([])
        y_range = np.ptp(vals) if len(vals) > 1 else 1.0
        if not np.isfinite(y_range) or y_range <= 0:
            y_range = 1.0
        y_spacing = y_range * 0.04
        x_step = 0.045
        sorted_idx = np.argsort(vals)
        offsets = np.zeros(len(vals), dtype=float)
        candidates = [0.0]
        for layer in range(1, max(3, len(vals) + 1)):
            candidates.extend([x_step * layer, -x_step * layer])
        for si, oi in enumerate(sorted_idx):
            for candidate in candidates:
                conflict = False
                for sj in range(si):
                    oj = sorted_idx[sj]
                    if abs(vals[oi] - vals[oj]) < y_spacing:
                        if abs(candidate - offsets[oj]) < x_step * 0.9:
                            conflict = True
                            break
                if not conflict:
                    offsets[oi] = candidate
                    break
        return base_x + offsets

    for i, (pos, data) in enumerate(zip(positions, data_list)):
        if len(data) < 1:
            continue
        if i in paired_left:
            xs = _beeswarm(data, pos, direction=1)
        elif i in paired_right:
            xs = _beeswarm(data, pos, direction=-1)
        else:
            xs = _centered_beeswarm(data, pos)
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
    show_only_significant: bool = False,
    sig_marker_offset_frac: float = 0.0001,
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
        bracket_levels = []
        if len(cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(cs_ss_paired, cs_cs_paired)
            bracket_levels.append([{'pos1': 2, 'pos2': 3, 'p': p}])
        if len(cs_all) >= 3 and len(non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_all, non_all)
            bracket_levels.append([{'pos1': 1, 'pos2': 4.5, 'p': p, 'color': ALL_COLOR}])
        if plot_cs_minus_ss and len(cs_ss) >= 3 and len(non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_ss, non_ss)
            bracket_levels.append([{'pos1': 2, 'pos2': 5.5, 'p': p, 'color': SS_COLOR}])

        y_range = ylim[1] - ylim[0]
        y_top = _draw_sig_bracket_levels(
            ax,
            bracket_levels,
            ylim,
            show_only_significant,
            sig_marker_offset_frac=sig_marker_offset_frac,
        )
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


def plot_moving_epoch_cs_cb_metrics_allcells_placecells(
    summary_df: pd.DataFrame,
    save_path: str,
    classification_df: pd.DataFrame | None = None,
    hist_bins: int = 10,
    count_bin_size: float = 5.0,
    within_pf: bool = True,
    fig_width: float = 4.8,
    fig_height: float = 3.6,
):
    """Plot moving-epoch complex spike and complex burst metric histograms."""
    moving_metric_summary_df = summary_df.copy()
    if classification_df is not None:
        key_cols = ['session', 'cell_idx']
        missing_keys = [col for col in key_cols if col not in moving_metric_summary_df.columns or col not in classification_df.columns]
        if missing_keys:
            raise KeyError(f"summary_df and classification_df must both contain keys: {missing_keys}")
        overlay_cols = [
            'is_place_cell',
            'is_cs_plc',
            'n_cb_in_pf',
            'cs_rate_loco_in_pf',
            'complex_burst_event_rate_loco_in_pf',
            'complex_burst_event_count_loco_in_pf',
            'complex_burst_time_loco_in_pf',
        ]
        overlay_cols = [col for col in overlay_cols if col in classification_df.columns]
        if overlay_cols:
            moving_metric_summary_df = moving_metric_summary_df.drop(
                columns=[col for col in overlay_cols if col in moving_metric_summary_df.columns],
            ).merge(
                classification_df[key_cols + overlay_cols],
                on=key_cols,
                how='left',
                validate='one_to_one',
            )

    required_cols = [
        'is_place_cell',
        'is_cs_plc',
        'cs_rate_loco',
        'complex_burst_event_rate_loco',
        'complex_burst_event_count_loco',
    ]
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise KeyError(f"summary_df is missing required columns: {missing_cols}")
    hist_bins = int(hist_bins)
    if hist_bins <= 0:
        raise ValueError("hist_bins must be a positive integer.")
    count_bin_size = float(count_bin_size)
    if (not np.isfinite(count_bin_size)) or count_bin_size <= 0:
        raise ValueError("count_bin_size must be a finite number > 0.")

    if 'cs_rate_loco_in_pf' not in moving_metric_summary_df.columns and 'cs_inout_loco_in' in moving_metric_summary_df.columns:
        moving_metric_summary_df['cs_rate_loco_in_pf'] = pd.to_numeric(
            moving_metric_summary_df['cs_inout_loco_in'],
            errors='coerce',
        )
    if 'complex_burst_event_count_loco_in_pf' not in moving_metric_summary_df.columns and 'n_cb_in_pf' in moving_metric_summary_df.columns:
        moving_metric_summary_df['complex_burst_event_count_loco_in_pf'] = pd.to_numeric(
            moving_metric_summary_df['n_cb_in_pf'],
            errors='coerce',
        )

    whole_epoch_specs = [
        ('cs_rate_loco', 'CS firing rate\n(Hz)'),
        ('complex_burst_event_rate_loco', 'CB event rate\n(Hz)'),
        ('complex_burst_event_count_loco', 'CB event count'),
    ]
    within_pf_specs = [
        ('cs_rate_loco_in_pf', 'CS firing rate\nin PF (Hz)'),
        ('complex_burst_event_rate_loco_in_pf', 'CB event rate\nin PF (Hz)'),
        ('complex_burst_event_count_loco_in_pf', 'CB event count\nin PF'),
    ]
    if within_pf:
        missing_within_pf_cols = [
            col for col, _ in within_pf_specs if col not in moving_metric_summary_df.columns
        ]
        if missing_within_pf_cols:
            raise KeyError(
                "summary_df is missing required within-PF columns: "
                f"{missing_within_pf_cols}. Rebuild pooled_stats with prepare_pooled_stats_tables()."
            )

    is_place_cell = moving_metric_summary_df['is_place_cell'] == True
    is_cs_minus_place_cell = is_place_cell & (moving_metric_summary_df['is_cs_plc'] == False)
    moving_metric_groups = [
        ('All cells', moving_metric_summary_df, whole_epoch_specs),
        (
            'Place cells',
            moving_metric_summary_df[is_place_cell],
            within_pf_specs if within_pf else whole_epoch_specs,
        ),
        (
            'CS- place cells',
            moving_metric_summary_df[is_cs_minus_place_cell],
            within_pf_specs if within_pf else whole_epoch_specs,
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    axes = np.asarray(axes)
    hist_color = CS_COLOR

    def _count_hist_bins_from_values(values: list[np.ndarray]) -> np.ndarray:
        finite_vals = []
        for arr in values:
            arr = np.asarray(arr, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                finite_vals.append(finite)
        if not finite_vals:
            return np.array([-0.5, count_bin_size + 0.5], dtype=float)
        max_val = float(np.nanmax(np.concatenate(finite_vals)))
        if max_val <= count_bin_size:
            return np.array([-0.5, count_bin_size + 0.5], dtype=float)
        n_extra_bins = int(np.ceil((max_val - count_bin_size) / count_bin_size))
        upper_edges = count_bin_size + 0.5 + np.arange(n_extra_bins + 1) * count_bin_size
        return np.concatenate(([-0.5], upper_edges.astype(float)))

    count_col_values = []
    for _group_label, group_df, metric_specs in moving_metric_groups:
        count_col = metric_specs[2][0]
        count_col_values.append(pd.to_numeric(group_df[count_col], errors='coerce').to_numpy(dtype=float))
    shared_count_bins = _count_hist_bins_from_values(count_col_values)

    def _hist_bins(metric_col: str):
        if metric_col.startswith('complex_burst_event_count_loco'):
            return shared_count_bins
        return hist_bins

    for row_idx, (group_label, group_df, metric_specs) in enumerate(moving_metric_groups):
        for col_idx, (metric_col, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            vals = pd.to_numeric(group_df[metric_col], errors='coerce').to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size > 0:
                ax.hist(
                    vals,
                    bins=_hist_bins(metric_col),
                    color=hist_color,
                    alpha=0.35,
                    edgecolor='black',
                    linewidth=0.4,
                )

            ax.text(
                0.97,
                0.95,
                f'n={vals.size}',
                ha='right',
                va='top',
                transform=ax.transAxes,
                fontsize=5,
                fontname='Arial',
            )
            ax.set_xlabel(ylabel, fontsize=6, fontname='Arial')
            ax.set_ylabel('Cell count', fontsize=6, fontname='Arial')
            ax.set_title(group_label, fontsize=6, fontname='Arial')
            if metric_col.startswith('complex_burst_event_count_loco'):
                ax.set_xlim(left=0.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_combined_cs_plus_minus_4panels(
    df_cs_plc: pd.DataFrame,
    df_non_cs_plc: pd.DataFrame,
    save_path: str,
    plot_cs_minus_ss: bool = False,
    fig_width: float = 9.6,
    fig_height: float = 1.4,
    show_only_significant: bool = True,
    sig_marker_offset_frac: float = 0.0001,
):
    sig_marker_offset_frac = float(sig_marker_offset_frac)
    if not np.isfinite(sig_marker_offset_frac):
        raise ValueError("sig_marker_offset_frac must be a finite number.")

    first_panel_ratio = 1.4
    fig, axes = plt.subplots(
        1,
        8,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [first_panel_ratio, 1, 1, 1, 1, 1, 1, first_panel_ratio]},
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
        bracket_levels = []
        paired_level = []
        if len(cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(cs_ss_paired, cs_cs_paired)
            paired_level.append({'pos1': 2, 'pos2': 3, 'p': p})
        if len(non_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(non_ss_paired, non_cs_paired)
            paired_level.append({'pos1': 5.5, 'pos2': 6.5, 'p': p})
        if paired_level:
            bracket_levels.append(paired_level)
        if len(cs_all) >= 3 and len(non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_all, non_all)
            bracket_levels.append([{'pos1': 1, 'pos2': 4.5, 'p': p, 'color': ALL_COLOR}])
        if len(cs_ss) >= 3 and len(non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_ss, non_ss)
            bracket_levels.append([{'pos1': 2, 'pos2': 5.5, 'p': p, 'color': SS_COLOR}])
        if len(cs_cs) >= 3 and len(non_cs) >= 3:
            p, _, _, _ = _unpaired_test_auto(cs_cs, non_cs)
            bracket_levels.append([{'pos1': 3, 'pos2': 6.5, 'p': p, 'color': CS_COLOR}])
        y_range = ylim[1] - ylim[0]
        y_top = _draw_sig_bracket_levels(
            ax,
            bracket_levels,
            ylim,
            show_only_significant,
            sig_marker_offset_frac=sig_marker_offset_frac,
        )
        ax.set_ylim(ylim[0], y_top + y_range * 0.05)

    def _get_pf_size_data(df_subset: pd.DataFrame, reducer):
        all_sizes, ss_sizes, cs_sizes = [], [], []
        for idx in df_subset.index:
            sizes = df_subset.loc[idx, 'place_field_sizes_cm2']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                all_sizes.append(reducer(sizes) / arena_area_cm2 * 100)
            else:
                all_sizes.append(np.nan)

            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_ss']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                ss_sizes.append(reducer(sizes) / arena_area_cm2 * 100)
            else:
                ss_sizes.append(np.nan)

            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_cs']
            if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0:
                cs_sizes.append(reducer(sizes) / arena_area_cm2 * 100)
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

    def _sum_pf_sizes(sizes):
        return float(np.nansum(np.asarray(sizes, dtype=float)))

    def _primary_pf_size(sizes):
        arr = np.asarray(sizes, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return float(arr[0]) if arr.size > 0 else np.nan

    pf_cs_all, pf_cs_ss, pf_cs_cs, pf_cs_ss_paired, pf_cs_cs_paired = _get_pf_size_data(df_cs_plc, _sum_pf_sizes)
    pf_non_all, pf_non_ss, pf_non_cs, pf_non_ss_paired, pf_non_cs_paired = _get_pf_size_data(df_non_cs_plc, _sum_pf_sizes)
    _plot_pf_style_panel(
        axes[1],
        (pf_cs_all, pf_cs_ss, pf_cs_cs, pf_cs_ss_paired, pf_cs_cs_paired),
        (pf_non_all, pf_non_ss, pf_non_cs, pf_non_ss_paired, pf_non_cs_paired),
        ylabel='PF size all PFs (% arena)',
        plot_cs_minus_ss=plot_cs_minus_ss,
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
    )

    primary_pf_cs_all, primary_pf_cs_ss, primary_pf_cs_cs, primary_pf_cs_ss_paired, primary_pf_cs_cs_paired = _get_pf_size_data(df_cs_plc, _primary_pf_size)
    primary_pf_non_all, primary_pf_non_ss, primary_pf_non_cs, primary_pf_non_ss_paired, primary_pf_non_cs_paired = _get_pf_size_data(df_non_cs_plc, _primary_pf_size)
    _plot_pf_style_panel(
        axes[2],
        (primary_pf_cs_all, primary_pf_cs_ss, primary_pf_cs_cs, primary_pf_cs_ss_paired, primary_pf_cs_cs_paired),
        (primary_pf_non_all, primary_pf_non_ss, primary_pf_non_cs, primary_pf_non_ss_paired, primary_pf_non_cs_paired),
        ylabel='Primary PF size (% arena)',
        plot_cs_minus_ss=plot_cs_minus_ss,
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
    )

    # Panel 4: Coherence.
    coh_cs = _get_numeric_triplet_data(df_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs')
    coh_non = _get_numeric_triplet_data(df_non_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs')
    _plot_pf_style_panel(
        axes[3],
        coh_cs,
        coh_non,
        ylabel='Coherence (z)',
        plot_cs_minus_ss=plot_cs_minus_ss,
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
    )

    # Panel 5: Sparsity.
    sparsity_cs = _get_numeric_triplet_data(df_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
    sparsity_non = _get_numeric_triplet_data(df_non_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
    sparsity_data_list = (
        [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0], sparsity_non[1]]
        if plot_cs_minus_ss
        else [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0]]
    )
    _plot_pf_style_panel(
        axes[4],
        sparsity_cs,
        sparsity_non,
        ylabel='Sparsity',
        plot_cs_minus_ss=plot_cs_minus_ss,
        clamp_unit_interval=_all_finite_in_unit_interval(sparsity_data_list),
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
    )

    # Panel 6: Spatial information.
    si_cs = _get_numeric_triplet_data(df_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs')
    si_non = _get_numeric_triplet_data(df_non_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs')
    _plot_pf_style_panel(
        axes[5],
        si_cs,
        si_non,
        ylabel='SI (bits/spike)',
        plot_cs_minus_ss=plot_cs_minus_ss,
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
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

    # Panel 7: Selectivity.
    sel_cs_all, sel_cs_ss, sel_cs_cs, sel_cs_ss_paired, sel_cs_cs_paired = _get_selectivity(df_cs_plc)
    sel_non_all, sel_non_ss, sel_non_cs, sel_non_ss_paired, sel_non_cs_paired = _get_selectivity(df_non_cs_plc)
    _plot_pf_style_panel(
        axes[6],
        (sel_cs_all, sel_cs_ss, sel_cs_cs, sel_cs_ss_paired, sel_cs_cs_paired),
        (sel_non_all, sel_non_ss, sel_non_cs, sel_non_ss_paired, sel_non_cs_paired),
        ylabel='Selectivity',
        plot_cs_minus_ss=plot_cs_minus_ss,
        zero_line=True,
        show_only_significant=show_only_significant,
        sig_marker_offset_frac=sig_marker_offset_frac,
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

    # Panel 8: In-field firing rate using all-spike PF reference mask.
    # Match Panel 1 layout/style: All/SS/CS for both CS+ and CS- PLC.
    ax = axes[7]
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
        bracket_levels = []
        paired_level = []
        if len(fr_cs_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(fr_cs_ss_paired, fr_cs_cs_paired)
            paired_level.append({'pos1': 2, 'pos2': 3, 'p': p})
        if len(fr_non_ss_paired) >= 3:
            p, _, _, _, _ = _paired_test(fr_non_ss_paired, fr_non_cs_paired)
            paired_level.append({'pos1': 5.5, 'pos2': 6.5, 'p': p})
        if paired_level:
            bracket_levels.append(paired_level)
        if len(fr_cs_all) >= 3 and len(fr_non_all) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_all, fr_non_all)
            bracket_levels.append([{'pos1': 1, 'pos2': 4.5, 'p': p, 'color': ALL_COLOR}])
        if len(fr_cs_ss) >= 3 and len(fr_non_ss) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_ss, fr_non_ss)
            bracket_levels.append([{'pos1': 2, 'pos2': 5.5, 'p': p, 'color': SS_COLOR}])
        if len(fr_cs_cs) >= 3 and len(fr_non_cs) >= 3:
            p, _, _, _ = _unpaired_test_auto(fr_cs_cs, fr_non_cs)
            bracket_levels.append([{'pos1': 3, 'pos2': 6.5, 'p': p, 'color': CS_COLOR}])
        y_range = ylim[1] - ylim[0]
        y_top = _draw_sig_bracket_levels(
            ax,
            bracket_levels,
            ylim,
            show_only_significant,
            sig_marker_offset_frac=sig_marker_offset_frac,
        )
        ax.set_ylim(ylim[0], y_top + y_range * 0.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_combined_cs_plus_minus_4panels_2sessions(
    session_tables: dict[str, dict[str, pd.DataFrame]],
    save_path: str,
    plot_cs_minus_ss: bool = False,
    fig_width: float = 7.5,
    fig_height: float = 2.4,
):
    """Plot the CS+/CS- pooled 7-panel summary separately for session 1 and session 2."""
    first_panel_ratio = 1.4
    fig, axes = plt.subplots(
        2,
        7,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [first_panel_ratio, 1, 1, 1, 1, 1, first_panel_ratio]},
    )
    axes = np.asarray(axes)
    positions_6 = [1, 2, 3, 4.5, 5.5, 6.5]
    arena_area_cm2 = 20.0 * 35.5

    required_cols = [
        'peak_rate_all', 'peak_rate_ss', 'peak_rate_cs',
        'place_field_sizes_cm2', 'place_field_sizes_cm2_ss', 'place_field_sizes_cm2_cs',
        'coherence_all', 'coherence_ss', 'coherence_cs',
        'sparsity_all', 'sparsity_ss', 'sparsity_cs',
        'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs',
        'all_inout_loco_in', 'all_inout_loco_out',
        'ss_inout_loco_in', 'ss_inout_loco_out',
        'cs_inout_loco_in', 'cs_inout_loco_out',
        'fr_in_allpf_all', 'fr_in_allpf_ss', 'fr_in_allpf_cs',
    ]

    def _coerce_df(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        return df.reindex(columns=list(dict.fromkeys(list(df.columns) + required_cols)))

    def _session_pair(session_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        block = session_tables.get(session_key, {})
        return _coerce_df(block.get('df_cs_plc')), _coerce_df(block.get('df_non_cs_plc'))

    def _get_pf_size_sum(df_subset: pd.DataFrame):
        all_sizes, ss_sizes, cs_sizes = [], [], []
        for idx in df_subset.index:
            sizes = df_subset.loc[idx, 'place_field_sizes_cm2']
            all_sizes.append(np.sum(sizes) / arena_area_cm2 * 100 if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0 else np.nan)
            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_ss']
            ss_sizes.append(np.sum(sizes) / arena_area_cm2 * 100 if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0 else np.nan)
            sizes = df_subset.loc[idx, 'place_field_sizes_cm2_cs']
            cs_sizes.append(np.sum(sizes) / arena_area_cm2 * 100 if isinstance(sizes, (list, np.ndarray)) and len(sizes) > 0 else np.nan)

        all_sizes = np.asarray(all_sizes, dtype=float)
        ss_sizes = np.asarray(ss_sizes, dtype=float)
        cs_sizes = np.asarray(cs_sizes, dtype=float)
        paired_mask = np.isfinite(ss_sizes) & np.isfinite(cs_sizes)
        return (
            all_sizes[np.isfinite(all_sizes)],
            ss_sizes[np.isfinite(ss_sizes)],
            cs_sizes[np.isfinite(cs_sizes)],
            ss_sizes[paired_mask],
            cs_sizes[paired_mask],
        )

    def _get_selectivity(df_subset: pd.DataFrame):
        all_in = pd.to_numeric(df_subset['all_inout_loco_in'], errors='coerce').to_numpy(dtype=float)
        all_out = pd.to_numeric(df_subset['all_inout_loco_out'], errors='coerce').to_numpy(dtype=float)
        ss_in = pd.to_numeric(df_subset['ss_inout_loco_in'], errors='coerce').to_numpy(dtype=float)
        ss_out = pd.to_numeric(df_subset['ss_inout_loco_out'], errors='coerce').to_numpy(dtype=float)
        cs_in = pd.to_numeric(df_subset['cs_inout_loco_in'], errors='coerce').to_numpy(dtype=float)
        cs_out = pd.to_numeric(df_subset['cs_inout_loco_out'], errors='coerce').to_numpy(dtype=float)

        def _sel(v_in, v_out):
            denom = v_in + v_out
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where((denom > 0) & np.isfinite(v_in) & np.isfinite(v_out), (v_in - v_out) / denom, np.nan)

        all_sel = _sel(all_in, all_out)
        ss_sel = _sel(ss_in, ss_out)
        cs_sel = _sel(cs_in, cs_out)
        paired_mask = np.isfinite(ss_sel) & np.isfinite(cs_sel)
        return (
            all_sel[np.isfinite(all_sel)],
            ss_sel[np.isfinite(ss_sel)],
            cs_sel[np.isfinite(cs_sel)],
            ss_sel[paired_mask],
            cs_sel[paired_mask],
        )

    def _get_fr_in_ref_pf_data(df_subset: pd.DataFrame):
        all_fr = pd.to_numeric(df_subset['fr_in_allpf_all'], errors='coerce').to_numpy(dtype=float)
        ss_fr = pd.to_numeric(df_subset['fr_in_allpf_ss'], errors='coerce').to_numpy(dtype=float)
        cs_fr = pd.to_numeric(df_subset['fr_in_allpf_cs'], errors='coerce').to_numpy(dtype=float)
        paired_mask = np.isfinite(ss_fr) & np.isfinite(cs_fr)
        return (
            all_fr[np.isfinite(all_fr)],
            ss_fr[np.isfinite(ss_fr)],
            cs_fr[np.isfinite(cs_fr)],
            ss_fr[paired_mask],
            cs_fr[paired_mask],
        )

    def _draw_six_position_panel(ax, cs_metric, non_metric, ylabel: str):
        cs_all, cs_ss, cs_cs, cs_ss_paired, cs_cs_paired = cs_metric
        non_all, non_ss, non_cs, non_ss_paired, non_cs_paired = non_metric
        ax.axvspan(0.5, 3.5, alpha=0.3, color=CS_PLC_BG, zorder=0)
        ax.axvspan(4.0, 7.0, alpha=0.3, color=NON_CS_PLC_BG, zorder=0)
        ax.text(2, -0.22, 'CS+ PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())
        ax.text(5.5, -0.22, 'CS- PLC', ha='center', va='top', fontsize=6, fontname='Arial', transform=ax.get_xaxis_transform())

        data_list = [cs_all, cs_ss, cs_cs, non_all, non_ss, non_cs]
        colors_list = [ALL_COLOR, SS_COLOR, CS_COLOR, ALL_COLOR, SS_COLOR, CS_COLOR]
        ylim = _global_ylim(data_list)
        _half_violin_panel(ax, positions_6, data_list, colors_list, paired_specs=[(1, 2), (4, 5)])
        ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
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
            y1 = ylim[1] + y_range * 0.05
            y_top = y1
            if len(cs_ss_paired) >= 3:
                p, _, _, _, _ = _paired_test(cs_ss_paired, cs_cs_paired)
                _add_bracket(ax, 2, 3, y1, h, _sig_label(p))
                y_top = max(y_top, y1 + h + text_h)
            if len(non_ss_paired) >= 3:
                p, _, _, _, _ = _paired_test(non_ss_paired, non_cs_paired)
                _add_bracket(ax, 5.5, 6.5, y1, h, _sig_label(p))
                y_top = max(y_top, y1 + h + text_h)
            y2 = y1 + h + text_h + gap
            if len(cs_all) >= 3 and len(non_all) >= 3:
                p, _, _, _ = _unpaired_test_auto(cs_all, non_all)
                _add_bracket(ax, 1, 4.5, y2, h, _sig_label(p), color=ALL_COLOR)
                y_top = max(y_top, y2 + h + text_h)
            y3 = y2 + h + text_h + gap
            if len(cs_ss) >= 3 and len(non_ss) >= 3:
                p, _, _, _ = _unpaired_test_auto(cs_ss, non_ss)
                _add_bracket(ax, 2, 5.5, y3, h, _sig_label(p), color=SS_COLOR)
                y_top = max(y_top, y3 + h + text_h)
            y4 = y3 + h + text_h + gap
            if len(cs_cs) >= 3 and len(non_cs) >= 3:
                p, _, _, _ = _unpaired_test_auto(cs_cs, non_cs)
                _add_bracket(ax, 3, 6.5, y4, h, _sig_label(p), color=CS_COLOR)
                y_top = max(y_top, y4 + h + text_h)
            ax.set_ylim(ylim[0], y_top + y_range * 0.05)

    def _draw_row(row_idx: int, session_key: str, row_label: str):
        df_cs_plc, df_non_cs_plc = _session_pair(session_key)
        row_axes = axes[row_idx]

        _draw_six_position_panel(
            row_axes[0],
            _get_peak_rate_data(df_cs_plc),
            _get_peak_rate_data(df_non_cs_plc),
            'Peak rate (Hz)',
        )

        _plot_pf_style_panel(
            row_axes[1],
            _get_pf_size_sum(df_cs_plc),
            _get_pf_size_sum(df_non_cs_plc),
            ylabel='PF size (% arena)',
            plot_cs_minus_ss=plot_cs_minus_ss,
            fixed_ylim=(0, 100),
            yticks=[0, 25, 50, 75, 100],
        )
        _plot_pf_style_panel(
            row_axes[2],
            _get_numeric_triplet_data(df_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs'),
            _get_numeric_triplet_data(df_non_cs_plc, 'coherence_all', 'coherence_ss', 'coherence_cs'),
            ylabel='Coherence (z)',
            plot_cs_minus_ss=plot_cs_minus_ss,
        )
        sparsity_cs = _get_numeric_triplet_data(df_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
        sparsity_non = _get_numeric_triplet_data(df_non_cs_plc, 'sparsity_all', 'sparsity_ss', 'sparsity_cs')
        sparsity_data_list = (
            [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0], sparsity_non[1]]
            if plot_cs_minus_ss
            else [sparsity_cs[0], sparsity_cs[1], sparsity_cs[2], sparsity_non[0]]
        )
        _plot_pf_style_panel(
            row_axes[3],
            sparsity_cs,
            sparsity_non,
            ylabel='Sparsity',
            plot_cs_minus_ss=plot_cs_minus_ss,
            clamp_unit_interval=_all_finite_in_unit_interval(sparsity_data_list),
        )
        _plot_pf_style_panel(
            row_axes[4],
            _get_numeric_triplet_data(df_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs'),
            _get_numeric_triplet_data(df_non_cs_plc, 'si_bits_per_spike', 'si_bits_per_spike_ss', 'si_bits_per_spike_cs'),
            ylabel='SI (bits/spike)',
            plot_cs_minus_ss=plot_cs_minus_ss,
        )
        _plot_pf_style_panel(
            row_axes[5],
            _get_selectivity(df_cs_plc),
            _get_selectivity(df_non_cs_plc),
            ylabel='Selectivity',
            plot_cs_minus_ss=plot_cs_minus_ss,
            zero_line=True,
        )
        _draw_six_position_panel(
            row_axes[6],
            _get_fr_in_ref_pf_data(df_cs_plc),
            _get_fr_in_ref_pf_data(df_non_cs_plc),
            'In-PF FR (Hz)',
        )
        row_axes[0].text(
            -0.42,
            0.5,
            row_label,
            transform=row_axes[0].transAxes,
            rotation=90,
            ha='center',
            va='center',
            fontsize=6,
            fontname='Arial',
        )

    _draw_row(0, 'session1', 'Session 1')
    _draw_row(1, 'session2', 'Session 2')

    for col_idx in range(axes.shape[1]):
        ylims = [axes[row_idx, col_idx].get_ylim() for row_idx in range(axes.shape[0])]
        y_min = min(y[0] for y in ylims)
        y_max = max(y[1] for y in ylims)
        for row_idx in range(axes.shape[0]):
            axes[row_idx, col_idx].set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_combined_cs_plus_minus_session_pair_metrics_14cols(
    session_tables: dict[str, dict[str, pd.DataFrame]],
    save_path: str,
    fig_width: float = 14.5,
    fig_height: float = 7.0,
    min_included_minutes_per_session: float | None = None,
):
    """Compare session 1 vs session 2 for each metric, spike type, and CS class.

    The CS+ CB row is populated only for FR columns and reports complex-burst
    event rate in the matching state/PF region.
    """
    fig, axes = plt.subplots(7, 14, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    axes = np.asarray(axes)
    arena_area_cm2 = 20.0 * 35.5

    def _get_table(session_key: str, class_key: str) -> pd.DataFrame:
        block = session_tables.get(session_key, {})
        table_key = 'df_cs_plc' if class_key == 'csplus' else 'df_non_cs_plc'
        df = block.get(table_key, pd.DataFrame())
        if df is None:
            df = pd.DataFrame()
        return df.copy()

    def _resolved_min_included_minutes() -> float:
        if min_included_minutes_per_session is None:
            return 0.0
        try:
            value = float(min_included_minutes_per_session)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    min_minutes = _resolved_min_included_minutes()

    def _included_minutes_from_paired_columns(df: pd.DataFrame, suffix: str) -> np.ndarray:
        for col in (f'included_minutes_{suffix}', f'included_time_min_{suffix}'):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
                if vals.size == len(df):
                    return vals

        kept_col = f'n_frames_kept_total_{suffix}'
        frame_rate_col = f'frame_rate_{suffix}'
        if kept_col in df.columns and frame_rate_col in df.columns:
            kept = pd.to_numeric(df[kept_col], errors='coerce').to_numpy(dtype=float)
            frame_rate = pd.to_numeric(df[frame_rate_col], errors='coerce').to_numpy(dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                vals = kept / frame_rate / 60.0
            vals[~np.isfinite(vals)] = np.nan
            return vals

        return np.full(len(df), np.nan, dtype=float)

    def _filter_paired_by_included_minutes(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or min_minutes <= 0:
            return df
        s1_minutes = _included_minutes_from_paired_columns(df, 's1')
        s2_minutes = _included_minutes_from_paired_columns(df, 's2')
        keep = (
            np.isfinite(s1_minutes)
            & np.isfinite(s2_minutes)
            & (s1_minutes >= min_minutes)
            & (s2_minutes >= min_minutes)
        )
        return df.loc[keep].copy()

    def _paired_frame(class_key: str) -> pd.DataFrame:
        s1 = _get_table('session1', class_key)
        s2 = _get_table('session2', class_key)
        required_keys = ['session', 'cell_idx']
        if any(col not in s1.columns for col in required_keys) or any(col not in s2.columns for col in required_keys):
            return pd.DataFrame()
        paired = s1.merge(
            s2,
            on=required_keys,
            how='inner',
            suffixes=('_s1', '_s2'),
            validate='one_to_one',
        )
        return _filter_paired_by_included_minutes(paired)

    paired_by_class = {
        'csplus': _paired_frame('csplus'),
        'csminus': _paired_frame('csminus'),
    }

    def _pooled_paired_frame() -> pd.DataFrame:
        frames = [df for df in paired_by_class.values() if not df.empty]
        if len(frames) == 0:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    pooled_pair = _pooled_paired_frame()

    def _numeric_pair(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        if df.empty or f'{col}_s1' not in df.columns or f'{col}_s2' not in df.columns:
            return np.array([], dtype=float), np.array([], dtype=float)
        s1 = pd.to_numeric(df[f'{col}_s1'], errors='coerce').to_numpy(dtype=float)
        s2 = pd.to_numeric(df[f'{col}_s2'], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(s1) & np.isfinite(s2)
        return s1[mask], s2[mask]

    def _pf_size_pair(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        if df.empty or f'{col}_s1' not in df.columns or f'{col}_s2' not in df.columns:
            return np.array([], dtype=float), np.array([], dtype=float)

        def _size_pct(value: Any) -> float:
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                return float(np.nansum(np.asarray(value, dtype=float)) / arena_area_cm2 * 100.0)
            return np.nan

        s1 = df[f'{col}_s1'].apply(_size_pct).to_numpy(dtype=float)
        s2 = df[f'{col}_s2'].apply(_size_pct).to_numpy(dtype=float)
        mask = np.isfinite(s1) & np.isfinite(s2)
        return s1[mask], s2[mask]

    def _selectivity_pair(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        in_col = f'{prefix}_inout_loco_in'
        out_col = f'{prefix}_inout_loco_out'
        required = [f'{in_col}_s1', f'{out_col}_s1', f'{in_col}_s2', f'{out_col}_s2']
        if df.empty or any(col not in df.columns for col in required):
            return np.array([], dtype=float), np.array([], dtype=float)

        def _sel(v_in, v_out):
            denom = v_in + v_out
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where((denom > 0) & np.isfinite(v_in) & np.isfinite(v_out), (v_in - v_out) / denom, np.nan)

        s1_in = pd.to_numeric(df[f'{in_col}_s1'], errors='coerce').to_numpy(dtype=float)
        s1_out = pd.to_numeric(df[f'{out_col}_s1'], errors='coerce').to_numpy(dtype=float)
        s2_in = pd.to_numeric(df[f'{in_col}_s2'], errors='coerce').to_numpy(dtype=float)
        s2_out = pd.to_numeric(df[f'{out_col}_s2'], errors='coerce').to_numpy(dtype=float)
        s1 = _sel(s1_in, s1_out)
        s2 = _sel(s2_in, s2_out)
        mask = np.isfinite(s1) & np.isfinite(s2)
        return s1[mask], s2[mask]

    def _pf_rank_fr_pair(df: pd.DataFrame, spike_key: str, pf_rank: int) -> tuple[np.ndarray, np.ndarray]:
        col = f'fr_in_pf{pf_rank}_{spike_key}'
        if df.empty or f'{col}_s1' not in df.columns or f'{col}_s2' not in df.columns:
            return np.array([], dtype=float), np.array([], dtype=float)
        s1 = pd.to_numeric(df[f'{col}_s1'], errors='coerce').to_numpy(dtype=float)
        s2 = pd.to_numeric(df[f'{col}_s2'], errors='coerce').to_numpy(dtype=float)
        if 'n_combined_reference_pfs_s1' in df.columns:
            n_pf_s1 = pd.to_numeric(df['n_combined_reference_pfs_s1'], errors='coerce').to_numpy(dtype=float)
        else:
            n_pf_s1 = np.full(len(df), np.nan)
        if 'n_combined_reference_pfs_s2' in df.columns:
            n_pf_s2 = pd.to_numeric(df['n_combined_reference_pfs_s2'], errors='coerce').to_numpy(dtype=float)
        else:
            n_pf_s2 = np.full(len(df), np.nan)
        has_ranked_combined_pf = (n_pf_s1 >= pf_rank) & (n_pf_s2 >= pf_rank)
        mask = has_ranked_combined_pf & np.isfinite(s1) & np.isfinite(s2)
        return s1[mask], s2[mask]

    def _cb_event_rate_pair(df: pd.DataFrame, metric_title: str) -> tuple[np.ndarray, np.ndarray]:
        metric_to_col = {
            'In-PF FR': 'complex_burst_event_rate_loco_in_pf',
            'Out-PF FR': 'complex_burst_event_rate_loco_out_pf',
            'Quiet FR': 'complex_burst_event_rate_quiet',
            'Quiet In-PF FR': 'complex_burst_event_rate_quiet_in_pf',
            'Quiet Out-PF FR': 'complex_burst_event_rate_quiet_out_pf',
        }
        col = metric_to_col.get(metric_title)
        if col is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        return _numeric_pair(df, col)

    def _cb_pf_rank_event_rate_pair(df: pd.DataFrame, pf_rank: int) -> tuple[np.ndarray, np.ndarray]:
        col = f'complex_burst_event_rate_loco_in_pf{pf_rank}'
        if df.empty or f'{col}_s1' not in df.columns or f'{col}_s2' not in df.columns:
            return np.array([], dtype=float), np.array([], dtype=float)
        s1 = pd.to_numeric(df[f'{col}_s1'], errors='coerce').to_numpy(dtype=float)
        s2 = pd.to_numeric(df[f'{col}_s2'], errors='coerce').to_numpy(dtype=float)
        if 'n_combined_reference_pfs_s1' in df.columns:
            n_pf_s1 = pd.to_numeric(df['n_combined_reference_pfs_s1'], errors='coerce').to_numpy(dtype=float)
        else:
            n_pf_s1 = np.full(len(df), np.nan)
        if 'n_combined_reference_pfs_s2' in df.columns:
            n_pf_s2 = pd.to_numeric(df['n_combined_reference_pfs_s2'], errors='coerce').to_numpy(dtype=float)
        else:
            n_pf_s2 = np.full(len(df), np.nan)
        has_ranked_combined_pf = (n_pf_s1 >= pf_rank) & (n_pf_s2 >= pf_rank)
        mask = has_ranked_combined_pf & np.isfinite(s1) & np.isfinite(s2)
        return s1[mask], s2[mask]

    metric_specs = [
        ('Peak rate', 'Peak rate (Hz)', lambda df, sp: _numeric_pair(df, f'peak_rate_{sp}' if sp != 'all' else 'peak_rate_all')),
        ('PF size', 'PF size (% arena)', lambda df, sp: _pf_size_pair(df, 'place_field_sizes_cm2' if sp == 'all' else f'place_field_sizes_cm2_{sp}')),
        ('Coherence', 'Coherence (z)', lambda df, sp: _numeric_pair(df, f'coherence_{sp}' if sp != 'all' else 'coherence_all')),
        ('Sparsity', 'Sparsity', lambda df, sp: _numeric_pair(df, f'sparsity_{sp}' if sp != 'all' else 'sparsity_all')),
        ('SI', 'SI (bits/spike)', lambda df, sp: _numeric_pair(df, 'si_bits_per_spike' if sp == 'all' else f'si_bits_per_spike_{sp}')),
        ('Selectivity', 'Selectivity', lambda df, sp: _selectivity_pair(df, sp)),
        ('In-PF FR', 'In-PF FR (Hz)', lambda df, sp: _numeric_pair(df, f'fr_in_allpf_{sp}' if sp != 'all' else 'fr_in_allpf_all')),
        ('Out-PF FR', 'Out-PF FR (Hz)', lambda df, sp: _numeric_pair(df, f'fr_out_allpf_{sp}' if sp != 'all' else 'fr_out_allpf_all')),
        ('PF1 FR', 'PF1 FR (Hz)', lambda df, sp: _pf_rank_fr_pair(df, sp, 1)),
        ('PF2 FR', 'PF2 FR (Hz)', lambda df, sp: _pf_rank_fr_pair(df, sp, 2)),
        ('Quiet FR', 'Quiet FR (Hz)', lambda df, sp: _numeric_pair(df, f'fr_quiet_all_allpf_{sp}' if sp != 'all' else 'fr_quiet_all_allpf_all')),
        ('Quiet In-PF FR', 'Quiet In-PF FR (Hz)', lambda df, sp: _numeric_pair(df, f'fr_quiet_in_allpf_{sp}' if sp != 'all' else 'fr_quiet_in_allpf_all')),
        ('Quiet Out-PF FR', 'Quiet Out-PF FR (Hz)', lambda df, sp: _numeric_pair(df, f'fr_quiet_out_allpf_{sp}' if sp != 'all' else 'fr_quiet_out_allpf_all')),
    ]
    row_specs = [
        ('csplus', 'all', 'CS+ All', ALL_COLOR, 'spike'),
        ('csplus', 'ss', 'CS+ SS', SS_COLOR, 'spike'),
        ('csplus', 'cs', 'CS+ CS', CS_COLOR, 'spike'),
        ('csplus', 'cb', 'CS+ CB', CB_COLOR, 'complex_burst'),
        ('csminus', 'all', 'CS- All', ALL_COLOR, 'spike'),
        ('csminus', 'ss', 'CS- SS', SS_COLOR, 'spike'),
        ('csminus', 'cs', 'CS- CS', CS_COLOR, 'spike'),
    ]
    cb_row_indices = {
        row_idx
        for row_idx, row_spec in enumerate(row_specs)
        if row_spec[4] == 'complex_burst'
    }

    has_data_grid = np.zeros(axes.shape, dtype=bool)

    def _draw_pair_panel(ax, s1_vals: np.ndarray, s2_vals: np.ndarray, color: str, ylabel: str) -> bool:
        data = [np.asarray(s1_vals, dtype=float), np.asarray(s2_vals, dtype=float)]
        for pos, vals in zip([1, 2], data):
            if vals.size >= 3:
                parts = ax.violinplot([vals], positions=[pos], showmedians=True, showextrema=False, widths=0.55)
                _style_violin_medians(parts)
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor('none')
                body.set_alpha(0.35)

        for i in range(min(len(s1_vals), len(s2_vals))):
            ax.plot([1, 2], [s1_vals[i], s2_vals[i]], color='black', alpha=0.22, linewidth=0.45, zorder=1)
        if len(s1_vals) > 0:
            jitter = np.linspace(-0.08, 0.08, len(s1_vals)) if len(s1_vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(s1_vals), 1.0) + jitter, s1_vals, s=4, color='black', alpha=0.55, linewidths=0, zorder=2)
            ax.scatter(np.full(len(s2_vals), 2.0) + jitter, s2_vals, s=4, color='black', alpha=0.55, linewidths=0, zorder=2)

        ylim = _global_ylim(data)
        if ylim is not None:
            y_range = ylim[1] - ylim[0]
            y_top = ylim[1]
            if len(s1_vals) >= 3:
                p, _, _, _, _ = _paired_test(s1_vals, s2_vals)
                h = y_range * 0.03
                y = ylim[1] + y_range * 0.06
                _add_bracket(ax, 1, 2, y, h, _sig_label(p))
                y_top = y + h + y_range * 0.13
            ax.set_ylim(ylim[0], y_top + y_range * 0.05)

        ax.set_xlim(0.55, 2.45)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['S1', 'S2'], fontsize=5, fontname='Arial')
        ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=5)
        ax.text(0.96, 0.94, f'n={len(s1_vals)}', transform=ax.transAxes, ha='right', va='top', fontsize=5, fontname='Arial')
        return len(s1_vals) > 0

    for col_idx, (title, ylabel, getter) in enumerate(metric_specs):
        axes[0, col_idx].set_title(title, fontsize=6, fontname='Arial')
        for row_idx, (class_key, spike_key, row_label, color, row_kind) in enumerate(row_specs):
            df_pair = paired_by_class[class_key]
            if row_kind == 'complex_burst':
                if 'FR' not in title:
                    axes[row_idx, col_idx].axis('off')
                    if col_idx == 0:
                        axes[row_idx, col_idx].text(
                            -0.45,
                            0.5,
                            row_label,
                            transform=axes[row_idx, col_idx].transAxes,
                            rotation=90,
                            ha='center',
                            va='center',
                            fontsize=6,
                            fontname='Arial',
                        )
                    continue
                if title == 'PF1 FR':
                    s1_vals, s2_vals = _cb_pf_rank_event_rate_pair(df_pair, 1)
                elif title == 'PF2 FR':
                    s1_vals, s2_vals = _cb_pf_rank_event_rate_pair(df_pair, 2)
                else:
                    s1_vals, s2_vals = _cb_event_rate_pair(df_pair, title)
                panel_ylabel = 'CB event rate (Hz)'
            else:
                s1_vals, s2_vals = getter(df_pair, spike_key)
                panel_ylabel = ylabel
            has_data_grid[row_idx, col_idx] = _draw_pair_panel(axes[row_idx, col_idx], s1_vals, s2_vals, color, panel_ylabel)
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    -0.45,
                    0.5,
                    row_label,
                    transform=axes[row_idx, col_idx].transAxes,
                    rotation=90,
                    ha='center',
                    va='center',
                    fontsize=6,
                    fontname='Arial',
                )

    pooled_col_idx = len(metric_specs)
    pooled_metric_specs = [
        ('Quiet FR', 'Quiet FR (Hz)', 'fr_quiet_all_allpf_all'),
        ('Quiet In-PF FR', 'Quiet In-PF FR (Hz)', 'fr_quiet_in_allpf_all'),
        ('Quiet Out-PF FR', 'Quiet Out-PF FR (Hz)', 'fr_quiet_out_allpf_all'),
        ('Locomotion FR', 'Locomotion FR (Hz)', 'fr_loco_all_allpf_all'),
        ('Loc IN-PF FR', 'Loc IN-PF FR (Hz)', 'fr_in_allpf_all'),
        ('Loc Out-PF FR', 'Loc Out-PF FR (Hz)', 'fr_out_allpf_all'),
    ]
    axes[0, pooled_col_idx].set_title('Pooled All', fontsize=6, fontname='Arial')
    for pooled_row_idx, (_title, ylabel, col) in enumerate(pooled_metric_specs):
        s1_vals, s2_vals = _numeric_pair(pooled_pair, col)
        has_data_grid[pooled_row_idx, pooled_col_idx] = _draw_pair_panel(
            axes[pooled_row_idx, pooled_col_idx],
            s1_vals,
            s2_vals,
            ALL_COLOR,
            ylabel,
        )
    for row_idx in range(len(pooled_metric_specs), axes.shape[0]):
        axes[row_idx, pooled_col_idx].axis('off')

    for col_idx in range(axes.shape[1]):
        data_rows = np.flatnonzero(has_data_grid[:, col_idx])
        shared_data_rows = [row_idx for row_idx in data_rows if row_idx not in cb_row_indices]
        ylims = [axes[row_idx, col_idx].get_ylim() for row_idx in shared_data_rows]
        y_min = min(y[0] for y in ylims) if ylims else None
        y_max = max(y[1] for y in ylims) if ylims else None
        for row_idx in range(axes.shape[0]):
            if y_min is not None and row_idx not in cb_row_indices:
                axes[row_idx, col_idx].set_ylim(y_min, y_max)
            if row_idx != axes.shape[0] - 1:
                axes[row_idx, col_idx].set_xticklabels([])
    axes[len(pooled_metric_specs) - 1, pooled_col_idx].set_xticklabels(['S1', 'S2'], fontsize=5, fontname='Arial')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_all_cells_session_state_rate_comparison_4cols(
    session_compare_groups,
    save_path,
    fig_width: float = 6.5,
    fig_height: float = 3.6,
    cs_min_rate_hz: float = 0.5,
    cb_min_event_rate_hz: float = 0.1,
    min_included_minutes_per_session: float | None = None,
    show_only_significant: bool = False,
):
    """Plot quiet/moving S1/S2 rate comparisons for all retained cells."""
    cs_min_rate_hz = float(cs_min_rate_hz)
    cb_min_event_rate_hz = float(cb_min_event_rate_hz)
    if (not np.isfinite(cs_min_rate_hz)) or cs_min_rate_hz < 0:
        raise ValueError("cs_min_rate_hz must be a finite number >= 0.")
    if (not np.isfinite(cb_min_event_rate_hz)) or cb_min_event_rate_hz < 0:
        raise ValueError("cb_min_event_rate_hz must be a finite number >= 0.")

    def _resolved_min_minutes() -> float:
        if min_included_minutes_per_session is None:
            return 0.0
        try:
            value = float(min_included_minutes_per_session)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    min_minutes = _resolved_min_minutes()

    def _included_minutes(cell: dict[str, Any]) -> float:
        for col in ("included_minutes", "included_time_min"):
            if col in cell:
                try:
                    value = float(cell.get(col, np.nan))
                except (TypeError, ValueError):
                    value = np.nan
                if np.isfinite(value):
                    return value
        try:
            kept = float(cell.get("n_frames_kept_total", np.nan))
            frame_rate = float(cell.get("frame_rate", np.nan))
        except (TypeError, ValueError):
            return np.nan
        if np.isfinite(kept) and np.isfinite(frame_rate) and frame_rate > 0:
            return kept / frame_rate / 60.0
        return np.nan

    def _passes_included_minutes(cell: dict[str, Any]) -> bool:
        if min_minutes <= 0:
            return True
        minutes = _included_minutes(cell)
        return bool(np.isfinite(minutes) and minutes >= min_minutes)

    def _cell_condition(cell: dict[str, Any]) -> str:
        label = str(cell.get("condition_label", "")).lower()
        condition = str(cell.get("condition", "")).lower()
        if "session 1" in label or condition == "session1":
            return "session1"
        if "session 2" in label or condition == "session2":
            return "session2"
        return ""

    def _session_cell(group, session_key: str, fallback_idx: int) -> dict[str, Any] | None:
        if not isinstance(group, (list, tuple)):
            return None
        for cell in group:
            if isinstance(cell, dict) and _cell_condition(cell) == session_key:
                return cell
        if len(group) >= 3:
            idx = 1 if session_key == "session1" else 2
            if isinstance(group[idx], dict):
                return group[idx]
        if len(group) > fallback_idx and isinstance(group[fallback_idx], dict):
            return group[fallback_idx]
        return None

    def _valid_session_pair(group) -> tuple[dict[str, Any], dict[str, Any]] | None:
        s1 = _session_cell(group, "session1", 0)
        s2 = _session_cell(group, "session2", 1)
        if not isinstance(s1, dict) or not isinstance(s2, dict):
            return None
        if bool(s1.get("is_na_panel", False)) or bool(s2.get("is_na_panel", False)):
            return None
        if not _passes_included_minutes(s1) or not _passes_included_minutes(s2):
            return None
        return s1, s2

    all_groups = []
    for key in ("csplus_groups", "csminus_groups", "nonpc_groups"):
        groups = session_compare_groups.get(key, []) if isinstance(session_compare_groups, dict) else []
        if isinstance(groups, (list, tuple)):
            all_groups.extend(groups)

    paired_cells = []
    for group in all_groups:
        pair = _valid_session_pair(group)
        if pair is not None:
            paired_cells.append(pair)

    def _numeric(cell: dict[str, Any], col: str) -> float:
        try:
            return float(cell.get(col, np.nan))
        except (TypeError, ValueError):
            return np.nan

    def _panel_matrix(quiet_col: str, moving_col: str, threshold: float | None = None) -> np.ndarray:
        rows = []
        for s1, s2 in paired_cells:
            vals = np.array(
                [
                    _numeric(s1, quiet_col),
                    _numeric(s1, moving_col),
                    _numeric(s2, quiet_col),
                    _numeric(s2, moving_col),
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(vals)):
                continue
            if threshold is not None and not np.any(vals >= float(threshold)):
                continue
            rows.append(vals)
        if not rows:
            return np.empty((0, 4), dtype=float)
        return np.vstack(rows).astype(float, copy=False)

    panel_specs = [
        ("All spikes", "all_rate_quiet", "all_rate_loco", None, ALL_COLOR),
        ("Simple spikes", "ss_rate_quiet", "ss_rate_loco", None, SS_COLOR),
        (f"Complex spikes\n>= {cs_min_rate_hz:g} Hz", "cs_rate_quiet", "cs_rate_loco", cs_min_rate_hz, CS_COLOR),
        (
            f"Complex bursts\n>= {cb_min_event_rate_hz:g} Hz",
            "complex_burst_event_rate_quiet",
            "complex_burst_event_rate_loco",
            cb_min_event_rate_hz,
            CB_COLOR,
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    axes = np.asarray(axes)
    positions = np.array([1.0, 2.0], dtype=float)
    xticklabels = ["S1", "S2"]

    def _draw_state_panel(
        ax,
        matrix: np.ndarray,
        value_indices: tuple[int, int],
        title: str,
        color: str,
        row_label: str,
        show_title: bool,
    ) -> None:
        if matrix.size:
            panel_matrix = matrix[:, list(value_indices)]
            finite_rows = np.all(np.isfinite(panel_matrix), axis=1)
            panel_matrix = panel_matrix[finite_rows]
        else:
            panel_matrix = np.empty((0, 2), dtype=float)
        data = [
            panel_matrix[:, idx] if panel_matrix.size else np.array([], dtype=float)
            for idx in range(2)
        ]

        for pos, vals in zip(positions, data):
            if vals.size >= 3:
                parts = ax.violinplot([vals], positions=[pos], showmedians=True, showextrema=False, widths=0.55)
                _style_violin_medians(parts)
                body = parts["bodies"][0]
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.35)

        for row in panel_matrix:
            ax.plot(positions, row, color="black", alpha=0.16, linewidth=0.45, zorder=1)

        for pos, vals in zip(positions, data):
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full(vals.size, pos) + jitter, vals, s=5, color="black", alpha=0.55, linewidths=0, zorder=2)

        ylim = _global_ylim(data)
        if ylim is None:
            ylim = (0.0, 1.0)
        bracket_levels = []
        if panel_matrix.shape[0] >= 3:
            p_s1_s2, _, _, _, _ = _paired_test(panel_matrix[:, 0], panel_matrix[:, 1])
            bracket_levels = [
                [{"pos1": positions[0], "pos2": positions[1], "p": p_s1_s2, "color": color}],
            ]

        y_range = ylim[1] - ylim[0]
        y_top = _draw_sig_bracket_levels(ax, bracket_levels, ylim, bool(show_only_significant))
        ax.set_ylim(ylim[0], y_top + y_range * 0.05)
        ax.set_xlim(0.55, 2.45)
        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels, fontsize=5, fontname="Arial")
        if show_title:
            ax.set_title(title, fontsize=6, fontname="Arial")
        ax.set_ylabel("Rate (Hz)", fontsize=6, fontname="Arial")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=5)
        ax.text(
            0.96,
            0.94,
            f"n={panel_matrix.shape[0]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5,
            fontname="Arial",
        )
        if row_label:
            ax.text(
                -0.38,
                0.5,
                row_label,
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
                clip_on=False,
            )

    for col_idx, (title, quiet_col, moving_col, threshold, color) in enumerate(panel_specs):
        matrix = _panel_matrix(quiet_col, moving_col, threshold=threshold)
        _draw_state_panel(
            axes[0, col_idx],
            matrix,
            (0, 2),
            title,
            color,
            "Quiet" if col_idx == 0 else "",
            show_title=True,
        )
        _draw_state_panel(
            axes[1, col_idx],
            matrix,
            (1, 3),
            title,
            color,
            "Moving" if col_idx == 0 else "",
            show_title=False,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_combined_cs_plus_minus_session_pair_metrics_13cols(
    session_tables: dict[str, dict[str, pd.DataFrame]],
    save_path: str,
    fig_width: float = 14.5,
    fig_height: float = 7.0,
    min_included_minutes_per_session: float | None = None,
):
    """Backward-compatible wrapper; the session-pair summary now includes 14 columns."""
    return plot_combined_cs_plus_minus_session_pair_metrics_14cols(
        session_tables=session_tables,
        save_path=save_path,
        fig_width=fig_width,
        fig_height=fig_height,
        min_included_minutes_per_session=min_included_minutes_per_session,
    )


def plot_combined_cs_plus_minus_session_pair_metrics_9cols(
    session_tables: dict[str, dict[str, pd.DataFrame]],
    save_path: str,
    fig_width: float = 14.5,
    fig_height: float = 7.0,
    min_included_minutes_per_session: float | None = None,
):
    """Backward-compatible wrapper; the session-pair summary now includes 14 columns."""
    return plot_combined_cs_plus_minus_session_pair_metrics_14cols(
        session_tables=session_tables,
        save_path=save_path,
        fig_width=fig_width,
        fig_height=fig_height,
        min_included_minutes_per_session=min_included_minutes_per_session,
    )


def plot_combined_cs_plus_minus_session_pair_metrics_7cols(
    session_tables: dict[str, dict[str, pd.DataFrame]],
    save_path: str,
    fig_width: float = 9.5,
    fig_height: float = 7.0,
    min_included_minutes_per_session: float | None = None,
):
    """Backward-compatible wrapper; the session-pair summary now includes 14 columns."""
    return plot_combined_cs_plus_minus_session_pair_metrics_14cols(
        session_tables=session_tables,
        save_path=save_path,
        fig_width=fig_width,
        fig_height=fig_height,
        min_included_minutes_per_session=min_included_minutes_per_session,
    )


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


def plot_in_pf_fr_vs_all_pf_size_by_cs_group(
    df_cs_plc: pd.DataFrame,
    df_non_cs_plc: pd.DataFrame,
    save_path: str,
    fig_width: float = 1.6,
    fig_height: float = 1.35,
    arena_area_cm2: float = 20.0 * 35.5,
    fr_metric: str = "mean",
):
    """Plot all-spike in-PF firing rate against summed all-spike PF size.

    fr_metric='mean' uses the mean all-spike in-field FR. fr_metric='peak'
    uses the all-spike spatial-map peak rate within the place-field map.
    """

    metric_key = str(fr_metric).strip().lower().replace("-", "_")
    metric_aliases = {
        "mean": "mean",
        "mean_fr": "mean",
        "in_pf_mean": "mean",
        "in_pf_mean_fr": "mean",
        "peak": "peak",
        "peak_fr": "peak",
        "in_pf_peak": "peak",
        "in_pf_peak_fr": "peak",
    }
    metric_key = metric_aliases.get(metric_key)
    if metric_key is None:
        raise ValueError("fr_metric must be one of {'mean', 'peak'}.")

    if metric_key == "mean":
        fr_label = "Mean In-PF FR"
        fr_column_label = "fr_in_allpf_all"
        x_label = "Mean In-PF FR (Hz)"
    else:
        fr_label = "Peak In-PF FR"
        fr_column_label = "peak_rate_all"
        x_label = "Peak In-PF FR (Hz)"

    def _extract_group(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
        if metric_key == "mean":
            fr_col = "fr_in_allpf_all" if "fr_in_allpf_all" in df.columns else "all_inout_loco_in"
            if fr_col not in df.columns:
                raise KeyError("df missing 'fr_in_allpf_all' and fallback 'all_inout_loco_in'")
        else:
            fr_col = "peak_rate_all"
            if fr_col not in df.columns:
                raise KeyError("df missing column 'peak_rate_all'")
        if "place_field_sizes_cm2" not in df.columns:
            raise KeyError("df missing column 'place_field_sizes_cm2'")

        x = pd.to_numeric(df[fr_col], errors="coerce").to_numpy(dtype=float)
        y = _pf_sizes_pct_from_col(df, "place_field_sizes_cm2", arena_area_cm2)
        rows = []
        for row_i, (_, row) in enumerate(df.iterrows()):
            session = row.get("session", "")
            cell_idx_raw = row.get("cell_idx", np.nan)
            try:
                cell_idx = int(cell_idx_raw)
            except Exception:
                cell_idx = np.nan
            cell_num = int(cell_idx + 1) if np.isfinite(cell_idx) else np.nan
            rows.append(
                {
                    "group": group_label,
                    "session": session,
                    "cell_idx": cell_idx,
                    "cell_num": cell_num,
                    "fr_metric": metric_key,
                    "fr_source_column": fr_col,
                    "in_pf_fr_hz": float(x[row_i]) if np.isfinite(x[row_i]) else np.nan,
                    "pf_size_all_pfs_pct_arena": float(y[row_i]) if np.isfinite(y[row_i]) else np.nan,
                }
            )
        out = pd.DataFrame(rows)
        return out[np.isfinite(out["in_pf_fr_hz"]) & np.isfinite(out["pf_size_all_pfs_pct_arena"])].copy()

    def _stats_for_group(data: pd.DataFrame, group_label: str) -> dict[str, Any]:
        x = data["in_pf_fr_hz"].to_numpy(dtype=float)
        y = data["pf_size_all_pfs_pct_arena"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        out = {
            "group": group_label,
            "fr_metric": metric_key,
            "fr_source_column": fr_column_label,
            "n": int(x.size),
            "pearson_r": np.nan,
            "p_value": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }
        if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return out
        lr = scipy_stats.linregress(x, y)
        r, p = scipy_stats.pearsonr(x, y)
        out.update(
            {
                "pearson_r": float(r),
                "p_value": float(p),
                "slope": float(lr.slope),
                "intercept": float(lr.intercept),
            }
        )
        return out

    data_df = pd.concat(
        [
            _extract_group(df_cs_plc, "CS+ PLC"),
            _extract_group(df_non_cs_plc, "CS- PLC"),
        ],
        ignore_index=True,
    )
    stats_df = pd.DataFrame(
        [
            _stats_for_group(data_df[data_df["group"] == "CS+ PLC"], "CS+ PLC"),
            _stats_for_group(data_df[data_df["group"] == "CS- PLC"], "CS- PLC"),
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_facecolor(CS_PLC_BG)
    ax.patch.set_alpha(0.22)
    ax.tick_params(labelsize=5)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontname("Arial")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    group_specs = [
        ("CS+ PLC", CS_COLOR, 0.82),
        ("CS- PLC", "#1F77B4", 0.70),
    ]
    for group_label, color, text_y in group_specs:
        sub = data_df[data_df["group"] == group_label]
        _plot_corr_series(
            ax,
            sub["in_pf_fr_hz"].to_numpy(dtype=float),
            sub["pf_size_all_pfs_pct_arena"].to_numpy(dtype=float),
            series_label=group_label,
            color=color,
            xlabel=x_label,
            ylabel="PF size all PFs (% arena)",
            panel_name=f"{fr_label} vs all PF size",
            text_y=text_y,
            linestyle="-",
        )

    ax.set_xlabel(x_label, fontsize=6, fontname="Arial")
    ax.set_ylabel("PF size all PFs (% arena)", fontsize=6, fontname="Arial")
    plt.tight_layout()

    figure_path = None
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        figure_path = save_path

    return {
        "fig": fig,
        "data_df": data_df,
        "stats_df": stats_df,
        "fr_metric": metric_key,
        "figure_path": figure_path,
    }


def plot_cs_plus_plc_correlations_overlay_cs_ss(
    df_cs_plc: pd.DataFrame,
    save_path: str,
    overlay_ss_metrics: bool = True,
    fr_metric: str = "peak",
    df_non_cs_plc: pd.DataFrame | None = None,
    csminus_spike_metric: str = "ss",
):
    cs_color_local = CS_COLOR
    ss_color_local = SS_COLOR
    all_color_local = ALL_COLOR
    cb_color_local = CB_COLOR
    # In the original pooled notebook this resolves from global `cs_plc_bg`,
    # which is set to light orange.
    cs_plc_bg_local = CS_PLC_BG
    non_cs_plc_bg_local = NON_CS_PLC_BG

    arena_area_cm2_local = 20.0 * 35.5

    metric_key = str(fr_metric).strip().lower().replace("-", "_")
    metric_aliases = {
        "peak": "peak",
        "peak_fr": "peak",
        "mean": "mean",
        "mean_fr": "mean",
        "in_pf_mean": "mean",
        "in_pf_mean_fr": "mean",
    }
    metric_key = metric_aliases.get(metric_key)
    if metric_key is None:
        raise ValueError("fr_metric must be one of {'peak', 'mean'}.")

    csminus_metric_key = str(csminus_spike_metric).strip().lower().replace("-", "_")
    csminus_metric_aliases = {
        "ss": "ss",
        "simple": "ss",
        "simple_spike": "ss",
        "simple_spikes": "ss",
        "all": "all",
        "all_spike": "all",
        "all_spikes": "all",
    }
    csminus_metric_key = csminus_metric_aliases.get(csminus_metric_key)
    if csminus_metric_key is None:
        raise ValueError("csminus_spike_metric must be one of {'ss', 'all'}.")

    if metric_key == "mean":
        rate_label = "Mean In-PF FR"
        rate_axis_label = "Mean In-PF FR (Hz)"
    else:
        rate_label = "Peak rate"
        rate_axis_label = "Peak rate (Hz)"

    def _column_label(spike_key: str, base_col: str) -> str:
        return base_col if spike_key == "all" else f"{base_col}_{spike_key}"

    def _mean_pf_fr_values(df: pd.DataFrame, spike_key: str, inout_key: str, df_label: str) -> np.ndarray:
        col = f"fr_{inout_key}_allpf_{spike_key}"
        fallback_col = f"{spike_key}_inout_loco_{inout_key}"
        if col not in df.columns:
            col = fallback_col
        if col not in df.columns:
            raise KeyError(
                f"{df_label} missing 'fr_{inout_key}_allpf_{spike_key}' and fallback '{fallback_col}'"
            )
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    def _rate_values(df: pd.DataFrame, spike_key: str, df_label: str) -> np.ndarray:
        if metric_key == "peak":
            col = f"peak_rate_{spike_key}"
            if col not in df.columns:
                raise KeyError(f"{df_label} missing column '{col}'")
        else:
            return _mean_pf_fr_values(df, spike_key, "in", df_label)
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    def _spike_metric_values(df: pd.DataFrame, spike_key: str, df_label: str) -> dict[str, Any]:
        pf_col = _column_label(spike_key, "place_field_sizes_cm2")
        si_col = _column_label(spike_key, "si_bits_per_spike")
        if si_col not in df.columns:
            raise KeyError(f"{df_label} missing column '{si_col}'")
        return {
            "rate": _rate_values(df, spike_key, df_label),
            "mean_in_fr": _mean_pf_fr_values(df, spike_key, "in", df_label),
            "mean_out_fr": _mean_pf_fr_values(df, spike_key, "out", df_label),
            "pf_size": _pf_sizes_pct_from_col(df, pf_col, arena_area_cm2_local),
            "si": pd.to_numeric(df[si_col], errors="coerce").to_numpy(dtype=float),
            "selectivity": _selectivity_from_cols(
                df,
                f"{spike_key}_inout_loco_in",
                f"{spike_key}_inout_loco_out",
            ),
            "label": {"all": "All", "ss": "SS", "cs": "CS"}.get(spike_key, spike_key.upper()),
            "color": {"all": all_color_local, "ss": ss_color_local, "cs": cs_color_local}.get(
                spike_key,
                "black",
            ),
        }

    cs_metrics = _spike_metric_values(df_cs_plc, "cs", "df_cs_plc")
    ss_metrics = None
    if overlay_ss_metrics:
        ss_metrics = _spike_metric_values(df_cs_plc, "ss", "df_cs_plc")

    if 'complex_burst_event_rate_loco_in_pf' not in df_cs_plc.columns:
        raise KeyError("df_cs_plc missing column 'complex_burst_event_rate_loco_in_pf'")
    cb_rate = pd.to_numeric(
        df_cs_plc['complex_burst_event_rate_loco_in_pf'],
        errors="coerce",
    ).to_numpy(dtype=float)

    has_csminus_row = df_non_cs_plc is not None
    n_rows = 2 if has_csminus_row else 1
    fig, axes = plt.subplots(n_rows, 9, figsize=(1.0 * 9, 1.2 * n_rows))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    axes[0, 5].sharey(axes[0, 4])
    axes[0, 7].sharey(axes[0, 0])
    axes[0, 8].sharey(axes[0, 2])

    def _style_corr_axis(ax, bg_color: str) -> None:
        ax.set_facecolor(bg_color)
        ax.patch.set_alpha(0.3)
        ax.tick_params(labelsize=5)
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_six_corr_panels(ax_row, primary, overlay, bg_color: str) -> None:
        panels = [
            (ax_row[0], primary["rate"], primary["pf_size"], None if overlay is None else overlay["rate"], None if overlay is None else overlay["pf_size"], rate_axis_label, 'PF size (% arena)', f'{rate_label} vs PF size'),
            (ax_row[1], primary["si"], primary["rate"], None if overlay is None else overlay["si"], None if overlay is None else overlay["rate"], 'SI (bits/spike)', rate_axis_label, f'SI vs {rate_label}'),
            (ax_row[2], primary["rate"], primary["selectivity"], None if overlay is None else overlay["rate"], None if overlay is None else overlay["selectivity"], rate_axis_label, 'Selectivity', f'{rate_label} vs Selectivity'),
            (ax_row[3], primary["mean_in_fr"], primary["mean_out_fr"], None if overlay is None else overlay["mean_in_fr"], None if overlay is None else overlay["mean_out_fr"], 'Mean In-PF FR (Hz)', 'Mean Out-PF FR (Hz)', 'Mean In-PF FR vs Mean Out-PF FR'),
            (ax_row[4], primary["si"], primary["pf_size"], None if overlay is None else overlay["si"], None if overlay is None else overlay["pf_size"], 'SI (bits/spike)', 'PF size (% arena)', 'SI vs PF size'),
            (ax_row[5], primary["selectivity"], primary["pf_size"], None if overlay is None else overlay["selectivity"], None if overlay is None else overlay["pf_size"], 'Selectivity', 'PF size (% arena)', 'Selectivity vs PF size'),
            (ax_row[6], primary["si"], primary["selectivity"], None if overlay is None else overlay["si"], None if overlay is None else overlay["selectivity"], 'SI (bits/spike)', 'Selectivity', 'SI vs Selectivity'),
        ]

        for ax, x_primary, y_primary, x_overlay, y_overlay, xlabel, ylabel, panel_name in panels:
            _style_corr_axis(ax, bg_color)
            any_plotted = False
            any_plotted |= _plot_corr_series(
                ax,
                x_primary,
                y_primary,
                series_label=primary["label"],
                color=primary["color"],
                xlabel=xlabel,
                ylabel=ylabel,
                panel_name=panel_name,
                text_y=0.82,
                linestyle='-',
            )

            if overlay is not None:
                any_plotted |= _plot_corr_series(
                    ax,
                    x_overlay,
                    y_overlay,
                    series_label=overlay["label"],
                    color=overlay["color"],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    panel_name=panel_name,
                    text_y=0.92,
                    linestyle='-',
                )

            if not any_plotted:
                ax.axis('off')

    _plot_six_corr_panels(axes[0], cs_metrics, ss_metrics, cs_plc_bg_local)

    cb_panels = [
        (axes[0, 7], cb_rate, cs_metrics["pf_size"], 'CB event rate in PF (Hz)', 'PF size (% arena)', 'CB rate vs PF size'),
        (axes[0, 8], cb_rate, cs_metrics["selectivity"], 'CB event rate in PF (Hz)', 'Selectivity', 'CB rate vs Selectivity'),
    ]
    for ax, x_cb, y_cb, xlabel, ylabel, panel_name in cb_panels:
        _style_corr_axis(ax, cs_plc_bg_local)
        any_plotted = False
        any_plotted |= _plot_corr_series(
            ax,
            x_cb,
            y_cb,
            series_label='CB',
            color=cb_color_local,
            xlabel=xlabel,
            ylabel=ylabel,
            panel_name=panel_name,
            text_y=0.82,
            linestyle='-',
        )
        if not any_plotted:
            ax.axis('off')

    if has_csminus_row:
        axes[1, 5].sharey(axes[1, 4])
        csminus_metrics = _spike_metric_values(
            df_non_cs_plc,
            csminus_metric_key,
            "df_non_cs_plc",
        )
        _plot_six_corr_panels(axes[1], csminus_metrics, None, non_cs_plc_bg_local)
        for ax in axes[1, 7:]:
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_map_pair_similarity_by_cell_type(
    plcs_csplus,
    plcs_csminus,
    non_plcs,
    save_path: str,
    min_firing_rate_map_peak_hz: float = 1.0,
    fig_width: float = 7.2,
    fig_height: float = 3.8,
    show_pairwise_stats: bool = True,
    show_only_significant_stats: bool = True,
):
    """Compare per-cell similarity between selected spatial maps by cell type."""
    group_specs = [
        ("CS+ PLC", list(plcs_csplus), "#D81B60"),
        ("CS- PLC", list(plcs_csminus), "#1F77B4"),
        ("Non-PLC", list(non_plcs), "#8A8A8A"),
    ]
    map_pair_specs = [
        ("ss_vs_cs", "SS vs CS", "ss_norm_map", "cs_norm_map", ("ss_rate_map", "cs_rate_map")),
        ("theta_vs_cs", "Theta vs CS", "theta_map", "cs_norm_map", ("cs_rate_map",)),
        ("slow_vs_all", "Slow Vm vs all", "slow_map", "rate_map", ("rate_map",)),
        ("slow_vs_ss", "Slow Vm vs SS", "slow_map", "ss_norm_map", ("ss_rate_map",)),
        ("slow_vs_cs", "Slow Vm vs CS", "slow_map", "cs_norm_map", ("cs_rate_map",)),
    ]
    csplus_paired_map_pair_specs = [
        ("all_vs_ss", "All vs SS", "rate_map", "ss_norm_map", ("rate_map", "ss_rate_map")),
        ("all_vs_cs", "All vs CS", "rate_map", "cs_norm_map", ("rate_map", "cs_rate_map")),
    ]
    similarity_specs = [
        ("pearson", "Pearson r"),
        ("spearman", "Spearman rho"),
        ("cosine", "Cosine similarity"),
    ]

    def _map_peak(cell: dict[str, Any], key: str) -> float:
        arr = np.asarray(cell.get(key, None), dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return np.nan
        return float(np.nanmax(arr))

    def _valid_flat_pair(map_a: Any, map_b: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(map_a, np.ndarray) or not isinstance(map_b, np.ndarray):
            return np.array([], dtype=float), np.array([], dtype=float)
        if map_a.shape != map_b.shape:
            return np.array([], dtype=float), np.array([], dtype=float)
        a = np.asarray(map_a, dtype=float).ravel()
        b = np.asarray(map_b, dtype=float).ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        return a[valid], b[valid]

    def _similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
        if a.size < 3 or b.size < 3:
            return np.nan
        if metric == "pearson":
            if np.nanstd(a) == 0 or np.nanstd(b) == 0:
                return np.nan
            r, _p = scipy_stats.pearsonr(a, b)
            return float(r) if np.isfinite(r) else np.nan
        if metric == "spearman":
            if np.nanstd(a) == 0 or np.nanstd(b) == 0:
                return np.nan
            rho, _p = scipy_stats.spearmanr(a, b)
            return float(rho) if np.isfinite(rho) else np.nan

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 0 or norm_b <= 0:
            return np.nan
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))

    def _holm_adjust(p_values: list[float]) -> list[float]:
        p = np.asarray(p_values, dtype=float)
        adjusted = np.full(p.shape, np.nan, dtype=float)
        finite = np.isfinite(p)
        if not np.any(finite):
            return adjusted.tolist()
        finite_idx = np.flatnonzero(finite)
        order = finite_idx[np.argsort(p[finite])]
        m = len(order)
        running = 0.0
        for rank, idx in enumerate(order):
            raw_adj = (m - rank) * p[idx]
            running = max(running, raw_adj)
            adjusted[idx] = min(running, 1.0)
        return adjusted.tolist()

    def _omnibus_test(groups: list[np.ndarray]) -> tuple[float, float, str, list[float]]:
        clean = [np.asarray(g, dtype=float) for g in groups]
        clean = [g[np.isfinite(g)] for g in clean]
        usable = [g for g in clean if g.size >= 2]
        shapiro_ps = [
            float(scipy_stats.shapiro(g).pvalue) if 3 <= g.size <= 5000 else np.nan
            for g in clean
        ]
        if len(usable) < 2:
            return np.nan, np.nan, "insufficient data", shapiro_ps
        all_normal = all(
            (g.size >= 3) and np.isfinite(sp) and sp >= 0.05
            for g, sp in zip(clean, shapiro_ps)
            if g.size >= 2
        )
        try:
            if all_normal:
                stat, p_val = scipy_stats.f_oneway(*usable)
                test_name = "one-way ANOVA"
            else:
                stat, p_val = scipy_stats.kruskal(*usable)
                test_name = "Kruskal-Wallis"
        except ValueError as exc:
            if "All numbers are identical" in str(exc):
                return 0.0, 1.0, "Kruskal-Wallis", shapiro_ps
            return np.nan, np.nan, "test failed", shapiro_ps
        return float(stat), float(p_val), test_name, shapiro_ps

    def _draw_axis_fraction_bracket(ax, pos1, pos2, y, h, text, color="black") -> None:
        trans = ax.get_xaxis_transform()
        ax.plot(
            [pos1, pos1, pos2, pos2],
            [y, y + h, y + h, y],
            color=color,
            linewidth=0.45,
            transform=trans,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (pos1 + pos2) / 2,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=4,
            fontname="Arial",
            color=color,
            transform=trans,
            clip_on=False,
            zorder=6,
        )

    def _dynamic_similarity_ylim(arrays: list[np.ndarray]) -> tuple[float, float]:
        finite_arrays = []
        for arr in arrays:
            vals = np.asarray(arr, dtype=float).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                finite_arrays.append(vals)
        if not finite_arrays:
            return -1.0, 1.0
        merged = np.concatenate(finite_arrays)
        lo = float(np.nanmin(merged))
        hi = float(np.nanmax(merged))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return -1.0, 1.0
        span = float(hi - lo)
        if span <= 1e-9:
            center = 0.5 * (lo + hi)
            pad = max(0.05, abs(center) * 0.05)
        else:
            pad = max(0.025, span * 0.18)
        y0 = max(-1.0, lo - pad)
        y1 = min(1.0, hi + pad)
        if y1 - y0 < 0.08:
            center = 0.5 * (y0 + y1)
            y0 = max(-1.0, center - 0.04)
            y1 = min(1.0, center + 0.04)
        if y1 <= y0:
            return -1.0, 1.0
        return float(y0), float(y1)

    min_peak = float(min_firing_rate_map_peak_hz)
    if not np.isfinite(min_peak) or min_peak < 0:
        raise ValueError("min_firing_rate_map_peak_hz must be a finite number >= 0.")

    value_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for group_label, cells, _color in group_specs:
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            animal_id = str(cell.get("animal_id", cell.get("session", "")))
            cell_idx = int(cell.get("cell_idx", -1))
            for pair_key, pair_label, map_a_key, map_b_key, cutoff_keys in map_pair_specs:
                cutoff_peaks = {key: _map_peak(cell, key) for key in cutoff_keys}
                passes_cutoff = all(
                    np.isfinite(val) and val >= min_peak
                    for val in cutoff_peaks.values()
                )
                if not passes_cutoff:
                    skipped_rows.append(
                        {
                            "group": group_label,
                            "animal_id": animal_id,
                            "session": animal_id,
                            "cell_idx": cell_idx,
                            "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                            "map_pair": pair_key,
                            "map_pair_label": pair_label,
                            "map_a": map_a_key,
                            "map_b": map_b_key,
                            "cutoff_map_keys": ",".join(cutoff_keys),
                            "min_firing_rate_map_peak_hz": min_peak,
                            **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                            "reason": "below_min_firing_rate_map_peak_hz",
                        }
                    )
                    continue

                a, b = _valid_flat_pair(cell.get(map_a_key, None), cell.get(map_b_key, None))
                if a.size < 3:
                    skipped_rows.append(
                        {
                            "group": group_label,
                            "animal_id": animal_id,
                            "session": animal_id,
                            "cell_idx": cell_idx,
                            "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                            "map_pair": pair_key,
                            "map_pair_label": pair_label,
                            "map_a": map_a_key,
                            "map_b": map_b_key,
                            "cutoff_map_keys": ",".join(cutoff_keys),
                            "min_firing_rate_map_peak_hz": min_peak,
                            **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                            "reason": "fewer_than_three_valid_bins_or_shape_mismatch",
                        }
                    )
                    continue

                for metric_key, metric_label in similarity_specs:
                    val = _similarity(a, b, metric_key)
                    if not np.isfinite(val):
                        continue
                    value_rows.append(
                        {
                            "group": group_label,
                            "animal_id": animal_id,
                            "session": animal_id,
                            "cell_idx": cell_idx,
                            "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                            "map_pair": pair_key,
                            "map_pair_label": pair_label,
                            "map_a": map_a_key,
                            "map_b": map_b_key,
                            "similarity_metric": metric_key,
                            "similarity_label": metric_label,
                            "similarity": float(val),
                            "n_valid_bins": int(a.size),
                            "cutoff_map_keys": ",".join(cutoff_keys),
                            "min_firing_rate_map_peak_hz": min_peak,
                            **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                        }
                    )

    for cell in list(plcs_csplus):
        if not isinstance(cell, dict):
            continue
        animal_id = str(cell.get("animal_id", cell.get("session", "")))
        cell_idx = int(cell.get("cell_idx", -1))
        for pair_key, pair_label, map_a_key, map_b_key, cutoff_keys in csplus_paired_map_pair_specs:
            cutoff_peaks = {key: _map_peak(cell, key) for key in cutoff_keys}
            passes_cutoff = all(
                np.isfinite(val) and val >= min_peak
                for val in cutoff_peaks.values()
            )
            if not passes_cutoff:
                skipped_rows.append(
                    {
                        "group": "CS+ PLC",
                        "animal_id": animal_id,
                        "session": animal_id,
                        "cell_idx": cell_idx,
                        "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                        "map_pair": pair_key,
                        "map_pair_label": pair_label,
                        "map_a": map_a_key,
                        "map_b": map_b_key,
                        "cutoff_map_keys": ",".join(cutoff_keys),
                        "min_firing_rate_map_peak_hz": min_peak,
                        **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                        "reason": "below_min_firing_rate_map_peak_hz",
                    }
                )
                continue

            a, b = _valid_flat_pair(cell.get(map_a_key, None), cell.get(map_b_key, None))
            if a.size < 3:
                skipped_rows.append(
                    {
                        "group": "CS+ PLC",
                        "animal_id": animal_id,
                        "session": animal_id,
                        "cell_idx": cell_idx,
                        "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                        "map_pair": pair_key,
                        "map_pair_label": pair_label,
                        "map_a": map_a_key,
                        "map_b": map_b_key,
                        "cutoff_map_keys": ",".join(cutoff_keys),
                        "min_firing_rate_map_peak_hz": min_peak,
                        **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                        "reason": "fewer_than_three_valid_bins_or_shape_mismatch",
                    }
                )
                continue

            for metric_key, metric_label in similarity_specs:
                val = _similarity(a, b, metric_key)
                if not np.isfinite(val):
                    continue
                value_rows.append(
                    {
                        "group": "CS+ PLC",
                        "animal_id": animal_id,
                        "session": animal_id,
                        "cell_idx": cell_idx,
                        "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                        "map_pair": pair_key,
                        "map_pair_label": pair_label,
                        "map_a": map_a_key,
                        "map_b": map_b_key,
                        "similarity_metric": metric_key,
                        "similarity_label": metric_label,
                        "similarity": float(val),
                        "n_valid_bins": int(a.size),
                        "cutoff_map_keys": ",".join(cutoff_keys),
                        "min_firing_rate_map_peak_hz": min_peak,
                        **{f"{key}_peak_hz": cutoff_peaks.get(key, np.nan) for key in cutoff_keys},
                    }
                )

    values_df = pd.DataFrame(value_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if values_df.empty:
        values_df = pd.DataFrame(
            columns=[
                "group",
                "animal_id",
                "session",
                "cell_idx",
                "cell_num",
                "map_pair",
                "map_pair_label",
                "map_a",
                "map_b",
                "similarity_metric",
                "similarity_label",
                "similarity",
                "n_valid_bins",
                "cutoff_map_keys",
                "min_firing_rate_map_peak_hz",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for metric_key, metric_label in similarity_specs:
        for pair_key, pair_label, _map_a_key, _map_b_key, _cutoff_keys in map_pair_specs:
            for group_label, _cells, _color in group_specs:
                vals = values_df[
                    (values_df["similarity_metric"] == metric_key)
                    & (values_df["map_pair"] == pair_key)
                    & (values_df["group"] == group_label)
                ]["similarity"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = int(vals.size)
                sd = float(np.nanstd(vals, ddof=1)) if n > 1 else np.nan
                summary_rows.append(
                    {
                        "similarity_metric": metric_key,
                        "similarity_label": metric_label,
                        "map_pair": pair_key,
                        "map_pair_label": pair_label,
                        "group": group_label,
                        "n": n,
                        "mean": float(np.nanmean(vals)) if n > 0 else np.nan,
                        "median": float(np.nanmedian(vals)) if n > 0 else np.nan,
                        "sd": sd,
                        "sem": float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan,
                        "min": float(np.nanmin(vals)) if n > 0 else np.nan,
                        "max": float(np.nanmax(vals)) if n > 0 else np.nan,
                    }
                )
        for pair_key, pair_label, _map_a_key, _map_b_key, _cutoff_keys in csplus_paired_map_pair_specs:
            vals = values_df[
                (values_df["similarity_metric"] == metric_key)
                & (values_df["map_pair"] == pair_key)
                & (values_df["group"] == "CS+ PLC")
            ]["similarity"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            sd = float(np.nanstd(vals, ddof=1)) if n > 1 else np.nan
            summary_rows.append(
                {
                    "similarity_metric": metric_key,
                    "similarity_label": metric_label,
                    "map_pair": pair_key,
                    "map_pair_label": pair_label,
                    "group": "CS+ PLC",
                    "n": n,
                    "mean": float(np.nanmean(vals)) if n > 0 else np.nan,
                    "median": float(np.nanmedian(vals)) if n > 0 else np.nan,
                    "sd": sd,
                    "sem": float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan,
                    "min": float(np.nanmin(vals)) if n > 0 else np.nan,
                    "max": float(np.nanmax(vals)) if n > 0 else np.nan,
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    comparison_specs = [
        ("CS+ PLC", "CS- PLC"),
        ("CS+ PLC", "Non-PLC"),
        ("CS- PLC", "Non-PLC"),
    ]
    omnibus_rows: list[dict[str, Any]] = []
    for metric_key, metric_label in similarity_specs:
        for pair_key, pair_label, _map_a_key, _map_b_key, _cutoff_keys in map_pair_specs:
            group_values = []
            group_ns = {}
            for group_label, _cells, _color in group_specs:
                vals = values_df[
                    (values_df["similarity_metric"] == metric_key)
                    & (values_df["map_pair"] == pair_key)
                    & (values_df["group"] == group_label)
                ]["similarity"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                group_values.append(vals)
                group_ns[group_label] = int(vals.size)
            stat, p_val, test_name, shapiro_ps = _omnibus_test(group_values)
            omnibus_rows.append(
                {
                    "similarity_metric": metric_key,
                    "similarity_label": metric_label,
                    "map_pair": pair_key,
                    "map_pair_label": pair_label,
                    "test": test_name,
                    "statistic": stat,
                    "p_value": p_val,
                    "significance": _sig_label(float(p_val)) if np.isfinite(p_val) else "",
                    **{f"n_{label}": group_ns[label] for label, _cells, _color in group_specs},
                    **{f"shapiro_p_{label}": shapiro_ps[idx] for idx, (label, _cells, _color) in enumerate(group_specs)},
                }
            )
    omnibus_df = pd.DataFrame(omnibus_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for metric_key, metric_label in similarity_specs:
        for pair_key, pair_label, _map_a_key, _map_b_key, _cutoff_keys in map_pair_specs:
            start_idx = len(pairwise_rows)
            for group1, group2 in comparison_specs:
                vals1 = values_df[
                    (values_df["similarity_metric"] == metric_key)
                    & (values_df["map_pair"] == pair_key)
                    & (values_df["group"] == group1)
                ]["similarity"].to_numpy(dtype=float)
                vals2 = values_df[
                    (values_df["similarity_metric"] == metric_key)
                    & (values_df["map_pair"] == pair_key)
                    & (values_df["group"] == group2)
                ]["similarity"].to_numpy(dtype=float)
                vals1 = vals1[np.isfinite(vals1)]
                vals2 = vals2[np.isfinite(vals2)]
                p_val, test_name, shapiro_p1, shapiro_p2 = _unpaired_test_auto(vals1, vals2)
                pairwise_rows.append(
                    {
                        "similarity_metric": metric_key,
                        "similarity_label": metric_label,
                        "map_pair": pair_key,
                        "map_pair_label": pair_label,
                        "comparison": f"{group1} vs {group2}",
                        "group1": group1,
                        "group2": group2,
                        "n_group1": int(vals1.size),
                        "n_group2": int(vals2.size),
                        "test": test_name,
                        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                        "p_adjust_method": "holm",
                        "p_value_adj": np.nan,
                        "significance": "",
                        "shapiro_p_group1": shapiro_p1,
                        "shapiro_p_group2": shapiro_p2,
                    }
                )
            p_adj = _holm_adjust([row["p_value"] for row in pairwise_rows[start_idx:]])
            for row, adj in zip(pairwise_rows[start_idx:], p_adj):
                row["p_value_adj"] = float(adj) if np.isfinite(adj) else np.nan
                row["significance"] = _sig_label(float(adj)) if np.isfinite(adj) else ""
    pairwise_df = pd.DataFrame(pairwise_rows)

    paired_rows: list[dict[str, Any]] = []
    paired_values_by_metric: dict[str, pd.DataFrame] = {}
    for metric_key, metric_label in similarity_specs:
        left = values_df[
            (values_df["similarity_metric"] == metric_key)
            & (values_df["map_pair"] == "all_vs_ss")
            & (values_df["group"] == "CS+ PLC")
        ][["animal_id", "session", "cell_idx", "cell_num", "similarity"]].rename(
            columns={"similarity": "all_vs_ss_similarity"}
        )
        right = values_df[
            (values_df["similarity_metric"] == metric_key)
            & (values_df["map_pair"] == "all_vs_cs")
            & (values_df["group"] == "CS+ PLC")
        ][["animal_id", "session", "cell_idx", "cell_num", "similarity"]].rename(
            columns={"similarity": "all_vs_cs_similarity"}
        )
        paired = left.merge(
            right,
            on=["animal_id", "session", "cell_idx", "cell_num"],
            how="inner",
        )
        paired_values_by_metric[metric_key] = paired
        vals_ss = paired["all_vs_ss_similarity"].to_numpy(dtype=float) if not paired.empty else np.array([], dtype=float)
        vals_cs = paired["all_vs_cs_similarity"].to_numpy(dtype=float) if not paired.empty else np.array([], dtype=float)
        p_val, statistic, test_name, n_valid, shapiro_p = _paired_test(vals_ss, vals_cs)
        paired_rows.append(
            {
                "similarity_metric": metric_key,
                "similarity_label": metric_label,
                "comparison": "CS+ PLC all_vs_ss vs all_vs_cs",
                "group": "CS+ PLC",
                "map_pair_1": "all_vs_ss",
                "map_pair_label_1": "All vs SS",
                "map_pair_2": "all_vs_cs",
                "map_pair_label_2": "All vs CS",
                "n_pairs": int(n_valid),
                "median_1": float(np.nanmedian(vals_ss)) if vals_ss.size else np.nan,
                "median_2": float(np.nanmedian(vals_cs)) if vals_cs.size else np.nan,
                "median_delta_1_minus_2": (
                    float(np.nanmedian(vals_ss) - np.nanmedian(vals_cs)) if vals_ss.size and vals_cs.size else np.nan
                ),
                "test": test_name,
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                "significance": _sig_label(float(p_val)) if np.isfinite(p_val) else "",
                "shapiro_p_diff": float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
            }
        )
    paired_df = pd.DataFrame(paired_rows)

    n_plot_cols = len(map_pair_specs) + 1
    fig, axes = plt.subplots(len(similarity_specs), n_plot_cols, figsize=(fig_width, fig_height), sharey=False)
    axes = np.asarray(axes)
    rng = np.random.default_rng(42)
    x = np.arange(len(group_specs), dtype=float)
    group_tick_base = ["CS+\nPLC", "CS-\nPLC", "Non-\nPLC"]

    for row_idx, (metric_key, metric_label) in enumerate(similarity_specs):
        for col_idx, (pair_key, pair_label, _map_a_key, _map_b_key, _cutoff_keys) in enumerate(map_pair_specs):
            ax = axes[row_idx, col_idx]
            ns_for_ticks: list[int] = []
            subplot_vals: list[np.ndarray] = []
            subplot_colors: list[str] = []
            for group_idx, (group_label, _cells, color) in enumerate(group_specs):
                vals = values_df[
                    (values_df["similarity_metric"] == metric_key)
                    & (values_df["map_pair"] == pair_key)
                    & (values_df["group"] == group_label)
                ]["similarity"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                ns_for_ticks.append(int(vals.size))
                subplot_vals.append(vals)
                subplot_colors.append(color)
            _half_violin_panel(
                ax,
                list(x),
                subplot_vals,
                subplot_colors,
                paired_specs=[],
                rng_seed=int(42 + row_idx * 17 + col_idx),
            )

            ax.axhline(0, color="black", linewidth=0.4, zorder=1)
            ax.set_ylim(*_dynamic_similarity_ylim(subplot_vals))
            ax.set_xlim(-0.6, len(group_specs) - 0.4)
            if row_idx == 0:
                title_y = 1.24 if show_pairwise_stats else 1.05
                ax.set_title(pair_label, fontsize=6, fontname="Arial", y=title_y, pad=0)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=6, fontname="Arial")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{label}\nn={n}" for label, n in zip(group_tick_base, ns_for_ticks)],
                fontsize=5,
                fontname="Arial",
            )
            if show_pairwise_stats and not pairwise_df.empty:
                bracket_specs = [
                    ("CS+ PLC", "CS- PLC", 0, 1, 1.01),
                    ("CS+ PLC", "Non-PLC", 0, 2, 1.065),
                    ("CS- PLC", "Non-PLC", 1, 2, 1.12),
                ]
                for group1, group2, pos1, pos2, y_bracket in bracket_specs:
                    stat_row = pairwise_df[
                        (pairwise_df["similarity_metric"] == metric_key)
                        & (pairwise_df["map_pair"] == pair_key)
                        & (pairwise_df["group1"] == group1)
                        & (pairwise_df["group2"] == group2)
                    ]
                    if stat_row.empty:
                        continue
                    p_adj = float(stat_row.iloc[0]["p_value_adj"])
                    if not np.isfinite(p_adj):
                        continue
                    if show_only_significant_stats and p_adj >= 0.05:
                        continue
                    _draw_axis_fraction_bracket(
                        ax,
                        pos1,
                        pos2,
                        y_bracket,
                        0.018,
                        _sig_label(p_adj),
                    )
            ax.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax = axes[row_idx, len(map_pair_specs)]
        paired = paired_values_by_metric.get(metric_key, pd.DataFrame())
        vals_ss = (
            paired["all_vs_ss_similarity"].to_numpy(dtype=float)
            if not paired.empty
            else np.array([], dtype=float)
        )
        vals_cs = (
            paired["all_vs_cs_similarity"].to_numpy(dtype=float)
            if not paired.empty
            else np.array([], dtype=float)
        )
        vals_ss_vs_cs = values_df[
            (values_df["similarity_metric"] == metric_key)
            & (values_df["map_pair"] == "ss_vs_cs")
            & (values_df["group"] == "CS+ PLC")
        ]["similarity"].to_numpy(dtype=float)
        paired_vals = [
            vals_ss[np.isfinite(vals_ss)],
            vals_cs[np.isfinite(vals_cs)],
            vals_ss_vs_cs[np.isfinite(vals_ss_vs_cs)],
        ]
        paired_colors = [SS_COLOR, CS_COLOR, "#D81B60"]
        paired_labels = ["All vs\nSS", "All vs\nCS", "SS vs\nCS"]
        pair_x = np.arange(3, dtype=float)
        _half_violin_panel(
            ax,
            list(pair_x),
            paired_vals,
            paired_colors,
            paired_specs=[(0, 1)],
            rng_seed=int(142 + row_idx),
        )
        ax.axhline(0, color="black", linewidth=0.4, zorder=1)
        ax.set_ylim(*_dynamic_similarity_ylim(paired_vals))
        ax.set_xlim(-0.55, 2.55)
        if row_idx == 0:
            title_y = 1.24 if show_pairwise_stats else 1.05
            ax.set_title("CS+ only", fontsize=6, fontname="Arial", y=title_y, pad=0)
        ns_for_ticks = [int(vals.size) for vals in paired_vals]
        ax.set_xticks(pair_x)
        ax.set_xticklabels(
            [f"{label}\nn={n}" for label, n in zip(paired_labels, ns_for_ticks)],
            fontsize=5,
            fontname="Arial",
        )
        if show_pairwise_stats and not paired_df.empty:
            stat_row = paired_df[paired_df["similarity_metric"] == metric_key]
            if not stat_row.empty:
                p_val = float(stat_row.iloc[0]["p_value"])
                if np.isfinite(p_val) and (not show_only_significant_stats or p_val < 0.05):
                    _draw_axis_fraction_bracket(
                        ax,
                        0,
                        1,
                        1.01,
                        0.018,
                        _sig_label(p_val),
                    )
        ax.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=0.5, h_pad=1.2)
    figure_path = str(save_path) if save_path is not None else None
    if save_path:
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return {
        "fig": fig,
        "values_df": values_df,
        "summary_df": summary_df,
        "omnibus_df": omnibus_df,
        "pairwise_df": pairwise_df,
        "paired_df": paired_df,
        "skipped_df": skipped_df,
        "figure_path": figure_path,
    }


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


def _collect_complex_burst_metric_pools(
    plcs_csplus,
    metric_specs,
    min_bursts_per_condition: int = 3,
):
    conditions = ['run_in', 'run_out', 'rest_in', 'rest_out']
    burst_data_all = {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs}

    for cell in plcs_csplus:
        bm = cell.get('burst_metrics')
        complex_by_cond = bm.get('complex') if isinstance(bm, dict) else None
        for cond in conditions:
            bursts = complex_by_cond.get(cond, []) if isinstance(complex_by_cond, dict) else []
            if not isinstance(bursts, (list, tuple)) or len(bursts) < int(min_bursts_per_condition):
                continue
            for burst in bursts:
                if not isinstance(burst, dict):
                    continue
                for metric_key, _ in metric_specs:
                    val = burst.get(metric_key, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    if np.isfinite(val):
                        burst_data_all[metric_key][cond].append(val)

    return burst_data_all, conditions


def _compute_hist_axis_config(metric_key, arrays):
    finite_arrays = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_arrays.append(finite)
    if not finite_arrays:
        return None

    combined = np.concatenate(finite_arrays)
    data_min = float(np.min(combined))
    data_max = float(np.max(combined))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return None

    combined_rounded = np.round(combined)
    is_integer_metric = np.all(np.isclose(combined, combined_rounded, atol=1e-8))
    int_min = None
    int_max = None

    if metric_key == 'n_spikes':
        bin_edges = np.arange(1.5, 10.5 + 1.0, 1.0, dtype=float)
        xlim = (2.0, 10.0)
        xticks = np.array([2, 4, 6, 8, 10], dtype=int)
    elif metric_key == 'duration_ms':
        if data_max == data_min:
            span = 10.0
            bin_edges = np.array([data_min - span, data_max + span], dtype=float)
            xlim = (float(data_min - span), float(data_max + span))
        else:
            bin_start = 10.0 * np.floor(data_min / 10.0)
            bin_end = 10.0 * np.ceil(data_max / 10.0)
            if bin_end <= bin_start:
                bin_end = bin_start + 10.0
            bin_edges = np.arange(bin_start, bin_end + 10.0, 10.0, dtype=float)
            x_pad = 0.03 * (data_max - data_min) if data_max > data_min else 1.0
            xlim = (float(data_min - x_pad), float(data_max + x_pad))
        xticks = None
    elif is_integer_metric:
        int_min = int(np.min(combined_rounded))
        int_max = int(np.max(combined_rounded))
        bin_edges = np.arange(int_min - 0.5, int_max + 1.5, 1.0, dtype=float)
        if bin_edges.size < 2:
            bin_edges = np.array([int_min - 0.5, int_min + 0.5], dtype=float)
        xlim = (float(bin_edges[0]), float(bin_edges[-1]))
        xticks = np.arange(int_min, int_max + 1, 1, dtype=int) if int_max >= int_min else None
    elif data_max == data_min:
        span = max(abs(data_max) * 0.1, 1.0)
        bin_edges = np.array([data_min - span, data_max + span], dtype=float)
        xlim = (float(data_min - span), float(data_max + span))
        xticks = None
    else:
        n_bins = int(np.clip(np.sqrt(combined.size), 10, 30))
        bin_edges = np.linspace(data_min, data_max, n_bins + 1)
        x_pad = 0.03 * (data_max - data_min) if data_max > data_min else 1.0
        xlim = (float(data_min - x_pad), float(data_max + x_pad))
        xticks = None

    if metric_key == 'duration_ms':
        tick_start = max(50, int(np.ceil(xlim[0] / 50.0) * 50))
        tick_end = int(np.floor(xlim[1] / 50.0) * 50)
        if tick_end >= tick_start:
            xticks = np.arange(tick_start, tick_end + 1, 50, dtype=int)

    return {
        'bin_edges': np.asarray(bin_edges, dtype=float),
        'xlim': xlim,
        'xticks': xticks,
    }


def _plot_overlapping_hist_panel(
    ax,
    vals_in,
    vals_out,
    axis_config,
    xlabel,
    ylabel='',
    sig_label='',
    show_legend=False,
    legend_loc='upper left',
    first_label='Out PF',
    second_label='In PF',
    first_color='gray',
    second_color='magenta',
    background_color=CS_PLC_BG,
    plot_mode='hist',
    fill_alpha=0.3,
):
    vals_in = np.asarray(vals_in, dtype=float)
    vals_out = np.asarray(vals_out, dtype=float)
    vals_in = vals_in[np.isfinite(vals_in)]
    vals_out = vals_out[np.isfinite(vals_out)]

    if len(vals_in) < 2 or len(vals_out) < 2 or axis_config is None:
        ax.axis('off')
        return

    bin_edges = axis_config['bin_edges']
    plot_mode = str(plot_mode).strip().lower()
    if plot_mode not in {'hist', 'cumulative'}:
        raise ValueError("plot_mode must be 'hist' or 'cumulative'.")

    if background_color is not None:
        ax.set_facecolor(background_color)
    if plot_mode == 'hist':
        weights_out = np.full(vals_out.shape, 100.0 / len(vals_out), dtype=float)
        weights_in = np.full(vals_in.shape, 100.0 / len(vals_in), dtype=float)
        ax.hist(
            vals_out,
            bins=bin_edges,
            weights=weights_out,
            color=first_color,
            alpha=fill_alpha,
            edgecolor='none',
            zorder=1,
            label=first_label,
        )
        ax.hist(
            vals_in,
            bins=bin_edges,
            weights=weights_in,
            color=second_color,
            alpha=fill_alpha,
            edgecolor='none',
            zorder=2,
            label=second_label,
        )
        ax.hist(
            vals_out,
            bins=bin_edges,
            weights=weights_out,
            histtype='step',
            color=first_color,
            linewidth=0.8,
            zorder=3,
        )
        ax.hist(
            vals_in,
            bins=bin_edges,
            weights=weights_in,
            histtype='step',
            color=second_color,
            linewidth=0.8,
            zorder=4,
        )
    else:
        vals_out_sorted = np.sort(vals_out)
        vals_in_sorted = np.sort(vals_in)
        cdf_out = 100.0 * np.arange(1, len(vals_out_sorted) + 1, dtype=float) / float(len(vals_out_sorted))
        cdf_in = 100.0 * np.arange(1, len(vals_in_sorted) + 1, dtype=float) / float(len(vals_in_sorted))

        ax.step(vals_out_sorted, cdf_out, where='post', color=first_color, linewidth=1.0, zorder=3, label=first_label)
        ax.step(vals_in_sorted, cdf_in, where='post', color=second_color, linewidth=1.0, zorder=4, label=second_label)
        ax.fill_between(vals_out_sorted, cdf_out, step='post', color=first_color, alpha=0.12, zorder=1)
        ax.fill_between(vals_in_sorted, cdf_in, step='post', color=second_color, alpha=0.12, zorder=2)

    if sig_label:
        ax.text(
            0.5,
            0.99,
            sig_label,
            ha='center',
            va='top',
            fontsize=6,
            fontname='Arial',
            transform=ax.transAxes,
        )

    ax.set_xlabel(xlabel, fontsize=6, fontname='Arial')
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    if show_legend:
        ax.legend(
            frameon=False,
            loc=legend_loc,
            fontsize=5,
            handlelength=1.0,
            borderaxespad=0.2,
        )
    ax.tick_params(labelsize=5)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontname('Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(*axis_config['xlim'])
    xticks = axis_config.get('xticks')
    if xticks is not None:
        ax.set_xticks(xticks)

    if plot_mode == 'hist':
        y_max = 0.0
        out_hist, _ = np.histogram(vals_out, bins=bin_edges, weights=np.full(vals_out.shape, 100.0 / len(vals_out), dtype=float))
        in_hist, _ = np.histogram(vals_in, bins=bin_edges, weights=np.full(vals_in.shape, 100.0 / len(vals_in), dtype=float))
        if out_hist.size > 0:
            y_max = max(y_max, float(np.nanmax(out_hist)))
        if in_hist.size > 0:
            y_max = max(y_max, float(np.nanmax(in_hist)))
        if np.isfinite(y_max) and y_max > 0:
            ax.set_ylim(0, y_max * 1.15)
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])


def _plot_paired_violin_panel(
    ax,
    vals_first,
    vals_second,
    ylabel='',
    sig_label='',
    first_label='Quiet',
    second_label='Moving',
    first_color='#563C25',
    second_color='#F9E800',
    background_color=None,
    ylim=None,
):
    vals_first = np.asarray(vals_first, dtype=float)
    vals_second = np.asarray(vals_second, dtype=float)

    if background_color is not None:
        ax.set_facecolor(background_color)

    if len(vals_first) < 2 or len(vals_second) < 2:
        ax.axis('off')
        return

    plot_data = []
    for vals in (vals_first, vals_second):
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([np.nan], dtype=float)
        plot_data.append(finite)

    if any(len(arr) < 2 for arr in plot_data):
        ax.axis('off')
        return

    positions = [1, 2]
    half_sides = ['left', 'right']
    colors = [first_color, second_color]
    for pos, d, color, side in zip(positions, plot_data, colors, half_sides):
        vp = ax.violinplot([d], positions=[pos], showmedians=True, showextrema=False)
        body = vp['bodies'][0]
        verts = body.get_paths()[0].vertices
        if side == 'left':
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
        else:
            verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
        body.set_facecolor(color)
        body.set_edgecolor('none')
        body.set_alpha(0.3)
        _style_violin_medians(vp)

        finite_mask = np.isfinite(d)
        if np.any(finite_mask):
            d_finite = d[finite_mask]
            y_range = np.ptp(d_finite) if len(d_finite) > 1 else 1.0
            y_spacing = max(y_range * 0.04, 1e-8)
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
            ax.scatter(xs, d_finite, s=4, color='black', alpha=0.6, linewidths=0, zorder=3)

    n_pairs = min(len(vals_first), len(vals_second))
    for idx in range(n_pairs):
        if np.isfinite(vals_first[idx]) and np.isfinite(vals_second[idx]):
            ax.plot([1.12, 1.88], [vals_first[idx], vals_second[idx]], color='black', alpha=0.3, linewidth=0.6, zorder=2)

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
    ax.set_xticklabels([first_label, second_label], fontsize=5, fontname='Arial')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    ax.tick_params(labelsize=5)
    for label in list(ax.get_yticklabels()):
        label.set_fontname('Arial')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_stacked_outcome_bar(
    ax,
    counts,
    total_count: int,
    ylabel='',
    show_legend=False,
):
    quiet_higher = int(counts.get('quiet_higher', 0))
    moving_higher = int(counts.get('moving_higher', 0))
    nonsig = int(counts.get('nonsig', 0))

    segments = [
        ('Quiet > moving', quiet_higher, '#563C25'),
        ('Moving > quiet', moving_higher, '#F9E800'),
        ('n.s.', nonsig, '#BDBDBD'),
    ]

    left = 0.0
    legend_handles = []
    for label, value, color in segments:
        bar = ax.barh(
            [1.0],
            [value],
            left=left,
            height=0.18,
            color=color,
            edgecolor='none',
            zorder=2,
            label=label,
        )
        legend_handles.append(bar[0])
        if value > 0:
            txt_color = 'white' if color == '#563C25' else 'black'
            ax.text(
                left + 0.5 * value,
                1.0,
                str(value),
                ha='center',
                va='center',
                fontsize=5,
                fontname='Arial',
                color=txt_color,
            )
        left += value

    ax.set_ylim(0.7, 1.3)
    ax.set_xlim(0, max(int(total_count), 1))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=6, fontname='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if show_legend:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc='upper right',
            fontsize=5,
            handlelength=1.0,
            borderaxespad=0.2,
        )


def _format_csplus_cell_hist_label(cell, fallback_idx):
    session = ''
    if isinstance(cell, dict):
        for key in ('session', 'animal_id', 'folder'):
            val = cell.get(key)
            if isinstance(val, str) and val.strip():
                session = val.strip()
                break

    cell_num = None
    if isinstance(cell, dict):
        cell_idx = cell.get('cell_idx', None)
        try:
            if cell_idx is not None and np.isfinite(float(cell_idx)):
                cell_num = int(cell_idx) + 1
        except (TypeError, ValueError):
            cell_num = None

    if session and cell_num is not None:
        return f'{session}\nCell {cell_num}'
    if session:
        return session
    if cell_num is not None:
        return f'Cell {cell_num}'
    return f'Cell {int(fallback_idx) + 1}'


def _collect_complex_burst_metric_cell_rows(
    plcs_csplus,
    metric_specs,
    state_key: str = 'run',
    min_bursts_per_condition: int = 3,
):
    conds = [f'{state_key}_in', f'{state_key}_out']
    rows = []

    for fallback_idx, cell in enumerate(plcs_csplus):
        bm = cell.get('burst_metrics') if isinstance(cell, dict) else None
        complex_by_cond = bm.get('complex') if isinstance(bm, dict) else None
        metric_arrays = {}
        any_plottable = False

        for metric_key, _ in metric_specs:
            cond_arrays = {}
            for cond in conds:
                bursts = complex_by_cond.get(cond, []) if isinstance(complex_by_cond, dict) else []
                if not isinstance(bursts, (list, tuple)) or len(bursts) < int(min_bursts_per_condition):
                    cond_arrays[cond] = np.array([], dtype=float)
                    continue

                vals = []
                for burst in bursts:
                    if not isinstance(burst, dict):
                        continue
                    val = burst.get(metric_key, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    if np.isfinite(val):
                        vals.append(val)
                cond_arrays[cond] = np.asarray(vals, dtype=float)

            metric_arrays[metric_key] = cond_arrays
            if len(cond_arrays[conds[0]]) >= 2 and len(cond_arrays[conds[1]]) >= 2:
                any_plottable = True

        if any_plottable:
            rows.append(
                {
                    'label': _format_csplus_cell_hist_label(cell, fallback_idx),
                    'data': metric_arrays,
                }
            )

    return rows


def _validate_quiet_moving_speed_threshold(cell, speed_threshold_cm_s: float) -> None:
    if not isinstance(cell, dict):
        return
    params = cell.get('params')
    if not isinstance(params, dict):
        return
    saved_threshold = params.get('speed_threshold', np.nan)
    try:
        saved_threshold = float(saved_threshold)
    except (TypeError, ValueError):
        return
    if not np.isfinite(saved_threshold):
        return
    if not np.isclose(saved_threshold, float(speed_threshold_cm_s), atol=1e-8):
        session = str(cell.get('session', cell.get('animal_id', 'unknown_session')))
        cell_idx = cell.get('cell_idx', cell.get('cell_id', '?'))
        raise ValueError(
            f"Cell {session}:{cell_idx} has saved speed_threshold={saved_threshold}, "
            f"expected {float(speed_threshold_cm_s)} for quiet/moving comparison."
        )


def _collect_complex_burst_metric_pools_quiet_vs_moving_allcells(
    plcs_csplus,
    plcs_csminus,
    non_plcs,
    metric_specs,
    speed_threshold_cm_s: float = 3.0,
    min_bursts_per_state: int = 10,
):
    conditions = ['quiet', 'moving']
    pooled = {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs}
    cell_means = {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs}
    cell_burst_values = {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs}
    eligible_cells = 0

    all_cells = list(plcs_csplus) + list(plcs_csminus) + list(non_plcs)
    for cell in all_cells:
        _validate_quiet_moving_speed_threshold(cell, speed_threshold_cm_s=speed_threshold_cm_s)

        bm = cell.get('burst_metrics') if isinstance(cell, dict) else None
        complex_by_cond = bm.get('complex') if isinstance(bm, dict) else None
        if not isinstance(complex_by_cond, dict):
            continue

        state_bursts = {
            'moving': [],
            'quiet': [],
        }
        for cond_name, bursts in complex_by_cond.items():
            if cond_name == 'params' or not isinstance(bursts, (list, tuple)):
                continue
            cond_name = str(cond_name).strip().lower()
            for burst in bursts:
                if not isinstance(burst, dict):
                    continue
                state = str(burst.get('state', '')).strip().lower()
                if state == 'run':
                    state_bursts['moving'].append(burst)
                elif state == 'rest':
                    state_bursts['quiet'].append(burst)
                elif cond_name.startswith('run_'):
                    state_bursts['moving'].append(burst)
                elif cond_name.startswith('rest_'):
                    state_bursts['quiet'].append(burst)

        if len(state_bursts['moving']) <= int(min_bursts_per_state) or len(state_bursts['quiet']) <= int(min_bursts_per_state):
            continue

        eligible_cells += 1
        per_cell_metric_values = {metric: {cond: [] for cond in conditions} for metric, _ in metric_specs}
        for state_name in conditions:
            bursts = state_bursts[state_name]
            for burst in bursts:
                if not isinstance(burst, dict):
                    continue
                for metric_key, _ in metric_specs:
                    val = burst.get(metric_key, np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    if np.isfinite(val):
                        pooled[metric_key][state_name].append(val)
                        per_cell_metric_values[metric_key][state_name].append(val)

        for metric_key, _ in metric_specs:
            for state_name in conditions:
                vals = np.asarray(per_cell_metric_values[metric_key][state_name], dtype=float)
                cell_burst_values[metric_key][state_name].append(vals.copy())
                cell_means[metric_key][state_name].append(float(np.nanmean(vals)) if vals.size > 0 else np.nan)

    for metric_key, _ in metric_specs:
        for state_name in conditions:
            cell_means[metric_key][state_name] = np.asarray(cell_means[metric_key][state_name], dtype=float)

    return pooled, cell_means, cell_burst_values, eligible_cells


def _classify_quiet_moving_cell_outcomes(
    cell_burst_values,
    metric_specs,
    alpha: float = 0.05,
):
    outcome_counts = {}
    for metric_key, _ in metric_specs:
        counts = {
            'quiet_higher': 0,
            'moving_higher': 0,
            'nonsig': 0,
        }
        quiet_arrays = cell_burst_values[metric_key]['quiet']
        moving_arrays = cell_burst_values[metric_key]['moving']
        for quiet_vals, moving_vals in zip(quiet_arrays, moving_arrays):
            p_val, _ = _unpaired_test_bursts(quiet_vals, moving_vals)
            quiet_mean = float(np.nanmean(quiet_vals)) if np.asarray(quiet_vals).size > 0 else np.nan
            moving_mean = float(np.nanmean(moving_vals)) if np.asarray(moving_vals).size > 0 else np.nan

            if np.isfinite(p_val) and p_val < float(alpha) and np.isfinite(quiet_mean) and np.isfinite(moving_mean):
                if quiet_mean > moving_mean:
                    counts['quiet_higher'] += 1
                elif moving_mean > quiet_mean:
                    counts['moving_higher'] += 1
                else:
                    counts['nonsig'] += 1
            else:
                counts['nonsig'] += 1
        outcome_counts[metric_key] = counts
    return outcome_counts


def _plot_complex_burst_metric_distribution_panels(
    burst_data_all,
    cell_rows,
    metric_specs,
    save_path: str,
    state_key: str = 'run',
):
    panel_w = 1.1
    panel_h = 1.1
    n_rows = 1 + len(cell_rows)
    fig, axes = plt.subplots(
        n_rows,
        len(metric_specs),
        figsize=(panel_w * len(metric_specs), panel_h * n_rows),
        sharex='col',
        sharey=False,
    )
    if len(metric_specs) == 1:
        axes = np.asarray(axes).reshape(n_rows, 1)
    elif n_rows == 1:
        axes = np.asarray(axes).reshape(1, len(metric_specs))

    conds = [f'{state_key}_in', f'{state_key}_out']
    axis_configs = {}
    for metric_key, _ in metric_specs:
        all_arrays = [
            burst_data_all[metric_key][conds[0]],
            burst_data_all[metric_key][conds[1]],
        ]
        for row in cell_rows:
            all_arrays.append(row['data'][metric_key][conds[0]])
            all_arrays.append(row['data'][metric_key][conds[1]])
        axis_configs[metric_key] = _compute_hist_axis_config(
            metric_key,
            all_arrays,
        )

    for ax_idx, (metric_key, xlabel) in enumerate(metric_specs):
        top_vals_in = burst_data_all[metric_key][conds[0]]
        top_vals_out = burst_data_all[metric_key][conds[1]]
        p_val_top, _ = _unpaired_test_bursts(top_vals_in, top_vals_out)
        _plot_overlapping_hist_panel(
            axes[0, ax_idx],
            top_vals_in,
            top_vals_out,
            axis_configs[metric_key],
            xlabel=xlabel,
            ylabel='% of bursts' if ax_idx == 0 else '',
            sig_label=_sig_label(p_val_top),
            show_legend=(ax_idx == 0),
        )

        for row_idx, row in enumerate(cell_rows, start=1):
            bottom_vals_in = row['data'][metric_key][conds[0]]
            bottom_vals_out = row['data'][metric_key][conds[1]]
            p_val_bottom, _ = _unpaired_test_bursts(bottom_vals_in, bottom_vals_out)
            _plot_overlapping_hist_panel(
                axes[row_idx, ax_idx],
                bottom_vals_in,
                bottom_vals_out,
                axis_configs[metric_key],
                xlabel=xlabel,
                ylabel='% of bursts' if ax_idx == 0 else '',
                sig_label=_sig_label(p_val_bottom),
                show_legend=False,
            )
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    row_labels = ['Pooled'] + [row['label'] for row in cell_rows]
    for row_idx, row_label in enumerate(row_labels):
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, -1]
        pos_left = left_ax.get_position()
        pos_right = right_ax.get_position()
        x_center = 0.5 * (pos_left.x0 + pos_right.x1)
        y_top = pos_left.y1 + 0.006
        fig.text(
            x_center,
            y_top,
            row_label,
            ha='center',
            va='bottom',
            fontsize=5,
            fontname='Arial',
        )

    fig.savefig(save_path, dpi=300)
    return fig


def plot_complex_burst_metrics_allbursts_csplus(
    plcs_csplus,
    save_path: str,
    state_key: str = 'run',
    min_bursts_per_condition: int = 3,
):
    burst_metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
    ]
    burst_data_all, _ = _collect_complex_burst_metric_pools(
        plcs_csplus,
        burst_metric_specs,
        min_bursts_per_condition=min_bursts_per_condition,
    )
    return _plot_complex_burst_metric_distribution_panels(
        burst_data_all,
        [],
        burst_metric_specs,
        save_path=save_path,
        state_key=state_key,
    )

def plot_complex_burst_metric_distributions_csplus(
    plcs_csplus,
    save_path: str,
    state_key: str = 'run',
    min_bursts_per_condition: int = 3,
):
    """Plot pooled and per-cell complex-burst metric distributions for CS+ place cells."""
    burst_metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
    ]
    burst_data_all, _ = _collect_complex_burst_metric_pools(
        plcs_csplus,
        burst_metric_specs,
        min_bursts_per_condition=min_bursts_per_condition,
    )
    cell_rows = _collect_complex_burst_metric_cell_rows(
        plcs_csplus,
        burst_metric_specs,
        state_key=state_key,
        min_bursts_per_condition=min_bursts_per_condition,
    )
    return _plot_complex_burst_metric_distribution_panels(
        burst_data_all,
        cell_rows,
        burst_metric_specs,
        save_path=save_path,
        state_key=state_key,
    )


def plot_complex_burst_metric_distributions_quiet_vs_moving_allcells(
    plcs_csplus,
    plcs_csminus,
    non_plcs,
    save_path: str,
    speed_threshold_cm_s: float = 3.0,
    min_bursts_per_state: int = 10,
    plot_mode: str = 'cumulative',
    fig_width: float = 4.4,
    fig_height: float = 3.15,
):
    """Plot pooled complex-burst metric distributions for quiet vs moving across all cell categories."""
    metric_specs = [
        ('n_spikes', 'Spks./burst'),
        ('peak_amp', 'Peak amp'),
        ('duration_ms', 'Duration (ms)'),
        ('auc', 'AUC'),
    ]
    pooled, cell_means, cell_burst_values, eligible_cells = _collect_complex_burst_metric_pools_quiet_vs_moving_allcells(
        plcs_csplus,
        plcs_csminus,
        non_plcs,
        metric_specs,
        speed_threshold_cm_s=speed_threshold_cm_s,
        min_bursts_per_state=min_bursts_per_state,
    )
    outcome_counts = _classify_quiet_moving_cell_outcomes(
        cell_burst_values,
        metric_specs,
    )

    fig, axes = plt.subplots(
        3,
        len(metric_specs),
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
        gridspec_kw={'height_ratios': [1.2, 1.15, 0.8]},
    )
    axes = np.asarray(axes)
    if len(metric_specs) == 1:
        axes = axes.reshape(3, 1)

    axis_configs = {}
    metric_ylims = {}
    for metric_key, _ in metric_specs:
        axis_configs[metric_key] = _compute_hist_axis_config(
            metric_key,
            [pooled[metric_key]['quiet'], pooled[metric_key]['moving']],
        )
        metric_ylims[metric_key] = _global_ylim(
            [cell_means[metric_key]['quiet'], cell_means[metric_key]['moving']],
        )

    for ax_idx, (metric_key, xlabel) in enumerate(metric_specs):
        vals_quiet = np.asarray(pooled[metric_key]['quiet'], dtype=float)
        vals_moving = np.asarray(pooled[metric_key]['moving'], dtype=float)
        p_val_hist, _ = _unpaired_test_bursts(vals_quiet, vals_moving)
        _plot_overlapping_hist_panel(
            axes[0, ax_idx],
            vals_in=vals_moving,
            vals_out=vals_quiet,
            axis_config=axis_configs[metric_key],
            xlabel=xlabel,
            ylabel='% of bursts' if ax_idx == 0 else '',
            sig_label=_sig_label(p_val_hist),
            show_legend=(ax_idx == 0),
            legend_loc='upper right',
            first_label='Quiet',
            second_label='Moving',
            first_color='#563C25',
            second_color='#F9E800',
            background_color=None,
            plot_mode=plot_mode,
            fill_alpha=0.18,
        )

        mean_quiet = np.asarray(cell_means[metric_key]['quiet'], dtype=float)
        mean_moving = np.asarray(cell_means[metric_key]['moving'], dtype=float)
        p_val_violin, _, _, _, _ = _paired_test(mean_moving, mean_quiet)
        _plot_paired_violin_panel(
            axes[1, ax_idx],
            vals_first=mean_moving,
            vals_second=mean_quiet,
            ylabel=xlabel,
            sig_label=_sig_label(p_val_violin),
            first_label='Loco.',
            second_label='Quiet',
            first_color='#F9E800',
            second_color='#563C25',
            background_color=None,
            ylim=metric_ylims[metric_key],
        )

        _plot_stacked_outcome_bar(
            axes[2, ax_idx],
            outcome_counts[metric_key],
            total_count=eligible_cells,
            ylabel='Cells' if ax_idx == 0 else '',
            show_legend=False,
        )

    if axes.size > 0:
        axes[0, 0].text(
            0.0,
            1.08,
            f'Eligible cells: {eligible_cells}',
            ha='left',
            va='bottom',
            fontsize=5,
            fontname='Arial',
            transform=axes[0, 0].transAxes,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if axes.size > 0:
        top_left = axes[0, 0].get_position()
        top_right = axes[0, -1].get_position()
        bottom_left = axes[1, 0].get_position()
        bottom_right = axes[1, -1].get_position()
        summary_left = axes[2, 0].get_position()
        summary_right = axes[2, -1].get_position()
        fig.text(
            0.5 * (top_left.x0 + top_right.x1),
            top_left.y1 + 0.008,
            'All CBs pooled',
            ha='center',
            va='bottom',
            fontsize=5,
            fontname='Arial',
        )
        fig.text(
            0.5 * (bottom_left.x0 + bottom_right.x1),
            bottom_left.y1 + 0.008,
            f'Per-cell mean (paired, n={eligible_cells})',
            ha='center',
            va='bottom',
            fontsize=5,
            fontname='Arial',
        )
        fig.text(
            0.5 * (summary_left.x0 + summary_right.x1),
            summary_left.y1 + 0.008,
            f'Per-cell burst test outcome (n={eligible_cells})',
            ha='center',
            va='bottom',
            fontsize=5,
            fontname='Arial',
        )
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


def _plot_inout_panel(
    df_cs_plc,
    df_non_cs_plc,
    prefix: str,
    ylabel: str,
    title: str,
    save_path: str,
    fig_width: float = 1.4,
    fig_height: float = 1.5,
):
    in_color = 'magenta'
    out_color = 'gray'

    df_csplus_paired = df_cs_plc[[f'{prefix}_loco_in', f'{prefix}_loco_out']].dropna()
    csplus_in_paired = df_csplus_paired[f'{prefix}_loco_in'].values
    csplus_out_paired = df_csplus_paired[f'{prefix}_loco_out'].values

    df_csminus_paired = df_non_cs_plc[[f'{prefix}_loco_in', f'{prefix}_loco_out']].dropna()
    csminus_in_paired = df_csminus_paired[f'{prefix}_loco_in'].values
    csminus_out_paired = df_csminus_paired[f'{prefix}_loco_out'].values

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
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


def plot_theta_inout_loco_csplus_vs_csminus(
    df_cs_plc: pd.DataFrame,
    df_non_cs_plc: pd.DataFrame,
    save_path: str,
    fig_width: float = 1.4,
    fig_height: float = 1.5,
):
    return _plot_inout_panel(
        df_cs_plc,
        df_non_cs_plc,
        prefix='theta',
        ylabel='Theta amp',
        title='Locomotion',
        save_path=save_path,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def plot_slow_vm_inout_loco_csplus_vs_csminus(
    df_cs_plc: pd.DataFrame,
    df_non_cs_plc: pd.DataFrame,
    save_path: str,
    fig_width: float = 1.4,
    fig_height: float = 1.5,
):
    return _plot_inout_panel(
        df_cs_plc,
        df_non_cs_plc,
        prefix='slow',
        ylabel='Slow Vm',
        title='Locomotion',
        save_path=save_path,
        fig_width=fig_width,
        fig_height=fig_height,
    )
