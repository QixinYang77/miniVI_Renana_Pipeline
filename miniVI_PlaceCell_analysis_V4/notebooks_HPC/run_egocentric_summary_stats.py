#!/usr/bin/env python3
"""
Step 5b: Summarize egocentric tuning statistics across all categories.

Extends the original CSplus-vs-CSminus comparison to include all-nonPLC.
Generates:
  1. Bar chart of pass counts (pass_95, pass_99, pass_100) per category
  2. Box + scatter plot of real_mrl per category with pairwise Mann-Whitney U tests
  3. Prints summary tables to stdout

Submitted by HPC_egocentric_job_submission.ipynb via sbatch.
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

# --------------- path surgery ---------------
HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_top_repo)]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Default color palette for categories
DEFAULT_COLORS = {
    'CSplus':     '#EE9B00',
    'CSminus':    '#026C80',
    'all-nonPLC': '#7B2D8E',
    'nonPLC':     '#7B2D8E',
}


def main():
    parser = argparse.ArgumentParser(description='Summarize egocentric tuning statistics')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Direction/spike-specific output dir with egocentric_tuning_summary.csv')
    parser.add_argument('--categories', nargs='+', default=['CSplus', 'CSminus', 'all-nonPLC'],
                        help='Categories to include in the summary')
    parser.add_argument('--save-formats', nargs='+', default=['svg', 'png'])
    parser.add_argument('--fig-width', type=float, default=4.5)
    parser.add_argument('--fig-height', type=float, default=2.5)
    args = parser.parse_args()

    import pandas as pd
    from scipy.stats import mannwhitneyu

    output_dir = Path(args.output_dir)
    summary_csv = output_dir / 'egocentric_tuning_summary.csv'

    if not summary_csv.exists():
        print(f'[ERROR] Summary CSV not found: {summary_csv}')
        sys.exit(1)

    df = pd.read_csv(summary_csv)
    print(f'Loaded {len(df)} rows from {summary_csv}')

    cats = list(args.categories)
    df = df[df['category'].astype(str).isin(cats)].copy()
    if len(df) == 0:
        print(f'[ERROR] No rows found for categories: {cats}')
        sys.exit(1)

    # Coerce pass columns to bool
    for col in ('pass_95', 'pass_99', 'pass_100'):
        if col in df.columns:
            df[col] = df[col].apply(_coerce_bool)
        else:
            df[col] = False

    save_dir = output_dir / 'summary_stats'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass counts table ----
    counts = (
        df.groupby('category')[['pass_95', 'pass_99', 'pass_100']]
        .sum().astype(int).reindex(cats).fillna(0).astype(int)
    )
    totals = df.groupby('category').size().reindex(cats).fillna(0).astype(int).rename('n_cells')
    counts['n_cells'] = totals

    print('\n=== Pass Counts ===')
    print(counts.to_string())

    # ---- MRL statistics table ----
    mrl_df = df[['category', 'real_mrl']].copy()
    mrl_df['real_mrl'] = pd.to_numeric(mrl_df['real_mrl'], errors='coerce')
    mrl_df = mrl_df[np.isfinite(mrl_df['real_mrl'])]

    if len(mrl_df) > 0:
        mrl_stats = (
            mrl_df.groupby('category')['real_mrl']
            .agg(['count', 'mean', 'median', 'std'])
            .reindex(cats)
        )
        print('\n=== MRL Statistics ===')
        print(mrl_stats.to_string())

        # Pairwise Mann-Whitney U tests
        print('\n=== Pairwise Mann-Whitney U Tests ===')
        pairwise_results = []
        for cat_a, cat_b in combinations(cats, 2):
            a = mrl_df.loc[mrl_df['category'] == cat_a, 'real_mrl'].to_numpy(dtype=float)
            b = mrl_df.loc[mrl_df['category'] == cat_b, 'real_mrl'].to_numpy(dtype=float)
            if a.size > 0 and b.size > 0:
                u, p = mannwhitneyu(a, b, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                print(f'  {cat_a} vs {cat_b}: U={u:.1f}, p={p:.4g} {sig}')
                pairwise_results.append({
                    'cat_a': cat_a, 'cat_b': cat_b,
                    'U': u, 'p': p, 'sig': sig,
                })
            else:
                print(f'  {cat_a} vs {cat_b}: insufficient data')

        # Save pairwise results
        pd.DataFrame(pairwise_results).to_csv(
            save_dir / 'pairwise_mannwhitney.csv', index=False
        )

    # ---- Plotting ----
    colors = {cat: DEFAULT_COLORS.get(cat, '#999999') for cat in cats}

    pooled_style = {
        'svg.fonttype': 'none',
        'font.family': 'Arial',
        'font.size': 6.0,
        'axes.labelsize': 6.0,
        'axes.titlesize': 6.0,
        'xtick.labelsize': 5.0,
        'ytick.labelsize': 5.0,
        'xtick.major.size': 1.75,
        'ytick.major.size': 1.75,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 5.0,
        'axes.linewidth': 0.5,
    }

    with plt.rc_context(pooled_style):
        # ---- Figure 1: Pass counts bar chart ----
        n_cats = len(cats)
        fig1, ax1 = plt.subplots(figsize=(args.fig_width, args.fig_height))
        x = np.arange(3, dtype=float)  # 3 pass thresholds
        width = 0.8 / n_cats
        for i, cat in enumerate(cats):
            offset = (i - (n_cats - 1) / 2.0) * width
            vals = counts.loc[cat, ['pass_95', 'pass_99', 'pass_100']].to_numpy(dtype=float)
            ax1.bar(
                x + offset, vals, width * 0.9,
                label=f'{cat} (n={int(totals.get(cat, 0))})',
                color=colors[cat],
            )
        ax1.set_xticks(x)
        ax1.set_xticklabels(['pass_95', 'pass_99', 'pass_100'])
        ax1.set_ylabel('Cell count')
        ax1.set_title('Egocentric pass counts by category')
        ax1.legend(frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()

        for fmt in args.save_formats:
            out_path = save_dir / f'egocentric_pass_counts_all_categories.{fmt}'
            fig1.savefig(out_path, bbox_inches='tight')
            print(f'Saved: {out_path}')
        plt.close(fig1)

        # ---- Figure 2: MRL box + scatter ----
        if len(mrl_df) > 0:
            fig2, ax2 = plt.subplots(figsize=(args.fig_width, args.fig_height))
            data_plot = [
                mrl_df.loc[mrl_df['category'] == cat, 'real_mrl'].to_numpy(dtype=float)
                for cat in cats
            ]
            bp = ax2.boxplot(
                data_plot,
                labels=cats,
                showfliers=False,
                widths=0.6,
                patch_artist=True,
            )
            for patch, cat in zip(bp.get('boxes', []), cats):
                patch.set_facecolor(colors[cat])
                patch.set_alpha(0.35)

            rng = np.random.default_rng(42)
            for i, (vals, cat) in enumerate(zip(data_plot, cats), start=1):
                if vals.size <= 0:
                    continue
                jitter = rng.uniform(-0.1, 0.1, size=vals.size)
                ax2.scatter(
                    np.full(vals.size, i) + jitter,
                    vals, s=12, alpha=0.8, linewidths=0,
                    color=colors[cat],
                )

            ax2.set_ylabel('real_mrl')
            ax2.set_title('Mean Resultant Length by category')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # Add significance annotations
            if pairwise_results:
                y_max = max(v.max() for v in data_plot if v.size > 0)
                y_step = y_max * 0.08
                for j, res in enumerate(pairwise_results):
                    if res['p'] >= 0.05:
                        continue
                    i_a = cats.index(res['cat_a']) + 1
                    i_b = cats.index(res['cat_b']) + 1
                    y_bar = y_max + y_step * (j + 1)
                    ax2.plot([i_a, i_a, i_b, i_b], [y_bar - y_step * 0.2, y_bar, y_bar, y_bar - y_step * 0.2],
                             lw=0.6, color='k')
                    ax2.text((i_a + i_b) / 2, y_bar, res['sig'],
                             ha='center', va='bottom', fontsize=5)

            plt.tight_layout()
            for fmt in args.save_formats:
                out_path = save_dir / f'egocentric_real_mrl_all_categories.{fmt}'
                fig2.savefig(out_path, bbox_inches='tight')
                print(f'Saved: {out_path}')
            plt.close(fig2)

    # Save summary tables as CSVs
    counts.to_csv(save_dir / 'pass_counts.csv')
    if len(mrl_df) > 0:
        mrl_stats.to_csv(save_dir / 'mrl_stats.csv')

    print(f'\nAll stats saved to: {save_dir}')


def _coerce_bool(v):
    """Coerce various representations to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes'):
        return True
    return False


if __name__ == '__main__':
    main()
