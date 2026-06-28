# Unified CKII Pipeline Notebook Outline

Source notebook: `Unified_CKII_Pipeline.ipynb`

This outline follows the notebook source. It does not rerun the notebook or validate the generated figures. The notebook currently has 135 cells: 58 markdown cells and 77 code cells.

## High-Level Purpose

The notebook is a unified entry point for CKII place-cell analysis. It:

1. Sets up the analysis environment, plotting style, animals, and shared parameters.
2. Validates or rebuilds per-animal cached analysis artifacts.
3. Classifies cells into CS+ place cells, CS- place cells, and non-place cells.
4. Generates pooled CKII summary figures and spatial heatmaps.
5. Runs trial-by-trial place-field analyses in time, distance-normalized, and DRZ-normalized coordinate systems.
6. Compares activity across session segments and movement directions.
7. Runs directionality, egocentric tuning, GLM, and LN model analyses.

## Notebook Structure

### 1. Setup and Imports

The first code cell enables autoreload, sets `force_recompute = True`, adds the parent project folder to `sys.path`, and applies a compact Arial matplotlib style used by downstream figures.

It imports pipeline functions from four main utility modules:

- `utils.placecell_pipeline`: cache management, trial extraction, PF-centered analyses, distance/DRZ analyses, directionality, egocentric tuning, GLM, and LN model functions.
- `utils.spatial_heatmaps`: cell classification and heatmap rendering helpers.
- `utils.pooled_figures_core`: pooled statistics and figure-generation helpers.
- `utils.session_compare_heatmaps`: session-split heatmap and classification comparison helpers.

### 2. Parameter Blocks

The parameter cell reloads the utility modules so the notebook gets current dataclass and helper definitions, then constructs the shared `PipelineConfig`.

Main configured inputs:

- `project_root`, `data_root`, `figures_root`, and `notebooks_root`.
- Six CKII animals:
  - `CKII_pAce21_PR_20250806`
  - `CKII_pAce38_PX_20251126`
  - `CKII_pAce45_PX_20260118`
  - `CKII_pAce47_PX_20260128`
  - `CKII_pAce46_PR_20260222`
  - `CKII_pAce50_PRL_20260317`

Main parameter groups:

- `AnalysisParams`: speed thresholds, behavior speed outlier cleaning, event duration and merge settings, SNR threshold, good-recording duration threshold, theta band, and slow Vm cutoff.
- `PlaceCellParams`: spatial bin size, place-field threshold, component peak rules, peak firing-rate threshold, field size gates, distance-defined firing-traversal gates, smoothing, shuffle count, and sparse-row trimming.
- `PFTraversalParams`: PF centering mode, PF component selection, traversal duration and distance gates, session indices, PF-centered time window, minimum traversals, firing-rate binning/smoothing, baseline subtraction, PF-distance gate, and plateau duration threshold.
- `PooledParams`: CS+ classification mode/thresholds, PSD controls, simple-event windows, PSD normalization, and PSD frequency bounds.
- `CachePolicy`: controlled by `force_recompute`, with validation-only and executed-notebook saving disabled.

### 3. Cache Validation/Rebuild

The notebook calls `ensure_cache_for_all_animals(config, force=config.cache.force_recompute)` and prints each animal's cache action, rebuild reasons, missing artifacts, and a summary from `summarize_statuses`.

Because `force_recompute` is set to `True`, this cell is intended to rebuild cached results rather than only reuse them.

### 4. Pooled Figures

This section is intended to run individual figure cells after the setup, parameter, cache, and spatial classification cells have been run.

#### SNR Frame-Removal Distribution

Uses `summarize_snr_removed_frames` to report frame-removal summaries, optionally across all valid cells rather than only place cells.

#### Spatial Heatmap Preparation

Builds `figure_save_folder = figures/CKII_pooled`, then:

- Calls `classify_spatial_cells` to split cells into `plcs_csplus`, `plcs_csminus`, and `non_plcs`.
- Computes shared theta and slow Vm heatmap limits with `compute_global_theta_slow_vlims`.
- Sets fixed display limits for selected heatmaps.
- Defines plateau heatmap controls used by subsequent spatial plots.

#### Cell-Type and State-Dependent Summaries

Generates:

- Cell-type pie chart with `plot_celltype_distribution_pie`.
- State-dependent firing-rate and plateau-count summary with `plot_state_dependent_firing_rates_with_plateau_count`.
- Animal movement boxplots with `plot_animal_movement_summary_boxplots`.
- First-vs-last 10 minute movement comparison with `plot_animal_movement_first_last_window_comparison`.

#### Spatial Heatmaps

Generates:

- Combined CS+ and CS- place-cell spatial heatmaps with `plot_selected_cells_figure`.
- Chunked non-place-cell heatmaps with `render_spatial_heatmap_chunks`.
- A manually selected, transposed CS+/CS- heatmap panel with `plot_selected_cells_transposed_figure`.

The selected-cell panel uses two notebook-local helper functions:

- `_animal_short(cell)`: extracts the short animal label from `cell["animal_id"]`.
- `_select_spatial_cells(cells, specs, group_label)`: selects cells by `(animal_short, one_based_cell_number)` and raises a `ValueError` if any requested cell is missing.

#### Pooled Stats and Burst/Firing Figures

Calls `prepare_pooled_stats_tables` to create pooled cell tables, including `df_cs_plc` and `df_non_cs_plc`.

Then it generates multiple pooled summary figures:

- Moving-epoch CS and complex-burst metrics for all cells/place cells.
- Four-panel CS+ vs CS- summary.
- CS+ correlations with overlaid SS metrics.
- CS+ complex-burst metrics and distributions during running.
- Quiet-vs-moving complex-burst metric distributions for all cells, CS+ only, and non-PLCs only.
- CS+ complex-burst rate correlations inside/outside PF.
- Theta and slow Vm inside/outside PF during locomotion for CS+ vs CS- cells.

### 5. Time-Based Trial-by-Trial Analysis

This section works in seconds around PF traversals.

#### Per-PF Traversal Plots

`generate_trial_by_trial_plc_plots` creates per-cell/per-PF trial-by-trial plots for CS+ and CS- PLCs. It uses no PF reliability dilation in this notebook (`trial_plot_dilation_bins = 0`), merges nearby traversals using `traversal_params.traversal_merge_gap_s`, saves a summary CSV, and reports counters and output folders.

#### PF-Centered Dataset and Heatmaps

`build_pf_centered_component_dataset` creates a dataset for primary and secondary PF components (`pf_ranks=(1, 2)`) for CS+ and CS- cells.

`generate_pf_centered_component_heatmaps` renders individual PF-centered heatmaps with merged SS/CS and plateau rows.

#### PF-Centered Average Plots

`plot_pf_centered_category_primary_secondary_from_dataset` generates average primary/secondary PF plots separately for:

- CS+ PLCs.
- CS- PLCs.

Both use smoothed traces, trial overlays, fixed x-limits around PF center, and a minimum traversal gate.

#### PF-Centered Selectivity Summary

`compute_pf_centered_selectivity_summary` computes a five-panel selectivity summary at the PF center (`timepoint_sec = 0.0`) with minimum traversal gates and an AUC window from -3 to +3 seconds.

`plot_pf_centered_selectivity_summary` saves the summary figures and CSVs.

#### Session Delta Analyses

The notebook computes and plots:

- CS+ PF1 vs PF2 session deltas with `compute_csplus_two_pf_session_delta` and `plot_csplus_two_pf_session_delta`.
- CS+ vs CS- primary-PF session deltas with `compute_csplus_vs_csminus_primary_pf_session_delta` and `plot_csplus_vs_csminus_primary_pf_session_delta`.

Both write per-cell and stats CSVs to the pooled figure folder.

### 6. Session Compare: Heatmaps

This section compares session segments using `SessionCompareParams`.

Configured behavior includes:

- S1/S2 panel mode.
- Cache version `v2`.
- Rebuilding the session-compare cache.
- Up to 10 cells per figure.
- Missing S2 panels shown as NA panels.
- Optional occupancy filtering disabled.
- S1/S2 peak-rate filter enabled at 0.5 Hz.
- Weighted Pearson heatmap similarity.
- Spike-shape panels enabled.
- Plateau rows enabled.
- Time-window split mode with 10 minute windows.
- Alignment to distance-normalized exports.

Execution flow:

1. `build_session_compare_analysis` builds or loads payloads, assembles groups, builds split 4-panel tables, and computes classification summaries.
2. `render_session_compare_heatmaps` writes CS+, CS-, and non-PLC heatmaps.
3. `plot_combined_cs_plus_minus_4panels_2sessions` renders session-split pooled metrics.
4. `plot_combined_cs_plus_minus_session_pair_metrics_14cols` renders a 14-column session-pair metrics figure.

### 7. Distance-Normalized Trial-by-Trial Analysis

This section centers activity by distance from PF peak or component rather than by time.

#### Distance Dataset Build

`PFDistanceCenteredParams` defines:

- 15 cm analysis window.
- 10 cm detection window in the dataclass, overridden to 8 cm for dataset generation.
- 1.5 cm bins.
- Movement-required traversal extraction.
- Bad-frame exclusion.
- Primary and secondary PF analysis.
- Plateau inclusion settings.

`generate_distance_defined_trials_and_dataset` creates both trial plots and a unified `distance_centered_dataset` for `CSplus`, `CSminus`, and `non-PLC` categories, with PF ranks 1 and 2. The dataset generation uses `distance_mode='euclidean_to_peak'`, optional resting plateaus, and alignment to the session-compare cache.

A diagnostic cell prints per-cell trial counts for PF1 and PF2.

#### Compare Sessions in Distance Space

The session-comparison branch uses 10 minute time windows and renders:

- Two-session distance-centered component heatmaps with `generate_pf_distance_centered_component_heatmaps_2sessions`.
- Trial-count diagnostics from exported average CSVs.
- CS+ and CS- primary/secondary average plots with `plot_pf_distance_centered_category_primary_secondary_from_dataset_2sessions`.

Statistics include:

- Session stability metrics with `compute_pf_distance_centered_session_stability` and `plot_pf_distance_centered_session_stability`.
- CS+ PF1/PF2 delta and width metrics with `compute_csplus_pf_distance_centered_session_delta_width` and `plot_csplus_pf_distance_centered_session_delta_width`.
- Absolute normalized FR change summaries for SS vs CS with `plot_pf_distance_centered_abs_norm_fr_change_ss_cs`.

#### Compare Directions in Distance Space

The direction-comparison branch compares CW vs CCW traversal directions.

It generates:

- Direction-split distance-centered heatmaps with `generate_pf_distance_centered_component_heatmaps_2directions`.
- Direction-split average plots with `plot_pf_distance_centered_category_primary_secondary_from_dataset_2directions`.
- Preferred-vs-nonpreferred direction statistics with `compute_pf_distance_centered_pref_nonpref_stats` and `plot_pf_distance_centered_pref_nonpref_stats`.

### 8. DRZ-Based Trial-by-Trial Analysis

This section repeats the distance-style workflow using directional rate zone (DRZ) coordinates.

#### DRZ Dataset Build

The notebook reloads `utils.placecell_core` and `utils.placecell_pipeline`, binds DRZ helper functions, and constructs `PFDRZParams`.

Key DRZ settings:

- Trial clip mode: `pf_entry_exit`.
- DRZ window: 1.0.
- DRZ bin: 0.1.
- Distance mode: `euclidean_to_peak`.
- Minimum traversals: 10 overall and 5 per type.
- Movement and bad-frame filters enabled.
- PF dilation of 4 bins for DRZ trial plots/dataset.
- Categories: `CSplus`, `CSminus`, and `non-PLC`.

`generate_drz_trials_and_dataset` creates `drz_dataset` and optional traversal SVGs.

#### Compare Sessions in DRZ Space

The session branch:

- Uses 10 minute session windows.
- Requires at least 4 trials per plotted average column.
- Renders two-session DRZ heatmaps with `generate_pf_drz_component_heatmaps_2sessions`.
- Renders CS+ and CS- average plots with `plot_pf_drz_category_primary_secondary_from_dataset_2sessions`.
- Computes session stability with `compute_pf_drz_session_stability`.
- Plots stability and SS/CS absolute normalized FR changes.

#### Compare Directions in DRZ Space

The direction branch:

- Uses `drz_direction_avg_mode = 'S1'`.
- Renders CW/CCW DRZ heatmaps with `generate_pf_drz_component_heatmaps_2directions`.
- Renders direction-split average plots with `plot_pf_drz_category_primary_secondary_from_dataset_2directions`.
- Computes preferred-vs-nonpreferred statistics with `compute_pf_drz_pref_nonpref_stats`.
- Plots directionality summaries with `plot_pf_drz_pref_nonpref_stats`.

### 9. Bin-by-Bin Directionality

This section generates vector and polar directionality plots for `CSplus`, `CSminus`, and `non-PLC` categories.

Main configuration:

- Direction mode: `head`.
- First 10 minutes only.
- Arena size: 35.5 x 20.0 cm.
- Vector bin size: 4 cm.
- Head-direction bin size: 15 degrees.
- Movement filtering uses the global analysis speed and event thresholds.
- Bin inclusion requires enough spikes, visited angles, angle separation, and occupancy.
- Null mode is `multinomial`; time-shift settings are defined but only relevant when using the time-shift null.

`generate_bin_by_bin_directionality_plots` writes category and animal-level directionality outputs and prints attempted/saved/skipped counts.

### 10. Egocentric Tuning

This section runs Carpenter-style egocentric tuning analyses for CS+ and CS- cells.

#### Egocentric Parameters

`EgocentricTuningParams` is configured with:

- Categories: `CSplus`, `CSminus`.
- First 10 minutes.
- Head-direction mode.
- 0.1 second time bins.
- Arena size: 35.5 x 20.0 cm.
- Local spatial bins of 5 cm and coarse spatial bins of 2 cm.
- 10 angle bins.
- Speed range from 3 to 60 cm/s.
- Occupancy and valid-bin gates.
- 100 optimizer restarts using Nelder-Mead.
- 1000 surrogates, parallelized across available CPUs.
- Null distributions saved.

#### Diagnostics, Fits, and Plots

The notebook:

1. Runs `generate_egocentric_valid_spatial_bin_diagnostic_plots` before fitting.
2. Runs pooled egocentric tuning with `run_pooled_egocentric_tuning_analysis`.
3. Builds `EgocentricSummaryPlotParams` and renders per-cell summary plots with `generate_egocentric_per_cell_summary_plots`.
4. Summarizes CS+ vs CS- egocentric statistics with `summarize_egocentric_csplus_csminus_stats`, including pass counts, MRL statistics, Mann-Whitney U testing, and saved summary figures.

### 11. GLM

The GLM section models place-cell firing responses for CS+ and CS- cells.

Configuration:

- Session mode: `session1`.
- Direction predictor: `head`.
- Response kinds: all spikes, simple spikes, and complex spikes.
- Time bin: 0.3 seconds.
- Temporal smoothing: 3 bins.
- Full valid bins required.
- Minimum total spikes: 20.
- Ridge-style regularization enabled with alpha 0.1.
- Spatial RBF basis sigma: 6 cm.

`run_pooled_placecell_glm_analysis` fits models, reports attempted/successful/skipped models by category and response type, writes saved outputs, and displays group summaries.

### 12. LN Model

The LN section runs a Hardcastle-style linear-nonlinear model using position, head direction, and speed predictors. The notebook comments that theta is omitted for this dataset.

Configuration:

- Session mode: `session1`.
- Direction predictor: `head`.
- Response kinds: all spikes, simple spikes, and complex spikes.
- Time bin: 0.4 seconds.
- Full valid bins required.
- Position bins: 18 x 10.
- Direction bins: 12.
- Speed bins: 10, up to 30 cm/s.
- Roughness penalties for position, direction, and speed.
- 10-fold cross-validation with 3 section repeats.
- Forward-selection alpha 0.05 and baseline alpha 0.2.
- Temporal and spatial smoothing settings.
- Minimum total spikes: 20.

`run_pooled_placecell_ln_model_analysis` fits models, reports attempted/successful/skipped models, writes saved outputs, and displays category-level LN summaries.

## Execution Dependencies

The notebook is not a set of fully independent cells. Important dependencies are:

- Setup/imports and parameter blocks must run before everything else.
- Cache validation/rebuild should run before downstream figures if cached artifacts are stale.
- Spatial classification must run before pooled spatial heatmaps, pooled stats, trial analyses, directionality, egocentric tuning, GLM, and LN modeling.
- `pooled_stats` is required before pooled statistics figures.
- The time-based PF-centered dataset is required before PF-centered heatmaps, average plots, selectivity summaries, and time-based session delta analyses.
- The distance-centered dataset is required before all distance-based session and direction comparisons.
- The two-session distance heatmap/average export step should run before distance-based stability and saved-export average plots.
- The DRZ dataset is required before all DRZ session and direction comparisons.
- Egocentric fitting should run before egocentric per-cell summary plots and summary statistics.
- LN model parameters must be created before running the LN model summary cell.

## Primary Outputs

Most outputs are written under `figures/CKII_pooled`. The notebook creates:

- SVG pooled summary figures.
- Spatial heatmap SVGs.
- Per-cell trial-by-trial traversal plots.
- PF-centered, distance-centered, and DRZ-centered heatmaps and average plots.
- Session-comparison and direction-comparison figures.
- Directionality and egocentric tuning diagnostics and summaries.
- GLM and LN model output folders.
- Several CSV exports for per-cell metrics, stats summaries, trial counts, and plot manifests.

## Source-Level Notes

- The notebook defines only two local helper functions: `_animal_short` and `_select_spatial_cells`.
- Most substantive behavior is delegated to the utility modules in `miniVI_PlaceCell_analysis_V4/utils`.
- The final code cell is empty.
- Some saved notebook outputs contain prior execution errors, but this outline is based on the notebook source rather than re-executing the pipeline.
