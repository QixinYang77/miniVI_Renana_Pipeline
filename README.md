# miniVI Renana Pipeline

Pipeline for voltage imaging data processing and place cell analysis.

---

## Part 1: Data Processing Pipeline

Converts raw `.raw` voltage imaging files into either VolPy or automatic ALI
demixing results.

**Relevant folders:** `notebooks/`, `utils/`, `utils_concat/`

### Quickstart
- **Full pipeline (recommended):** `notebooks/Batch_concat_pipeline.ipynb`
- **Full-session automatic ALI:** `notebooks/Batch_concat_pipeline_ALI.ipynb`
- **Step-by-step walkthrough:** `notebooks/Batch_pipeline.ipynb`

The ALI notebook keeps the established concatenation, patterned-illumination
preprocessing, and rigid NoRMCorre stages, but removes manual cell ROI drawing.
It performs all-pixel motion-artifact projection and then runs positive-going
ALI over the complete concatenated movie. SLM masks constrain only the search
region. The cluster entry point is `utils/run_ali_concat.py`; its cache-aware
helpers are `utils/ali_motion_projection.py`, `utils/ali_pipeline_utils.py`,
and `utils/ali_dashboard_utils.py`.

The upstream Python ALI functions are vendored under `third_party/pyALI` at
commit `a348549f1601686c9c4e3bd5fbb7e832c2958502` with their GPL-3.0 license.
The existing CaImAn environment must also provide NumPy, SciPy, scikit-image,
tifffile, and Matplotlib.

---

## Part 2: Data Analysis with Behavior

Place cell analysis pipeline located in `miniVI_PlaceCell_analysis_V4/`.

### Step 1 — Preprocessing

Run the notebooks in order inside `miniVI_PlaceCell_analysis_V4/notebooks_preprocessing/`:

1. `step1_CS_detection.ipynb` — Complex spike detection
2. `step2_preprocess.ipynb` — Preprocessing
3. `step3_movedata_to_local.ipynb` — Move data to local machine

### Step 2 — Analysis

Main analysis pipeline:

**`miniVI_PlaceCell_analysis_V4/notebooks_PCs/Unified_CKII_Pipeline.ipynb`**

Associated functions are stored in `miniVI_PlaceCell_analysis_V4/utils/`, including:
- `placecell_pipeline.py` — main pipeline functions
- `placecell_core.py` — core place cell computations
- `spatial_analysis_func.py` — spatial analysis utilities
- `preprocess_neural.py`, `preprocess_behavior.py`, `preprocess_thorsync.py` — preprocessing helpers
- `pooled_figures_core.py` — pooled figure generation
