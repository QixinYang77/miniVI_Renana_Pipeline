# miniVI Renana Pipeline

Pipeline for voltage imaging data processing and place cell analysis.

---

## Part 1: Data Processing Pipeline

Converts raw `.raw` voltage imaging files into a VolPy demixed pickle file.

**Relevant folders:** `notebooks/`, `utils/`, `utils_concat/`

### Quickstart
- **Full pipeline (recommended):** `notebooks/Batch_concat_pipeline.ipynb`
- **Step-by-step walkthrough:** `notebooks/Batch_pipeline.ipynb`

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
