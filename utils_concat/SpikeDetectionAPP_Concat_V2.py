# app.py — Demix viewer (Overview, Per-cell, Spike detection)
# Fixes:
# 1) Uses cache_resource for loading large pickles (no repeated serialization)
# 2) Per-trace normalization only (no huge T×N normalized matrices)
# 3) "Tabs" via radio so Overview isn't recomputed when working in Spike tab
# 4) Good-cell selection wrapped in a form so clicks on multiselect don't immediately apply
# 5) Bad-epoch editing wrapped in a form so epochs only change when "Add Epoch" is clicked
#    (no explicit st.rerun anywhere)

import os, io, math, pickle, sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Optional deps ----
try:
    from skimage.measure import find_contours
    _HAS_SKI = True
except Exception:
    _HAS_SKI = False

try:
    from scipy.signal import butter, filtfilt, find_peaks
    _HAS_SCIpY = True
except Exception:
    _HAS_SCIpY = False

st.set_page_config(page_title="Demix Viewer", layout="wide")

# ---------------- Caching Functions ----------------
@st.cache_resource(show_spinner="Loading data...")
def load_data_from_paths(paths):
    """Load multiple pickle files from disk (cached as a resource)."""
    data = []
    for p in paths:
        with open(p, "rb") as f:
            data.append(pickle.load(f))
    return data

@st.cache_resource(show_spinner="Loading data...")
def load_data_single(path=None, file_bytes=None):
    """Load a single pickle from path or bytes (cached as a resource)."""
    if file_bytes is not None:
        return pickle.load(io.BytesIO(file_bytes))
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ---------------- Utilities (NaN-safe) ----------------
def zscore(x):
    x = np.asarray(x)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)

def norm01(x):
    x = np.asarray(x)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

def roi_contours(mask):
    if _HAS_SKI:
        return find_contours(mask.astype(float), 0.5)
    r = mask.astype(bool)
    up = np.zeros_like(r);  up[1:]   = r[:-1]
    dn = np.zeros_like(r);  dn[:-1]  = r[1:]
    lf = np.zeros_like(r);  lf[:,1:] = r[:,:-1]
    rt = np.zeros_like(r);  rt[:,:-1]= r[:,1:]
    edge = r & ~(up & dn & lf & rt)
    ys, xs = np.where(edge)
    return [np.vstack([ys, xs]).T] if len(xs) else []

def roi_centroid(mask):
    ys, xs = np.nonzero(mask)
    return (ys.mean(), xs.mean()) if xs.size > 0 else (0.0, 0.0)

def palette_colors(n):
    from plotly.colors import qualitative as qual
    base = getattr(qual, "Alphabet", qual.Plotly)
    return (base * ((n // len(base)) + 1))[:n]

def robust_mad(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med, 1.4826 * mad

def butter_highpass(x, fs, fc, order=3):
    if not _HAS_SCIpY or fc <= 0 or fs <= 0 or fc >= fs/2:
        return x
    b, a = butter(order, fc/(fs/2.0), btype='highpass')
    return filtfilt(b, a, np.nan_to_num(x))

# --- Cached spike detection ---
@st.cache_data(show_spinner=False)
def detect_spikes_segmented(trace, fs, n_segments=1, hp_hz=20.0, mad_k=4.0,
                            min_isi_ms=2.0, segment_edges=None):
    """
    Cached version of spike detection.
    Recalculates only if trace data or parameters change.
    """
    T = trace.size
    hp = butter_highpass(trace, fs, hp_hz)
    if segment_edges is not None:
        seg_edges = np.asarray(segment_edges, dtype=int)
    else:
        seg_edges = np.linspace(0, T, n_segments+1, dtype=int)

    peaks_all = []
    min_distance = int(max(1, round((min_isi_ms/1000.0)*fs)))
    for j in range(len(seg_edges)-1):
        s0, s1 = seg_edges[j], seg_edges[j+1]
        seg = hp[s0:s1]
        med, mad = robust_mad(seg)
        thr = med + mad_k * mad
        if _HAS_SCIpY:
            pks, _ = find_peaks(seg, height=thr, distance=min_distance)
            peaks_all.append(pks + s0)
        else:
            with np.errstate(invalid='ignore'):
                idx = np.where(seg > thr)[0]
            keep = []
            for k in idx:
                if not keep or (k - keep[-1]) >= min_distance:
                    keep.append(k)
            peaks_all.append(np.array(keep, dtype=int) + s0)
    peaks = np.sort(np.concatenate(peaks_all)) if peaks_all else np.array([], dtype=int)
    return peaks, hp, seg_edges

# ---------------- Load logic ----------------
# Accept demix files directly via command line arguments
# Usage: streamlit run SpikeDetectionAPP_Concat.py -- file1.pickle file2.pickle ...

results_list = []
paths_list = []
labels_list = []

# Get demix files from command line arguments
demix_files = sys.argv[1:] if len(sys.argv) > 1 else []

st.sidebar.header("Load results")

if demix_files:
    # Load files provided via command line
    valid_files = [f for f in demix_files if os.path.isfile(f)]
    if valid_files:
        st.sidebar.success(f"Loading {len(valid_files)} file(s) from command line.")
        loaded_data_list = load_data_from_paths(valid_files)
        for i, r in enumerate(loaded_data_list):
            results_list.append(r)
            paths_list.append(valid_files[i])
            labels_list.append(
                os.path.basename(os.path.dirname(valid_files[i])) or os.path.basename(valid_files[i])
            )
    else:
        st.sidebar.warning("No valid files found from command line arguments.")

# Allow additional file upload or manual path entry
st.sidebar.markdown("---")
st.sidebar.subheader("Add additional files")
upload = st.sidebar.file_uploader("Upload pickle file", type=["pickle", "pkl"])
path_text_input = st.sidebar.text_input("…or path to pickle on disk", value="")

if upload:
    f_bytes = upload.getvalue()
    loaded_res = load_data_single(file_bytes=f_bytes)
    if loaded_res:
        results_list.append(loaded_res)
        paths_list.append("")
        labels_list.append("Uploaded file")
elif path_text_input and os.path.isfile(path_text_input):
    loaded_res = load_data_single(path=path_text_input)
    if loaded_res:
        results_list.append(loaded_res)
        paths_list.append(path_text_input)
        labels_list.append(
            os.path.basename(os.path.dirname(path_text_input)) or os.path.basename(path_text_input)
        )

if not results_list:
    st.info("Provide demix file(s) via command line:\n\n`streamlit run SpikeDetectionAPP_Concat.py -- file1.pickle file2.pickle ...`\n\nOr upload/paste a path above.")
    st.stop()

# ---------------- Extract arrays ----------------
required_core = ["mean_img_raw", "mean_img_mc", "weights", "ROIs", "shift_distances", "mc_traces"]
missing_any = []
for si, r in enumerate(results_list):
    for k in required_core:
        if k not in r:
            missing_any.append(f"session {si+1} ({labels_list[si]}): missing '{k}'")
if missing_any:
    st.error("Missing keys:\n" + "\n".join(missing_any))
    st.stop()

base0 = results_list[0]
multi_session = (len(results_list) > 1)
n_sessions = len(results_list)

results = {
    "mean_img_raw": np.asarray(base0["mean_img_raw"]),
    "mean_img_mc": np.asarray(base0["mean_img_mc"]),
    "mean_img_mc_denoised": np.asarray(base0.get("mean_img_mc_denoised", base0["mean_img_mc"])),
    "weights": base0["weights"],
    "ROIs": list(base0["ROIs"]),
    "shift_distances": None,
    "frame_rate": float(base0.get("frame_rate", 500.0)),
}

ROIs = list(results["ROIs"])
N = len(ROIs)
colors = palette_colors(N)

frame_rate_global = float(results.get("frame_rate", 500.0))
frame_rate = st.sidebar.number_input("Frame rate (Hz)", value=float(frame_rate_global), step=1.0)
time_is_seconds = frame_rate and frame_rate > 0

def _get_session_length(r, n_cells):
    base_keys = [
        "weighted_mc_denoised_traces",
        "weighted_mc_traces",
        "mc_denoised_traces",
        "mc_traces",
        "raw_traces",
        "volpy_trace",
        "volpy_sub_trace",
    ]
    for k in base_keys:
        arr = r.get(k, None)
        if arr is None:
            continue
        A = np.asarray(arr)
        if A.ndim != 2:
            continue
        if A.shape[0] == n_cells and A.shape[1] != n_cells:
            return A.shape[1]
        elif A.shape[1] == n_cells:
            return A.shape[0]
        elif A.shape[0] == n_cells and A.shape[1] == n_cells:
            return A.shape[1]
    sd = np.asarray(r["shift_distances"]).ravel()
    if sd.size > 0:
        return sd.size
    return 0

session_lengths = [_get_session_length(r, N) for r in results_list]
if any(L == 0 for L in session_lengths):
    st.error("Time length error.")
    st.stop()
T_total = int(sum(session_lengths))
session_edges_frames = np.concatenate(([0], np.cumsum(session_lengths)))
T = T_total
t = (np.arange(T) / frame_rate) if time_is_seconds else np.arange(T)
time_label = "Time (s)" if time_is_seconds else "Samples"
if time_is_seconds:
    session_boundary_times = [session_edges_frames[i] / frame_rate for i in range(1, n_sessions)]
else:
    session_boundary_times = [float(session_edges_frames[i]) for i in range(1, n_sessions)]

# ---- RECONSTRUCT METADATA (Spikes & Bad Epochs) ----
# 1. Good Cells
saved_good = base0.get("good_cells", None)
if saved_good is not None and len(saved_good) > 0:
    loaded_good = saved_good
else:
    loaded_good = list(range(N))

# 2. Params
saved_params = base0.get("spike_detection_params_per_cell", {})

# 3. Verified Spikes & Bad Epochs (Stitching)
stitched_sv = [[] for _ in range(N)]
stitched_be = []
has_stitched_sv = False

offset_samples = 0
for i_sess, r in enumerate(results_list):
    s_len = session_lengths[i_sess]

    # Spikes
    if "spikes_verified_array" in r:
        has_stitched_sv = True
        for i in range(N):
            local_spikes = r["spikes_verified_array"][i]
            global_spikes = [int(x) + offset_samples for x in local_spikes]
            stitched_sv[i].extend(global_spikes)

    # Bad Epochs
    local_be = r.get("bad_epochs", [])
    if local_be:
        if time_is_seconds:
            time_offset = offset_samples / frame_rate
        else:
            time_offset = offset_samples
        for (e0, e1) in local_be:
            stitched_be.append((float(e0) + time_offset, float(e1) + time_offset))

    offset_samples += s_len

if multi_session:
    sv = stitched_sv if has_stitched_sv else None
else:
    sv = base0.get("spikes_verified_array", None)

if sv is None:
    sv_dict = results.get("spikes_verified", {})
    sv = [[] for _ in range(N)]
    if isinstance(sv_dict, dict):
        for k, v in sv_dict.items():
            try:
                sv[int(k)] = list(map(int, v))
            except:
                pass
else:
    if not isinstance(sv, (list, tuple)) or len(sv) != N:
        sv = [[] for _ in range(N)]
spikes_verified_array = [list(map(int, s)) for s in sv]

trace_keys_all = [
    "raw_traces",
    "mc_traces",
    "mc_denoised_traces",
    "weighted_mc_traces",
    "weighted_mc_denoised_traces",
    "volpy_trace",
    "volpy_sub_trace",
]
sources_raw = {}

def _as_2d_TN(arr, n_cells, T_target):
    if arr is None:
        return np.full((T_target, n_cells), np.nan, dtype=float)
    A = np.asarray(arr)
    if A.ndim != 2:
        return np.full((T_target, n_cells), np.nan, dtype=float)
    if A.shape[1] == n_cells:
        pass
    elif A.shape[0] == n_cells:
        A = A.T
    else:
        return np.full((T_target, n_cells), np.nan, dtype=float)
    if A.shape[0] > T_target:
        A = A[:T_target, :]
    elif A.shape[0] < T_target:
        pad = np.full((T_target - A.shape[0], n_cells), np.nan, dtype=float)
        A = np.concatenate([A, pad], axis=0)
    return A.astype(float)

for k in trace_keys_all:
    if any(r.get(k, None) is not None for r in results_list):
        per_sess = []
        for r, T_s in zip(results_list, session_lengths):
            per_sess.append(_as_2d_TN(r.get(k, None), N, T_s))
        sources_raw[k] = np.concatenate(per_sess, axis=0)

shift_concat_list = []
for r, T_s in zip(results_list, session_lengths):
    sd = np.asarray(r["shift_distances"]).ravel()
    if sd.size > T_s:
        sd = sd[:T_s]
    elif sd.size < T_s:
        pad = np.full(T_s - sd.size, np.nan, dtype=float)
        sd = np.concatenate([sd, pad])
    shift_concat_list.append(sd)
shift_distances = np.concatenate(shift_concat_list)

results["shift_distances"] = shift_distances
for k, A in sources_raw.items():
    results[k] = A

mean_img_raw_list = [np.asarray(r["mean_img_raw"]) for r in results_list]
mean_img_mc_list = [np.asarray(r["mean_img_mc"]) for r in results_list]
mean_img_mc_denoised_list = [
    np.asarray(r.get("mean_img_mc_denoised", r["mean_img_mc"])) for r in results_list
]

mean_img_raw_base = mean_img_raw_list[0]
mean_img_mc_base = mean_img_mc_list[0]
mean_img_mc_denoised_base = mean_img_mc_denoised_list[0]
H, W = mean_img_mc_base.shape

weights_img_list = []
for r in results_list:
    w = np.asarray(r["weights"])
    if w.ndim == 2:
        w_img = w.max(axis=0).reshape(H, W)
    elif w.ndim == 3:
        w_img = w.max(axis=0)
    else:
        st.error("weights error")
        st.stop()
    weights_img_list.append(w_img)
weights_img_base = weights_img_list[0]

# ---- Bad epochs Init ----
def make_bad_mask(t_axis, epochs):
    if not epochs or t_axis.size == 0:
        return np.zeros_like(t_axis, dtype=bool)
    mask = np.zeros_like(t_axis, dtype=bool)
    for t0, t1 in epochs:
        lo, hi = min(float(t0), float(t1)), max(float(t0), float(t1))
        mask |= (t_axis >= lo) & (t_axis <= hi)
    return mask

if "bad_epochs" not in st.session_state:
    if multi_session:
        st.session_state.bad_epochs = stitched_be
    else:
        stored = base0.get("bad_epochs", [])
        parsed = []
        try:
            for e in stored:
                if len(e) >= 2:
                    parsed.append((float(e[0]), float(e[1])))
        except:
            parsed = []
        st.session_state.bad_epochs = parsed

bad_mask_full = make_bad_mask(t, st.session_state.bad_epochs)

if "good_cells" not in st.session_state:
    try:
        st.session_state.good_cells = sorted(
            set(int(i) for i in loaded_good if 0 <= int(i) < N)
        )
    except:
        st.session_state.good_cells = list(range(N))
if "show_good_only" not in st.session_state:
    st.session_state.show_good_only = False

def _apply_multiselect_to_state(labels):
    st.session_state.good_cells = sorted({int(lbl.split()[-1]) - 1 for lbl in labels})

# ---------------- Sidebar controls ----------------
st.sidebar.header("Display controls")
norm_mode = st.sidebar.radio("Normalization", ["0–1", "z-score"], index=0, horizontal=True)
trace_row_height = st.sidebar.number_input("Overview row height (px)", value=50, step=5)
norm_fn = norm01 if norm_mode == "0–1" else zscore

st.sidebar.markdown("---")
st.sidebar.subheader("Filter & Selection")

# Initialize view_mode radio state once
if "view_mode_radio" not in st.session_state:
    st.session_state.view_mode_radio = "Only Good" if st.session_state.show_good_only else "All cells"

good_options = [f"Cell {i+1}" for i in range(N)]
default_good_labels = [f"Cell {i+1}" for i in st.session_state.good_cells]

# Wrap good-cell selection in a form so individual clicks don't apply until submit
with st.sidebar.form("good_cells_form"):
    st.radio(
        "View Mode",
        ["All cells", "Only Good"],
        key="view_mode_radio",
        horizontal=True,
    )

    st.multiselect(
        "Good cells",
        options=good_options,
        default=default_good_labels,
        key="good_multiselect_form",
    )

    apply_btn = st.form_submit_button("Apply selection")
    select_all_btn = st.form_submit_button("Mark all as Good")

# Only when user clicks a button do we update state
if apply_btn:
    _apply_multiselect_to_state(st.session_state.good_multiselect_form)
    st.session_state.show_good_only = (st.session_state.view_mode_radio == "Only Good")

elif select_all_btn:
    st.session_state.good_cells = list(range(N))
    st.session_state.show_good_only = True
    st.session_state.good_multiselect_form = [f"Cell {i+1}" for i in range(N)]
    st.session_state.view_mode_radio = "Only Good"

# Helper: per-trace normalization (no full T×N matrices)
def get_normalized_trace(source_key, cell_idx, bad_mask=None):
    A = sources_raw[source_key]
    y = A[:, cell_idx].astype(float)
    if bad_mask is not None and bad_mask.any():
        y = y.copy()
        y[bad_mask] = np.nan
    return norm_fn(y)

# ---------------- "Tabs" via radio (only active tab code runs) ----------------
tab = st.radio(
    "View",
    ["🔭 Overview", "🧫 Per-cell", "⚡ Spike detection & Good cells"],
    horizontal=True,
)

# -----------------------------------------------------------------------------#
#                               TAB 1: OVERVIEW                                #
# -----------------------------------------------------------------------------#
if tab == "🔭 Overview":
    st.subheader("Trace source (for overview)")
    if not sources_raw:
        st.warning("No trace sources available.")
        src_over_default = None
    else:
        src_over_default = (
            "weighted_mc_denoised_traces"
            if "weighted_mc_denoised_traces" in sources_raw
            else list(sources_raw.keys())[0]
        )
        src_over = st.selectbox(
            "Trace source",
            list(sources_raw.keys()),
            index=list(sources_raw.keys()).index(src_over_default),
        )

    st.markdown("---")

    def image_with_contours_figure(img, ROIs, colors, title, magma=False):
        cs = "Magma" if magma else "gray"
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=img, colorscale=cs, showscale=False))
        idx_draw = (
            st.session_state.good_cells
            if st.session_state.show_good_only
            else range(len(ROIs))
        )
        for i in idx_draw:
            mask = ROIs[i]
            for C in roi_contours(mask):
                fig.add_trace(
                    go.Scatter(
                        x=C[:, 1],
                        y=C[:, 0],
                        mode="lines",
                        line=dict(width=1, color=colors[i]),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            cy, cx = roi_centroid(mask)
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="text",
                    text=[str(i + 1)],
                    textfont=dict(color=colors[i], size=12),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(
            visible=False, autorange="reversed", scaleanchor="x", scaleratio=1
        )
        fig.update_layout(title=title, margin=dict(l=0, r=0, t=32, b=0), height=320)
        return fig

    st.markdown("### Mean images per session")
    for si in range(n_sessions):
        st.markdown(f"**Session {si+1}** — `{labels_list[si]}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.plotly_chart(
            image_with_contours_figure(mean_img_raw_list[si], ROIs, colors, "Mean Image – Raw"),
            use_container_width=True,
            config={"displaylogo": False},
        )
        c2.plotly_chart(
            image_with_contours_figure(mean_img_mc_list[si], ROIs, colors, "Mean Image – MC"),
            use_container_width=True,
            config={"displaylogo": False},
        )
        c3.plotly_chart(
            image_with_contours_figure(
                mean_img_mc_denoised_list[si], ROIs, colors, "MC Denoised"
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        c4.plotly_chart(
            image_with_contours_figure(
                weights_img_list[si], ROIs, colors, "Max Weight Map", magma=True
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.markdown("---")

    if sources_raw and T > 0:
        # Recompute bad_mask_full from current epochs
        bad_mask_full = make_bad_mask(t, st.session_state.bad_epochs)

        idx_all = (
            st.session_state.good_cells
            if (st.session_state.show_good_only and st.session_state.good_cells)
            else list(range(N))
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Pagination")
        default_cells = len(idx_all) if idx_all else 1
        cells_per_page = st.sidebar.number_input(
            "Cells per page",
            min_value=1,
            max_value=max(1, len(idx_all)),
            value=max(1, default_cells),
            step=1,
        )
        total_pages = int(math.ceil(len(idx_all) / cells_per_page)) if idx_all else 1
        page_idx = st.sidebar.number_input(
            "Page", min_value=1, max_value=max(1, total_pages), value=1, step=1
        )

        if len(idx_all) == 0:
            st.warning("No cells to display.")
        else:
            st.markdown("### Bad Epochs")

            # ---- BAD EPOCH FORM: epochs only change on 'Add Epoch' ----
            with st.form("bad_epoch_form"):
                c_bad_in1, c_bad_in2, c_bad_btn = st.columns([1, 1, 1])
                if T > 0:
                    t_min, t_max = float(t[0]), float(t[-1])
                else:
                    t_min, t_max = 0.0, 1.0
                step_val = float(1.0 / frame_rate) if time_is_seconds and frame_rate > 0 else 1.0
                default_width = (t_max - t_min) * 0.02

                with c_bad_in1:
                    start_val = st.number_input(
                        f"Start ({time_label})",
                        min_value=t_min,
                        max_value=t_max,
                        value=t_min,
                        step=step_val,
                        key="be_s_form",
                    )
                with c_bad_in2:
                    end_val = st.number_input(
                        f"End ({time_label})",
                        min_value=t_min,
                        max_value=t_max,
                        value=min(t_max, t_min + default_width),
                        step=step_val,
                        key="be_e_form",
                    )
                with c_bad_btn:
                    st.write("")
                    st.write("")
                    add_epoch_clicked = st.form_submit_button("Add Epoch")

            if add_epoch_clicked:
                s0, s1 = float(start_val), float(end_val)
                st.session_state.bad_epochs.append((s0, s1))
                st.session_state.bad_epochs.sort(key=lambda x: x[0])
                # no explicit st.rerun; button already causes rerun

            if st.session_state.bad_epochs:
                st.markdown("**Current Bad Epochs:**")
                for i, (e0, e1) in enumerate(st.session_state.bad_epochs):
                    c_list_txt, c_list_del = st.columns([4, 1])
                    with c_list_txt:
                        st.text(f"[{i+1}] {e0:.3f} – {e1:.3f} {time_label}")
                    with c_list_del:
                        if st.button("🗑️", key=f"del_be_{i}"):
                            del st.session_state.bad_epochs[i]
                            # click itself triggers rerun, state already modified

                bad_mask_full = make_bad_mask(t, st.session_state.bad_epochs)
            else:
                st.caption("No bad epochs defined.")

            st.markdown("---")

            start_i = (page_idx - 1) * cells_per_page
            end_i = min(len(idx_all), start_i + cells_per_page)
            idx_show = idx_all[start_i:end_i]

            rows = len(idx_show) + 1
            shift_frac = 0.15
            cell_frac = (1.0 - shift_frac) / len(idx_show) if len(idx_show) > 0 else 0.85
            row_heights = [cell_frac] * len(idx_show) + [shift_frac]

            fig_stack = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                row_heights=row_heights,
                vertical_spacing=0.005,
                subplot_titles=tuple(
                    [f"Cell {i+1}" for i in idx_show] + ["Shift distances"]
                ),
            )

            # Plot per-cell normalized traces (only those on this page)
            for r, i in enumerate(idx_show, start=1):
                y = get_normalized_trace(src_over, i, bad_mask_full)
                fig_stack.add_trace(
                    go.Scatter(
                        x=t,
                        y=y,
                        mode="lines",
                        line=dict(width=1, color=colors[i]),
                        hovertemplate="ROI %{name}<br>val=%{y:.3f}<extra></extra>",
                        name=f"{i+1}",
                    ),
                    row=r,
                    col=1,
                )

            sd = shift_distances
            sd = sd[: len(t)] if sd.size > len(t) else np.pad(
                sd, (0, len(t) - sd.size), constant_values=np.nan
            )
            fig_stack.add_trace(
                go.Scatter(
                    x=t, y=sd, mode="lines", line=dict(width=1), showlegend=False
                ),
                row=rows,
                col=1,
            )
            fig_stack.update_yaxes(title_text="Shift", row=rows, col=1)
            fig_stack.update_xaxes(title_text=time_label, row=rows, col=1)

            for (t0, t1) in st.session_state.bad_epochs:
                for r in range(1, rows + 1):
                    fig_stack.add_vrect(
                        x0=t0,
                        x1=t1,
                        fillcolor="red",
                        opacity=0.15,
                        line_width=0,
                        layer="below",
                        row=r,
                        col=1,
                    )
            for bt in session_boundary_times:
                for r in range(1, rows + 1):
                    fig_stack.add_vline(
                        x=bt,
                        line_width=1,
                        line_dash="dot",
                        line_color="yellow",
                        row=r,
                        col=1,
                    )

            fig_stack.update_annotations(font=dict(size=9))
            total_h = max(600, trace_row_height * rows)
            fig_stack.update_layout(
                height=total_h,
                margin=dict(l=40, r=10, t=30, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(
                fig_stack, use_container_width=True, config={"displaylogo": False}
            )

# -----------------------------------------------------------------------------#
#                               TAB 2: PER-CELL                                #
# -----------------------------------------------------------------------------#
elif tab == "🧫 Per-cell":
    st.subheader("Per-cell")
    cA, cB, cC, cD = st.columns([1, 1, 1, 1.2])
    with cA:
        cell_label = st.number_input(
            "Cell (1-based)", min_value=1, max_value=N, value=1, step=1
        )
        cell_i = cell_label - 1
    with cB:
        available_sources = list(sources_raw.keys())
        src_defaults = [
            s
            for s in [
                "raw_traces",
                "mc_traces",
                "mc_denoised_traces",
                "weighted_mc_denoised_traces",
            ]
            if s in available_sources
        ]
        if not src_defaults:
            src_defaults = [available_sources[0]] if available_sources else []
        selected_sources = st.multiselect(
            "Trace sources", available_sources, default=src_defaults
        )
    with cC:
        ysync = st.checkbox("Sync Y across traces", value=False)
    with cD:
        st.markdown(
            f"**Normalization:** {norm_mode}<br>**N cells:** {N}",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    def image_with_contours_focus(img, title):
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=img,
                colorscale="gray" if title != "Max Weight Map" else "Magma",
                showscale=False,
            )
        )
        for i, mask in enumerate(ROIs):
            alpha = 1.0 if i == cell_i else 0.2
            for C in roi_contours(mask):
                fig.add_trace(
                    go.Scatter(
                        x=C[:, 1],
                        y=C[:, 0],
                        mode="lines",
                        line=dict(width=1, color=colors[i]),
                        opacity=alpha,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            cy, cx = roi_centroid(mask)
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="text",
                    text=[str(i + 1)],
                    textfont=dict(color=colors[i], size=12),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=alpha,
                )
            )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(
            visible=False, autorange="reversed", scaleanchor="x", scaleratio=1
        )
        fig.update_layout(title=title, margin=dict(l=0, r=0, t=32, b=0), height=320)
        return fig

    c1.plotly_chart(
        image_with_contours_focus(mean_img_raw_base, "Mean Raw"),
        use_container_width=True,
        config={"displaylogo": False},
    )
    c2.plotly_chart(
        image_with_contours_focus(mean_img_mc_base, "Mean MC"),
        use_container_width=True,
        config={"displaylogo": False},
    )
    c3.plotly_chart(
        image_with_contours_focus(mean_img_mc_denoised_base, "MC Denoised"),
        use_container_width=True,
        config={"displaylogo": False},
    )
    c4.plotly_chart(
        image_with_contours_focus(weights_img_base, "Max Weight Map"),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.markdown("---")

    series = {}
    if len(t) > 0 and selected_sources:
        # recompute bad mask in case it changed
        bad_mask_full = make_bad_mask(t, st.session_state.bad_epochs)
        for k in selected_sources:
            if k in sources_raw:
                series[k] = get_normalized_trace(k, cell_i, bad_mask_full)

    sidx_saved = (
        np.array(spikes_verified_array[cell_i], dtype=int)
        if 0 <= cell_i < N
        else np.array([], dtype=int)
    )
    mask_sp = np.zeros(len(t), dtype=bool)
    if len(sidx_saved) > 0:
        mask_sp[(sidx_saved[(sidx_saved >= 0) & (sidx_saved < len(t))])] = True
    spikes_overlay_idx = np.where(mask_sp)[0]

    if series and len(t) > 0:
        rows = len(series)
        fig_cell = make_subplots(
            rows=rows + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=tuple(list(series.keys()) + ["Shift distances"]),
        )
        for r, k in enumerate(series.keys(), start=1):
            y = series[k]
            fig_cell.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    line=dict(width=1, color=colors[cell_i]),
                    showlegend=False,
                    hovertemplate=time_label + "=%{x:.3f}<br>val=%{y:.3f}<extra></extra>",
                ),
                row=r,
                col=1,
            )
            if spikes_overlay_idx.size > 0:
                fig_cell.add_trace(
                    go.Scatter(
                        x=t[spikes_overlay_idx],
                        y=y[spikes_overlay_idx],
                        mode="markers",
                        marker=dict(size=6, color="#111", symbol="x"),
                        showlegend=False,
                    ),
                    row=r,
                    col=1,
                )
            if ysync:
                ymin = float(
                    np.nanmin([np.nanmin(series[k2]) for k2 in series.keys()])
                )
                ymax = float(
                    np.nanmax([np.nanmax(series[k2]) for k2 in series.keys()])
                )
                if ymax <= ymin:
                    ymax = ymin + 1.0
                fig_cell.update_yaxes(range=[ymin, ymax], row=r, col=1)
            fig_cell.update_yaxes(title_text=k, row=r, col=1)

        sd = shift_distances
        sd = sd[: len(t)] if sd.size > len(t) else np.pad(
            sd, (0, len(t) - sd.size), constant_values=np.nan
        )
        fig_cell.add_trace(
            go.Scatter(
                x=t, y=sd, mode="lines", line=dict(width=1), showlegend=False
            ),
            row=rows + 1,
            col=1,
        )
        for bt in session_boundary_times:
            for r in range(1, rows + 2):
                fig_cell.add_vline(
                    x=bt,
                    line_width=1,
                    line_dash="dot",
                    line_color="yellow",
                    row=r,
                    col=1,
                )
        fig_cell.update_layout(
            margin=dict(l=60, r=10, t=40, b=10),
            height=max(320, 220 * (rows + 1)),
        )
        st.plotly_chart(
            fig_cell, use_container_width=True, config={"displaylogo": False}
        )

# -----------------------------------------------------------------------------#
#                       TAB 3: SPIKE DETECTION & GOOD CELLS                    #
# -----------------------------------------------------------------------------#
elif tab == "⚡ Spike detection & Good cells":
    st.subheader("Spike detection")
    if not sources_raw:
        st.error("No sources.")
        st.stop()

    src_spike = list(sources_raw.keys())[0]
    if "weighted_mc_denoised_traces" in sources_raw:
        src_spike = "weighted_mc_denoised_traces"
    src_spike = st.selectbox(
        "Source",
        list(sources_raw.keys()),
        index=list(sources_raw.keys()).index(src_spike),
        key="spike_source_select",
    )

    TR = np.asarray(sources_raw[src_spike], dtype=float)
    bad_mask_full = make_bad_mask(t, st.session_state.bad_epochs)
    if bad_mask_full.any() and TR.shape[0] == bad_mask_full.size:
        TR = TR.copy()
        TR[bad_mask_full, :] = np.nan

    if "params_per_cell" not in st.session_state:
        st.session_state.params_per_cell = {
            i: {
                "frame_rate": float(frame_rate),
                "n_segments": 3,
                "hp_hz": 20.0,
                "mad_k": 4.0,
                "min_isi_ms": 2.0,
            }
            for i in range(N)
        }
        for k, v in saved_params.items():
            try:
                st.session_state.params_per_cell[int(k)].update(v)
            except:
                pass

    if "spikes_verified_array" not in st.session_state:
        st.session_state.spikes_verified_array = [
            list(map(int, s)) for s in spikes_verified_array
        ]

    cols_sel = st.columns([1, 2])
    good_1based = [i + 1 for i in st.session_state.good_cells]
    with cols_sel[0]:
        dd = st.selectbox("Pick Good Cell", options=good_1based if good_1based else [1])
    cell_i = int(dd) - 1

    p = st.session_state.params_per_cell[cell_i]
    c_p = st.columns(4)
    with c_p[0]:
        hp_hz_ui = st.number_input(
            "HP Hz", value=int(p["hp_hz"]), step=1, key=f"hpf_{cell_i}"
        )
    with c_p[1]:
        mad_k = st.number_input(
            "Mad K",
            value=float(p["mad_k"]),
            step=0.1,
            format="%.1f",
            key=f"kmad_{cell_i}",
        )
    with c_p[2]:
        min_isi = st.number_input(
            "Min ISI", value=int(p["min_isi_ms"]), step=1, key=f"isi_{cell_i}"
        )
    with c_p[3]:
        n_seg_ui = st.number_input(
            "Segments",
            value=int(p["n_segments"]),
            min_value=1,
            step=1,
            key=f"seg_{cell_i}",
        )

    st.session_state.params_per_cell[cell_i].update(
        {
            "hp_hz": hp_hz_ui,
            "mad_k": mad_k,
            "min_isi_ms": min_isi,
            "n_segments": n_seg_ui,
        }
    )

    peaks, hp_tr, segs = detect_spikes_segmented(
        np.nan_to_num(TR[:, cell_i]),
        float(frame_rate),
        int(n_seg_ui),
        float(hp_hz_ui),
        float(mad_k),
        float(min_isi),
        segment_edges=session_edges_frames if multi_session else None,
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(
        go.Scatter(x=t, y=hp_tr, line=dict(width=1), showlegend=False),
        row=1,
        col=1,
    )
    if peaks.size:
        fig.add_trace(
            go.Scatter(
                x=t[peaks],
                y=hp_tr[peaks],
                mode="markers",
                marker=dict(symbol="x", size=8, color="red"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=t, y=np.nan_to_num(TR[:, cell_i]), line=dict(width=1), showlegend=False
        ),
        row=2,
        col=1,
    )
    if peaks.size:
        fig.add_trace(
            go.Scatter(
                x=t[peaks],
                y=TR[:, cell_i][peaks],
                mode="markers",
                marker=dict(symbol="x", size=8, color="red"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    n_eff = len(segs) - 1
    for j in range(n_eff):
        s0, s1 = segs[j], segs[j + 1]
        m, mad = robust_mad(hp_tr[s0:s1])
        th = m + mad_k * mad
        fig.add_trace(
            go.Scatter(
                x=t[s0:s1],
                y=np.full(s1 - s0, th),
                mode="lines",
                line=dict(dash="dash", color="green", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    st.plotly_chart(fig, use_container_width=True)

    c_act = st.columns([1, 1, 2])
    with c_act[0]:
        if st.button("Confirm Spikes"):
            st.session_state.spikes_verified_array[cell_i] = list(map(int, peaks))
            st.success("Saved.")
    with c_act[1]:
        if st.button("Clear Spikes"):
            st.session_state.spikes_verified_array[cell_i] = []
            st.info("Cleared.")

    st.markdown("---")
    st.subheader("Save Results")
    fname = st.text_input("Filename", "volpy_demix_results_annotated.pickle")

    if st.button("Save Annotations"):
        def _local_epochs(s_idx):
            s0, s1 = session_edges_frames[s_idx], session_edges_frames[s_idx + 1]
            loc = []
            t0g, t1g = (
                (s0 / frame_rate, s1 / frame_rate)
                if time_is_seconds
                else (s0, s1)
            )
            for (b0, b1) in st.session_state.bad_epochs:
                lo, hi = max(min(b0, b1), t0g), min(max(b0, b1), t1g)
                if hi > lo:
                    loc.append((lo - t0g, hi - t0g))
            return loc

        for s in range(n_sessions):
            if s >= len(paths_list) or not paths_list[s]:
                continue
            base = dict(results_list[s])
            s0, s1 = session_edges_frames[s], session_edges_frames[s + 1]
            ep = _local_epochs(s)
            mask_l = make_bad_mask(
                np.arange(s1 - s0) / frame_rate if time_is_seconds else np.arange(s1 - s0),
                ep,
            )

            for k in trace_keys_all:
                if k in results:
                    dat_processed = results[k][s0:s1, :].copy()
                    if mask_l.any():
                        dat_processed[mask_l, :] = np.nan
                    orig_val = base.get(k)
                    if hasattr(orig_val, "shape") and len(orig_val.shape) == 2:
                        if orig_val.shape[1] == N and orig_val.shape[0] != N:
                            base[k] = dat_processed.T
                        else:
                            base[k] = dat_processed
                    else:
                        base[k] = dat_processed

            spk_local = []
            for i in range(N):
                g = np.array(st.session_state.spikes_verified_array[i])
                valid = g[(g >= s0) & (g < s1)]
                spk_local.append((valid - s0).tolist())
            base["spikes_verified_array"] = spk_local
            base["bad_epochs"] = ep
            base["good_cells"] = st.session_state.good_cells
            base["spike_detection_params_per_cell"] = st.session_state.params_per_cell

            out = os.path.join(os.path.dirname(paths_list[s]), fname)
            with open(out, "wb") as f:
                pickle.dump(base, f)
        st.success("Saved!")
