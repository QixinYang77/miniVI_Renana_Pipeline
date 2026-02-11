import math
import numpy as np


def zscore(x):
    arr = np.asarray(x, dtype=np.float32)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    return (arr - m) / (s if s > 0 else 1.0)


def norm01(x):
    arr = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def make_bad_mask(t_axis, epochs):
    t_arr = np.asarray(t_axis)
    if t_arr.size == 0 or not epochs:
        return np.zeros(t_arr.shape[0], dtype=bool)
    mask = np.zeros(t_arr.shape[0], dtype=bool)
    for t0, t1 in epochs:
        lo, hi = min(float(t0), float(t1)), max(float(t0), float(t1))
        mask |= (t_arr >= lo) & (t_arr <= hi)
    return mask


def _naninterp_1d(y):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    finite = np.isfinite(y)
    if finite.all():
        return y
    if not np.any(finite):
        return np.full_like(y, np.nan, dtype=np.float32)
    x = np.arange(y.size, dtype=np.float32)
    out = y.copy()
    out[~finite] = np.interp(x[~finite], x[finite], out[finite]).astype(np.float32)
    return out


def _smooth_blocks(x, k, method="median"):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or int(k) <= 1:
        return x
    k = int(max(1, k))
    half = k // 2
    if method == "mean":
        valid = np.isfinite(x)
        x0 = np.where(valid, x, 0.0).astype(np.float32, copy=False)
        kernel = np.ones(k, dtype=np.float32)
        sumv = np.convolve(x0, kernel, mode="same")
        cnt = np.convolve(valid.astype(np.float32), kernel, mode="same")
        out = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan, dtype=np.float32), where=(cnt > 0))
        return out.astype(np.float32, copy=False)

    # Median path: sliding window median with NaN support.
    pad_left = half
    pad_right = k - 1 - half
    padded = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.nanmedian(windows, axis=1).astype(np.float32, copy=False)


def baseline_correct_trace(
    trace,
    fs,
    percentile=50.0,
    window_s=60.0,
    smooth="median",
    bad_mask=None,
):
    x = np.asarray(trace, dtype=np.float32).reshape(-1)
    t_len = int(x.size)
    if t_len == 0 or fs is None or float(fs) <= 0:
        return x, np.full_like(x, np.nan, dtype=np.float32)

    p = min(50.0, max(0.0, float(percentile)))
    win_frames = int(max(1, round(float(window_s) * float(fs))))

    if bad_mask is not None:
        bm = np.asarray(bad_mask, dtype=bool).reshape(-1)
        if bm.size == t_len and np.any(bm):
            x = x.copy()
            x[bm] = np.nan

    block_frames = int(max(64, win_frames // 10))
    n_blocks = int(math.ceil(float(t_len) / float(block_frames)))
    if n_blocks <= 1:
        base = np.nanpercentile(x, p) if np.any(np.isfinite(x)) else np.nan
        baseline = np.full_like(x, base, dtype=np.float32)
        return x - baseline, baseline

    # Vectorized block percentile (faster than Python loops for long traces).
    pad = n_blocks * block_frames - t_len
    if pad > 0:
        x_pad = np.pad(x, (0, pad), mode="constant", constant_values=np.nan)
    else:
        x_pad = x
    blocks = x_pad.reshape(n_blocks, block_frames)
    block_vals = np.nanpercentile(blocks, p, axis=1).astype(np.float32, copy=False)

    k_blocks = int(max(1, round(float(win_frames) / float(block_frames))))
    k_blocks = int(min(n_blocks, k_blocks))
    block_smooth = _smooth_blocks(block_vals, k_blocks, method=str(smooth))
    block_smooth = _naninterp_1d(block_smooth)

    centers = (np.arange(n_blocks, dtype=np.float32) + 0.5) * float(block_frames)
    centers = np.clip(centers, 0.0, float(t_len - 1))
    baseline = np.interp(np.arange(t_len, dtype=np.float32), centers, block_smooth).astype(
        np.float32
    )
    return x - baseline, baseline


def highpass_filter_trace(trace, fs, cutoff_hz=20.0, bad_mask=None):
    x = np.asarray(trace, dtype=np.float32).reshape(-1)
    if x.size == 0 or fs is None or float(fs) <= 0:
        return x

    fc = float(cutoff_hz)
    if not np.isfinite(fc) or fc <= 0:
        return x.copy()

    fs = float(fs)
    nyq = 0.5 * fs
    if nyq <= 0:
        return x.copy()
    if fc >= nyq:
        fc = nyq * 0.99

    x_in = x.copy()
    if bad_mask is not None:
        bm = np.asarray(bad_mask, dtype=bool).reshape(-1)
        if bm.size == x_in.size and np.any(bm):
            x_in[bm] = np.nan

    x_fill = _naninterp_1d(x_in)
    if not np.any(np.isfinite(x_fill)):
        return np.zeros_like(x, dtype=np.float32)

    n = int(x_fill.size)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x_fill.astype(np.float32, copy=False))
    spec[freqs < fc] = 0
    y = np.fft.irfft(spec, n=n).astype(np.float32, copy=False)
    return y


def _motion_regress_chunk_pca(
    y_chunk,
    x_chunk,
    bad_mask_chunk=None,
    ridge_lambda=0.0,
    n_components=5,
):
    y = np.asarray(y_chunk, dtype=np.float32)
    x = np.asarray(x_chunk, dtype=np.float32)
    if y.ndim != 2 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        return y
    t_len, n_cols = y.shape
    if t_len < 4 or n_cols < 1 or x.shape[1] < 1:
        return y

    k_req = int(max(1, int(n_components)))
    fit_rows = np.isfinite(x).all(axis=1)
    if bad_mask_chunk is not None:
        bm = np.asarray(bad_mask_chunk, dtype=bool).reshape(-1)
        if bm.size == t_len:
            fit_rows &= ~bm

    # Prepare full predictors (for subtraction on all rows), with robust NaN handling.
    x_full = x.astype(np.float32, copy=True)
    for fi in range(x_full.shape[1]):
        x_full[:, fi] = _naninterp_1d(x_full[:, fi])
        if not np.any(np.isfinite(x_full[:, fi])):
            x_full[:, fi] = 0.0
    x_mu = np.nanmean(x_full, axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sd = np.nanstd(x_full, axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sd[x_sd <= 1e-8] = 1.0
    xz_full = (x_full - x_mu) / x_sd

    xz_fit = xz_full[fit_rows, :]
    if xz_fit.shape[0] < 3:
        return y

    try:
        _, svals, vt = np.linalg.svd(xz_fit.astype(np.float64, copy=False), full_matrices=False)
    except Exception:
        return y
    rank = int(np.count_nonzero(svals > 1e-8))
    k = int(min(k_req, rank, xz_fit.shape[1], max(1, xz_fit.shape[0] - 1)))
    if k < 1:
        return y

    vt_k = vt[:k, :].astype(np.float32, copy=False)  # (k, p)
    s_k = svals[:k].astype(np.float32, copy=False)   # (k,)
    inv_s = np.zeros_like(s_k, dtype=np.float32)
    nz = s_k > 1e-8
    inv_s[nz] = 1.0 / s_k[nz]
    ub_full = (xz_full @ vt_k.T) * inv_s[None, :]    # (T, k), VolPy-like temporal basis

    y_finite_all = np.isfinite(y).all(axis=1)
    fit_rows_y = fit_rows & y_finite_all
    min_rows = int(k + 1)
    if int(np.count_nonzero(fit_rows_y)) < min_rows:
        return y

    uv = ub_full[fit_rows_y, :].astype(np.float64, copy=False)
    y_fit = y[fit_rows_y, :].astype(np.float64, copy=False)
    y_mean = np.mean(y_fit, axis=0, keepdims=True)
    y_fit_centered = y_fit - y_mean

    lam = float(max(0.0, ridge_lambda))
    if lam > 0.0:
        xtx = uv.T @ uv
        xtx += lam * np.eye(int(xtx.shape[0]), dtype=np.float64)
        xty = uv.T @ y_fit_centered
        try:
            coef = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    else:
        coef = np.linalg.lstsq(uv, y_fit_centered, rcond=None)[0]  # (k, n_cells)

    motion_term = (ub_full.astype(np.float64, copy=False) @ coef).astype(np.float32, copy=False)
    out = y.copy()
    finite = np.isfinite(y) & np.isfinite(motion_term)
    out[finite] = out[finite] - motion_term[finite]
    return out.astype(np.float32, copy=False)


def _smooth_shift_vector(v, smooth_frames):
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    k = int(max(0, smooth_frames))
    if arr.size == 0 or k <= 1:
        return arr
    if k % 2 == 0:
        k += 1
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    filled = arr if valid.all() else _naninterp_1d(arr)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    sm = np.convolve(filled.astype(np.float32, copy=False), kernel, mode="same").astype(np.float32, copy=False)
    sm[~valid] = np.nan
    return sm


def motion_regress_traces_chunked(
    traces,
    shifts_xy,
    chunk_frames=5000,
    overlap_frames=0,
    ridge_lambda=0.0,
    shift_smooth_frames=0,
    n_components=5,
    bad_mask=None,
):
    y = np.asarray(traces, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1:
        return np.asarray(traces, dtype=np.float32)

    sxy = np.asarray(shifts_xy, dtype=np.float32)
    if sxy.ndim != 2 or sxy.shape[1] < 2 or sxy.shape[0] != y.shape[0]:
        return y.astype(np.float32, copy=False)

    t_len = int(y.shape[0])
    chunk = int(max(1, int(chunk_frames)))
    overlap = int(max(0, int(overlap_frames)))
    if overlap >= chunk:
        overlap = max(0, chunk - 1)

    dx_raw = sxy[:, 0]
    dy_raw = sxy[:, 1]
    dx_use = _smooth_shift_vector(dx_raw, shift_smooth_frames)
    dy_use = _smooth_shift_vector(dy_raw, shift_smooth_frames)
    dx = dx_use - float(np.nanmean(dx_use))
    dy = dy_use - float(np.nanmean(dy_use))
    x = np.column_stack([dx, dy, dx * dx, dy * dy, dx * dy]).astype(np.float32, copy=False)

    bm_full = None
    if bad_mask is not None:
        bm = np.asarray(bad_mask, dtype=bool).reshape(-1)
        if bm.size == t_len:
            bm_full = bm

    def _denoise(start, end):
        y_c = y[start:end, :]
        x_c = x[start:end, :]
        bm_c = bm_full[start:end] if bm_full is not None else None
        return _motion_regress_chunk_pca(
            y_c,
            x_c,
            bad_mask_chunk=bm_c,
            ridge_lambda=ridge_lambda,
            n_components=n_components,
        )

    if overlap <= 0:
        parts = []
        for start in range(0, t_len, chunk):
            end = min(start + chunk, t_len)
            parts.append(_denoise(start, end))
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    stride = chunk - overlap
    if stride < 1:
        stride = 1

    out_parts = []
    prev_tail = None
    start = 0
    while start < t_len:
        end = min(start + chunk, t_len)
        out_chunk = _denoise(start, end)
        t_chunk = int(end - start)

        if start == 0 and end >= t_len:
            out_parts.append(out_chunk)
            break

        eff_ov = int(min(overlap, t_chunk))
        if eff_ov <= 0:
            out_parts.append(out_chunk)
            prev_tail = None
            start += stride
            continue

        if start == 0:
            head_len = t_chunk - eff_ov
            if head_len > 0:
                out_parts.append(out_chunk[:head_len, :])
            prev_tail = out_chunk[head_len:, :]
            start += stride
            continue

        if prev_tail is None:
            prev_tail = out_chunk[:eff_ov, :]

        prev_len = int(prev_tail.shape[0])
        eff_ov2 = int(min(eff_ov, prev_len))
        if eff_ov2 > 0:
            curr_ov = out_chunk[:eff_ov2, :]
            w = np.linspace(0.0, 1.0, eff_ov2, dtype=np.float32)[:, None]
            blended = (1.0 - w) * prev_tail[-eff_ov2:, :] + w * curr_ov
            out_parts.append(blended.astype(np.float32, copy=False))

        if end >= t_len:
            if t_chunk > eff_ov2:
                out_parts.append(out_chunk[eff_ov2:, :])
            break

        mid_start = eff_ov2
        mid_end = t_chunk - eff_ov
        if mid_end > mid_start:
            out_parts.append(out_chunk[mid_start:mid_end, :])

        prev_tail = out_chunk[mid_end:, :]
        start += stride

    if not out_parts:
        return y.astype(np.float32, copy=False)
    out = np.concatenate(out_parts, axis=0)
    if out.shape[0] > t_len:
        out = out[:t_len, :]
    elif out.shape[0] < t_len:
        pad = np.full((t_len - out.shape[0], y.shape[1]), np.nan, dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    return out.astype(np.float32, copy=False)


def minmax_envelope(x, y, target_points=3000):
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n = int(y_arr.size)
    tgt = int(max(50, target_points))
    if n <= tgt:
        return x_arr, y_arr

    # Two points per bin (local min and max), preserving sharp events.
    bin_size = int(max(1, math.ceil(n / float(tgt // 2))))
    n_bins = int(math.ceil(n / float(bin_size)))

    x_out = []
    y_out = []
    for bi in range(n_bins):
        s0 = bi * bin_size
        s1 = min(n, (bi + 1) * bin_size)
        ys = y_arr[s0:s1]
        xs = x_arr[s0:s1]
        if ys.size == 0:
            continue

        finite_idx = np.where(np.isfinite(ys))[0]
        if finite_idx.size == 0:
            x_out.extend([float(xs[0]), float(xs[-1])])
            y_out.extend([np.nan, np.nan])
            continue

        y_fin = ys[finite_idx]
        i_min = int(finite_idx[int(np.argmin(y_fin))])
        i_max = int(finite_idx[int(np.argmax(y_fin))])
        if i_min <= i_max:
            order = [i_min, i_max]
        else:
            order = [i_max, i_min]

        for i_loc in order:
            x_out.append(float(xs[i_loc]))
            y_out.append(float(ys[i_loc]))

    return np.asarray(x_out, dtype=np.float32), np.asarray(y_out, dtype=np.float32)
