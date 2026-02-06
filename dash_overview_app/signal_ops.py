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
    percentile=5.0,
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
