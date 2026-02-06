# Dash Overview App (Separate from Streamlit)

This folder contains a Dash-based overview UI for long recordings, built to keep your existing output pickle schema compatible with downstream code.

Current overview defaults:
- No temporal downsampling in the trace view (full-resolution points in the active window).
- Confirmed bad epochs are hidden from trace display.
- Default normalization is `0-1`.
- Default window is 60 seconds. The bottom slider scrolls continuously across recording time.

## Files
- `app.py`: Dash app entrypoint (overview panel).
- `demix_io.py`: loading/stitching plus save compatibility writer.
- `trace_store.py`: fast windowed/downsampled trace access.
- `signal_ops.py`: baseline correction, normalization, bad-mask, min/max envelope.

## Run
From project root:

```bash
python dash_overview_app/app.py /path/to/volpy_demix_results.pickle
```

Multiple sessions:

```bash
python dash_overview_app/app.py /path/s1/volpy_demix_results.pickle /path/s2/volpy_demix_results.pickle
```

Or auto-discover in a main folder:

```bash
python dash_overview_app/app.py --main-folder /path/to/Awake
```

Optional:

```bash
python dash_overview_app/app.py --main-folder /path/to/Awake --host 0.0.0.0 --port 8050 --debug
```

By default, the app auto-opens in your browser when it starts. To disable:

```bash
python dash_overview_app/app.py --main-folder /path/to/Awake --no-browser
```

## Dependencies

```bash
pip install dash plotly numpy
```

Optional for smoother ROI contours:

```bash
pip install scikit-image
```

## Save compatibility
`Save Annotations` writes the same key structure used by your current Streamlit app, including:
- `spikes_verified_array`
- `bad_epochs`
- `good_cells`
- `spike_detection_params_per_cell`
- `baseline_correction_params` (when enabled in loaded metadata)
- processed trace keys with original orientation preserved (`T x N` vs `N x T`)

This is designed so downstream code does not need to change.
