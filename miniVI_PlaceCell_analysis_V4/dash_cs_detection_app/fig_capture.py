"""Capture matplotlib figures as Plotly figures for Dash display.

Safe to import from a Jupyter notebook — does NOT change the global
matplotlib backend on import.  The backend is switched to Agg only
inside ``capture_current_figures()`` and ``pipeline_context()``.
"""

import warnings
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

# Keep a reference to the real plt.show so we can restore it.
_original_show = plt.show


@contextmanager
def pipeline_context():
    """Context manager that suppresses ``plt.show()`` and switches to Agg.

    Use this around any call to the spike-detection functions so that
    their internal ``plt.show()`` calls become no-ops and figures are
    rendered to the Agg (in-memory) backend.

    The previous backend and ``plt.show`` are restored on exit.
    """
    prev_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    plt.show = lambda *_a, **_kw: None
    try:
        yield
    finally:
        plt.show = _original_show
        try:
            matplotlib.use(prev_backend)
        except Exception:
            pass  # some backends can't be re-activated; safe to ignore


def _conversion_error_figure(message):
    """Build a small placeholder Plotly figure when conversion fails."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 12, "color": "#444"},
        align="center",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=260,
    )
    return fig


def _fallback_image_figure(fig):
    """Convert a Matplotlib figure canvas to a zoomable Plotly image."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    pfig = go.Figure(data=[go.Image(z=rgba)])
    pfig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=max(280, int(fig.get_figheight() * 120)),
    )
    pfig.update_xaxes(visible=False)
    pfig.update_yaxes(visible=False, scaleanchor="x")
    return pfig


def capture_current_figures():
    """Capture all open matplotlib figures as Plotly figure dicts, then close them."""
    results = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        try:
            pfig = mpl_to_plotly(fig)
            pfig.update_layout(
                template="plotly_white",
                margin={"l": 40, "r": 20, "t": 30, "b": 40},
                height=max(280, int(fig.get_figheight() * 120)),
            )
            results.append(pfig.to_dict())
        except Exception as exc:
            warnings.warn(f"Matplotlib->Plotly conversion failed: {exc}")
            try:
                fallback = _fallback_image_figure(fig)
            except Exception:
                fallback = _conversion_error_figure(
                    "Figure conversion failed for one panel.\n"
                    "Re-run and check terminal for details."
                )
            results.append(fallback.to_dict())
        finally:
            plt.close(fig)
    return results


def close_all():
    """Close all open matplotlib figures."""
    plt.close("all")
