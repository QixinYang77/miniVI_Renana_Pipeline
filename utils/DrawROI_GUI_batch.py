# roi_gui_multi.py
import os, re, json, glob
import numpy as np
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Optional

from matplotlib.path import Path as MplPath
from PIL import Image, ImageTk, ImageDraw
try:
    import tifffile as tiff
    _USE_TIFFILE = True
except Exception:
    _USE_TIFFILE = False

TIFF_NAME_MC = "mean_image_mc.tif"
ROI_TIFF_NAME = "Image_001_001_ROIs.tif"
SAVE_NAME = "manual_ROIs.npz"
META_NAME = "manual_ROIs_meta.json"

_BTN_KW = dict(bg="#f0f0f0", fg="black",
               activebackground="#e0e0e0", activeforeground="black",
               relief="raised", bd=1)

def _load_mean_image(folder, tiff_name: str):
    fp = Path(folder) / tiff_name
    if not fp.exists():
        raise FileNotFoundError(f"Missing {tiff_name} in {folder}")
    if _USE_TIFFILE:
        arr = tiff.imread(str(fp))
    else:
        arr = np.array(Image.open(fp))
    if arr.ndim > 2:
        arr = arr.squeeze()
    return arr.astype(np.float32)

def _overlay_rois_on_pil(
    pil_rgb: Image.Image, roi_stack: Optional[np.ndarray], *, alpha: float = 0.2
) -> Image.Image:
    if roi_stack is None:
        return pil_rgb
    roi_stack = np.asarray(roi_stack)
    if roi_stack.ndim == 2:
        roi_stack = roi_stack[None, ...]
    if roi_stack.ndim != 3:
        return pil_rgb

    def _roi_stack_to_shape(stack: np.ndarray, target_hw) -> np.ndarray:
        stack = np.asarray(stack)
        if stack.ndim == 2:
            stack = stack[None, ...]
        if stack.ndim != 3:
            return stack

        th, tw = int(target_hw[0]), int(target_hw[1])
        sh, sw = int(stack.shape[1]), int(stack.shape[2])
        if (sh, sw) == (th, tw):
            return stack

        # Integer downsample via max-pooling (preserve any ROI pixel).
        if sh % th == 0 and sw % tw == 0 and th > 0 and tw > 0:
            fh, fw = sh // th, sw // tw
            if fh >= 1 and fw >= 1:
                try:
                    pooled = (
                        stack.astype(bool)
                        .reshape(stack.shape[0], th, fh, tw, fw)
                        .max(axis=(2, 4))
                    )
                    return pooled.astype(bool)
                except Exception:
                    pass

        # Integer upsample via repeat.
        if th % sh == 0 and tw % sw == 0 and sh > 0 and sw > 0:
            rh, rw = th // sh, tw // sw
            try:
                up = np.repeat(np.repeat(stack.astype(bool), rh, axis=1), rw, axis=2)
                return up.astype(bool)
            except Exception:
                pass

        # Fallback: nearest-neighbor resize per ROI.
        out = np.zeros((stack.shape[0], th, tw), dtype=bool)
        for k in range(stack.shape[0]):
            m = stack[k].astype(np.uint8) * 255
            pil = Image.fromarray(m)
            pil = pil.resize((tw, th), resample=Image.NEAREST)
            out[k] = np.asarray(pil) > 0
        return out

    base = np.asarray(pil_rgb)
    if base.ndim != 3 or base.shape[2] != 3:
        return pil_rgb
    roi_stack = _roi_stack_to_shape(roi_stack, (base.shape[0], base.shape[1]))

    overlay = roi_stack.astype(bool).sum(axis=0).astype(np.float32)
    if overlay.size == 0 or float(overlay.max()) <= 0:
        return pil_rgb

    try:
        from matplotlib import cm as mpl_cm

        cmap = mpl_cm.get_cmap("jet")
        norm = overlay / float(overlay.max())
        colors = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        colors = np.zeros((overlay.shape[0], overlay.shape[1], 3), dtype=np.uint8)
        colors[..., 0] = 255

    base = np.asarray(pil_rgb, dtype=np.float32)
    mask = overlay > 0
    base[mask] = (1.0 - float(alpha)) * base[mask] + float(alpha) * colors[mask].astype(np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _label_rois_on_pil(pil_rgb: Image.Image, roi_stack: Optional[np.ndarray]) -> Image.Image:
    if roi_stack is None:
        return pil_rgb
    roi_stack = np.asarray(roi_stack)
    if roi_stack.ndim == 2:
        roi_stack = roi_stack[None, ...]
    if roi_stack.ndim != 3:
        return pil_rgb

    # Resize ROI masks to match the image for correct centroids.
    base = np.asarray(pil_rgb)
    th, tw = int(base.shape[0]), int(base.shape[1])
    sh, sw = int(roi_stack.shape[1]), int(roi_stack.shape[2])
    if (sh, sw) != (th, tw):
        out = np.zeros((roi_stack.shape[0], th, tw), dtype=bool)
        for k in range(roi_stack.shape[0]):
            m = roi_stack[k].astype(np.uint8) * 255
            pil = Image.fromarray(m)
            pil = pil.resize((tw, th), resample=Image.NEAREST)
            out[k] = np.asarray(pil) > 0
        roi_stack = out

    draw = ImageDraw.Draw(pil_rgb)
    for idx, mask in enumerate(roi_stack.astype(bool)):
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        label = str(idx + 1)
        # crude outline for readability
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.text((cx + dx, cy + dy), label, fill="black")
        draw.text((cx, cy), label, fill="white")
    return pil_rgb


def _make_session_preview(
    mean_mc: np.ndarray,
    roi_stack: Optional[np.ndarray],
    *,
    thumb_h: int,
    vmax_factor: float = 0.95,
) -> Image.Image:
    def _to_rgb(arr: np.ndarray) -> Image.Image:
        arr = np.asarray(arr, dtype=np.float32)
        vmin = 0.0
        vmax = float(arr.max()) * float(vmax_factor) if float(arr.max()) > 0 else 1.0
        disp = _normalize_for_display(arr, vmin, vmax)
        rgb = np.stack([disp] * 3, axis=-1)
        return Image.fromarray(rgb)

    pil_mc = _to_rgb(mean_mc)

    pil_mc = _overlay_rois_on_pil(pil_mc, roi_stack, alpha=0.2)
    pil_mc = _label_rois_on_pil(pil_mc, roi_stack)

    # Resize to consistent thumbnail height
    H = pil_mc.size[1]
    if H <= 0:
        return pil_mc
    scale = float(thumb_h) / float(H)
    thumb_w = int(round(pil_mc.size[0] * scale))
    pil_mc = pil_mc.resize((thumb_w, thumb_h), Image.BILINEAR)
    return pil_mc

def _normalize_for_display(arr, vmin, vmax):
    arr = np.asarray(arr, dtype=np.float32)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = (arr - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

def _polygon_to_mask(verts, shape):
    if len(verts) < 3:
        return np.zeros(shape, dtype=bool)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    path = MplPath(verts, closed=True)
    return path.contains_points(pts).reshape(H, W)

class _ROIManager:
    def __init__(self):
        self.roi_vertices = []   # list of list[(x,y)]
        self.current_poly = []
    def add_point(self, x, y): self.current_poly.append((x, y))
    def finish_current(self):
        if len(self.current_poly) >= 3:
            self.roi_vertices.append(self.current_poly.copy())
        self.current_poly = []
    def undo_point(self):
        if self.current_poly: self.current_poly.pop()
    def undo_roi(self):
        if self.roi_vertices: self.roi_vertices.pop()
    def clear(self): self.roi_vertices.clear(); self.current_poly = []
    def as_masks(self, shape):
        masks = [_polygon_to_mask(v, shape) for v in self.roi_vertices]
        return np.stack(masks, axis=0) if masks else np.zeros((0, *shape), dtype=bool)

class _MultiROIGUI(tk.Tk):
    def __init__(self, folders, FigureCanvasTkAgg, plt):
        super().__init__()
        self.title("Multi-session ROI Drawer (max mean image)")
        self.configure(bg="white")
        self.geometry("1800x1000")

        # --- Load mean images from folders ---
        all_folders = [Path(f) for f in folders]
        self.folders = []
        self.images_mc = []
        self.raw_roi_stacks = []  # list of (n_rois, H, W) or None
        missing = []

        for f in all_folders:
            try:
                img_mc = _load_mean_image(f, TIFF_NAME_MC)
                self.images_mc.append(img_mc)
                self.folders.append(f)

                roi_fp = f / ROI_TIFF_NAME
                if roi_fp.exists() and _USE_TIFFILE:
                    try:
                        roi_stack = tiff.imread(str(roi_fp))
                        if roi_stack.ndim == 2:
                            roi_stack = roi_stack[np.newaxis, ...]
                        self.raw_roi_stacks.append(roi_stack.astype(bool))
                    except Exception:
                        self.raw_roi_stacks.append(None)
                else:
                    self.raw_roi_stacks.append(None)

            except Exception as e:
                missing.append(f"{f}: {e}")

        if missing:
            messagebox.showwarning("Missing TIFF",
                                   "Some folders were skipped:\n" + "\n".join(missing))

        if not self.images_mc:
            messagebox.showerror("Error", f"No valid {TIFF_NAME_MC} found in any folder.")
            self.destroy()
            return

        # Check shape consistency
        shapes = {im.shape for im in self.images_mc}
        if len(shapes) != 1:
            messagebox.showerror("Error",
                                 f"mean_image_mc.tif shapes differ between folders: {shapes}")
            self.destroy()
            return

        self.stack = np.stack(self.images_mc, axis=0)
        self.max_img = np.max(self.stack, axis=0)

        # global contrast
        self.vmin, self.vmax = np.percentile(self.stack, [1, 99])

        self.roi = _ROIManager()

        # Layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # ---- Top info bar ----
        top_info = tk.Frame(self, bg="white")
        top_info.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        lbl = tk.Label(
            top_info,
            text=f"{len(self.images_mc)} sessions loaded | drawing ROIs on max(mean_image_mc) | overlays from {ROI_TIFF_NAME}",
            font=("Arial", 13, "bold"),
            bg="white", fg="black"
        )
        lbl.pack(side="left", padx=4)

        # ---- Top thumbnails row (scrollable) ----
        thumbs_canvas = tk.Canvas(self, bg="white", highlightthickness=0, height=320)
        thumbs_canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        thumbs_scroll = tk.Scrollbar(self, orient="horizontal", command=thumbs_canvas.xview)
        thumbs_scroll.grid(row=2, column=0, sticky="ew", padx=10)
        thumbs_canvas.configure(xscrollcommand=thumbs_scroll.set)

        thumbs_frame = tk.Frame(thumbs_canvas, bg="white")
        thumbs_canvas.create_window((0, 0), window=thumbs_frame, anchor="nw")

        self._thumb_images = []  # keep references
        thumb_h = 260  # thumbnail height

        for i, folder in enumerate(self.folders):
            pil = _make_session_preview(
                self.images_mc[i],
                self.raw_roi_stacks[i],
                thumb_h=thumb_h,
                vmax_factor=0.95,
            )
            tk_img = ImageTk.PhotoImage(pil)
            self._thumb_images.append(tk_img)

            cell = tk.Frame(thumbs_frame, bg="white", bd=1, relief="solid")
            cell.pack(side="left", padx=4)
            lbl_img = tk.Label(cell, image=tk_img, bg="white")
            lbl_img.pack()
            lbl_txt = tk.Label(cell, text=folder.name, bg="white", fg="black", font=("Arial", 9))
            lbl_txt.pack()

        def _on_thumb_configure(_evt=None):
            bbox = thumbs_canvas.bbox("all")
            if bbox is not None:
                thumbs_canvas.configure(scrollregion=bbox)

        thumbs_frame.bind("<Configure>", _on_thumb_configure)
        _on_thumb_configure()

        # ---- Center: Matplotlib canvas for max image ----
        center = tk.Frame(self, bg="white")
        center.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=100)
        self.ax.set_axis_off()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ---- Bottom controls ----
        bottom = tk.Frame(self, bg="white")
        bottom.grid(row=4, column=0, sticky="ew", padx=10, pady=8)
        bottom.columnconfigure(5, weight=1)

        self.save_btn = tk.Button(bottom, text="💾 Save ROIs",
                                  command=self.on_save, **_BTN_KW)
        self.save_btn.grid(row=0, column=0, padx=4)

        self.clear_btn = tk.Button(bottom, text="🧹 Clear All",
                                   command=self.on_clear, **_BTN_KW)
        self.clear_btn.grid(row=0, column=1, padx=4)

        self.undo_point_btn = tk.Button(bottom, text="↩ Undo Point",
                                        command=self.on_undo_point, **_BTN_KW)
        self.undo_point_btn.grid(row=0, column=2, padx=4)

        self.undo_roi_btn = tk.Button(bottom, text="⟲ Undo ROI",
                                      command=self.on_undo_roi, **_BTN_KW)
        self.undo_roi_btn.grid(row=0, column=3, padx=4)

        self.help_lbl = tk.Label(
            bottom,
            text=("Left-click: add vertex  |  Double-click: finish ROI  |  "
                  "Right-click: SAVE to all folders  |  Keys: u=undo point, "
                  "z=undo ROI, c=clear"),
            bg="white", fg="#333"
        )
        self.help_lbl.grid(row=0, column=5, padx=12, sticky="w")

        # Events
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key   = self.canvas.mpl_connect("key_press_event", self.on_key)

        # Clean exit on window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.set_axis_off()
        # draw max image
        self.ax.imshow(self.max_img, cmap="gray", vmin=self.vmin, vmax=self.vmax,
                       interpolation="nearest")
        # finished ROIs
        for verts in self.roi.roi_vertices:
            if len(verts) >= 2:
                xs, ys = zip(*verts)
                self.ax.plot(xs + (verts[0][0],), ys + (verts[0][1],),
                             '-', lw=1.5, color="lime")
                self.ax.plot(xs, ys, 'o', ms=3, color="lime")
        # current ROI
        if self.roi.current_poly:
            xs, ys = zip(*self.roi.current_poly)
            self.ax.plot(xs, ys, 'r-', lw=1)
            self.ax.plot(xs, ys, 'ro', ms=3)
        self.ax.set_title("Max of mean_image_mc.tif across sessions")
        self.canvas.draw_idle()

    # Events
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            # left-click
            if event.dblclick:
                if len(self.roi.current_poly) >= 3:
                    self.roi.finish_current()
                    self.redraw()
                    self.help_lbl.config(
                        text="ROI finished. Right-click to SAVE to all folders, or continue drawing."
                    )
                else:
                    self.help_lbl.config(text="Need ≥3 points to finish ROI.")
            else:
                self.roi.add_point(event.xdata, event.ydata)
                self.redraw()
        elif event.button == 3:
            # right-click: save
            self.on_save()

    def on_key(self, event):
        if event.key == 'u':
            self.roi.undo_point()
        elif event.key == 'z':
            self.roi.undo_roi()
        elif event.key == 'c':
            self.roi.clear()
        self.redraw()

    # Save / Clear / Undo
    def on_save(self):
        if self.max_img is None:
            return
        H, W = self.max_img.shape
        masks = self.roi.as_masks((H, W))
        n_rois = masks.shape[0]
        if n_rois == 0:
            self.help_lbl.config(text="No ROIs to save.")
            return

        # save same ROIs into each folder
        for folder in self.folders:
            save_npz = folder / SAVE_NAME
            save_meta = folder / META_NAME
            np.savez(
                save_npz,
                ROIs=masks.astype(np.uint8),
                vertices=np.array(self.roi.roi_vertices, dtype=object)
            )
            meta = {
                "folders": [str(f) for f in self.folders],
                "base_folder": str(folder),
                "tiff_name": TIFF_NAME_MC,
                "shape": [H, W],
                "n_rois": int(n_rois),
                "vertices_per_roi": [len(v) for v in self.roi.roi_vertices],
            }
            with open(save_meta, "w") as f:
                json.dump(meta, f, indent=2)

        self.help_lbl.config(
            text=f"Saved {n_rois} ROI(s) to {SAVE_NAME} in each folder."
        )

    def on_clear(self):
        self.roi.clear()
        self.redraw()
        self.help_lbl.config(text="Cleared all ROIs (unsaved).")

    def on_undo_point(self):
        self.roi.undo_point()
        self.redraw()

    def on_undo_roi(self):
        self.roi.undo_roi()
        self.redraw()

    # Clean close
    def on_close(self):
        try:
            self.canvas.mpl_disconnect(self.cid_click)
            self.canvas.mpl_disconnect(self.cid_key)
        except Exception:
            pass
        try:
            self._plt.close(self.fig)
        except Exception:
            pass
        self.quit()
        self.destroy()

def launch_roi_gui(folders):
    """
    Launch the multi-session ROI GUI.

    folders: list of folder paths, each containing mean_image_mc.tif

    Example:
        import roi_gui_multi
        roi_gui_multi.launch_roi_gui([
            "/path/to/session1",
            "/path/to/session2",
        ])
    """
    import matplotlib
    if "tkagg" not in matplotlib.get_backend().lower():
        matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    app = _MultiROIGUI(folders, None, plt)
    app.mainloop()

if __name__ == "__main__":
    import sys
    folders = sys.argv[1:]
    if not folders:
        print("Usage: python roi_gui_multi.py /path/to/folder1 /path/to/folder2 ...")
    else:
        launch_roi_gui(folders)
