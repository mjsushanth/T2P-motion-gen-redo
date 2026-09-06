"""Compatibility shim: moviepy 1.0.3 (needed for its old .editor/mplfig_to_npimage API, which
MDM's vendored plot_script.py imports) calls FigureCanvasAgg.tostring_rgb(), which recent
matplotlib versions removed in favor of buffer_rgba(). Rather than downgrade matplotlib (shared
across other projects in this conda env) or patch the vendored MDM file, restore the old method
as a thin wrapper around the new one. Import this before anything that renders via plot_script.py.
"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

if not hasattr(FigureCanvasAgg, "tostring_rgb"):
    def _tostring_rgb(self):
        buf = np.asarray(self.buffer_rgba())
        return buf[:, :, :3].tobytes()

    FigureCanvasAgg.tostring_rgb = _tostring_rgb
