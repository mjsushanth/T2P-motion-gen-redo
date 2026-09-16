"""Generate the matplotlib figures for the T2P-Reboot slide deck.

Every figure is drawn straight from measured numbers or from raw per-observation
data recovered from a source notebook -- never from a fitted or smoothed curve
through the points, and never as fabricated individual observations standing in
for summary statistics. See each `fig0N_...` function for the specific numbers
and their source.

Before saving, every figure is passed through `check_collisions`, which draws
the figure and inspects the actual rendered bounding boxes of every text and
data-mark artist for pairwise overlap -- this is not an eyeball check.

Run:
    conda run -n mjs_mlcvdl_unified_m5 python deck/img/make_figures.py

Writes <name>.svg and <name>.png (150 dpi, transparent background) for every
figure into this directory.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# --------------------------------------------------------------------------
# Shared visual system
# --------------------------------------------------------------------------

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#eb6c36"
SECONDARY = "#4a7ba7"

OUT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_FIGSIZE = (6.0, 3.5)
DPI = 150


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "svg.fonttype": "none",
    })


def despine(ax: plt.Axes, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        spine = ax.spines[side]
        if side in keep:
            spine.set_linewidth(0.8)
            spine.set_color(INK)
        else:
            spine.set_visible(False)


def new_fig(figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# --------------------------------------------------------------------------
# Collision check -- empirical, not eyeballed
# --------------------------------------------------------------------------

@dataclass
class CollisionReport:
    name: str
    overlaps: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.overlaps


def _text_boxes(fig: Figure, renderer) -> list[tuple[Text, Bbox]]:
    boxes = []
    for artist in fig.findobj(match=lambda a: isinstance(a, Text)):
        if not artist.get_visible():
            continue  # e.g. a Tick's unused label2 -- never positioned, reports a bogus (0,0)-(1,1) bbox
        s = artist.get_text().strip()
        if not s:
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        boxes.append((artist, bbox))
    return boxes


@dataclass
class Mark:
    label: str
    bbox: Bbox              # quick-reject bounding box
    segment: tuple[tuple[float, float], tuple[float, float]] | None = None
    pad: float = 0.0        # only meaningful when `segment` is set


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _segment_intersects_rect(p0, p1, rect: Bbox, pad: float) -> bool:
    """True geometric test: does the actual line ink from p0 to p1 (thickened
    by `pad`) pass through `rect`? This is deliberately not just a bbox
    overlap -- the axis-aligned bounding box of a diagonal segment covers a
    lot of empty space that has no ink in it, which would make text near one
    endpoint of a line falsely 'collide' with that line even when the line
    itself runs off in a different direction."""
    r = Bbox([[rect.x0 - pad, rect.y0 - pad], [rect.x1 + pad, rect.y1 + pad]])
    if r.contains(*p0) or r.contains(*p1):
        return True
    corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(p0, p1, e0, e1) for e0, e1 in edges)


def _line_marks(line, renderer) -> list[Mark]:
    """Build precise marks for a Line2D: real line segments (checked via
    exact geometry, not their bounding box) plus small boxes around any
    markers."""
    path = line.get_path()
    if len(path.vertices) == 0:
        return []
    disp = line.get_transform().transform(path.vertices)
    lw = line.get_linewidth() or 0.0
    ms = line.get_markersize() or 0.0
    pad = max(lw, ms) / 2.0 + 1.0
    linestyle = line.get_linestyle()
    has_line = linestyle not in (None, "None", "none", "")
    marker = line.get_marker()
    has_marker = marker not in (None, "None", "none", "")
    label = line.get_label()

    marks = []
    if has_line:
        for p0, p1 in zip(disp[:-1], disp[1:]):
            x0, x1 = sorted([p0[0], p1[0]])
            y0, y1 = sorted([p0[1], p1[1]])
            quick_bbox = Bbox([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]])
            marks.append(Mark(f"line:{label}", quick_bbox, segment=(tuple(p0), tuple(p1)), pad=pad))
    if has_marker or not has_line:
        for p in disp:
            marks.append(Mark(f"line:{label}", Bbox([[p[0] - pad, p[1] - pad], [p[0] + pad, p[1] + pad]])))
    return marks


def _all_marks(fig: Figure, renderer) -> list[Mark]:
    """Data marks: plotted lines, error bars, scatter points, and small
    (non-background) patches such as bars. Full-span axvspan/axhspan
    rectangles are excluded -- text is deliberately placed inside those."""
    marks: list[Mark] = []
    for ax in fig.axes:
        ax_bbox = ax.get_window_extent(renderer)
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            marks.extend(_line_marks(line, renderer))
        for coll in ax.collections:
            if not coll.get_visible():
                continue
            bbox = coll.get_window_extent(renderer)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            marks.append(Mark(f"collection:{type(coll).__name__}", bbox))
        for patch in ax.patches:
            if not patch.get_visible():
                continue
            bbox = patch.get_window_extent(renderer)
            covers_height = bbox.height >= 0.85 * ax_bbox.height
            covers_width = bbox.width >= 0.85 * ax_bbox.width
            if covers_height or covers_width:
                continue  # background shading (axvspan/axhspan), not a data mark
            marks.append(Mark(f"patch:{type(patch).__name__}", bbox))
    return marks


def check_collisions(fig: Figure, name: str, pad: float = -1.0) -> CollisionReport:
    """pad < 0 shrinks boxes slightly before testing, so light kissing of
    bounding boxes (which always have a little font-metric slack) doesn't
    register as a false positive -- only real overlap does."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    report = CollisionReport(name=name)

    texts = _text_boxes(fig, renderer)
    marks = _all_marks(fig, renderer)

    def shrink(b: Bbox) -> Bbox:
        return b.expanded(1.0, 1.0).padded(pad)

    for i, (t1, b1) in enumerate(texts):
        for t2, b2 in texts[i + 1:]:
            if shrink(b1).overlaps(shrink(b2)):
                report.overlaps.append(
                    f"TEXT '{t1.get_text()!r}' overlaps TEXT '{t2.get_text()!r}'"
                )
        for mark in marks:
            if not shrink(b1).overlaps(mark.bbox):
                continue  # cheap reject before the precise geometric test
            if mark.segment is not None:
                if _segment_intersects_rect(mark.segment[0], mark.segment[1], b1, mark.pad):
                    report.overlaps.append(f"TEXT '{t1.get_text()!r}' overlaps MARK {mark.label}")
            else:
                if shrink(b1).overlaps(shrink(mark.bbox)):
                    report.overlaps.append(f"TEXT '{t1.get_text()!r}' overlaps MARK {mark.label}")

    return report


def save_fig(fig: Figure, name: str) -> None:
    report = check_collisions(fig, name)
    if not report.clean:
        print(f"[{name}] COLLISIONS DETECTED:")
        for line in report.overlaps:
            print(f"    - {line}")
    else:
        print(f"[{name}] no collisions detected")

    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(svg_path, transparent=True, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg_path.name} and {png_path.name}")


# --------------------------------------------------------------------------
# FIG 5 -- published vs measured reproduction
# --------------------------------------------------------------------------

def fig05_reproduction() -> None:
    fig, ax = new_fig((6.0, 2.6))

    y_published, y_measured = 1, 0
    published = (0.611, 0.007)
    measured = (0.6172, 0.0072)

    ax.errorbar(
        published[0], y_published, xerr=published[1],
        fmt="o", color=MUTED, ecolor=MUTED, elinewidth=1.2, capsize=4,
        markersize=6, markeredgewidth=0,
    )
    ax.errorbar(
        measured[0], y_measured, xerr=measured[1],
        fmt="o", color=ACCENT, ecolor=ACCENT, elinewidth=1.4, capsize=4,
        markersize=7, markeredgewidth=0,
    )

    ax.set_yticks([y_published, y_measured])
    ax.set_yticklabels(["Published (MDM paper)", "This harness (n=4,640)"])
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlim(0.59, 0.64)
    ax.set_xlabel("R-Precision top-3  (error bars: ±1 SE)")

    # Convention-free separation annotation: state the gap itself and compare
    # it directly to the published interval's own half-width, rather than to
    # any sigma statistic. (A one-sample z against the published SE alone
    # gives 0.87 sigma; the conventional two-sample statistic, pooling both
    # SEs in quadrature, gives 0.62 sigma -- a weaker number to defend either
    # way, and the more standard one is the more favourable one. Simplest to
    # just show the two lengths side by side and let the reader compare.)
    gap = measured[0] - published[0]

    y_gap = 0.62
    ax.annotate(
        "", xy=(measured[0], y_gap), xytext=(published[0], y_gap),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8),
    )
    ax.text(
        (published[0] + measured[0]) / 2, y_gap + 0.14, f"gap = {gap:.4f}",
        ha="center", va="bottom", fontsize=9, color=INK,
    )

    y_ref = 0.40
    ax.annotate(
        "", xy=(published[0] + published[1], y_ref), xytext=(published[0], y_ref),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8),
    )
    ax.text(
        published[0] + published[1] / 2, y_ref - 0.13,
        f"published's own ±{published[1]:.3f}",
        ha="center", va="top", fontsize=8, color=MUTED,
    )

    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_fig(fig, "fig05_reproduction")


# --------------------------------------------------------------------------
# FIG 6 -- CLIP spatial vs non-spatial separation
# --------------------------------------------------------------------------
# Raw per-pair cosine similarities recovered by re-running the exact,
# deterministic code path of notebooks/01_clip_spatial_blindness.ipynb
# (cells 3, 4, 7, 8: frozen CLIP ViT-B/32 text encoder, the same hardcoded
# pair lists, no randomness). The recomputed mean/std/min/max matched the
# notebook's own printed summary stats to 4 decimal places for all three
# groups, so these are treated as the real n=40 observations, not a
# reconstruction.

SPATIAL_SIMS = [
    0.969907, 0.963046, 0.942892, 0.975006, 0.990952, 0.948881, 0.979132,
    0.972268, 0.933042, 0.947801, 0.983486, 0.963720, 0.949743, 0.985809,
    0.968984, 0.971174, 0.989613, 0.974547, 0.989170, 0.969158, 0.989257,
    0.978891, 0.971691, 0.986644, 0.949460, 0.974268, 0.972893, 0.971330,
    0.972158, 0.951520, 0.970788, 0.985302, 0.978901, 0.965677, 0.993212,
    0.943945, 0.966243, 0.976331, 0.982580, 0.979062,
]

NONSPATIAL_VERB_SIMS = [
    0.924283, 0.935430, 0.947029, 0.927303, 0.956700, 0.967613, 0.955411,
    0.911475, 0.951591, 0.877799, 0.891227, 0.976189, 0.897796, 0.875941,
    0.928063, 0.948958, 0.863042, 0.931526, 0.884109, 0.958294, 0.880041,
    0.957639, 0.816788, 0.929229, 0.947722, 0.857965, 0.901351, 0.906644,
    0.970597, 0.989883, 0.855167, 0.894019, 0.908333, 0.925052, 0.980434,
    0.926208, 0.908058, 0.949315, 0.875634, 0.932573,
]

NONSPATIAL_MODIFIER_SIMS = [
    0.989004, 0.967613, 0.955059, 0.939414, 0.813143, 0.961333, 0.922767,
    0.972472, 0.916799, 0.932668, 0.949884, 0.947255, 0.949761, 0.963635,
    0.979000, 0.953402, 0.829518, 0.902718, 0.977100, 0.946284, 0.947934,
    0.969117, 0.953873, 0.941827, 0.985002, 0.969255, 0.955950, 0.933676,
    0.973251, 0.924595, 0.950945, 0.934697, 0.939574, 0.976513, 0.929547,
    0.940088, 0.942155, 0.927537, 0.940552, 0.964323,
]

UNRELATED_TOPICS_ANCHOR = 0.7190


def fig06_clip_separation() -> None:
    fig, ax = new_fig((7.4, 4.0))

    rng = np.random.default_rng(0)  # jitter only -- every y-value plotted is a real datum
    groups = [
        ("Spatial\n(n=40)", np.array(SPATIAL_SIMS), ACCENT),
        ("Non-spatial\nverb (n=40)", np.array(NONSPATIAL_VERB_SIMS), SECONDARY),
        ("Non-spatial\nmodifier (n=40)", np.array(NONSPATIAL_MODIFIER_SIMS), MUTED),
    ]

    for i, (label, values, color) in enumerate(groups):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter, values,
            s=14, color=color, alpha=0.65, linewidths=0, zorder=2,
        )
        ax.plot(
            [i - 0.22, i + 0.22], [values.mean(), values.mean()],
            color=INK, linewidth=2.0, zorder=3,
        )

    ax.axhline(
        UNRELATED_TOPICS_ANCHOR, color=MUTED, linestyle=":", linewidth=1.0, zorder=1,
    )
    ax.text(
        -0.55, UNRELATED_TOPICS_ANCHOR + 0.012, "two unrelated topics", fontsize=8,
        color=MUTED, va="bottom", ha="left",
    )

    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("cosine similarity within pair")

    ax.text(
        0.0, -0.28,
        "encoding: real per-pair values recovered from "
        "notebooks/01_clip_spatial_blindness.ipynb (n=40/group); black bar = group mean",
        transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="top",
    )

    despine(ax)
    fig.tight_layout()
    save_fig(fig, "fig06_clip_separation")


# --------------------------------------------------------------------------
# FIG 7 -- does removing pooling recover the distinction
# --------------------------------------------------------------------------

def fig07_pooling() -> None:
    fig, ax1 = new_fig((6.6, 3.8))

    labels = ["pooled\n(what the model uses)", "mean over\ntokens", "most-divergent\ntoken"]
    raw_gap = [0.0265, 0.0185, 0.1106]
    cohens_d = [0.995, 0.589, 0.724]
    x = np.arange(len(labels))

    # Accent marks the pooled representation, not the tallest bar: pooled is
    # what the generator actually conditions on, which is what makes this
    # probe relevant to the model rather than an off-label property of CLIP.
    # The other two bars are diagnostic comparisons, not the headline claim.
    bar_colors = [ACCENT, MUTED, MUTED]
    ax1.bar(x, raw_gap, width=0.5, color=bar_colors, zorder=2)
    ax1.set_ylabel("raw gap in cosine similarity\n(spatial minus non-spatial)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 0.15)

    ax2 = ax1.twinx()
    ax2.scatter(x, cohens_d, color=SECONDARY, marker="D", s=45, zorder=3, label="Cohen's d")
    ax2.set_ylabel("Cohen's d (effect size)", color=SECONDARY)
    ax2.tick_params(axis="y", colors=SECONDARY)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlim(-0.6, len(labels) - 0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(SECONDARY)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["left"].set_visible(False)

    for xi, g in zip(x, raw_gap):
        ax1.text(xi, g + 0.006, f"{g:+.4f}", ha="center", fontsize=8, color=INK)

    # The bar (ax1 scale) and the d-marker (ax2 scale) share the same pixel
    # column but not the same numeric scale, so "above the marker" can still
    # land inside a tall bar. Compare both in fraction-of-axis-height terms
    # and clear whichever is taller.
    bar_top_frac = [g / ax1.get_ylim()[1] for g in raw_gap]
    for i, (xi, g, d) in enumerate(zip(x, raw_gap, cohens_d)):
        d_frac = d / ax2.get_ylim()[1]
        if bar_top_frac[i] > d_frac:
            # the bar is taller than the marker here: put the label above the
            # bar's own top (converted to ax2 units), not above the marker
            y = (bar_top_frac[i] + 0.16) * ax2.get_ylim()[1]
            ax2.text(xi, y, f"d={d:+.3f}", ha="center", va="bottom", fontsize=8, color=SECONDARY)
        else:
            ax2.text(xi + 0.28, d, f"d={d:+.3f}", ha="left", va="center", fontsize=8, color=SECONDARY)

    ax1.text(
        0.0, 1.16,
        "all three clear the Bonferroni threshold (p < 0.0167)",
        transform=ax1.transAxes, fontsize=8, color=INK, ha="left",
    )

    despine(ax1, keep=("bottom", "left"))
    fig.tight_layout()
    save_fig(fig, "fig07_pooling")


# --------------------------------------------------------------------------
# FIG 8 -- measured effect against its own noise floor
# --------------------------------------------------------------------------

def fig08_effect_vs_floor() -> None:
    fig, ax = new_fig((6.0, 2.6))

    estimate = 0.0070
    se = 0.0147
    ci = 1.96 * se

    ax.axvspan(-se, se, color=MUTED, alpha=0.25, zorder=1)
    ax.text(
        -0.013, 0.68, "noise floor (±1 SE)", ha="right", fontsize=7.5,
        color=MUTED, transform=ax.get_xaxis_transform(),
    )

    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    ax.text(
        0.006, 0.95, "zero effect", ha="left", fontsize=8, color=INK,
        transform=ax.get_xaxis_transform(),
    )

    ax.errorbar(
        estimate, 0, xerr=ci, fmt="o", color=ACCENT, ecolor=ACCENT,
        elinewidth=1.6, capsize=5, markersize=7, zorder=3,
    )
    # The shaded band is +/-1 SE; this error bar is a 95% CI (+/-1.96 SE) --
    # a different, wider quantity, and the figure must not let the two get
    # silently conflated.
    ax.annotate(
        "95% CI", xy=(estimate + ci, 0), xytext=(10, 0), textcoords="offset points",
        ha="left", va="center", fontsize=8, color=ACCENT,
    )

    ax.set_xlim(-0.045, 0.05)
    ax.set_ylim(-1, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("difference in R-Precision top-3 (spatial minus non-spatial)")

    despine(ax, keep=("bottom",))
    fig.tight_layout()
    save_fig(fig, "fig08_effect_vs_floor")


# --------------------------------------------------------------------------
# FIG 9 -- two evaluators, opposite signs (highest-value figure)
# --------------------------------------------------------------------------

def fig09_two_evaluators() -> None:
    fig, ax = new_fig((6.6, 3.0))

    guo = dict(y=1, estimate=0.0070, se=0.0147, z=0.48, label="Guo evaluator (primary)")
    tmr = dict(y=0, estimate=-0.0263, se=0.0135, z=-1.95, label="TMR evaluator (independent)")

    for row, color in [(guo, INK), (tmr, ACCENT)]:
        ci = 1.96 * row["se"]
        ax.errorbar(
            row["estimate"], row["y"], xerr=ci, fmt="o", color=color, ecolor=color,
            elinewidth=1.6, capsize=5, markersize=7, zorder=3,
        )
        # offset the z-annotation away from x=0 so it never sits on the zero line
        x_off = 0.012 if row["estimate"] >= 0 else -0.012
        ha = "left" if row["estimate"] >= 0 else "right"
        ax.text(
            row["estimate"] + x_off, row["y"] + 0.3, f"z = {row['z']:+.2f}",
            ha=ha, fontsize=8.5, color=color,
        )

    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)

    ax.set_yticks([guo["y"], tmr["y"]])
    ax.set_yticklabels([guo["label"], tmr["label"]])
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlim(-0.08, 0.06)
    ax.set_xlabel("difference in R-Precision top-3 (spatial minus non-spatial)")

    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_fig(fig, "fig09_two_evaluators")


# --------------------------------------------------------------------------
# FIG 10 -- caption length interaction
# --------------------------------------------------------------------------

def fig10_terciles() -> None:
    fig, ax = new_fig((6.0, 3.8))

    spatial_x = [7.35, 12.51, 23.04]
    spatial_y = [0.5089, 0.5871, 0.5982]
    spatial_n = [902, 902, 904]

    nonspatial_x = [5.89, 8.77, 16.00]
    nonspatial_y = [0.5719, 0.5437, 0.5687]
    nonspatial_n = [644, 644, 644]

    ax.plot(
        spatial_x, spatial_y, marker="o", color=ACCENT, linewidth=1.0,
        markersize=6, zorder=3, label="spatial",
    )
    ax.plot(
        nonspatial_x, nonspatial_y, marker="o", color=SECONDARY, linewidth=1.0,
        markersize=6, zorder=2, label="non-spatial",
    )

    # per-point offsets: default is straight above/below, but a couple of
    # points sit right where a steep segment (their own line's next leg, or
    # the other series passing close by) would otherwise run through the
    # label, so those are nudged sideways as well as vertically
    spatial_offsets = [(-6, 14), (0, 12), (0, 12)]
    # point index 1 (8.77, 0.5437) is the non-spatial line's local minimum,
    # sitting close enough to the spatial line passing beneath it that a
    # below-point label would cross that other line; above clears both
    nonspatial_offsets = [(-6, -16), (0, 16), (0, -14)]

    for (x, y, n), (dx, dy) in zip(zip(spatial_x, spatial_y, spatial_n), spatial_offsets):
        ha = "right" if dx < 0 else "center"
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=7.5, color=ACCENT)
    for (x, y, n), (dx, dy) in zip(zip(nonspatial_x, nonspatial_y, nonspatial_n), nonspatial_offsets):
        ha = "right" if dx < 0 else "center"
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=7.5, color=SECONDARY)

    ax.set_xlabel("mean caption length (words), tercile")
    ax.set_ylabel("R-Precision top-3")
    ax.set_ylim(0.49, 0.615)
    ax.set_yticks([0.50, 0.52, 0.54, 0.56, 0.58, 0.60])  # explicit: avoids the
    # default locator placing a tick just below ylim (e.g. 0.48) whose label
    # then renders below the axis and collides with the x-axis label
    ax.legend(frameon=False, loc="lower right")

    despine(ax)
    fig.tight_layout()
    save_fig(fig, "fig10_terciles")


# --------------------------------------------------------------------------
# FIG 11 -- FID floor versus sample size
# --------------------------------------------------------------------------

def fig11_fid_floor() -> None:
    fig, ax = new_fig((6.0, 3.6))

    n_values = [941, 1378, 2320]
    means = [0.2573, 0.1791, 0.0957]
    stds = [0.1170, 0.0643, 0.0338]

    ax.errorbar(
        n_values, means, yerr=stds, fmt="o", color=ACCENT, ecolor=ACCENT,
        elinewidth=1.4, capsize=5, markersize=7, linestyle="none", zorder=3,
    )

    ax.set_xlabel("samples per side")
    ax.set_ylabel("real-vs-real FID floor")
    ax.set_xlim(600, 2650)
    ax.set_ylim(0, 0.45)

    # placed in the empty region to the right, clear of every point and error bar
    ax.text(
        1750, 0.40, "30 trials per point",
        fontsize=8, color=MUTED, ha="left", va="top",
    )
    ax.text(
        1750, 0.34,
        "points and error bars only -- no fitted curve",
        fontsize=7, color=MUTED, ha="left", va="top",
    )

    despine(ax)
    fig.tight_layout()
    save_fig(fig, "fig11_fid_floor")


# --------------------------------------------------------------------------
# FIG 12 -- what was excluded and what remains open
# --------------------------------------------------------------------------

def fig12_exclusion() -> None:
    fig, ax = new_fig((6.6, 2.8))

    ax.axvspan(0.117, 0.25, color=MUTED, alpha=0.28, zorder=1)
    ax.axvspan(0.175, 0.25, color=MUTED, alpha=0.35, zorder=2)

    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("true effect size in R-Precision top-3")

    ax.text(0.0585, 0.5, "still open", ha="center", va="center", fontsize=9, color=INK)
    # kept clear of the retrieval-effect line at x=0.145 by sitting left of it
    ax.text(0.128, 0.82, "excluded\nat 2σ", ha="center", va="center", fontsize=8, color=INK)
    ax.text(0.2125, 0.5, "excluded\nat 3σ", ha="center", va="center", fontsize=8, color=INK)

    retrieval_effect = 0.145
    ax.axvline(retrieval_effect, color=ACCENT, linewidth=1.4, zorder=3)
    ax.annotate(
        f"retrieval-space effect\n(motivating): {retrieval_effect:.3f}",
        xy=(retrieval_effect, 1.0), xytext=(retrieval_effect, 1.3),
        ha="center", fontsize=8, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8),
        annotation_clip=False,
    )

    despine(ax, keep=("bottom",))
    fig.tight_layout()
    save_fig(fig, "fig12_exclusion")


def main() -> None:
    apply_style()
    fig05_reproduction()
    fig06_clip_separation()
    fig07_pooling()
    fig08_effect_vs_floor()
    fig09_two_evaluators()
    fig10_terciles()
    fig11_fid_floor()
    fig12_exclusion()


if __name__ == "__main__":
    main()
