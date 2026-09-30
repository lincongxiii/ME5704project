"""Shared figure helpers (English labels, report-ready PNGs).

Colours: pressure uses a one-hue sequential blue ramp (light = low, dark = high); the p = 0
contour is red (solid) and p = 50 black (dashed), both with a white halo; text stays in neutral ink.
Any model object with ``predict(x, y)`` can be drawn.
"""
import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are only written to files
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import data

# ---- palette ------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RED = "#e34948"
SEQ = LinearSegmentedColormap.from_list(
    "pressure_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
P_MIN, P_MAX = 0.0, 60.0          # fixed colour range so all figures are directly comparable
Z_CLIP = (-20.0, 90.0)            # display range of the 3-D axis
PLOT_GRID = 161                   # grid resolution of the contour maps
HALO = [pe.withStroke(linewidth=4, foreground="white")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    "hatch.color": RED, "hatch.linewidth": 0.7,
})


def _predict(model, gx, gy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(gx, gy)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_contour(ax, model, title, colorbar=True, small=False, hull_max=None):
    """Filled contour map over [0, 2.5]^2 with sensors, hull, p = 0 / p = 50 contours, hatched p <= 0
    area and (``hull_max`` = (value, x, y)) a star at the maximum inside the hull."""
    gx, gy = data.region_grid(PLOT_GRID)
    z = _predict(model, gx, gy)
    zc = np.clip(np.where(np.isfinite(z), z, np.nan), P_MIN, P_MAX)
    cf = ax.contourf(gx, gy, np.ma.masked_invalid(zc), levels=np.linspace(P_MIN, P_MAX, 13), cmap=SEQ,
                     extend="both")
    inside = data.in_hull(gx, gy)
    ax.contourf(gx, gy, (~inside).astype(float), levels=[0.5, 1.5], colors=["#ffffff"], alpha=0.55)  # veil
    ax.contour(gx, gy, np.ma.masked_invalid(z), levels=np.arange(10, 51, 10), colors=[INK2],
               linewidths=0.4, alpha=0.6)
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    has0 = np.isfinite(zmin) and zmin <= data.P_ZERO < zmax
    has50 = np.isfinite(zmax) and zmin < data.P_LIMIT < zmax
    if has0:
        zm = np.ma.masked_invalid(z)
        low = [np.nanmin(z) - 1.0, data.P_ZERO]
        ax.contourf(gx, gy, zm, levels=low, colors=[RED], alpha=0.12)             # p <= 0 area
        ax.contourf(gx, gy, zm, levels=low, colors="none", hatches=["////"])      # hatched on top
        if hasattr(model, "ellipse"):  # Hertz: p = 0 on a whole area; its boundary is the exact ellipse
            x0, y0, a, b = model.ellipse()
            t = np.linspace(0, 2 * np.pi, 4000)
            ax.plot(x0 + a * np.cos(t), y0 + b * np.sin(t), color=RED, lw=2.2, path_effects=HALO)
        else:
            ax.contour(gx, gy, zm, levels=[data.P_ZERO], colors=[RED], linewidths=2.2).set_path_effects(HALO)
    if has50:
        ax.contour(gx, gy, np.ma.masked_invalid(z), levels=[data.P_LIMIT], colors=[INK], linewidths=1.8,
                   linestyles="--").set_path_effects(HALO)
    poly = data.hull_polygon()
    ax.plot(poly[:, 0], poly[:, 1], color=INK, lw=1.2)
    ax.scatter(data.X, data.Y, s=46 if not small else 26, c=INK, edgecolor="white", linewidth=1.2, zorder=5)
    for name, x, y, p in zip(data.NAMES, data.X, data.Y, data.P):
        right_edge = x > 0.8 * data.REGION[1]  # label to the left of sensors near the right edge
        ax.annotate(f"{name} ({p:g})", (x, y), xytext=(-5, 5) if right_edge else (5, 5),
                    ha="right" if right_edge else "left", textcoords="offset points",
                    fontsize=6.5 if small else 8, color=INK, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    if hull_max is not None:
        ax.scatter([hull_max[1]], [hull_max[2]], marker="*", s=150 if not small else 80, c=ORANGE,
                   edgecolor=INK, linewidth=0.8, zorder=7)
    ax.set_xlim(data.REGION[0], data.REGION[1])
    ax.set_ylim(data.REGION[2], data.REGION[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5 if small else 11)
    ax.set_xlabel("x", fontsize=8 if small else 10)
    ax.set_ylabel("y", fontsize=8 if small else 10)
    if small:
        ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color=RED if has0 else MUTED, lw=2, ls="-" if has0 else ":"),
               Line2D([0], [0], color=INK if has50 else MUTED, lw=1.6, ls="--" if has50 else ":"),
               Line2D([0], [0], color=INK, lw=1.2), Patch(facecolor="white", alpha=0.55, edgecolor=MUTED)]
    labels = ["p = 0" + ("" if has0 else " (not reached)"), "p = 50" + ("" if has50 else " (not reached)"),
              "Convex hull", "Extrapolation"]
    if has0:
        handles.append(Patch(facecolor=RED, alpha=0.25, hatch="////", edgecolor=RED))
        labels.append("p <= 0")
    if hull_max is not None:
        handles.append(Line2D([0], [0], marker="*", ls="", ms=11, mfc=ORANGE, mec=INK))
        labels.append(f"Max in hull ({hull_max[0]:.3g})")
    if not small:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)
    if colorbar:
        cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, ticks=np.arange(0, 61, 10))
        cb.set_label("Pressure p" + (" (clipped to 0-60)" if (zmin < P_MIN or zmax > P_MAX) else ""), fontsize=8)
    return cf


def draw_surface(ax, model, title):
    """3-D surface; the part outside the hull (extrapolation) is drawn semi-transparent."""
    gx, gy = data.region_grid(151)
    z = _predict(model, gx, gy)
    clipped = bool(np.nanmin(z) < Z_CLIP[0] or np.nanmax(z) > Z_CLIP[1])
    zp = np.clip(z, *Z_CLIP)
    rgba = SEQ(np.clip((np.nan_to_num(zp, nan=P_MIN) - P_MIN) / (P_MAX - P_MIN), 0, 1))
    rgba[..., 3] = np.where(data.in_hull(gx, gy), 1.0, 0.35)
    rgba[~np.isfinite(z)] = 0.0
    ax.plot_surface(gx, gy, np.where(np.isfinite(z), zp, np.nan), facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    for x, y, p in zip(data.X, data.Y, data.P):
        ax.plot([x, x], [y, y], [Z_CLIP[0], p], color=MUTED, lw=0.8, ls=":")
    ax.scatter(data.X, data.Y, data.P, s=36, c=INK, edgecolor="white", depthshade=False)
    poly = data.hull_polygon()
    ax.plot(poly[:, 0], poly[:, 1], np.full(len(poly), Z_CLIP[0]), color=INK, lw=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("p")
    ax.set_zlim(*Z_CLIP)
    ax.view_init(elev=26, azim=-58)
    ax.set_title(title + ("\n(z clipped to [%g, %g] for display)" % Z_CLIP if clipped else ""))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURFACE)
        a.pane.set_edgecolor(GRID)


def plot_hertz(model, info, path):
    """Hertz surface in the hand region and the fitted contact ellipse (``info``: text lines)."""
    x0, y0, a, b = model.ellipse()
    t = np.linspace(0, 2 * np.pi, 400)
    ex, ey = x0 + a * np.cos(t), y0 + b * np.sin(t)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    draw_contour(axes[0], model, "Hertz surface in the hand region", colorbar=True)
    ax = axes[1]
    ax.plot(ex, ey, color=RED, lw=2, label="Contact ellipse (p = 0)")
    ax.fill(ex, ey, color="#fde7dc", zorder=0)
    ax.add_patch(plt.Rectangle((0, 0), 2.5, 2.5, fill=False, ec=INK, lw=1.2, label="Hand region [0,2.5]^2"))
    ax.scatter(data.X, data.Y, s=30, c=INK, zorder=5, label="Sensors")
    ax.scatter([x0], [y0], marker="+", s=120, c=ORANGE, zorder=5, label="Ellipse centre")
    # an elongated ellipse would be a thin sliver at equal scale: axes are NOT to scale here
    ax.set_xlim(min(ex.min(), 0) - 0.3, max(ex.max(), 2.5) + 0.3)
    ax.set_ylim(min(ey.min(), 0) - 1, max(ey.max(), 2.5) + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y  (axes not to scale)")
    ax.set_title("Fitted contact ellipse (zoomed out)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.5, -0.02, "\n".join(info), ha="center", va="top", fontsize=9, color=INK2)
    fig.suptitle("Hertz contact model", fontsize=12)
    fig.tight_layout()
    save(fig, path)
