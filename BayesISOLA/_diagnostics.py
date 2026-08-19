"""Diagnostic summaries and plotting utilities for automated BayesISOLA runs.

The functions in this module operate on the structured mapping returned by
:func:`BayesISOLA.workflows.run_auto_cmt`.  They do not modify the inversion,
posterior, covariance matrix, or preferred moment tensor.  The plotting layer is
kept separate from :mod:`BayesISOLA.workflows` so workflow orchestration remains
focused on acquisition, Green-function preparation, inversion, and result
construction.

Depth and centroid-time plots use exact marginalization of the discrete
space-time posterior.  Moment-tensor uncertainty plots use the conditional
Gaussian samples stored in ``run["results"]["uncertainty"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

__all__ = [
    "plot_cmt_summary",
    "plot_posterior_summary",
    "plot_adaptive_history",
    "plot_station_qc",
    "summarize_uncertainty",
    "plot_uncertainty_summary",
    "plot_station_fit_summary",
]

PLOT_RC = {
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11,
}


def _require_dataframe(container, key, *, where="run", allow_empty=False):
    """Return a required DataFrame and raise a targeted error when unavailable."""
    if not isinstance(container, Mapping):
        raise TypeError(f"{where} must be a mapping; got {type(container).__name__}.")
    value = container.get(key)
    if not isinstance(value, pd.DataFrame):
        raise TypeError(
            f"{where}[{key!r}] must be a pandas DataFrame; "
            f"got {type(value).__name__}."
        )
    if not allow_empty and value.empty:
        raise ValueError(f"{where}[{key!r}] is empty.")
    return value


def _require_columns(df, columns: Sequence[str], *, name: str):
    """Validate the columns required by one diagnostic without mutating ``df``."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _get_results(run):
    """Return ``run['results']`` after validating the workflow mapping."""
    if not isinstance(run, Mapping):
        raise TypeError(
            "Diagnostic plotting requires the mapping returned by run_auto_cmt; "
            f"got {type(run).__name__}."
        )
    results = run.get("results")
    if not isinstance(results, Mapping):
        raise TypeError("run['results'] must be a mapping.")
    return results

def _posterior_marginal(df, group_cols):
    """Marginalize normalized log posterior mass over discrete grid columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Posterior-cell table containing ``log_posterior`` and every column in
        ``group_cols``. ``log_posterior`` may be normalized or unnormalized;
        the returned marginal is normalized explicitly in log space.
    group_cols : sequence of str
        Columns defining the retained nonlinear coordinate(s).

    Returns
    -------
    pandas.DataFrame
        One row per unique group with ``log_probability`` and normalized
        ``posterior_probability`` columns.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Posterior-cell table must be a non-empty DataFrame.")

    group_cols = list(group_cols)
    _require_columns(df, [*group_cols, "log_posterior"], name="posterior-cell table")

    logp = pd.to_numeric(df["log_posterior"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(logp).any():
        raise ValueError("Posterior-cell table contains no finite log-posterior values.")

    table = df.loc[np.isfinite(logp), [*group_cols, "log_posterior"]].copy()
    table["log_posterior"] = pd.to_numeric(table["log_posterior"], errors="coerce")

    out = (
        table.groupby(group_cols, sort=True, dropna=False)["log_posterior"]
        .agg(lambda x: logsumexp(x.to_numpy(dtype=float)))
        .reset_index(name="log_probability")
    )
    normalization = logsumexp(out["log_probability"].to_numpy(dtype=float))
    if not np.isfinite(normalization):
        raise ValueError("Posterior marginal could not be normalized.")

    out["posterior_probability"] = np.exp(
        out["log_probability"].to_numpy(dtype=float) - normalization
    )
    return out



def _discrete_interval(marginal, value_col, probs=(0.05, 0.50, 0.95)):
    """Return quantiles from a discrete one-dimensional posterior marginal."""
    if not isinstance(marginal, pd.DataFrame) or marginal.empty:
        raise ValueError("marginal must be a non-empty pandas DataFrame.")
    _require_columns(marginal, [value_col, "posterior_probability"], name="marginal")

    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1 or np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probs must contain finite probabilities in [0, 1].")

    values = pd.to_numeric(marginal[value_col], errors="coerce").to_numpy(dtype=float)
    probability = pd.to_numeric(
        marginal["posterior_probability"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(values) & np.isfinite(probability) & (probability >= 0.0)
    values = values[valid]
    probability = probability[valid]

    if values.size == 0:
        raise ValueError(f"No valid values available for {value_col!r}.")

    total_probability = probability.sum()
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        raise ValueError(f"Posterior probability for {value_col!r} does not have positive finite mass.")

    order = np.argsort(values)
    values = values[order]
    probability = probability[order]

    cdf = np.cumsum(probability)
    cdf /= cdf[-1]

    out = []
    for q in probs:
        idx = min(np.searchsorted(cdf, q, side="left"), len(values) - 1)
        out.append(values[idx])
    return tuple(out)



def _sample_interval(df, col, probs=(0.05, 0.50, 0.95)):
    """Return finite-sample quantiles for one uncertainty-sample column."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Uncertainty samples must be a non-empty pandas DataFrame.")
    _require_columns(df, [col], name="uncertainty samples")

    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1 or np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probs must contain finite probabilities in [0, 1].")

    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"No valid uncertainty samples available for {col!r}.")
    return tuple(np.quantile(x, probs))



def _angular_difference(values, reference):
    """
    Return the shortest signed angular difference from a reference angle.

    The result lies in [-180, 180) degrees and is suitable for both
    strike and rake uncertainty relative to a preferred solution.
    """
    values = np.asarray(values, dtype=float)
    reference = float(reference)
    return (values - reference + 180.0) % 360.0 - 180.0


def plot_cmt_summary(
    summary,
    centroid,
    uncertainty=None,
    *,
    tensor_mode="full",
    facecolor="white",
    bgcolor="red",
    show_dc_overlay=True,
    figsize=(11.5, 6.2),
    dpi=300,
    output_file=None,
    show=False,
):
    """
    Plot a publication-style summary of a BayesISOLA CMT solution.

    This is the operational replacement for BayesISOLA's basic single-beachball
    figure in the helper's ``summary`` plot preset. It does not alter the native
    inversion or decomposition. The figure combines the preferred solution with
    a zero-trace beachball, P/T labels, the DC nodal-plane overlay, source and
    quality metrics, moment-tensor components, nodal planes and principal axes.

    BayesISOLA's native ``plot_MT`` passes the preferred six-component tensor to
    ObsPy ``beach`` without overriding ``plot_zerotrace``; ObsPy therefore plots
    the zero-trace/deviatoric representation even when ``deviatoric=False`` was
    used for a full six-component inversion. This helper follows that zero-trace
    plotting convention explicitly. For ``tensor_mode='full'`` the source table
    and component table still report the full six-component solution, including
    ISO, while the beachball and principal axes represent its zero-trace part.
    For ``tensor_mode='deviatoric'`` the inversion itself is five-component and
    ISO is reported as constrained to zero.

    The beachball fill colours are controlled independently through ``facecolor``
    and ``bgcolor``. Their defaults are white and red, respectively, providing
    the preferred visual polarity convention for this summary figure without
    altering or negating the moment tensor.

    Parameters
    ----------
    summary : mapping, pandas.Series, or one-row pandas.DataFrame
        BayesISOLA solution summary, normally ``results["summary"]``. Required
        fields are ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``, ``Mtp``,
        ``M0_Nm``, ``Mw``, ``DC_percent``, ``CLVD_percent``, ``ISO_percent``,
        ``variance_reduction``, ``condition_number`` and both NP1/NP2
        strike/dip/rake triplets. Tensor components are in the USE
        (Up-South-East; r-theta-phi) convention and moments are in N·m.

    centroid : mapping, pandas.Series, or one-row pandas.DataFrame
        Preferred centroid information, normally ``results["centroid"]``.
        ``centroid_depth_km`` is required.

    uncertainty : optional
        Retained for backward call compatibility and intentionally ignored by
        the deterministic 0.2 CMT summary. Posterior/MT uncertainty is reported
        separately by :func:`summarize_uncertainty` and
        :func:`plot_uncertainty_summary`.

    tensor_mode : {"full", "deviatoric"}
        Inversion type that produced ``summary``. ``"full"`` denotes the native
        six-component inversion, whereas ``"deviatoric"`` denotes the
        five-component zero-trace inversion. This controls source reporting and
        decomposition interpretation. The beachball itself is plotted zero-trace
        in both modes, matching BayesISOLA/ObsPy's native tensor-display
        convention; its colour assignment is controlled separately by
        ``facecolor`` and ``bgcolor``.

    facecolor : matplotlib color, default="white"
        Foreground beachball colour passed directly to ObsPy ``beach`` as
        ``facecolor``. Together with the default red ``bgcolor``, this defines
        the preferred fill convention used by the summary figure. The moment
        tensor itself is not negated or otherwise modified.

    bgcolor : matplotlib color, default="red"
        Background beachball colour passed directly to ObsPy ``beach`` as
        ``bgcolor``. The default red background combined with white
        ``facecolor`` intentionally reverses the native ObsPy/BayesISOLA colour
        assignment while leaving the moment-tensor geometry and P/T axes
        unchanged.

    show_dc_overlay : bool, default=True
        Overlay the double-couple nodal-plane geometry derived from NP1. For a
        non-pure-DC solution these curves need not coincide exactly with the
        zero-trace moment-tensor beachball boundaries; the overlay is retained as
        a useful visual comparison with the decomposed DC mechanism.

    figsize : tuple, default=(11.5, 6.2)
        Matplotlib figure size in inches.

    dpi : int, default=300
        Figure resolution.

    output_file : str or pathlib.Path, optional
        Save the figure to this path. Parent directories are created as needed.

    show : bool, default=False
        Display the figure. In IPython/Jupyter, ``display(fig)`` is used so an
        Agg backend does not emit a non-interactive ``plt.show`` warning.

    Returns
    -------
    matplotlib.figure.Figure
        Generated CMT summary figure.
    """
    import matplotlib.pyplot as plt
    from obspy.imaging.beachball import beach, MomentTensor, mt2axes

    def as_record(obj, name):
        if isinstance(obj, pd.DataFrame):
            if len(obj) != 1:
                raise ValueError(f"{name} must contain exactly one row.")
            return obj.iloc[0].to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return dict(obj)

    def format_value(value, fmt, unit=""):
        return f"{format(float(value), fmt)}{unit}"

    def format_scientific(value, unit=""):
        return f"{float(value):.3e}{unit}"

    def project_axis(axis):
        azimuth = np.deg2rad(axis.strike)
        radius = np.sqrt(2.0) * np.sin(np.deg2rad((90.0 - axis.dip) / 2.0))
        return radius * np.sin(azimuth), radius * np.cos(azimuth)

    def rim_xy(azimuth, radius=1.0):
        azimuth = np.deg2rad(azimuth)
        return radius * np.sin(azimuth), radius * np.cos(azimuth)

    def choose_plane_label_azimuths(strike1, strike2):
        candidates1 = (strike1 % 360.0, (strike1 + 180.0) % 360.0)
        candidates2 = (strike2 % 360.0, (strike2 + 180.0) % 360.0)
        best_pair, best_distance = None, -np.inf
        for az1 in candidates1:
            for az2 in candidates2:
                distance = np.linalg.norm(np.asarray(rim_xy(az1)) - np.asarray(rim_xy(az2)))
                if distance > best_distance:
                    best_pair, best_distance = (az1, az2), distance
        return best_pair

    def style_table(table, *, first_col_bold=False, header=True, fontsize=9):
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("0.72")
            cell.set_linewidth(0.55)
            cell.visible_edges = "B"
            if header and row == 0:
                cell.get_text().set_fontweight("bold")
            if first_col_bold and col == 0 and (not header or row > 0):
                cell.get_text().set_fontweight("bold")

    summary = as_record(summary, "summary")
    centroid = as_record(centroid, "centroid")
    tensor_mode = str(tensor_mode).lower().strip()
    if tensor_mode not in {"full", "deviatoric"}:
        raise ValueError("tensor_mode must be 'full' or 'deviatoric'.")

    mt = np.array([
        summary["Mrr"], summary["Mtt"], summary["Mpp"],
        summary["Mrt"], summary["Mrp"], summary["Mtp"],
    ], dtype=float)

    # BayesISOLA's native beachball is zero-trace even after a six-component
    # inversion because ObsPy beach() defaults to plot_zerotrace=True. Make that
    # behavior explicit and use the same tensor for the plotted principal axes.
    mt_plot = mt.copy()
    mt_plot[:3] -= np.mean(mt_plot[:3])

    if tensor_mode == "deviatoric":
        tensor_label = "Deviatoric (5-component)"
        mt_table_values = mt_plot
    else:
        tensor_label = "Full MT (6-component)"
        mt_table_values = mt

    mt_obj = MomentTensor(*mt_plot, 0)
    T, N, P = mt2axes(mt_obj)

    nodal_values = [
        summary.get("NP1_strike_deg"), summary.get("NP1_dip_deg"), summary.get("NP1_rake_deg"),
        summary.get("NP2_strike_deg"), summary.get("NP2_dip_deg"), summary.get("NP2_rake_deg"),
    ]
    has_nodal_planes = all(
        value is not None and np.isfinite(float(value)) for value in nodal_values
    )
    draw_dc_overlay = bool(show_dc_overlay and has_nodal_planes)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.25))
    ax_ball = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    ball = beach(
        mt_plot,
        xy=(0, 0),
        width=2.0,
        size=300,
        linewidth=1.15,
        facecolor=facecolor,
        bgcolor=bgcolor,
        edgecolor="black",
        zorder=1,
    )
    ax_ball.add_collection(ball)

    if draw_dc_overlay:
        planes = beach(
            (summary["NP1_strike_deg"], summary["NP1_dip_deg"], summary["NP1_rake_deg"]),
            xy=(0, 0), width=2.0, size=300, linewidth=0.65,
            nofill=True, edgecolor="0.35", zorder=2,
        )
        ax_ball.add_collection(planes)

    for label, axis in (("P", P), ("T", T)):
        x, y = project_axis(axis)
        ax_ball.text(x, y, label, ha="center", va="center", fontsize=15, zorder=5)

    if draw_dc_overlay:
        np1_az, np2_az = choose_plane_label_azimuths(summary["NP1_strike_deg"], summary["NP2_strike_deg"])
        for label, azimuth in (("NP1", np1_az), ("NP2", np2_az)):
            x0, y0 = rim_xy(azimuth, 1.00)
            x1, y1 = rim_xy(azimuth, 1.11)
            ax_ball.annotate(
                label, xy=(x0, y0), xytext=(x1, y1),
                ha="left" if x1 >= 0 else "right", va="center", fontsize=9,
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "linewidth": 0.7, "shrinkA": 2, "shrinkB": 0},
                zorder=6,
            )

    ax_ball.set_xlim(-1.18, 1.18)
    ax_ball.set_ylim(-1.18, 1.18)
    ax_ball.set_aspect("equal")
    ax_ball.axis("off")
    ax_ball.set_title("Centroid Moment Tensor", fontsize=13, pad=10)

    depth_text = format_value(centroid["centroid_depth_km"], ".1f", " km")
    moment_text = format_scientific(summary["M0_Nm"], " N·m")
    magnitude_text = format_value(summary["Mw"], ".2f", " Mw")
    dc_text = format_value(summary["DC_percent"], ".1f", "%")
    clvd_text = format_value(summary["CLVD_percent"], ".1f", "%")
    iso_text = (
        format_value(summary["ISO_percent"], ".1f", "%")
        if tensor_mode == "full" else "0.0% (constrained)"
    )
    vr = float(summary["variance_reduction"])

    source_rows = [
        ["Tensor", tensor_label],
        ["Moment", moment_text],
        ["Magnitude", magnitude_text],
        ["Centroid depth", depth_text],
        ["DC", dc_text],
        ["CLVD", clvd_text],
        ["ISO", iso_text],
        ["Variance reduction", f"{100.0 * vr:.1f}%"],
        ["Condition number", f"{summary['condition_number']:.2f}"],
    ]

    ax_info.text(0.0, 0.98, "Source", fontsize=12, fontweight="bold", transform=ax_info.transAxes)
    source_table = ax_info.table(
        cellText=source_rows, cellLoc="left", colWidths=[0.42, 0.56],
        bbox=[0.0, 0.62, 1.0, 0.32],
    )
    style_table(source_table, first_col_bold=True, header=False, fontsize=9.2)

    ax_info.text(
        0.0, 0.575, "Moment Tensor Components (N·m)",
        fontsize=11, fontweight="bold", transform=ax_info.transAxes,
    )
    mt_rows = [
        ["Mrr", f"{mt_table_values[0]:.2e}", "Mtt", f"{mt_table_values[1]:.2e}", "Mpp", f"{mt_table_values[2]:.2e}"],
        ["Mrt", f"{mt_table_values[3]:.2e}", "Mrp", f"{mt_table_values[4]:.2e}", "Mtp", f"{mt_table_values[5]:.2e}"],
    ]
    mt_table = ax_info.table(
        cellText=mt_rows, cellLoc="center",
        colWidths=[0.09, 0.23, 0.09, 0.23, 0.09, 0.23],
        bbox=[0.0, 0.448, 1.0, 0.10],
    )
    mt_table.auto_set_font_size(False)
    mt_table.set_fontsize(8.6)
    for (row, col), cell in mt_table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.5)
        cell.visible_edges = "B"
        if col in (0, 2, 4):
            cell.get_text().set_fontweight("bold")

    ax_info.text(0.0, 0.375, "Nodal Planes", fontsize=11, fontweight="bold", transform=ax_info.transAxes)
    def format_angle(value):
        return f"{float(value):.1f}°" if value is not None and np.isfinite(float(value)) else "—"

    nodal_rows = [
        ["NP1", format_angle(summary.get("NP1_strike_deg")), format_angle(summary.get("NP1_dip_deg")), format_angle(summary.get("NP1_rake_deg"))],
        ["NP2", format_angle(summary.get("NP2_strike_deg")), format_angle(summary.get("NP2_dip_deg")), format_angle(summary.get("NP2_rake_deg"))],
    ]
    nodal_table = ax_info.table(
        cellText=nodal_rows, colLabels=["Plane", "Strike", "Dip", "Rake"],
        cellLoc="center", colLoc="center", bbox=[0.0, 0.075, 0.47, 0.265],
    )
    style_table(nodal_table, header=True, fontsize=8.4)

    ax_info.text(0.52, 0.375, "Principal Axes", fontsize=11, fontweight="bold", transform=ax_info.transAxes)
    axis_rows = [
        ["T", f"{T.val:.2e}", f"{T.dip:.1f}°", f"{T.strike:.1f}°"],
        ["N", f"{N.val:.2e}", f"{N.dip:.1f}°", f"{N.strike:.1f}°"],
        ["P", f"{P.val:.2e}", f"{P.dip:.1f}°", f"{P.strike:.1f}°"],
    ]
    axes_table = ax_info.table(
        cellText=axis_rows, colLabels=["Axis", "Value (N·m)", "Plunge", "Azimuth"],
        cellLoc="center", colLoc="center", bbox=[0.52, 0.075, 0.48, 0.265],
    )
    style_table(axes_table, header=True, fontsize=8.0)

    if output_file is not None:
        output_file = Path(output_file).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")

    if show:
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                display(fig)
            else:
                plt.show()
        except ImportError:
            plt.show()

    return fig


def plot_posterior_summary(run, figsize=(8.5, 3.8), dpi=200):
    """Plot horizontal, depth, and centroid-time posterior marginals.

    The posterior is marginalized directly from ``results['posterior_cells']``
    in log space; no Monte Carlo uncertainty table is used.  The horizontal
    panel therefore represents the exact discrete posterior mass on the final
    BayesISOLA search grid, while the depth and time panels are exact discrete
    one-dimensional marginals.

    Parameters
    ----------
    run : mapping
        Result returned by :func:`BayesISOLA.workflows.run_auto_cmt`.
    figsize : tuple, default=(8.5, 3.8)
        Matplotlib figure size in inches.
    dpi : int, default=200
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        Generated posterior-summary figure.
    """
    results = _get_results(run)
    cells = _require_dataframe(results, "posterior_cells", where="run['results']")
    _require_columns(
        cells,
        ["log_posterior", "centroid_depth_km", "centroid_time_shift_s",
         "offset_east_m", "offset_north_m", "centroid_lon", "centroid_lat"],
        name="results['posterior_cells']",
    )

    depth = _posterior_marginal(cells, ["centroid_depth_km"])
    time = _posterior_marginal(cells, ["centroid_time_shift_s"])
    horizontal = _posterior_marginal(
        cells, ["offset_east_m", "offset_north_m", "centroid_lon", "centroid_lat"]
    )

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
        ax_map = fig.add_subplot(gs[:, 0])
        ax_depth = fig.add_subplot(gs[0, 1])
        ax_time = fig.add_subplot(gs[1, 1])

        east = horizontal["offset_east_m"].to_numpy(dtype=float) / 1000.0
        north = horizontal["offset_north_m"].to_numpy(dtype=float) / 1000.0
        pxy = horizontal["posterior_probability"].to_numpy(dtype=float)
        sc = ax_map.scatter(east, north, c=pxy, s=18)

        imode = int(np.argmax(pxy))
        ax_map.scatter(
            east[imode], north[imode], marker="*", s=80,
            edgecolor="k", linewidth=0.6, label="Posterior mode",
        )
        ax_map.scatter(
            0.0, 0.0, marker="+", s=55, linewidth=1.1,
            label="Catalogue epicentre",
        )
        ax_map.set_xlabel("East offset (km)")
        ax_map.set_ylabel("North offset (km)")
        ax_map.set_title("Horizontal posterior")
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(loc="upper right", frameon=True)
        cbar = fig.colorbar(sc, ax=ax_map, pad=0.02, fraction=0.045)
        cbar.set_label(r"$P(x,y)$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax_depth.plot(
            depth["centroid_depth_km"], depth["posterior_probability"],
            marker="o", markersize=2.5, linewidth=1.0,
        )
        ax_depth.set_xlabel("Depth (km)")
        ax_depth.set_ylabel(r"$P(z)$")
        ax_depth.set_title("Depth marginal")

        ax_time.plot(
            time["centroid_time_shift_s"], time["posterior_probability"],
            marker="o", markersize=2.5, linewidth=1.0,
        )
        ax_time.set_xlabel("Time shift (s)")
        ax_time.set_ylabel(r"$P(t)$")
        ax_time.set_title("Time marginal")

    return fig



def plot_adaptive_history(run, figsize=(10.0, 2.8), dpi=200):
    """Plot the evolution of an adaptive BayesISOLA space-time search.

    Four panels show the spatial/depth grid scales, horizontal centroid track,
    searched depth interval, and preferred variance reduction at each recorded
    stage.  Boundary markers use the stage-level active-boundary diagnostic and
    therefore do not misclassify a deliberately fixed horizontal grid as an edge.

    ``adaptive_history`` always contains the initial inversion stage, so the
    function is also valid for a one-stage run in which no expansion or
    refinement was ultimately required.

    Parameters
    ----------
    run : mapping
        Result returned by :func:`BayesISOLA.workflows.run_auto_cmt`.
    figsize : tuple, default=(10.0, 2.8)
        Matplotlib figure size in inches.
    dpi : int, default=200
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        Generated adaptive-search diagnostic figure.
    """
    if not isinstance(run, Mapping):
        raise TypeError("run must be the mapping returned by run_auto_cmt.")
    h = _require_dataframe(run, "adaptive_history", where="run").copy()
    _require_columns(
        h,
        ["stage_index", "stage_type", "radius_km", "step_x_km", "step_z_km",
         "centroid_east_km", "centroid_north_km", "depth_min_km", "depth_max_km",
         "centroid_depth_km", "variance_reduction"],
        name="run['adaptive_history']",
    )

    x = np.arange(len(h))
    labels = [f"{int(i)}\n{stage}" for i, stage in zip(h["stage_index"], h["stage_type"])]

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi, constrained_layout=True)

        ax = axes[0]
        ax.plot(x, h["radius_km"], marker="o", markersize=4, label="Radius")
        ax.plot(x, h["step_x_km"], marker="o", markersize=4, label="XY step")
        ax.plot(x, h["step_z_km"], marker="o", markersize=4, label="Z step")
        ax.set_ylabel("Distance (km)")
        ax.set_title("Search grid")
        ax.legend(frameon=False)

        ax = axes[1]
        east = pd.to_numeric(h["centroid_east_km"], errors="coerce").to_numpy(dtype=float)
        north = pd.to_numeric(h["centroid_north_km"], errors="coerce").to_numpy(dtype=float)
        ax.plot(east, north, marker="o", markersize=4)
        ax.scatter(0.0, 0.0, marker="+", s=45, linewidth=1.0)
        for i, (e, n) in enumerate(zip(east, north)):
            if np.isfinite(e) and np.isfinite(n):
                ax.annotate(str(i), (e, n), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_title("Centroid track")
        ax.set_aspect("equal", adjustable="datalim")

        ax = axes[2]
        ax.fill_between(x, h["depth_min_km"], h["depth_max_km"], alpha=0.18)
        ax.plot(x, h["centroid_depth_km"], marker="o", markersize=4)
        ax.set_ylabel("Depth (km)")
        ax.set_title("Centroid depth")
        ax.invert_yaxis()

        ax = axes[3]
        vr = 100.0 * pd.to_numeric(h["variance_reduction"], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, vr, marker="o", markersize=4)
        if "on_active_spatial_boundary" in h:
            boundary = h["on_active_spatial_boundary"].fillna(False).astype(bool).to_numpy()
            boundary &= np.isfinite(vr)
            if boundary.any():
                ax.scatter(x[boundary], vr[boundary], marker="x", s=45, label="Boundary")
                ax.legend(frameon=False)
        ax.set_ylabel("VR (%)")
        ax.set_title("Solution")

        for ax in (axes[0], axes[2], axes[3]):
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

    return fig



def plot_station_qc(run, figsize=(8.8, 3.8), dpi=200):
    """Plot station geometry/selection and leave-one-station-out sensitivity.

    The left polar panel combines the station-selection audit with the components
    that actually entered the final inversion.  Selected stations with no active
    components are distinguished from actively used stations, while stations
    rejected by the selection/QC layer remain visible for geometry context.

    When ``results['station_jackknife']`` is available, the right panel shows
    centroid displacement versus Kagan-angle change for each omitted station.
    Marker size scales with the held-out whitened residual RMS and is therefore a
    visual influence diagnostic rather than a posterior probability.

    Parameters
    ----------
    run : mapping
        Result returned by :func:`BayesISOLA.workflows.run_auto_cmt`.
    figsize : tuple, default=(8.8, 3.8)
        Matplotlib figure size in inches.
    dpi : int, default=200
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        Generated station-QC diagnostic figure.
    """
    results = _get_results(run)
    station_fit = _require_dataframe(results, "station_fit", where="run['results']").copy()
    selection = _require_dataframe(run, "station_selection", where="run").copy()
    jackknife = results.get("station_jackknife")

    key = ["network", "station", "location"]
    _require_columns(station_fit, [*key, "used"], name="results['station_fit']")
    _require_columns(
        selection,
        [*key, "distance_km", "azimuth_deg", "selection_status", "selection_reason"],
        name="run['station_selection']",
    )

    for df in (station_fit, selection):
        for col in key:
            df[col] = df[col].fillna("").astype(str)

    station_fit["used"] = station_fit["used"].fillna(False).astype(bool)
    used = station_fit.groupby(key, dropna=False).agg(n_components=("used", "sum")).reset_index()

    geometry = (
        selection[key + ["distance_km", "azimuth_deg", "selection_status", "selection_reason"]]
        .drop_duplicates(key)
        .merge(used, on=key, how="left")
    )
    geometry["distance_km"] = pd.to_numeric(geometry["distance_km"], errors="coerce")
    geometry["azimuth_deg"] = pd.to_numeric(geometry["azimuth_deg"], errors="coerce")
    geometry["n_components"] = geometry["n_components"].fillna(0).astype(int)

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.15))

        ax = fig.add_subplot(gs[0], projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45))
        ax.set_rlabel_position(150.5)

        selected = geometry["selection_status"].eq("selected")
        active = selected & geometry["n_components"].gt(0)
        inactive = selected & geometry["n_components"].eq(0)
        excluded = ~selected
        groups = [
            (excluded, "Excluded", "o", 12, 0.20),
            (active, "Used", "o", 32, 0.90),
            (inactive, "Selected, unused", "x", 45, 1.00),
        ]

        for mask, label, marker, size, alpha in groups:
            finite = mask & np.isfinite(geometry["azimuth_deg"]) & np.isfinite(geometry["distance_km"])
            if not finite.any():
                continue
            theta = np.deg2rad(geometry.loc[finite, "azimuth_deg"].to_numpy(dtype=float))
            radius = geometry.loc[finite, "distance_km"].to_numpy(dtype=float)
            ax.scatter(theta, radius, marker=marker, s=size, alpha=alpha, label=label)

        annotate = selected & np.isfinite(geometry["azimuth_deg"]) & np.isfinite(geometry["distance_km"])
        for _, row in geometry.loc[annotate].iterrows():
            ax.annotate(
                row["station"],
                (np.deg2rad(float(row["azimuth_deg"])), float(row["distance_km"])),
                xytext=(3, 2), textcoords="offset points", fontsize=6.5,
            )
        ax.set_title("Station geometry", pad=10)
        if ax.collections:
            ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.14), frameon=False)

        ax = fig.add_subplot(gs[1])
        required_jk = {"station", "centroid_shift_km", "kagan_angle_deg", "heldout_rms_whitened_residual"}
        if isinstance(jackknife, pd.DataFrame) and not jackknife.empty and required_jk.issubset(jackknife.columns):
            jk = jackknife.copy()
            x = pd.to_numeric(jk["centroid_shift_km"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(jk["kagan_angle_deg"], errors="coerce").to_numpy(dtype=float)
            rms = pd.to_numeric(jk["heldout_rms_whitened_residual"], errors="coerce").to_numpy(dtype=float)

            finite_rms = np.isfinite(rms)
            sizes = np.full(len(jk), 35.0)
            if finite_rms.sum() > 1:
                lo, hi = np.nanquantile(rms[finite_rms], [0.10, 0.90])
                if hi > lo:
                    scaled = np.clip((rms - lo) / (hi - lo), 0.0, 1.0)
                    sizes = np.where(np.isfinite(scaled), 25.0 + 75.0 * scaled, 35.0)

            valid_xy = np.isfinite(x) & np.isfinite(y)
            if valid_xy.any():
                ax.scatter(x[valid_xy], y[valid_xy], s=sizes[valid_xy], alpha=0.75)
                for idx in np.flatnonzero(valid_xy):
                    ax.annotate(
                        str(jk.iloc[idx]["station"]), (x[idx], y[idx]),
                        xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                    )
            ax.text(
                0.02, 0.98, "marker size ∝ held-out RMS", transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
            )
            ax.set_xlabel("LOO centroid shift (km)")
            ax.set_ylabel("LOO Kagan angle (°)")
            ax.set_title("Station omission sensitivity")
        else:
            ax.text(0.5, 0.5, "Jackknife not requested", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    return fig



def summarize_uncertainty(run):
    """Summarize finite conditional uncertainty samples with central 90% intervals.

    Parameters
    ----------
    run : mapping
        Result returned by :func:`BayesISOLA.workflows.run_auto_cmt`.

    Returns
    -------
    pandas.DataFrame or None
        One row per available scalar variable with mean, sample standard
        deviation, q05, median, and q95.  ``None`` is returned when uncertainty
        sampling was not requested.  This helper summarizes the realization
        table only; exact discrete depth/time intervals should be taken from the
        posterior-cell marginal rather than from this table when reporting the
        nonlinear search uncertainty.
    """
    results = _get_results(run)
    df = results.get("uncertainty")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    if not isinstance(df, pd.DataFrame):
        raise TypeError("results['uncertainty'] must be a pandas DataFrame or None.")

    variables = [
        "Mw", "centroid_depth_km", "centroid_time_shift_s",
        "dc_percent", "clvd_percent", "iso_percent",
    ]
    rows = []
    for col in (column for column in variables if column in df.columns):
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        q05, q50, q95 = np.quantile(x, [0.05, 0.50, 0.95])
        rows.append({
            "parameter": col,
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            "q05": float(q05),
            "median": float(q50),
            "q95": float(q95),
        })
    return pd.DataFrame(rows, columns=["parameter", "mean", "std", "q05", "median", "q95"])



def plot_uncertainty_summary(run, figsize=(8.5, 6.6), dpi=200, bins=30):
    """Plot exact nonlinear marginals and conditional moment-tensor uncertainty.

    Depth and centroid-time panels are exact marginals of the discrete
    space-time posterior. Magnitude, DC/CLVD/ISO, and nodal-plane panels are
    calculated from the conditional moment-tensor realization table generated
    by ``run_auto_cmt(..., n_uncertainty=...)``. Nodal-plane strike/rake are
    shown as shortest angular deviations from each preferred plane so circular
    wraparound does not artificially broaden the distributions.

    Parameters
    ----------
    run : mapping
        Result returned by :func:`BayesISOLA.workflows.run_auto_cmt`.
    figsize : tuple, default=(8.5, 6.6)
        Matplotlib figure size in inches.
    dpi : int, default=200
        Figure resolution.
    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        Generated uncertainty-summary figure.
    """
    results = _get_results(run)
    cells = _require_dataframe(results, "posterior_cells", where="run['results']")
    unc = results.get("uncertainty")
    summary_df = _require_dataframe(results, "summary", where="run['results']")
    centroid_df = _require_dataframe(results, "centroid", where="run['results']")

    if not isinstance(unc, pd.DataFrame) or unc.empty:
        raise ValueError(
            "plot_uncertainty_summary requires uncertainty samples; "
            "run with n_uncertainty > 0."
        )
    if int(bins) <= 0:
        raise ValueError("bins must be a positive integer.")
    bins = int(bins)

    required_unc = [
        "Mw", "dc_percent", "clvd_percent", "iso_percent",
        "NP1_strike_deg", "NP1_dip_deg", "NP1_rake_deg",
        "NP2_strike_deg", "NP2_dip_deg", "NP2_rake_deg",
    ]
    _require_columns(unc, required_unc, name="results['uncertainty']")
    _require_columns(cells, ["log_posterior", "centroid_depth_km", "centroid_time_shift_s"], name="results['posterior_cells']")

    summary = summary_df.iloc[0]
    centroid = centroid_df.iloc[0]
    depth = _posterior_marginal(cells, ["centroid_depth_km"])
    time = _posterior_marginal(cells, ["centroid_time_shift_s"])

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            3, 3,
            figsize=figsize,
            dpi=dpi,
            constrained_layout=True,
        )

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1 = colors[0]
        c2 = colors[1]

        # ----------------------------------------------------------
        # Mw
        # ----------------------------------------------------------
        ax = axes[0, 0]

        x = unc["Mw"].dropna().to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        q05, _, q95 = _sample_interval(unc, "Mw")

        ax.hist(x, bins=bins, density=True, alpha=0.65)
        ax.axvline(float(summary["Mw"]), linewidth=1.2, label="Preferred")
        ax.axvline(q05, linestyle="--", linewidth=0.8)
        ax.axvline(q95, linestyle="--", linewidth=0.8)

        ax.set_xlabel("Mw")
        ax.set_ylabel("Density")
        ax.set_title("Magnitude")
        ax.text(
            0.98, 0.95,
            f"90%: {q05:.3f}–{q95:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

        # ----------------------------------------------------------
        # Exact depth marginal
        # ----------------------------------------------------------
        ax = axes[0, 1]

        q05, _, q95 = _discrete_interval(depth, "centroid_depth_km")

        ax.plot(
            depth["centroid_depth_km"],
            depth["posterior_probability"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
        )
        ax.axvline(
            float(centroid["centroid_depth_km"]),
            linewidth=1.2,
        )
        ax.axvline(q05, linestyle="--", linewidth=0.8)
        ax.axvline(q95, linestyle="--", linewidth=0.8)

        ax.set_xlabel("Depth (km)")
        ax.set_ylabel("Probability")
        ax.set_title("Centroid depth")
        ax.text(
            0.98, 0.95,
            f"90%: {q05:.2f}–{q95:.2f} km",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

        # ----------------------------------------------------------
        # Exact time marginal
        # ----------------------------------------------------------
        ax = axes[0, 2]

        q05, _, q95 = _discrete_interval(time, "centroid_time_shift_s")

        ax.plot(
            time["centroid_time_shift_s"],
            time["posterior_probability"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
        )
        ax.axvline(
            float(centroid["centroid_time_shift_s"]),
            linewidth=1.2,
        )
        ax.axvline(q05, linestyle="--", linewidth=0.8)
        ax.axvline(q95, linestyle="--", linewidth=0.8)

        ax.set_xlabel("Time shift (s)")
        ax.set_ylabel("Probability")
        ax.set_title("Centroid time")
        ax.text(
            0.98, 0.95,
            f"90%: {q05:.2f}–{q95:.2f} s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

        # ----------------------------------------------------------
        # MT decomposition
        # ----------------------------------------------------------
        specs = [
            ("dc_percent", "DC", "DC_percent"),
            ("clvd_percent", "CLVD", "CLVD_percent"),
            ("iso_percent", "ISO", "ISO_percent"),
        ]

        for ax, (col, title, preferred_col) in zip(axes[1], specs):
            x = unc[col].dropna().to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            q05, _, q95 = _sample_interval(unc, col)

            ax.hist(x, bins=bins, density=True, alpha=0.65)
            ax.axvline(
                float(summary[preferred_col]),
                linewidth=1.2,
            )
            ax.axvline(q05, linestyle="--", linewidth=0.8)
            ax.axvline(q95, linestyle="--", linewidth=0.8)

            ax.set_xlabel(f"{title} (%)")
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.text(
                0.98, 0.95,
                f"90%: {q05:.1f}–{q95:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )

        # ----------------------------------------------------------
        # Nodal-plane uncertainty
        #
        # Plot each sampled NP relative to its own preferred value.
        # This makes the distribution width visible and avoids using
        # most of the axis to display the separation between NP1/NP2.
        # ----------------------------------------------------------
        nodal_specs = [
            ("strike", "Strike deviation (°)"),
            ("dip", "Dip deviation (°)"),
            ("rake", "Rake deviation (°)"),
        ]

        for ax, (parameter, xlabel) in zip(axes[2], nodal_specs):
            col1 = f"NP1_{parameter}_deg"
            col2 = f"NP2_{parameter}_deg"

            preferred1 = float(summary[col1])
            preferred2 = float(summary[col2])

            x1 = unc[col1].dropna().to_numpy(dtype=float)
            x2 = unc[col2].dropna().to_numpy(dtype=float)

            x1 = x1[np.isfinite(x1)]
            x2 = x2[np.isfinite(x2)]

            if x1.size == 0 or x2.size == 0:
                raise ValueError(
                    f"No valid nodal-plane uncertainty samples for {parameter!r}."
                )

            if parameter in {"strike", "rake"}:
                d1 = _angular_difference(x1, preferred1)
                d2 = _angular_difference(x2, preferred2)
            else:
                d1 = x1 - preferred1
                d2 = x2 - preferred2

            q1_05, q1_50, q1_95 = np.quantile(d1, [0.05, 0.50, 0.95])
            q2_05, q2_50, q2_95 = np.quantile(d2, [0.05, 0.50, 0.95])

            # Common symmetric limits within each parameter panel make
            # NP1 and NP2 uncertainty directly comparable.
            span = max(
                np.quantile(np.abs(d1), 0.995),
                np.quantile(np.abs(d2), 0.995),
            )
            span = max(1.15 * span, 0.1)

            edges = np.linspace(-span, span, bins + 1)

            # NP1
            ax.hist(
                d1,
                bins=edges,
                density=True,
                histtype="stepfilled",
                alpha=0.20,
                color=c1,
            )
            ax.hist(
                d1,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.3,
                color=c1,
                label="NP1",
            )

            # NP2
            ax.hist(
                d2,
                bins=edges,
                density=True,
                histtype="stepfilled",
                alpha=0.20,
                color=c2,
            )
            ax.hist(
                d2,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.3,
                color=c2,
                label="NP2",
            )

            # Both preferred solutions correspond to zero deviation.
            ax.axvline(
                0.0,
                color="0.25",
                linewidth=1.0,
            )

            ax.set_xlim(-span, span)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"Nodal-plane {parameter}")

            ax.text(
                0.98, 0.95,
                f"NP1 {preferred1:.1f}°: "
                f"[{q1_05:+.2f}°, {q1_95:+.2f}°]\n"
                f"NP2 {preferred2:.1f}°: "
                f"[{q2_05:+.2f}°, {q2_95:+.2f}°]",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )

        axes[0, 0].legend(frameon=False)
        axes[2, 0].legend(frameon=False)

    return fig


def plot_station_fit_summary(
    run,
    figsize=(8.5, 4.2),
    dpi=200,
    output_file=None,
    show=False,
):
    """
    Plot station/component fit and full-solution whitened residual diagnostics.

    Left panel
    ----------
    BayesISOLA native component variance reduction for each station actually
    used in the inversion. Individual Z/N/E components are shown together
    with the station mean and the range across active components. Negative
    variance reduction is retained because it is a valid indicator that a
    component is fitted worse than a zero-synthetic reference.

    Right panel
    -----------
    Full-solution whitened residual RMS by station. When station-jackknife
    results are available, the corresponding fraction of the total preferred-
    solution misfit is annotated beside each station.

    Notes
    -----
    The residual quantities plotted here are the *full-solution* diagnostics
    stored in ``results["station_jackknife"]``:

        full_station_whitened_rms
        full_station_misfit_fraction

    They are not the held-out LOO residuals.

    Parameters
    ----------
    run : dict
        Output dictionary returned by ``run_auto_cmt``.
    figsize : tuple, optional
        Matplotlib figure size.
    dpi : int, optional
        Figure resolution.
    output_file : str or pathlib.Path, optional
        If supplied, save the figure to this path.
    show : bool, optional
        Display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    results = run["results"]

    station_fit = results.get("station_fit")
    if station_fit is None or station_fit.empty:
        raise ValueError("No station-fit results are available.")

    station_fit = station_fit.copy()

    required = {
        "network",
        "station",
        "location",
        "component",
        "distance_km",
        "used",
        "variance_reduction",
    }
    missing = required.difference(station_fit.columns)
    if missing:
        raise ValueError(
            "station_fit is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------------
    # Keep only components that actually entered the inversion.
    # This deliberately excludes stations such as MSZ when all
    # components were disabled by BayesISOLA QC.
    # --------------------------------------------------------------
    active = station_fit.loc[
        station_fit["used"].fillna(False).astype(bool)
    ].copy()

    active = active.loc[
        np.isfinite(pd.to_numeric(active["variance_reduction"], errors="coerce"))
    ].copy()

    if active.empty:
        raise ValueError("No active station components have finite variance reduction.")

    active["variance_reduction"] = pd.to_numeric(
        active["variance_reduction"], errors="coerce"
    )
    active["vr_percent"] = 100.0 * active["variance_reduction"]

    # Stable station identifier. The plotted label remains the short station
    # code unless duplicate codes occur.
    for col in ("network", "station", "location"):
        active[col] = active[col].fillna("").astype(str)

    active["_station_key"] = (
        active["network"]
        + "."
        + active["station"]
        + "."
        + active["location"]
    )

    # --------------------------------------------------------------
    # Station-level summaries
    # --------------------------------------------------------------
    station_summary = (
        active.groupby("_station_key", sort=False)
        .agg(
            network=("network", "first"),
            station=("station", "first"),
            location=("location", "first"),
            distance_km=("distance_km", "first"),
            n_components=("component", "size"),
            mean_vr_percent=("vr_percent", "mean"),
            min_vr_percent=("vr_percent", "min"),
            max_vr_percent=("vr_percent", "max"),
        )
        .reset_index()
        .sort_values(["distance_km", "station"])
        .reset_index(drop=True)
    )

    # Short labels unless a station code occurs more than once.
    duplicate_station = station_summary["station"].duplicated(keep=False)

    station_summary["plot_label"] = station_summary["station"]
    station_summary.loc[duplicate_station, "plot_label"] = (
        station_summary.loc[duplicate_station, "network"]
        + "."
        + station_summary.loc[duplicate_station, "station"]
        + "."
        + station_summary.loc[duplicate_station, "location"]
    )

    station_summary["y"] = np.arange(len(station_summary), dtype=float)

    y_lookup = dict(
        zip(station_summary["_station_key"], station_summary["y"])
    )

    # --------------------------------------------------------------
    # Full-solution whitened residuals.
    #
    # These currently live in the jackknife table because that calculation
    # reconstructs the exact preferred-solution residual. Do not substitute
    # heldout_rms_whitened_residual here.
    # --------------------------------------------------------------
    jackknife = results.get("station_jackknife")

    if jackknife is not None and not jackknife.empty:
        jk = jackknife.copy()

        residual_cols = {
            "network",
            "station",
            "location",
            "full_station_whitened_rms",
            "full_station_misfit_fraction",
        }

        if residual_cols.issubset(jk.columns):
            for col in ("network", "station", "location"):
                jk[col] = jk[col].fillna("").astype(str)

            jk["_station_key"] = (
                jk["network"]
                + "."
                + jk["station"]
                + "."
                + jk["location"]
            )

            station_summary = station_summary.merge(
                jk[
                    [
                        "_station_key",
                        "full_station_whitened_rms",
                        "full_station_misfit_fraction",
                    ]
                ].drop_duplicates("_station_key"),
                on="_station_key",
                how="left",
            )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    with plt.rc_context(PLOT_RC):
        fig, (ax_fit, ax_res) = plt.subplots(
            1,
            2,
            figsize=figsize,
            dpi=dpi,
            sharey=True,
            gridspec_kw={"width_ratios": (1.35, 1.0)},
            constrained_layout=True,
        )

        # ==========================================================
        # LEFT: component variance reduction
        # ==========================================================

        # Range across active components.
        ax_fit.hlines(
            station_summary["y"],
            station_summary["min_vr_percent"],
            station_summary["max_vr_percent"],
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

        component_markers = {
            "Z": "o",
            "N": "s",
            "E": "^",
        }

        for component, marker in component_markers.items():
            subset = active.loc[active["component"] == component]

            if subset.empty:
                continue

            y = subset["_station_key"].map(y_lookup).to_numpy(dtype=float)

            ax_fit.scatter(
                subset["vr_percent"],
                y,
                marker=marker,
                s=24,
                label=component,
                zorder=3,
            )

        # Station mean.
        ax_fit.scatter(
            station_summary["mean_vr_percent"],
            station_summary["y"],
            marker="|",
            s=110,
            linewidths=1.6,
            label="Mean",
            zorder=4,
        )

        ax_fit.set_yticks(station_summary["y"])
        ax_fit.set_yticklabels(station_summary["plot_label"])

        # Component VR is not bounded below: a poor component fit can be
        # negative.  Keep 100% as the physical upper reference while extending
        # the lower limit far enough to expose any negative values rather than
        # silently clipping the most diagnostic misfits.
        vr_min = float(np.nanmin(active["vr_percent"].to_numpy(dtype=float)))
        if vr_min < 0.0:
            margin = max(0.05 * (100.0 - vr_min), 2.0)
            vr_left = vr_min - margin
        else:
            vr_left = 0.0
        ax_fit.set_xlim(vr_left, 100.0)
        ax_fit.axvline(0.0, linewidth=0.7, alpha=0.45, zorder=0)
        ax_fit.set_xlabel("Variance reduction (%)")
        ax_fit.set_ylabel("Station")
        ax_fit.set_title("Component fit")

        ax_fit.legend(
            frameon=False,
            ncol=4,
            fontsize=7,
            loc="upper left",  # or your adjusted location
        )

        ax_fit.grid(
            axis="x",
            linewidth=0.4,
            alpha=0.25,
        )

        # ==========================================================
        # RIGHT: full-solution whitened residual
        # ==========================================================
        have_residuals = False
        if "full_station_whitened_rms" in station_summary.columns:
            residual_values = pd.to_numeric(
                station_summary["full_station_whitened_rms"], errors="coerce"
            ).to_numpy(dtype=float)
            have_residuals = bool(np.isfinite(residual_values).any())

        if have_residuals:
            rms = pd.to_numeric(
                station_summary["full_station_whitened_rms"],
                errors="coerce",
            ).to_numpy(dtype=float)

            fraction = pd.to_numeric(
                station_summary["full_station_misfit_fraction"],
                errors="coerce",
            ).to_numpy(dtype=float)

            valid = np.isfinite(rms)

            # Lollipop-style presentation.
            ax_res.hlines(
                station_summary.loc[valid, "y"],
                0.0,
                rms[valid],
                linewidth=1.0,
                alpha=0.45,
                zorder=1,
            )

            ax_res.scatter(
                rms[valid],
                station_summary.loc[valid, "y"],
                s=28,
                zorder=3,
            )

            rms_max = np.nanmax(rms[valid])
            text_offset = max(0.02 * rms_max, 0.05)

            # Annotate contribution to preferred-solution misfit.
            for y, x, frac in zip(
                station_summary.loc[valid, "y"],
                rms[valid],
                fraction[valid],
            ):
                if np.isfinite(frac):
                    ax_res.text(
                        x + text_offset,
                        y,
                        f"{100.0 * frac:.1f}%",
                        va="center",
                        ha="left",
                        fontsize=6.5,
                    )

            ax_res.set_xlim(
                0.0,
                max(rms_max * 1.23, rms_max + 4.0 * text_offset),
            )

            ax_res.set_xlabel("Whitened residual RMS")
            ax_res.set_title("Full-solution residual")

            ax_res.text(
                0.98,
                0.02,
                "labels = misfit contribution",
                transform=ax_res.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
            )

            ax_res.grid(
                axis="x",
                linewidth=0.4,
                alpha=0.25,
            )

        else:
            ax_res.text(
                0.5,
                0.5,
                "Full-solution residual metrics\nnot available",
                transform=ax_res.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
            ax_res.set_title("Full-solution residual")
            ax_res.set_xlabel("Whitened residual RMS")
            ax_res.set_xticks([])

        # Nearest station at top.
        ax_fit.invert_yaxis()

        # ----------------------------------------------------------
        # Output
        # ----------------------------------------------------------
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                output_file,
                dpi=dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()

    return fig
