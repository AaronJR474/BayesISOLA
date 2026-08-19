"""Curated HTML reporting for automated BayesISOLA CMT inversions.

This module complements the historical :mod:`BayesISOLA._html` renderer rather
than replacing it.  ``write_html_report`` consumes the structured mapping
returned by :func:`BayesISOLA.workflows.run_auto_cmt` and combines the new 0.2
posterior, uncertainty, adaptive-search, station-selection, and jackknife
diagnostics into a self-contained scientific summary.

The native ``plot.html_log()`` report remains available independently through
``run_auto_cmt(..., html_output=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
import base64
import mimetypes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BayesISOLA._diagnostics import (
    _discrete_interval,
    _posterior_marginal,
    _sample_interval,
    plot_adaptive_history,
    plot_cmt_summary,
    plot_posterior_summary,
    plot_station_fit_summary,
    plot_station_qc,
    plot_uncertainty_summary,
)

__all__ = ["write_html_report"]


# ---------------------------------------------------------------------
# Report validation helpers
# ---------------------------------------------------------------------
def _has_rows(value):
    """Return ``True`` when ``value`` is a non-empty DataFrame."""
    return isinstance(value, pd.DataFrame) and not value.empty


def _require_dataframe(container, key, *, where="results", allow_empty=False):
    """Return a required report table after validating type and emptiness."""
    value = container.get(key)

    if not isinstance(value, pd.DataFrame):
        raise TypeError(
            f"{where}[{key!r}] must be a pandas DataFrame; "
            f"got {type(value).__name__}."
        )

    if not allow_empty and value.empty:
        raise ValueError(f"{where}[{key!r}] is empty.")

    return value


def _require_columns(df, columns, *, name):
    """Raise a targeted error when a report table lacks required columns."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required column(s): "
            + ", ".join(missing)
        )


def _validate_report_run(run):
    """Validate the minimum run structure required by the HTML report."""
    if not isinstance(run, Mapping):
        raise TypeError(
            "write_html_report requires the mapping returned by run_auto_cmt; "
            f"got {type(run).__name__}."
        )

    missing_top = [key for key in ("results", "inputs", "solution") if key not in run]
    if missing_top:
        raise KeyError(
            "run is missing required key(s): "
            + ", ".join(missing_top)
        )

    results = run["results"]
    if not isinstance(results, Mapping):
        raise TypeError("run['results'] must be a mapping.")

    summary = _require_dataframe(results, "summary")
    centroid = _require_dataframe(results, "centroid")
    cells = _require_dataframe(results, "posterior_cells")
    station_fit = _require_dataframe(results, "station_fit")

    _require_columns(
        summary,
        [
            "Mw", "M0_Nm", "variance_reduction", "condition_number",
            "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
            "DC_percent", "CLVD_percent", "ISO_percent",
            "NP1_strike_deg", "NP1_dip_deg", "NP1_rake_deg",
            "NP2_strike_deg", "NP2_dip_deg", "NP2_rake_deg",
        ],
        name="results['summary']",
    )

    _require_columns(
        centroid,
        [
            "centroid_time", "centroid_time_shift_s",
            "centroid_lat", "centroid_lon", "centroid_depth_km",
            "offset_north_m", "offset_east_m",
        ],
        name="results['centroid']",
    )

    _require_columns(
        cells,
        ["centroid_depth_km", "centroid_time_shift_s", "log_posterior"],
        name="results['posterior_cells']",
    )

    _require_columns(
        station_fit,
        ["network", "station", "location", "used", "variance_reduction"],
        name="results['station_fit']",
    )

    inputs = run["inputs"]
    if not hasattr(inputs, "outdir"):
        raise AttributeError("run['inputs'] is missing the required 'outdir' attribute.")
    if not hasattr(inputs, "event"):
        raise AttributeError("run['inputs'] is missing the required 'event' attribute.")
    if not isinstance(inputs.event, Mapping):
        raise TypeError("run['inputs'].event must be a mapping.")
    if "depth" not in inputs.event:
        raise KeyError("run['inputs'].event is missing required key 'depth'.")

    solution = run["solution"]
    if not hasattr(solution, "deviatoric"):
        raise AttributeError("run['solution'] is missing the required 'deviatoric' attribute.")

    optional_frames = [
        "uncertainty",
        "posterior_diagnostics",
        "uncertainty_diagnostics",
        "station_jackknife",
    ]
    for key in optional_frames:
        value = results.get(key)
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"results[{key!r}] must be a pandas DataFrame or None; "
                f"got {type(value).__name__}."
            )

    for key in ("station_selection", "adaptive_history"):
        value = run.get(key)
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"run[{key!r}] must be a pandas DataFrame or None; "
                f"got {type(value).__name__}."
            )

    return results, inputs, solution


# ---------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------
def _report_value(value):
    """Format one scalar for compact, human-readable HTML table output."""
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if not np.isfinite(value):
            return "—"

        a = abs(value)

        if a != 0.0 and (a >= 1e5 or a < 1e-3):
            return f"{value:.3e}"

        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def _pretty_column(name):
    """Return a readable display label for a result-table column name."""
    special = {
        "Mw": "Mw",
        "M0_Nm": "M0 (N·m)",
        "DC_percent": "DC (%)",
        "CLVD_percent": "CLVD (%)",
        "ISO_percent": "ISO (%)",
        "NP1_strike_deg": "NP1 strike (°)",
        "NP1_dip_deg": "NP1 dip (°)",
        "NP1_rake_deg": "NP1 rake (°)",
        "NP2_strike_deg": "NP2 strike (°)",
        "NP2_dip_deg": "NP2 dip (°)",
        "NP2_rake_deg": "NP2 rake (°)",
    }

    if name in special:
        return special[name]

    return str(name).replace("_", " ").strip().capitalize()


def _dataframe_html(df, *, columns=None, rename=None, max_rows=None):
    """Render a DataFrame as escaped HTML using report-oriented formatting."""
    if df is None:
        return '<p class="muted">Not available.</p>'

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"_dataframe_html expected a pandas DataFrame or None; "
            f"got {type(df).__name__}."
        )

    if df.empty:
        return '<p class="muted">Not available.</p>'

    if rename is not None and not isinstance(rename, Mapping):
        raise TypeError("rename must be a mapping or None.")

    table = df.copy()

    if columns is not None:
        columns = [col for col in columns if col in table.columns]
        table = table.loc[:, columns]

    if max_rows is not None:
        table = table.iloc[:max_rows]

    if rename is None:
        rename = {col: _pretty_column(col) for col in table.columns}

    table = table.rename(columns=rename)

    for col in table.columns:
        table[col] = table[col].map(_report_value)

    return table.to_html(
        index=False,
        escape=True,
        border=0,
        classes="data-table",
    )


def _record_html(rows):
    """Render an iterable of ``(label, value)`` pairs as a two-column table."""
    body = "\n".join(
        f"<tr><th>{escape(str(label))}</th>"
        f"<td>{escape(_report_value(value))}</td></tr>"
        for label, value in rows
    )

    return f'<table class="record-table">{body}</table>'


def _image_src(path, html_dir, *, embed=False):
    """Return an embedded data URI or browser-safe path for one image."""
    path = Path(path).expanduser()

    if not path.is_file():
        return None

    if embed:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    try:
        relative = os.path.relpath(path, html_dir)
    except ValueError:
        return path.resolve().as_uri()

    return relative.replace("\\", "/")


def _figure_html(path, title, caption, html_dir, *, embed=False):
    """Render one saved diagnostic image and caption as an HTML figure."""
    src = _image_src(path, html_dir, embed=embed)

    if src is None:
        return ""

    src_escaped = escape(src, quote=True)

    if embed:
        image = f'<img src="{src_escaped}" alt="{escape(title)}">'
    else:
        image = (
            f'<a href="{src_escaped}" target="_blank">'
            f'<img src="{src_escaped}" alt="{escape(title)}">'
            "</a>"
        )

    return f"""
    <figure class="report-figure">
        {image}
        <figcaption>
            <strong>{escape(title)}</strong><br>
            {escape(caption)}
        </figcaption>
    </figure>
    """


def _save_report_figure(fig, path, dpi=200):
    """Save and close a Matplotlib figure, verifying that output was written."""
    if fig is None or not hasattr(fig, "savefig"):
        raise TypeError("Expected a Matplotlib figure with a savefig() method.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)

    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Report figure was not written correctly: {path}")

    return path


# ---------------------------------------------------------------------
# Compact posterior/uncertainty interval table
# ---------------------------------------------------------------------
def _build_report_intervals(run):
    """Build the compact exact-posterior and conditional-MT interval table."""
    results = run["results"]
    summary = results["summary"].iloc[0]
    centroid = results["centroid"].iloc[0]
    cells = results["posterior_cells"]
    unc = results.get("uncertainty")

    rows = []

    # Exact discrete nonlinear marginals
    depth = _posterior_marginal(cells, ["centroid_depth_km"])
    time = _posterior_marginal(cells, ["centroid_time_shift_s"])

    q05, q50, q95 = _discrete_interval(
        depth, "centroid_depth_km"
    )
    rows.append({
        "parameter": "Centroid depth",
        "preferred": float(centroid["centroid_depth_km"]),
        "q05": q05,
        "median": q50,
        "q95": q95,
        "unit": "km",
        "source": "Exact discrete posterior",
    })

    q05, q50, q95 = _discrete_interval(
        time, "centroid_time_shift_s"
    )
    rows.append({
        "parameter": "Centroid time shift",
        "preferred": float(centroid["centroid_time_shift_s"]),
        "q05": q05,
        "median": q50,
        "q95": q95,
        "unit": "s",
        "source": "Exact discrete posterior",
    })

    # Conditional MT samples
    if unc is not None and not unc.empty:
        specs = [
            ("Mw", "Magnitude", "Mw", ""),
            ("dc_percent", "DC", "DC_percent", "%"),
            ("clvd_percent", "CLVD", "CLVD_percent", "%"),
            ("iso_percent", "ISO", "ISO_percent", "%"),
        ]

        for sample_col, label, preferred_col, unit in specs:
            q05, q50, q95 = _sample_interval(unc, sample_col)

            rows.append({
                "parameter": label,
                "preferred": float(summary[preferred_col]),
                "q05": q05,
                "median": q50,
                "q95": q95,
                "unit": unit,
                "source": "Conditional MT samples",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# BayesISOLA 0.2 HTML summary report
# ---------------------------------------------------------------------
def write_html_report(
    run,
    output_file=None,
    *,
    title=None,
    embed_images=False,
    reuse_existing_figures=False,
    dpi=200,
):
    """
    Write a curated BayesISOLA 0.2 inversion report.

    The report generates/uses the standard summary figures and contains:
      - event and inversion overview
      - preferred CMT solution
      - posterior and uncertainty summaries
      - adaptive-search diagnostics
      - station geometry, fit and jackknife diagnostics
      - waveform fit when available
      - detailed result tables
      - links to saved CSV outputs

    Parameters
    ----------
    run : dict
        Result returned by run_auto_cmt.
    output_file : path-like, optional
        HTML destination. Defaults to <outdir>/report.html.
    title : str, optional
        Report title.
    embed_images : bool, default=False
        If True, encode figures directly into the HTML file. If False,
        use browser-safe relative paths to the saved figures.
    reuse_existing_figures : bool, default=False
        Reuse non-empty standard diagnostic PNGs already present in
        ``<outdir>/figures``. ``run_auto_cmt`` enables this only after its plot
        preset has just written those figures, avoiding duplicate rendering.
        Standalone report calls default to False so stale figures from an older
        run are never reused implicitly.
    dpi : int, default=200
        Resolution used when saving report figures.

    Returns
    -------
    pathlib.Path
        Path to the generated report.
    """
    results, inputs, solution = _validate_report_run(run)

    if not isinstance(dpi, (int, np.integer)) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer.")
    dpi = int(dpi)

    if not isinstance(embed_images, (bool, np.bool_)):
        raise TypeError("embed_images must be True or False.")
    if not isinstance(reuse_existing_figures, (bool, np.bool_)):
        raise TypeError("reuse_existing_figures must be True or False.")
    embed_images = bool(embed_images)
    reuse_existing_figures = bool(reuse_existing_figures)

    outdir = Path(inputs.outdir).expanduser()

    if output_file is None:
        output_file = outdir / "report.html"
    else:
        output_file = Path(output_file).expanduser()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    event_id = outdir.name
    if title is None:
        title = f"BayesISOLA CMT — {event_id}"

    summary = results["summary"].iloc[0]
    centroid = results["centroid"].iloc[0]
    event = inputs.event

    tensor_mode = (
        "deviatoric"
        if bool(solution.deviatoric)
        else "full"
    )

    # ==============================================================
    # Generate standard 0.2 summary figures
    # ==============================================================
    figures = {}

    def report_figure(key, filename, builder):
        path = figure_dir / filename
        if reuse_existing_figures and path.is_file() and path.stat().st_size > 0:
            figures[key] = path
            return
        figures[key] = _save_report_figure(builder(), path, dpi=dpi)

    report_figure(
        "cmt",
        "cmt_summary.png",
        lambda: plot_cmt_summary(
            results["summary"],
            results["centroid"],
            uncertainty=results.get("uncertainty"),
            tensor_mode=tensor_mode,
            show=False,
        ),
    )

    report_figure(
        "posterior",
        "posterior_summary.png",
        lambda: plot_posterior_summary(run),
    )

    uncertainty = results.get("uncertainty")
    if _has_rows(uncertainty):
        report_figure(
            "uncertainty",
            "uncertainty_summary.png",
            lambda: plot_uncertainty_summary(run),
        )

    station_selection = run.get("station_selection")
    if _has_rows(station_selection):
        report_figure(
            "station_qc",
            "station_qc_summary.png",
            lambda: plot_station_qc(run),
        )

    report_figure(
        "station_fit",
        "station_fit_summary.png",
        lambda: plot_station_fit_summary(run, show=False),
    )

    adaptive_history = run.get("adaptive_history")
    if adaptive_history is not None and not adaptive_history.empty:
        report_figure(
            "adaptive",
            "adaptive_grid_summary.png",
            lambda: plot_adaptive_history(run),
        )

    # ==============================================================
    # Native waveform figure, if the BayesISOLA plot object exists
    # ==============================================================
    native_waveform = None
    plot_object = run.get("plot")

    if plot_object is not None:
        native_plots = getattr(plot_object, "plots", {}) or {}

        for key in ("seismo", "seismo_sharey", "seismo_cova"):
            value = native_plots.get(key)

            if not value:
                continue

            value = str(value).replace("$outdir", str(outdir))
            candidate = Path(value).expanduser()

            if candidate.is_file():
                native_waveform = candidate
                break

    # ==============================================================
    # Preferred-solution and method summaries
    # ==============================================================
    station_fit = results["station_fit"].copy()

    station_keys = (
        station_fit["network"].fillna("").astype(str)
        + "."
        + station_fit["station"].fillna("").astype(str)
        + "."
        + station_fit["location"].fillna("").astype(str)
    )

    used = station_fit["used"].fillna(False).astype(bool)
    n_active_components = int(used.sum())
    n_active_stations = int(station_keys[used].nunique())

    posterior_diag_df = results.get("posterior_diagnostics")
    posterior_diag = (
        posterior_diag_df.iloc[0]
        if posterior_diag_df is not None and not posterior_diag_df.empty
        else None
    )

    mw_text = f"{float(summary['Mw']):.3f}"
    depth_text = f"{float(centroid['centroid_depth_km']):.2f} km"
    time_text = f"{float(centroid['centroid_time_shift_s']):+.2f} s"

    if _has_rows(uncertainty):
        q05, _, q95 = _sample_interval(uncertainty, "Mw")
        mw_text += f" [{q05:.3f}, {q95:.3f}]"

    depth_marginal = _posterior_marginal(
        results["posterior_cells"],
        ["centroid_depth_km"],
    )
    q05, _, q95 = _discrete_interval(
        depth_marginal,
        "centroid_depth_km",
    )
    depth_text += f" [{q05:.2f}, {q95:.2f}]"

    time_marginal = _posterior_marginal(
        results["posterior_cells"],
        ["centroid_time_shift_s"],
    )
    q05, _, q95 = _discrete_interval(
        time_marginal,
        "centroid_time_shift_s",
    )
    time_text += f" [{q05:+.2f}, {q95:+.2f}]"

    overview_cards = [
        ("Mw", mw_text),
        ("Centroid depth", depth_text),
        ("Time shift", time_text),
        ("Variance reduction", f"{100.0 * float(summary['variance_reduction']):.1f}%"),
        ("DC", f"{float(summary['DC_percent']):.1f}%"),
        ("CLVD", f"{float(summary['CLVD_percent']):.1f}%"),
        ("ISO", f"{float(summary['ISO_percent']):.1f}%"),
        ("Active stations", str(n_active_stations)),
    ]

    if (
        posterior_diag is not None
        and "posterior_effective_cells" in posterior_diag.index
        and pd.notna(posterior_diag["posterior_effective_cells"])
    ):
        overview_cards.append(
            (
                "Posterior Neff",
                f"{float(posterior_diag['posterior_effective_cells']):.3f}",
            )
        )

    cards_html = "\n".join(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
        </div>
        """
        for label, value in overview_cards
    )

    # ==============================================================
    # Run notes / explicit QC states
    # ==============================================================
    notes = []

    if bool(centroid.get("on_grid_edge", False)):
        notes.append(
            "Preferred centroid lies on an active searched spatial-grid boundary."
        )

    if bool(centroid.get("horizontal_search_fixed", False)):
        notes.append(
            "Horizontal source position was fixed; horizontal uncertainty was not estimated."
        )

    if posterior_diag is not None and bool(
        posterior_diag.get("time_search_fixed", False)
    ):
        notes.append(
            "Centroid time was fixed; time uncertainty was not estimated."
        )

    inactive_station_keys = station_keys.groupby(station_keys).apply(
        lambda s: not used.loc[s.index].any()
    )

    n_inactive = int(inactive_station_keys.sum())

    if n_inactive:
        notes.append(
            f"{n_inactive} retained station(s) had no active components in the final inversion."
        )

    jackknife = results.get("station_jackknife")

    if (
        jackknife is not None
        and not jackknife.empty
        and "loo_on_grid_edge" in jackknife.columns
    ):
        loo_edge = jackknife["loo_on_grid_edge"].fillna(False).astype(bool)
        n_loo_edge = int(loo_edge.sum())

        if n_loo_edge:
            notes.append(
                f"{n_loo_edge} leave-one-station-out solution(s) lie on the "
                "fixed final grid boundary; their location shifts are therefore "
                "boundary-censored diagnostics."
            )

    if notes:
        notes_html = (
            '<div class="notice"><strong>Run notes</strong><ul>'
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul></div>"
        )
    else:
        notes_html = ""

    # ==============================================================
    # Tables
    # ==============================================================

    event_rows = [
        ("Event ID", event_id),
        ("Agency", event.get("agency")),
        ("Origin time", event.get("t")),
        ("Latitude", event.get("lat")),
        ("Longitude", event.get("lon")),
        ("Hypocentral depth (km)", float(event["depth"]) / 1000.0),
        ("Catalogue magnitude", event.get("mag")),
    ]

    source_rows = [
        ("Tensor", "Deviatoric (5-component)" if solution.deviatoric else "Full MT (6-component)"),
        ("M0 (N·m)", summary["M0_Nm"]),
        ("Mw", summary["Mw"]),
        ("Centroid time", centroid["centroid_time"]),
        ("Centroid time shift (s)", centroid["centroid_time_shift_s"]),
        ("Centroid latitude", centroid["centroid_lat"]),
        ("Centroid longitude", centroid["centroid_lon"]),
        ("Centroid depth (km)", centroid["centroid_depth_km"]),
        ("North offset (m)", centroid["offset_north_m"]),
        ("East offset (m)", centroid["offset_east_m"]),
        ("DC (%)", summary["DC_percent"]),
        ("CLVD (%)", summary["CLVD_percent"]),
        ("ISO (%)", summary["ISO_percent"]),
        ("Variance reduction (%)", 100.0 * float(summary["variance_reduction"])),
        ("Condition number", summary["condition_number"]),
        (
            "NP1 strike / dip / rake",
            f"{summary['NP1_strike_deg']:.1f}° / "
            f"{summary['NP1_dip_deg']:.1f}° / "
            f"{summary['NP1_rake_deg']:.1f}°",
        ),
        (
            "NP2 strike / dip / rake",
            f"{summary['NP2_strike_deg']:.1f}° / "
            f"{summary['NP2_dip_deg']:.1f}° / "
            f"{summary['NP2_rake_deg']:.1f}°",
        ),
    ]

    gf = run.get("gf", {})
    if gf is None:
        gf = {}
    if not isinstance(gf, Mapping):
        raise TypeError("run['gf'] must be a mapping or None.")

    cova = run.get("cova")

    def _nonempty(value):
        if value is None:
            return False
        try:
            return len(value) > 0
        except TypeError:
            return bool(value)

    has_covariance = (
        bool(getattr(cova, "has_covariance", False))
        or _nonempty(getattr(cova, "Cd_inv", None))
        or _nonempty(getattr(cova, "Cd_inv_shifts", None))
    )

    crosscovariance = _nonempty(
        getattr(cova, "LT3", None)
    )

    method_rows = [
        ("Green's-function source", gf.get("source")),
        ("Green's-function model", gf.get("model")),
        ("Green's functions reused", gf.get("reused")),
        ("Covariance matrix", has_covariance),
        ("Cross-component covariance", crosscovariance),
        ("Active stations", n_active_stations),
        ("Active components", n_active_components),
        ("Adaptive search", run.get("adaptive_grid")),
        (
            "Adaptive stages",
            len(adaptive_history)
            if adaptive_history is not None
            else 0,
        ),
        (
            "Station jackknife",
            jackknife is not None and not jackknife.empty,
        ),
    ]

    interval_df = _build_report_intervals(run)

    # --------------------------------------------------------------
    # Station fit table
    # --------------------------------------------------------------
    station_fit_table = station_fit.copy()

    station_fit_table["VR (%)"] = (
        100.0
        * pd.to_numeric(
            station_fit_table["variance_reduction"],
            errors="coerce",
        )
    )

    station_fit_columns = [
        "network",
        "station",
        "location",
        "component",
        "used",
        "distance_km",
        "azimuth_deg",
        "fmin_hz",
        "fmax_hz",
        "weight",
        "VR (%)",
    ]

    # --------------------------------------------------------------
    # Station selection table
    # --------------------------------------------------------------
    selection_columns = [
        "network",
        "station",
        "location",
        "distance_km",
        "azimuth_deg",
        "azimuth_sector",
        "selection_status",
        "selection_reason",
        "download_status",
    ]

    # --------------------------------------------------------------
    # Jackknife table
    # --------------------------------------------------------------
    jackknife_table = None

    if jackknife is not None and not jackknife.empty:
        jackknife_table = jackknife.copy()

        if "loo_variance_reduction" in jackknife_table:
            jackknife_table["LOO VR (%)"] = (
                100.0
                * pd.to_numeric(
                    jackknife_table["loo_variance_reduction"],
                    errors="coerce",
                )
            )

        if "full_station_misfit_fraction" in jackknife_table:
            jackknife_table["Full misfit contribution (%)"] = (
                100.0
                * pd.to_numeric(
                    jackknife_table["full_station_misfit_fraction"],
                    errors="coerce",
                )
            )

    jackknife_columns = [
        "network",
        "station",
        "location",
        "full_station_fit",
        "full_station_whitened_rms",
        "Full misfit contribution (%)",
        "delta_Mw",
        "delta_depth_km",
        "centroid_shift_km",
        "delta_time_s",
        "kagan_angle_deg",
        "LOO VR (%)",
        "loo_on_grid_edge",
        "qc_flags",
    ]

    # --------------------------------------------------------------
    # Adaptive history
    # --------------------------------------------------------------
    adaptive_table = None

    if adaptive_history is not None and not adaptive_history.empty:
        adaptive_table = adaptive_history.copy()

        if "variance_reduction" in adaptive_table:
            adaptive_table["VR (%)"] = (
                100.0
                * pd.to_numeric(
                    adaptive_table["variance_reduction"],
                    errors="coerce",
                )
            )

    adaptive_columns = [
        "stage_index",
        "stage_type",
        "radius_km",
        "step_x_km",
        "step_z_km",
        "depth_min_km",
        "depth_max_km",
        "centroid_east_km",
        "centroid_north_km",
        "centroid_depth_km",
        "centroid_time_shift_s",
        "VR (%)",
        "on_active_spatial_boundary",
        "next_action",
    ]

    # ==============================================================
    # Saved result links
    # ==============================================================
    links = []
    result_paths = run.get("result_paths") or {}

    if not isinstance(result_paths, Mapping):
        raise TypeError("run['result_paths'] must be a mapping when supplied.")

    for key, path in result_paths.items():
        path = Path(path).expanduser()

        if not path.is_file():
            continue

        try:
            href = os.path.relpath(path, output_file.parent)
        except ValueError:
            href = path.resolve().as_uri()
        else:
            href = href.replace("\\", "/")

        links.append(
            f'<li><a href="{escape(href, quote=True)}">'
            f'{escape(str(key))}</a></li>'
        )

    links_html = (
        "<ul>" + "\n".join(links) + "</ul>"
        if links
        else '<p class="muted">No saved result tables found.</p>'
    )

    # ==============================================================
    # Figures
    # ==============================================================
    figure_html = {}

    figure_html["cmt"] = _figure_html(
        figures["cmt"],
        "Centroid moment tensor",
        "Preferred BayesISOLA source solution.",
        output_file.parent,
        embed=embed_images,
    )

    figure_html["posterior"] = _figure_html(
        figures["posterior"],
        "Posterior summary",
        "Horizontal, depth and centroid-time posterior marginals.",
        output_file.parent,
        embed=embed_images,
    )

    figure_html["uncertainty"] = (
        _figure_html(
            figures["uncertainty"],
            "Posterior and moment-tensor uncertainty",
            "Exact nonlinear depth/time marginals and conditional MT uncertainty.",
            output_file.parent,
            embed=embed_images,
        )
        if "uncertainty" in figures
        else ""
    )

    figure_html["station_qc"] = (
        _figure_html(
            figures["station_qc"],
            "Station geometry and omission sensitivity",
            "Azimuthal station geometry and leave-one-station-out sensitivity.",
            output_file.parent,
            embed=embed_images,
        )
        if "station_qc" in figures
        else ""
    )

    figure_html["station_fit"] = _figure_html(
        figures["station_fit"],
        "Station fit",
        "Component variance reduction and full-solution whitened residual diagnostics.",
        output_file.parent,
        embed=embed_images,
    )

    figure_html["adaptive"] = (
        _figure_html(
            figures["adaptive"],
            "Adaptive-grid evolution",
            "Evolution of the spatial/depth search and preferred solution.",
            output_file.parent,
            embed=embed_images,
        )
        if "adaptive" in figures
        else ""
    )

    figure_html["waveform"] = (
        _figure_html(
            native_waveform,
            "Waveform fit",
            "Observed and synthetic waveforms for the preferred solution.",
            output_file.parent,
            embed=embed_images,
        )
        if native_waveform is not None
        else ""
    )

    # ==============================================================
    # HTML
    # ==============================================================
    css = """
    :root {
        color-scheme: light;
        --text: #202124;
        --muted: #666;
        --line: #d9dde3;
        --soft: #f5f6f8;
        --card: #ffffff;
        --notice: #fff7df;
        --accent: #303846;
    }

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        font-family:
            -apple-system, BlinkMacSystemFont, "Segoe UI",
            Roboto, Helvetica, Arial, sans-serif;
        color: var(--text);
        background: #fff;
        line-height: 1.45;
    }

    main {
        width: min(1180px, calc(100% - 40px));
        margin: 0 auto;
        padding: 32px 0 64px;
    }

    h1 {
        margin: 0 0 4px;
        font-size: 28px;
        font-weight: 650;
    }

    h2 {
        margin-top: 38px;
        padding-bottom: 7px;
        border-bottom: 1px solid var(--line);
        font-size: 20px;
    }

    h3 {
        margin-top: 26px;
        font-size: 16px;
    }

    .subtitle {
        color: var(--muted);
        margin: 0 0 24px;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 18px 0 24px;
    }

    .metric-card {
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 11px 13px;
        background: var(--card);
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 16px;
        font-weight: 600;
    }

    .overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 18px;
    }

    .panel {
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 14px 16px;
        background: var(--card);
    }

    .panel h3 {
        margin-top: 0;
    }

    .notice {
        padding: 12px 15px;
        margin: 18px 0;
        border: 1px solid #e6d08c;
        border-radius: 7px;
        background: var(--notice);
    }

    .notice ul {
        margin-bottom: 0;
    }

    .report-figure {
        margin: 22px auto 30px;
        max-width: 1080px;
    }

    .report-figure img {
        display: block;
        width: 100%;
        height: auto;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fff;
    }

    figcaption {
        margin-top: 7px;
        color: var(--muted);
        font-size: 12px;
    }

    table {
        border-collapse: collapse;
        width: 100%;
    }

    th, td {
        padding: 6px 8px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
        font-size: 12px;
    }

    th {
        font-weight: 600;
        background: var(--soft);
    }

    .record-table th {
        width: 48%;
    }

    .table-scroll {
        overflow-x: auto;
    }

    details {
        margin: 12px 0;
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 10px 12px;
    }

    summary {
        cursor: pointer;
        font-weight: 600;
    }

    details > .table-scroll,
    details > div {
        margin-top: 10px;
    }

    .muted {
        color: var(--muted);
    }

    .footer {
        margin-top: 44px;
        padding-top: 12px;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 11px;
    }

    @media (max-width: 650px) {
        main {
            width: min(100% - 20px, 1180px);
        }

        h1 {
            font-size: 23px;
        }

        th, td {
            font-size: 11px;
        }
    }
    """

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
{css}
</style>
</head>

<body>
<main>

<header>
    <h1>{escape(title)}</h1>
    <p class="subtitle">
        BayesISOLA centroid moment-tensor inversion summary
    </p>
</header>

<section class="metrics">
    {cards_html}
</section>

{notes_html}

<section>
    <h2>Event and inversion</h2>

    <div class="overview-grid">
        <div class="panel">
            <h3>Catalogue event</h3>
            {_record_html(event_rows)}
        </div>

        <div class="panel">
            <h3>Method</h3>
            {_record_html(method_rows)}
        </div>
    </div>
</section>

<section>
    <h2>Preferred solution</h2>

    {figure_html["cmt"]}

    <div class="panel">
        {_record_html(source_rows)}
    </div>
</section>

<section>
    <h2>Posterior and uncertainty</h2>

    {figure_html["posterior"]}
    {figure_html["uncertainty"]}

    <h3>90% posterior intervals</h3>

    <div class="table-scroll">
        {_dataframe_html(
            interval_df,
            columns=[
                "parameter",
                "preferred",
                "q05",
                "median",
                "q95",
                "unit",
                "source",
            ],
        )}
    </div>

    <details>
        <summary>Posterior diagnostics</summary>
        <div class="table-scroll">
            {_dataframe_html(
                results.get("posterior_diagnostics").T.reset_index()
                if results.get("posterior_diagnostics") is not None
                else None,
                rename={"index": "Diagnostic", 0: "Value"},
            )}
        </div>
    </details>

    <details>
        <summary>Uncertainty sampling diagnostics</summary>
        <div class="table-scroll">
            {_dataframe_html(
                results.get("uncertainty_diagnostics").T.reset_index()
                if results.get("uncertainty_diagnostics") is not None
                else None,
                rename={"index": "Diagnostic", 0: "Value"},
            )}
        </div>
    </details>
</section>

<section>
    <h2>Station diagnostics</h2>

    {figure_html["station_qc"]}
    {figure_html["station_fit"]}
    {figure_html["waveform"]}

    <details>
        <summary>Station/component fit</summary>
        <div class="table-scroll">
            {_dataframe_html(
                station_fit_table,
                columns=station_fit_columns,
            )}
        </div>
    </details>

    <details>
        <summary>Station selection audit</summary>
        <div class="table-scroll">
            {_dataframe_html(
                station_selection,
                columns=selection_columns,
            )}
        </div>
    </details>

    <details>
        <summary>Station jackknife</summary>
        <div class="table-scroll">
            {_dataframe_html(
                jackknife_table,
                columns=jackknife_columns,
            )}
        </div>
    </details>
</section>

<section>
    <h2>Adaptive search</h2>

    {figure_html["adaptive"]}

    <div class="table-scroll">
        {_dataframe_html(
            adaptive_table,
            columns=adaptive_columns,
        )}
    </div>
</section>

<section>
    <h2>Saved outputs</h2>
    {links_html}
</section>

<footer class="footer">
    Generated by BayesISOLA.
    Intervals shown as q05–q95 correspond to central 90% posterior intervals.
    Depth and centroid-time intervals are calculated from the exact discrete
    space-time posterior; MT-derived intervals are calculated from the
    conditional moment-tensor samples.
</footer>

</main>
</body>
</html>
"""

    required_html_markers = (
        "<!DOCTYPE html>",
        "<h2>Event and inversion</h2>",
        "<h2>Preferred solution</h2>",
        "<h2>Posterior and uncertainty</h2>",
        "<h2>Station diagnostics</h2>",
        "</html>",
    )
    missing_markers = [marker for marker in required_html_markers if marker not in html_text]
    if missing_markers:
        raise RuntimeError(
            "Generated report HTML is incomplete; missing marker(s): "
            + ", ".join(missing_markers)
        )

    output_file.write_text(
        html_text,
        encoding="utf-8",
        newline="\n",
    )

    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise OSError(f"HTML report was not written correctly: {output_file}")

    return output_file
