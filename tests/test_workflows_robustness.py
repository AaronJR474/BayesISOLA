import importlib.util
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BayesISOLA" / "workflows.py"


def _load_workflows():
    spec = importlib.util.spec_from_file_location("bayesisola_workflows_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_station_csv_preserves_seed_identifiers_and_literal_na(tmp_path):
    module = _load_workflows()
    csv = tmp_path / "stations.csv"
    csv.write_text(
        'network,station,location,channel_prefix,station_id,component_scheme,selected_channels,channels,gf_model,distance_km\n'
        'NA,0123,00,HH,NA.0123.00.HH,ZNE,"HHZ,HHN,HHE","HHZ,HHN,HHE",m001,12.3\n',
        encoding="utf-8",
    )

    table = module._coerce_station_df(csv)
    row = table.iloc[0]

    assert row["network"] == "NA"
    assert row["station"] == "0123"
    assert row["location"] == "00"
    assert row["channel_prefix"] == "HH"
    assert row["station_id"] == "NA.0123.00.HH"
    assert row["gf_model"] == "m001"
    assert float(row["distance_km"]) == pytest.approx(12.3)


def test_gf_helpers_import_does_not_fall_back_to_unrelated_top_level_module(monkeypatch):
    module = _load_workflows()
    package = types.ModuleType("BayesISOLA")
    package.__path__ = []
    poison = types.ModuleType("gf_helpers")

    monkeypatch.setitem(sys.modules, "BayesISOLA", package)
    monkeypatch.setitem(sys.modules, "gf_helpers", poison)
    sys.modules.pop("BayesISOLA.gf_helpers", None)

    with pytest.raises(ImportError):
        module._get_gf_helpers_module()


def _install_fake_beachball(monkeypatch, calls):
    obspy = types.ModuleType("obspy")
    imaging = types.ModuleType("obspy.imaging")
    beachball = types.ModuleType("obspy.imaging.beachball")

    def beach(
        fm, xy=(0, 0), width=200, size=100, linewidth=2,
        facecolor="b", bgcolor="w", edgecolor="k", nofill=False, zorder=100,
    ):
        calls.append({"fm": fm, "nofill": nofill})
        return PatchCollection([])

    class MomentTensor:
        def __init__(self, *args):
            self.args = args

    axis = lambda val, dip, strike: types.SimpleNamespace(val=val, dip=dip, strike=strike)

    def mt2axes(moment_tensor):
        return axis(1.0, 25.0, 10.0), axis(0.0, 0.0, 90.0), axis(-1.0, 35.0, 190.0)

    beachball.beach = beach
    beachball.MomentTensor = MomentTensor
    beachball.mt2axes = mt2axes
    imaging.beachball = beachball
    obspy.imaging = imaging

    monkeypatch.setitem(sys.modules, "obspy", obspy)
    monkeypatch.setitem(sys.modules, "obspy.imaging", imaging)
    monkeypatch.setitem(sys.modules, "obspy.imaging.beachball", beachball)


def _summary(**updates):
    values = {
        "Mrr": 1.0e15,
        "Mtt": -0.6e15,
        "Mpp": -0.4e15,
        "Mrt": 0.1e15,
        "Mrp": -0.2e15,
        "Mtp": 0.05e15,
        "M0_Nm": 1.2e15,
        "Mw": 4.0,
        "DC_percent": 80.0,
        "CLVD_percent": 15.0,
        "ISO_percent": 5.0,
        "variance_reduction": 0.72,
        "condition_number": 12.0,
        "NP1_strike_deg": 10.0,
        "NP1_dip_deg": 45.0,
        "NP1_rake_deg": 90.0,
        "NP2_strike_deg": 190.0,
        "NP2_dip_deg": 45.0,
        "NP2_rake_deg": 90.0,
    }
    values.update(updates)
    return values


def test_plot_cmt_summary_is_compatible_with_beach_without_plot_zerotrace(tmp_path, monkeypatch):
    module = _load_workflows()
    calls = []
    _install_fake_beachball(monkeypatch, calls)

    output = tmp_path / "summary.png"
    fig = module.plot_cmt_summary(
        _summary(), {"centroid_depth_km": 8.0}, output_file=output, show=False,
    )
    plt.close(fig)

    assert output.is_file()
    assert len(calls) == 2
    assert calls[0]["nofill"] is False
    assert calls[1]["nofill"] is True


def test_plot_cmt_summary_handles_undefined_isotropic_nodal_planes(tmp_path, monkeypatch):
    module = _load_workflows()
    calls = []
    _install_fake_beachball(monkeypatch, calls)

    output = tmp_path / "isotropic.png"
    fig = module.plot_cmt_summary(
        _summary(
            DC_percent=0.0,
            CLVD_percent=0.0,
            ISO_percent=100.0,
            NP1_strike_deg=np.nan,
            NP1_dip_deg=np.nan,
            NP1_rake_deg=np.nan,
            NP2_strike_deg=np.nan,
            NP2_dip_deg=np.nan,
            NP2_rake_deg=np.nan,
        ),
        {"centroid_depth_km": 8.0},
        output_file=output,
        show=False,
    )
    plt.close(fig)

    assert output.is_file()
    assert len(calls) == 1  # no undefined DC nodal-plane overlay
