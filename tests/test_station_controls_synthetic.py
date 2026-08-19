"""Synthetic regression tests for BayesISOLA 0.2 station controls."""

import numpy as np
import pandas as pd
import pytest

from BayesISOLA import workflows as wf


def test_public_option_normalization():
    adaptive = wf._normalize_adaptive_grid_search(True)
    assert adaptive["adaptive_grid"] is True
    assert wf._normalize_adaptive_grid_search(None)["adaptive_grid"] is False
    assert wf._normalize_adaptive_grid_search(False)["adaptive_grid"] is False
    assert (
        wf._normalize_adaptive_grid_search({"adaptive_refine_factor": 0})[
            "adaptive_refine_factor"
        ]
        == 0.0
    )

    azimuth = wf._normalize_azimuth_control(True)
    assert azimuth == {
        "azimuth_selection": True,
        "azimuth_min_sectors": 3,
        "azimuth_max_stations_per_sector": 2,
        "minimum_stations": 4,
    }
    assert wf._normalize_azimuth_control(None)["azimuth_selection"] is False

    jackknife = wf._normalize_station_jackknife(True)
    assert jackknife == {"enabled": True, "jackknife_min_stations": 4}
    assert wf._normalize_station_jackknife(None)["enabled"] is False
    assert wf._normalize_station_jackknife(False)["enabled"] is False


def test_dynamic_channel_priority_preserves_legacy_and_rule_semantics():
    legacy = wf._normalize_channel_priority(
        ("HH", "BH", "LH"),
        ("HH?", "BH?", "LH?"),
    )
    assert legacy["mode"] == "static"
    assert legacy["query_patterns"] == ("HH?", "BH?", "LH?")

    rules = wf._normalize_channel_priority(
        {
            "mag_range": [[4.0, 5.0], [5.0, 6.0]],
            "dist_range": [[10, 250], [40, 300]],
            "channels": [["BH", "HH"], ["HN", "BN"]],
            "default": ["HH", "BH", "LH"],
        },
        ("HH?", "BH?", "LH?"),
    )
    assert rules["query_patterns"] == ("HH?", "BH?", "LH?", "HN?", "BN?")
    assert wf._resolve_channel_priority(rules, magnitude=4.5, distance_km=100)[0][:2] == (
        "BH",
        "HH",
    )
    # Half-open magnitude ranges: Mw=5.0 belongs to [5, 6), not [4, 5).
    assert wf._resolve_channel_priority(rules, magnitude=5.0, distance_km=100)[0][:2] == (
        "HN",
        "BN",
    )
    assert wf._resolve_channel_priority(rules, magnitude=5.5, distance_km=350)[0] == (
        "HH",
        "BH",
        "LH",
    )

    with pytest.raises(ValueError):
        wf._normalize_channel_priority(
            {
                "mag_range": [[4, 5]],
                "dist_range": [[10, 20], [20, 30]],
                "channels": [["BH"]],
            },
            ("HH?",),
        )


def _station_table():
    return pd.DataFrame(
        [
            # Sector 0: A is preferred by HH, B is fallback BH.
            {
                "network": "NZ",
                "station": "AAA",
                "location": "",
                "channel_prefix": "HH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 30.0,
                "azimuth_deg": 10.0,
            },
            {
                "network": "NZ",
                "station": "BBB",
                "location": "",
                "channel_prefix": "BH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 20.0,
                "azimuth_deg": 20.0,
            },
            # Sector 1: HH is farther than BH; channel precedence must win.
            {
                "network": "NZ",
                "station": "CCC",
                "location": "",
                "channel_prefix": "BH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 10.0,
                "azimuth_deg": 55.0,
            },
            {
                "network": "NZ",
                "station": "DDD",
                "location": "",
                "channel_prefix": "HH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 40.0,
                "azimuth_deg": 60.0,
            },
            {
                "network": "NZ",
                "station": "EEE",
                "location": "",
                "channel_prefix": "HH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 50.0,
                "azimuth_deg": 110.0,
            },
            {
                "network": "NZ",
                "station": "FFF",
                "location": "",
                "channel_prefix": "HH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 60.0,
                "azimuth_deg": 155.0,
            },
            {
                "network": "NZ",
                "station": "GGG",
                "location": "",
                "channel_prefix": "HH",
                "station_lat": -45.0,
                "station_lon": 170.0,
                "distance_km": 70.0,
                "azimuth_deg": 205.0,
            },
        ]
    )


def test_manual_drop_precedes_azimuth_selection_and_backfills_sector():
    stations = _station_table()
    config = wf._normalize_azimuth_control(
        {
            "azimuth_selection": True,
            "azimuth_min_sectors": 3,
            "azimuth_max_stations_per_sector": 1,
            "minimum_stations": 4,
        }
    )
    selected, audit = wf._apply_station_controls(
        stations,
        drop_stations=["AAA"],
        azimuth_config=config,
        channel_priority=("HH", "BH", "LH"),
        channels=("HH?", "BH?", "LH?"),
        magnitude=4.5,
        event_lat=-45.0,
        event_lon=170.0,
    )

    selected_codes = set(selected["station"])
    assert "AAA" not in selected_codes
    assert "BBB" in selected_codes  # sector back-fill after explicit drop
    assert "DDD" in selected_codes and "CCC" not in selected_codes
    assert set(audit.loc[audit.selection_status == "manual_drop", "station"]) == {"AAA"}
    assert "CCC" in set(audit.loc[audit.selection_status == "azimuth_excluded", "station"])


def test_station_omission_sufficient_statistics_match_explicit_direct_solve():
    rng = np.random.default_rng(91)
    n_stations, n_parameters, n_shifts = 5, 6, 7
    rows_per_station = [12, 15, 11, 14, 13]
    g_blocks = [rng.normal(size=(n, n_parameters)) for n in rows_per_station]
    d_blocks = [rng.normal(size=(n, n_shifts)) for n in rows_per_station]
    G = np.vstack(g_blocks)
    D = np.vstack(d_blocks)
    A_total = G.T @ G
    B_total = G.T @ D
    q_total = np.sum(D * D, axis=0)

    for heldout in range(n_stations):
        Gs, Ds = g_blocks[heldout], d_blocks[heldout]
        fast = wf._solve_omission_normal_equations(
            A_total,
            B_total,
            q_total,
            Gs.T @ Gs,
            Gs.T @ Ds,
            np.sum(Ds * Ds, axis=0),
        )

        Gminus = np.vstack([block for i, block in enumerate(g_blocks) if i != heldout])
        Dminus = np.vstack([block for i, block in enumerate(d_blocks) if i != heldout])
        A_direct = Gminus.T @ Gminus
        B_direct = Gminus.T @ Dminus
        coefficients = np.linalg.solve(A_direct, B_direct)
        residual = Dminus - Gminus @ coefficients
        misfit = np.sum(residual * residual, axis=0)
        q_direct = np.sum(Dminus * Dminus, axis=0)
        vr = 1.0 - misfit / q_direct

        np.testing.assert_allclose(fast["A"], A_direct, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fast["coefficients"], coefficients, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(fast["misfit"], misfit, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(fast["variance_reduction"], vr, rtol=1e-11, atol=1e-11)

class _Channel:
    def __init__(self, code):
        self.code = code
        self.location_code = ""
        self.sample_rate = 100.0
        self.depth = 0.0
        self.latitude = -43.0
        self.longitude = 172.0
        self.elevation = 10.0


class _Station:
    def __init__(self, codes):
        self.code = "ONLY"
        self.latitude = -43.0
        self.longitude = 172.0
        self.elevation = 10.0
        self.channels = [_Channel(code) for code in codes]


class _Network:
    def __init__(self, codes):
        self.code = "NZ"
        self.stations = [_Station(codes)]

    def __iter__(self):
        return iter(self.stations)


class _Inventory:
    def __init__(self, codes):
        self.networks = [_Network(codes)]

    def __iter__(self):
        return iter(self.networks)


def test_rule_only_family_is_not_selected_outside_matching_rule():
    rules = wf._normalize_channel_priority(
        {
            "mag_range": [[5.0, 6.0]],
            "dist_range": [[0.0, 1000.0]],
            "channels": [["HN", "BN"]],
            "default": ["HH", "BH", "LH"],
        },
        ("HH?", "BH?", "LH?"),
    )
    inventory = _Inventory(["HNZ", "HNN", "HNE"])

    outside = wf._inventory_channel_families(
        inventory,
        rules,
        ground_level=True,
        channels=rules["query_patterns"],
        magnitude=4.5,
        event_lat=-43.0,
        event_lon=172.0,
    )
    assert outside.empty

    inside = wf._inventory_channel_families(
        inventory,
        rules,
        ground_level=True,
        channels=rules["query_patterns"],
        magnitude=5.5,
        event_lat=-43.0,
        event_lon=172.0,
    )
    assert len(inside) == 1
    assert inside.iloc[0]["channel_prefix"] == "HN"
