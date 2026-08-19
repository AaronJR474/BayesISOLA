"""End-to-end synthetic test of the fast exact 0.2 station jackknife."""

from types import SimpleNamespace

import numpy as np

from BayesISOLA.MT_comps import a2mt, decompose
from BayesISOLA import inverse_problem
from BayesISOLA import workflows as wf


def test_cached_station_jackknife_matches_brute_force_fixed_grid(monkeypatch):
    rng = np.random.default_rng(91)
    n_stations, npts, n_parameters, n_shifts = 5, 9, 6, 3
    n_values = n_stations * npts
    shifts = np.array([-0.2, 0.0, 0.2])
    D = rng.normal(size=(n_values, n_shifts))

    stations = [
        {
            "network": "NZ",
            "code": f"S{i}",
            "location": "",
            "dist": (i + 1) * 10000.0,
            "az": i * 70.0,
            "useZ": True,
            "useN": False,
            "useE": False,
            "VR_Z": 0.8 - i * 0.02,
            "VR_N": None,
            "VR_E": None,
        }
        for i in range(n_stations)
    ]

    grid = []
    G_by_id = {}
    for i, depth in enumerate((9000.0, 10000.0, 11000.0)):
        G = rng.normal(size=(n_values, n_parameters))
        if i == 1:
            true_a = np.array([1.3, -0.8, 0.6, 0.3, -0.4, 2.0])
            D[:, 1] = G @ true_a + rng.normal(scale=0.15, size=n_values)
        point_id = str(i).zfill(4)
        G_by_id[point_id] = G
        station_indices, station_GtG, station_Gtd = inverse_problem._station_normal_equations(
            G, D, stations, npts
        )
        grid.append(
            {
                "id": point_id,
                "x": 0.0,
                "y": 0.0,
                "z": depth,
                "err": 0,
                "edge": False,
                "_station_normal_indices": station_indices,
                "_station_GtG": station_GtG,
                "_station_Gtd": station_Gtd,
            }
        )

    q = np.sum(D * D, axis=0)
    full_best = None
    for gp in grid:
        G = G_by_id[gp["id"]]
        A = G.T @ G
        B = G.T @ D
        coefficients = np.linalg.solve(A, B)
        misfit = q - np.sum(B * coefficients, axis=0)
        vr = 1.0 - misfit / q
        shift_index = int(np.argmax(vr))
        gp["shifts"] = {
            s: {
                "a": coefficients[:, s, None],
                "misfit": float(misfit[s]),
                "VR": float(vr[s]),
                "CN": float(np.sqrt(np.linalg.cond(A))),
                "GtGinv": np.linalg.inv(A),
                "det_Ca": float(np.linalg.det(np.linalg.inv(A))),
                "log_det_Ca": float(np.linalg.slogdet(np.linalg.inv(A))[1]),
                "c": 1.0,
            }
            for s in range(n_shifts)
        }
        gp["shift_idx"] = shift_index
        gp["shift"] = float(shifts[shift_index])
        gp["a"] = coefficients[:, shift_index, None]
        gp["misfit"] = float(misfit[shift_index])
        gp["VR"] = float(vr[shift_index])
        gp["CN"] = float(np.sqrt(np.linalg.cond(A)))
        if full_best is None or gp["VR"] > full_best["VR"]:
            full_best = gp

    solution = SimpleNamespace()
    solution.inp = SimpleNamespace(stations=stations, nr=n_stations)
    solution.deviatoric = False
    solution.centroid = full_best
    solution.grid = grid
    solution.mt_decomp = decompose(a2mt(full_best["a"]))
    solution.g = SimpleNamespace(
        grid=grid,
        step_x=1000.0,
        step_z=1000.0,
        radius=0.0,
        circle_shape=True,
    )

    data = SimpleNamespace(
        npts_slice=npts,
        d_shifts=[D[:, i, None] for i in range(n_shifts)],
        shifts=shifts,
        components=n_stations,
    )
    cova = SimpleNamespace(
        Cd_inv_shifts=[],
        factorized_noise=False,
        has_covariance=False,
        LT=[],
        LT3=[],
    )

    # The fast path must not reopen/refilter Green functions.
    monkeypatch.setattr(
        wf,
        "_jackknife_design_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached jackknife unexpectedly reconstructed Green functions")
        ),
    )

    output = wf.compute_station_jackknife(
        solution,
        data,
        cova,
        jackknife_min_stations=4,
    )
    assert len(output) == n_stations
    assert set(output["station"]) == {f"S{i}" for i in range(n_stations)}
    assert output["loo_Mw"].notna().all()
    assert output["heldout_whitened_misfit"].notna().all()
    assert (output["n_stations_remaining"] == 4).all()

    for heldout in range(n_stations):
        keep = np.ones(n_values, dtype=bool)
        keep[heldout * npts : (heldout + 1) * npts] = False
        Dminus = D[keep]
        qminus = np.sum(Dminus * Dminus, axis=0)
        brute = None

        for gp in grid:
            Gminus = G_by_id[gp["id"]][keep]
            A = Gminus.T @ Gminus
            B = Gminus.T @ Dminus
            coefficients = np.linalg.solve(A, B)
            misfit = qminus - np.sum(B * coefficients, axis=0)
            vr = 1.0 - misfit / qminus
            shift_index = int(np.argmax(vr))
            if brute is None or vr[shift_index] > brute["vr"]:
                brute = {
                    "gp": gp,
                    "shift": shift_index,
                    "coeff": coefficients[:, shift_index, None],
                    "vr": float(vr[shift_index]),
                }

        row = output.loc[output.station == f"S{heldout}"].iloc[0]
        brute_decomp = decompose(a2mt(brute["coeff"]))
        assert np.isclose(row.loo_variance_reduction, brute["vr"], rtol=1e-10, atol=1e-10)
        assert np.isclose(row.loo_depth_km, brute["gp"]["z"] / 1000.0)
        assert np.isclose(row.loo_Mw, brute_decomp["Mw"], rtol=1e-10, atol=1e-10)
        assert np.isclose(
            row.delta_time_s,
            shifts[brute["shift"]] - float(full_best["shift"]),
        )
