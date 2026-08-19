#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solves inverse problem in a single grid point for multiple time shifts.
"""

import numpy as np

from obspy import UTCDateTime

from BayesISOLA.fileformats import read_elemse, read_elemse_from_files
from BayesISOLA.helpers import my_filter
from BayesISOLA.MT_comps import decompose, a2mt
from BayesISOLA._paths import green_path


def _covariance_determinant_metrics(matrix):
    """Return a finite-log determinant and a backward-compatible determinant.

    ``GtGinv`` is a covariance matrix and should therefore have a positive
    determinant.  ``slogdet`` keeps the logarithm numerically stable even when
    the determinant itself would overflow or underflow in floating point.
    """
    sign, log_det = np.linalg.slogdet(np.asarray(matrix, dtype=float))
    if sign <= 0 or not np.isfinite(log_det):
        return np.nan, -np.inf

    log_det = float(log_det)
    log_max = float(np.log(np.finfo(float).max))
    log_min = float(np.log(np.nextafter(0.0, 1.0)))

    if log_det > log_max:
        det = np.inf
    elif log_det < log_min:
        det = 0.0
    else:
        det = float(np.exp(log_det))
    return det, log_det


def whiten_covariance_array(values, covariance_factors, stations, npts):
    """Apply BayesISOLA's stored covariance whitening factors block-by-block.

    The factors satisfy ``W.T @ W = C_D^{-1}``.  ``LT3`` contains one full
    three-component station factor when cross-covariance is enabled; otherwise
    ``LT`` contains independent factors for the used Z/N/E component blocks.
    The transformation therefore preserves the original generalized
    least-squares objective exactly, apart from floating-point roundoff.
    """
    LT = covariance_factors.get("LT", [])
    LT3 = covariance_factors.get("LT3", [])
    output = np.empty_like(values, dtype=np.result_type(values.dtype, np.float64))
    offset = 0

    for r, station in enumerate(stations):
        used = [
            comp for comp, key in enumerate(("useZ", "useN", "useE"))
            if station[key]
        ]
        size = len(used) * npts
        if not size:
            continue

        if LT3:
            output[offset:offset+size] = np.dot(LT3[r], values[offset:offset+size])
        else:
            local = 0
            for comp in used:
                sl = slice(offset + local*npts, offset + (local+1)*npts)
                output[sl] = np.dot(LT[r][comp], values[sl])
                local += 1
        offset += size

    return output


def _station_normal_equations(G, D, stations, npts):
    """Return station-wise contributions to ``G.T@G`` and ``G.T@D``.

    Only components that participate in the inversion are included.  The
    returned station indices preserve the same active-station order used by
    BayesISOLA's concatenated data vector.  These small sufficient statistics
    allow an exact fixed-grid leave-one-station-out calculation later without
    rereading or refiltering the elementary seismograms.
    """
    G = np.asarray(G, dtype=float)
    D = np.asarray(D, dtype=float)
    if G.ndim != 2 or D.ndim != 2 or G.shape[0] != D.shape[0]:
        raise ValueError("G and D must be two-dimensional with equal row counts.")

    station_indices = []
    station_GtG = []
    station_Gtd = []
    offset = 0
    for station_index, station in enumerate(stations):
        n_components = sum(bool(station[key]) for key in ("useZ", "useN", "useE"))
        size = int(n_components) * int(npts)
        if not size:
            continue
        sl = slice(offset, offset + size)
        Gs = G[sl, :]
        Ds = D[sl, :]
        station_indices.append(station_index)
        station_GtG.append(Gs.T @ Gs)
        station_Gtd.append(Gs.T @ Ds)
        offset += size

    if offset != G.shape[0]:
        raise RuntimeError(
            "Station normal-equation blocks do not span the assembled design matrix: "
            f"covered={offset}, rows={G.shape[0]}."
        )

    ne = G.shape[1]
    n_shifts = D.shape[1]
    if station_GtG:
        A = np.stack(station_GtG, axis=0)
        B = np.stack(station_Gtd, axis=0)
    else:
        A = np.empty((0, ne, ne), dtype=float)
        B = np.empty((0, ne, n_shifts), dtype=float)
    return np.asarray(station_indices, dtype=int), A, B


def invert(point_id, d_shifts, norm_d, Cd_inv, Cd_inv_shifts, nr, comps, stations,
           npts_elemse, npts_slice, elemse_start_origin, origin_time, samprate,
           deviatoric=False, decomp=True, invert_displacement=False,
           elemse_path=None, covariance_factors=None, data_whitened=False, green_dir=None,
           store_station_normal_equations=False):
    """
    Solves inverse problem in a single grid point for multiple time shifts.

    ``covariance_factors`` is the factorized-noise path used by the modernized
    BayesISOLA workflow.  It applies the already-computed covariance whitening
    factors rather than explicitly multiplying by ``C_D^{-1}``.  The legacy
    ``Cd_inv``/``Cd_inv_shifts`` path remains unchanged for compatibility and
    for shift-dependent ACF covariance matrices. ``green_dir`` selects the Axitra
    workspace when elementary seismograms are read from native files.

    When ``store_station_normal_equations`` is true, the inversion also returns
    each active station's contributions to ``G.T@G`` and ``G.T@d`` for all time
    shifts.  This does not change the inversion objective; it only retains small
    sufficient statistics needed by the workflow's exact station jackknife.
    """
    if deviatoric:
        ne = 5
    else:
        ne = 6

    if elemse_path:
        elemse = read_elemse_from_files(
            nr, elemse_path, stations, origin_time, samprate,
            npts_elemse, invert_displacement,
        )
    else:
        elemse = read_elemse(
            nr, npts_elemse, green_path(green_dir if green_dir is not None else 'green', 'elemse'+point_id+'.dat'),
            stations, invert_displacement,
        )

    for r in range(nr):
        for i in range(ne):
            my_filter(elemse[r][i], stations[r]['fmin'], stations[r]['fmax'])

    if npts_slice != npts_elemse:
        for st6 in elemse:
            for st in st6:
                st.trim(UTCDateTime(0)+elemse_start_origin)
        npts = npts_slice
    else:
        npts = npts_elemse

    c = 0
    G = np.empty((comps*npts, ne))
    for r in range(nr):
        for comp in range(3):
            if stations[r][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
                weight = stations[r][{0:'weightZ', 1:'weightN', 2:'weightE'}[comp]]
                for i in range(npts):
                    for e in range(ne):
                        G[c*npts+i, e] = elemse[r][e][comp].data[i] * weight
                c += 1

    factorized = bool(covariance_factors) and not Cd_inv_shifts
    station_normal = None

    if factorized:
        G_work = whiten_covariance_array(G, covariance_factors, stations, npts)
        Gt = G_work.T
        GtG = np.dot(Gt, G_work)
        CN = np.sqrt(np.linalg.cond(GtG))
        GtGinv = np.linalg.inv(GtG)
        det_Ca, log_det_Ca = _covariance_determinant_metrics(GtGinv)

        if store_station_normal_equations:
            if data_whitened:
                D_normal = np.column_stack([
                    np.asarray(d_shift, dtype=float).reshape(-1)
                    for d_shift in d_shifts
                ])
            else:
                D_normal = np.column_stack([
                    whiten_covariance_array(
                        d_shift, covariance_factors, stations, npts
                    ).reshape(-1)
                    for d_shift in d_shifts
                ])
            station_normal = _station_normal_equations(
                G_work, D_normal, stations, npts
            )

    elif store_station_normal_equations and not Cd_inv and not Cd_inv_shifts:
        D_normal = np.column_stack([
            np.asarray(d_shift, dtype=float).reshape(-1)
            for d_shift in d_shifts
        ])
        station_normal = _station_normal_equations(G, D_normal, stations, npts)

    res = {}
    for shift in range(len(d_shifts)):
        d_shift = d_shifts[shift]

        if factorized:
            if data_whitened:
                d_work = d_shift
            else:
                d_work = whiten_covariance_array(
                    d_shift, covariance_factors, stations, npts
                )
            Gtd = np.dot(Gt, d_work)
            a = np.dot(GtGinv, Gtd)
            if deviatoric:
                a = np.append(a, [[0.]], axis=0)
            residual = d_work - np.dot(G_work, a[:ne])
            misfit = np.dot(residual.T, residual)[0, 0]

        else:
            if Cd_inv_shifts:
                Cd_inv = Cd_inv_shifts[shift]

            if 'Gt' in vars() and not Cd_inv_shifts:
                pass
            elif Cd_inv:
                idx = 0
                GtCd = []
                for C in Cd_inv:
                    size = len(C)
                    GtCd.append(np.dot(G[idx:idx+size, :].T, C))
                    idx += size
                Gt = np.concatenate(GtCd, axis=1)
            else:
                Gt = G.transpose()

            if not 'det_Ca' in vars() or Cd_inv_shifts:
                GtG = np.dot(Gt, G)
                CN = np.sqrt(np.linalg.cond(GtG))
                GtGinv = np.linalg.inv(GtG)
                det_Ca, log_det_Ca = _covariance_determinant_metrics(GtGinv)

            Gtd = np.dot(Gt, d_shift)
            a = np.dot(GtGinv, Gtd)
            if deviatoric:
                a = np.append(a, [[0.]], axis=0)

            if Cd_inv:
                dGm = d_shift - np.dot(G, a[:ne])
                idx = 0
                dGmCd_blocks = []
                for C in Cd_inv:
                    size = len(C)
                    dGmCd_blocks.append(np.dot(dGm[idx:idx+size, :].T, C))
                    idx += size
                dGmCd = np.concatenate(dGmCd_blocks, axis=1)
                misfit = np.dot(dGmCd, dGm)[0, 0]
            else:
                synt = np.zeros(comps*npts)
                for i in range(ne):
                    synt += G[:, i] * a[i]
                misfit = 0
                for i in range(npts*comps):
                    misfit += (d_shift[i, 0]-synt[i])**2

        VR = 1 - misfit / norm_d[shift]
        res[shift] = {
            'a': a.copy(),
            'misfit': misfit,
            'VR': VR,
            'CN': CN,
            'GtGinv': GtGinv,
            'det_Ca': det_Ca,
            'log_det_Ca': log_det_Ca,
        }

    shift = max(res, key=lambda s: res[s]['VR'])
    r = {
        'shift': shift,
        'a': res[shift]['a'].copy(),
        'VR': res[shift]['VR'],
        'misfit': res[shift]['misfit'],
        'CN': res[shift]['CN'],
        'GtGinv': res[shift]['GtGinv'],
        'det_Ca': res[shift]['det_Ca'],
        'log_det_Ca': res[shift]['log_det_Ca'],
        'shifts': res,
    }
    if station_normal is not None:
        r['_station_normal_indices'] = station_normal[0]
        r['_station_GtG'] = station_normal[1]
        r['_station_Gtd'] = station_normal[2]

    if decomp:
        r.update(decompose(a2mt(r['a'])))
    return r
