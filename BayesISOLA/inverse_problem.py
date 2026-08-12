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


def invert(point_id, d_shifts, norm_d, Cd_inv, Cd_inv_shifts, nr, comps, stations,
           npts_elemse, npts_slice, elemse_start_origin, origin_time, samprate,
           deviatoric=False, decomp=True, invert_displacement=False,
           elemse_path=None, covariance_factors=None, data_whitened=False, green_dir=None):
    """
    Solves inverse problem in a single grid point for multiple time shifts.

    ``covariance_factors`` is the factorized-noise path used by the modernized
    BayesISOLA workflow.  It applies the already-computed covariance whitening
    factors rather than explicitly multiplying by ``C_D^{-1}``.  The legacy
    ``Cd_inv``/``Cd_inv_shifts`` path remains unchanged for compatibility and
    for shift-dependent ACF covariance matrices. ``green_dir`` selects the Axitra
    workspace when elementary seismograms are read from native files.
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
    if factorized:
        G_work = whiten_covariance_array(G, covariance_factors, stations, npts)
        Gt = G_work.T
        GtG = np.dot(Gt, G_work)
        CN = np.sqrt(np.linalg.cond(GtG))
        GtGinv = np.linalg.inv(GtG)
        det_Ca = np.linalg.det(GtGinv)

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
                det_Ca = np.linalg.det(GtGinv)

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
        'shifts': res,
    }
    if decomp:
        r.update(decompose(a2mt(r['a'])))
    return r
