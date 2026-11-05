#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate Green's function using ``Axitra`` code.

"""

from pathlib import Path
import re
import subprocess
import hashlib

from BayesISOLA._paths import axitra_executable, green_path


def Axitra_wrapper(i, model, x, y, z, npts_exp, elemse_start_origin,
                   logfile='output/log_green.txt', green_dir=None,
                   gr_xyz_executable=None, elemse_executable=None):
    r"""
    Evaluate Green's function using ``Axitra`` code (programs ``gr_xyz`` and ``elemse``) in a given grid point.

    :param i: number (identifier) of grid point
    :type i: integer
    :param model: identifier of crust model
    :type model: string
    :param x: source coordinate in N-S direction [m] (positive to the north)
    :type x: float
    :param y: source coordinate in E-W direction [m] (positive to the east)
    :type y: float
    :param z: source depth [m] (positive down)
    :type z: float
    :param npts_exp: the number of samples in the computation is :math:`2^{\mathrm{npts\_exp}}`
    :type npts_exp: integer
    :param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
    :type elemse_start_origin: float
    :param logfile: path to text file, where are details about computation logged
    :type logfile: string, optional
    :param green_dir: writable Axitra working directory
    :type green_dir: string or path-like, optional
    :param gr_xyz_executable: explicit path to the ``gr_xyz`` executable
    :type gr_xyz_executable: string or path-like, optional
    :param elemse_executable: explicit path to the ``elemse`` executable
    :type elemse_executable: string or path-like, optional

    Remark: because of paralelisation, this wrapper cannot be part of class :class:`BayesISOLA`.
    """
    # gr_xyz is deterministic since its uninitialised `lastik` was fixed, so a
    # single retry only covers transient I/O problems
    iter_max = 2
    point_id = str(i).zfill(4)
    if model:
        point_id += '-' + model

    workspace = Path(green_dir) if green_dir is not None else Path('green')
    gr_xyz = Path(gr_xyz_executable) if gr_xyz_executable is not None else axitra_executable('gr_xyz')
    elemse = Path(elemse_executable) if elemse_executable is not None else axitra_executable('elemse')

    log = open(logfile, 'a', encoding='utf-8', newline='\n')
    for iter in range(iter_max):
        process = subprocess.Popen(
            [str(gr_xyz), '{0:1.3f}'.format(x/1e3), '{0:1.3f}'.format(y/1e3), '{0:1.3f}'.format(z/1e3), point_id, model],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(workspace)
        )
        out, err = process.communicate()
        if process.returncode == 0 and not out:
            if err:
                log.write('grid point {0:3d}: gr_xyz stderr (non-fatal): '.format(i) + err.decode(errors='replace').strip() + '\n')
            break
        else:
            if iter == iter_max-1:
                log.write('grid point {0:3d}, gr_xyz failed {1:2d} times, POINT SKIPPED\n'.format(i, iter))
                if out:
                    log.write('gr_xyz stdout: ' + out.decode(errors='replace').strip() + '\n')
                if err:
                    log.write('gr_xyz stderr: ' + err.decode(errors='replace').strip() + '\n')
                log.close()
                return False
    log.write('grid point {0:3d}, {1:2d} calculation(s)\n'.format(i, iter+1))
    process = subprocess.Popen(
        [str(elemse), str(npts_exp), point_id, "{0:8.3f}".format(elemse_start_origin)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(workspace)
    )
    out, err = process.communicate()
    if process.returncode != 0 or out:
        log.write('grid point {0:3d}: elemse FAILED\n'.format(i, iter))
        if out:
            log.write('elemse stdout: ' + out.decode(errors='replace').strip() + '\n')
        if err:
            log.write('elemse stderr: ' + err.decode(errors='replace').strip() + '\n')
        log.close()
        return False
    if err:
        log.write('grid point {0:3d}: elemse stderr (non-fatal): '.format(i) + err.decode(errors='replace').strip() + '\n')
    log.close()

    meta = open(green_path(workspace, 'elemse'+point_id+'.txt'), 'w', encoding='utf-8', newline='\n')
    md5_crustal = hashlib.md5(open(green_path(workspace, 'crustal.dat'), 'rb').read()).hexdigest()
    md5_station = hashlib.md5(open(green_path(workspace, 'station.dat'), 'rb').read()).hexdigest()
    txt_soutype = open(green_path(workspace, 'soutype.dat'), encoding='utf-8').read().strip().replace('\n', '_')
    meta.write('{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(x/1e3, y/1e3, z/1e3, md5_crustal, md5_station, txt_soutype))
    meta.close()

    return True


# Receivers per gr_xyz run; must not exceed ``nrp`` in fortran/axitra/param.inc
AXITRA_MAX_RECEIVERS = 1000


def Axitra_depth_chunk(chunk_id, model, z, receivers, npts_exp, elemse_start_origin,
                       green_dir, gr_xyz_executable=None, elemse_executable=None):
    """
    Run ``gr_xyz`` and ``elemse`` once for many grid points at one source depth.

    In a 1-D model the Green's functions only depend on the source-receiver
    offset, so every (grid point, station) pair of a depth becomes one receiver
    at its relative offset with the source at (0, 0, z). The output for one grid
    point is then exactly the block of records the per-point run would write.

    Parameters
    ----------
    chunk_id : str
        Identifier of this run, also its working subdirectory name
    model : str
        Crustal model identifier, '' for the single-model case
    z : str
        Source depth [km] formatted as for the per-point run ('{:1.3f}')
    receivers : list of tuple
        Receiver offsets (north, east, depth) [km] relative to the source
    npts_exp : int
        Number of samples is ``2**npts_exp``
    elemse_start_origin : float
        Time between elementary seismogram start and origin time [s]
    green_dir : str or path-like
        Axitra workspace holding grdat, crustal and soutype files
    gr_xyz_executable : str or path-like, optional
        Path to ``gr_xyz``, by default the packaged one
    elemse_executable : str or path-like, optional
        Path to ``elemse``, by default the packaged one

    Returns
    -------
    elemse_file : Path or None
        Path to the elemse output (receivers in the given order), None on failure
    message : str
        Log text (empty when everything ran cleanly)
    """
    green_dir = Path(green_dir)
    suffix = '-' + model if model else ''
    work = green_dir / 'depth_runs' / chunk_id
    work.mkdir(parents=True, exist_ok=True)
    gr_xyz = Path(gr_xyz_executable) if gr_xyz_executable is not None else axitra_executable('gr_xyz')
    elemse = Path(elemse_executable) if elemse_executable is not None else axitra_executable('elemse')

    # same inputs as the per-point run, except the receiver list
    for name in ('crustal' + suffix + '.dat', 'soutype.dat'):
        (work / name).write_bytes((green_dir / name).read_bytes())
    grdat = (green_dir / ('grdat' + suffix + '.hed')).read_text(encoding='utf-8')
    grdat = re.sub(r'\bnr=\d+', 'nr={0:d}'.format(len(receivers)), grdat)
    (work / ('grdat' + suffix + '.hed')).write_text(grdat, encoding='utf-8', newline='\n')
    lines = [' Station co-ordinates', ' x(N>0,km),y(E>0,km),z(km),azim.,dist.,stat.']
    lines += ['{0:10.4f} {1:10.4f} {2:10.4f} {3:10.4f} {4:10.4f} R ?'.format(x, y, zr, 0.0, 0.0) for x, y, zr in receivers]
    (work / ('station' + suffix + '.dat')).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

    message = ''
    for attempt in range(2):
        process = subprocess.run(
            [str(gr_xyz), '0.000', '0.000', z, chunk_id, model],
            capture_output=True, cwd=str(work),
        )
        if process.returncode == 0 and not process.stdout:
            break
        message = 'depth {0:s} km ({1:s}): gr_xyz failed (attempt {2:d}): {3:s}\n'.format(
            z, chunk_id, attempt + 1, (process.stdout + process.stderr).decode(errors='replace').strip())
    else:
        return None, message
    process = subprocess.run(
        [str(elemse), str(npts_exp), chunk_id, '{0:8.3f}'.format(elemse_start_origin)],
        capture_output=True, cwd=str(work),
    )
    # the big frequency-domain file is not needed once elemse has run
    (work / ('gr' + chunk_id + '.hes')).unlink(missing_ok=True)
    if process.returncode != 0 or process.stdout:
        return None, message + 'depth {0:s} km ({1:s}): elemse FAILED: {2:s}\n'.format(
            z, chunk_id, (process.stdout + process.stderr).decode(errors='replace').strip())
    return work / ('elemse' + chunk_id + '.dat'), message
