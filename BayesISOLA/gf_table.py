#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Green's function tables on a regular (source depth, epicentral distance) grid.

In a 1-D model the elementary seismograms of a receiver at distance ``r`` and
azimuth ``phi`` follow exactly from those at azimuth 0: the six elementary
moment tensors are rotated by ``phi`` and the horizontal components rotated
back. A table computed once per crustal model and Axitra parameter set (with
receivers at azimuth 0) therefore gives any receiver by interpolation in
distance and depth followed by that exact rotation. Only the interpolation is
an approximation.
"""

import math
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np

from BayesISOLA.axitra import AXITRA_MAX_RECEIVERS, Axitra_depth_chunk
from BayesISOLA.fileformats import ELEMSE_RECORD

# Elementary moment tensors of BayesISOLA (NEZ, see MT_comps.a2mt)
ELEMENTARY_MT = np.array([
	[[0, 1, 0], [1, 0, 0], [0, 0, 0]],
	[[0, 0, 1], [0, 0, 0], [1, 0, 0]],
	[[0, 0, 0], [0, 0, -1], [0, -1, 0]],
	[[-1, 0, 0], [0, 0, 0], [0, 0, 1]],
	[[0, 0, 0], [0, -1, 0], [0, 0, 1]],
	[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
], dtype=float)


def mt_coefficients(mt):
	"""
	Coefficients of a moment tensor in the elementary basis (inverse of a2mt)

	Parameters
	----------
	mt : np.ndarray
		Moment tensors, shape (..., 3, 3), NEZ

	Returns
	-------
	a : np.ndarray
		Coefficients, shape (..., 6)
	"""
	a6 = (mt[..., 0, 0] + mt[..., 1, 1] + mt[..., 2, 2]) / 3
	return np.stack([mt[..., 0, 1], mt[..., 0, 2], -mt[..., 1, 2], a6 - mt[..., 0, 0], a6 - mt[..., 1, 1], a6], axis=-1)


def rotate_from_azimuth_zero(g0, azimuth):
	"""
	Elementary seismograms at an azimuth from those of a receiver at azimuth 0

	Parameters
	----------
	g0 : np.ndarray
		Elementary seismograms at azimuth 0 (receiver due north), shape
		(n, 6, npts, 3) with components N, E, Z
	azimuth : np.ndarray
		Receiver azimuth from the source [rad, clockwise from north], shape (n,)

	Returns
	-------
	g : np.ndarray
		Elementary seismograms at the azimuths, same shape as ``g0``
	"""
	cos, sin = np.cos(azimuth), np.sin(azimuth)
	zero, one = np.zeros_like(cos), np.ones_like(cos)
	# R maps the rotated frame (x' towards the receiver) to NEZ
	rot = np.stack([np.stack([cos, -sin, zero], -1), np.stack([sin, cos, zero], -1), np.stack([zero, zero, one], -1)], -2)
	# each elementary tensor seen in the rotated frame, in the elementary basis
	rotated = np.einsum('nij,kil,nlm->nkjm', rot, ELEMENTARY_MT, rot, optimize=True)
	coefficients = mt_coefficients(rotated)
	gp = np.einsum('nkj,njtc->nktc', coefficients, g0, optimize=True)
	g = np.empty_like(gp)
	g[..., 0] = cos[:, None, None] * gp[..., 0] - sin[:, None, None] * gp[..., 1]
	g[..., 1] = sin[:, None, None] * gp[..., 0] + cos[:, None, None] * gp[..., 1]
	g[..., 2] = gp[..., 2]
	return g


class GreenTable:
	"""
	Elementary seismograms on a (source depth, distance) grid at azimuth 0.

	Parameters
	----------
	depths : np.ndarray
		Source depths [km], increasing
	distances : np.ndarray
		Epicentral distances [km], increasing and evenly spaced
	data : np.ndarray
		float32, shape (depths, distances, 6, npts, 3), components N, E, Z
	times : np.ndarray
		Time of each sample [s], shape (npts,)
	param_hash : str
		:func:`BayesISOLA.gf_store.axitra_parameter_hash` of the inputs
	"""

	def __init__(self, depths, distances, data, times, param_hash):
		self.depths = np.asarray(depths, dtype=float)
		self.distances = np.asarray(distances, dtype=float)
		self.data = data
		self.times = np.asarray(times, dtype=np.float32)
		self.param_hash = str(param_hash)

	@classmethod
	def build(cls, green_dir, model, depths, distances, npts_exp, elemse_start_origin, param_hash, threads=1,
			gr_xyz_executable=None, elemse_executable=None):
		"""
		Compute a table with Axitra, one run per depth (chunked by receivers)

		Parameters
		----------
		green_dir : str or path-like
			Axitra workspace with the crustal, grdat and soutype files
		model : str
			Crustal model identifier, '' for the single-model case
		depths : list of str
			Source depths [km] formatted as for gr_xyz ('{:1.3f}')
		distances : np.ndarray
			Epicentral distances [km]
		npts_exp : int
			Number of samples is ``2**npts_exp``
		elemse_start_origin : float
			Time between elementary seismogram start and origin time [s]
		param_hash : str
			Hash of the Axitra inputs, stored with the table
		threads : int, optional
			Parallel Axitra processes, by default 1
		gr_xyz_executable, elemse_executable : str, optional
			Paths to the Axitra executables

		Returns
		-------
		table : GreenTable
			The computed table
		"""
		npts = 2 ** npts_exp
		receivers = [(float('{0:.4f}'.format(r)), 0.0, 0.0) for r in distances]
		per_run = min(AXITRA_MAX_RECEIVERS, len(receivers))
		tasks = []
		for d, z in enumerate(depths):
			for c, start in enumerate(range(0, len(receivers), per_run)):
				chunk_id = 't{0:04d}_{1:02d}'.format(d, c)
				tasks.append((chunk_id, model, z, receivers[start:start+per_run], npts_exp, elemse_start_origin,
							  str(green_dir), gr_xyz_executable, elemse_executable))
		data = np.empty((len(depths), len(receivers), 6, npts, 3), dtype=np.float32)
		times = None
		with mp.Pool(processes=max(int(threads), 1)) as pool:
			results = [pool.apply_async(Axitra_depth_chunk, args=task) for task in tasks]
			for task, result in zip(tasks, results):
				elemse_file, message = result.get()
				if elemse_file is None:
					raise RuntimeError('Axitra failed while building a Green\'s function table: ' + message)
				d = int(task[0][1:5])
				start = int(task[0][6:]) * per_run
				records = np.fromfile(elemse_file, dtype=ELEMSE_RECORD).reshape(len(task[3]), 6, npts)
				data[d, start:start+len(task[3])] = np.stack([records['N'], records['E'], records['Z']], axis=-1)
				times = records['t'][0, 0]
				shutil.rmtree(Path(green_dir) / 'depth_runs' / task[0], ignore_errors=True)
		return cls([float(z) for z in depths], np.asarray(distances, dtype=float), data, times, param_hash)

	def save(self, path):
		"""
		Save the table as an uncompressed npz file

		Parameters
		----------
		path : str or path-like
			Output file
		"""
		np.savez(path, depths=self.depths, distances=self.distances, data=self.data, times=self.times,
				 param_hash=np.array(self.param_hash))

	@classmethod
	def load(cls, path):
		"""
		Load a table saved with :meth:`save`

		Parameters
		----------
		path : str or path-like
			Table file

		Returns
		-------
		table : GreenTable
			The table
		"""
		with np.load(path) as f:
			return cls(f['depths'], f['distances'], f['data'], f['times'], str(f['param_hash']))

	@staticmethod
	def _brackets(axis, values):
		"""
		Linear interpolation indices and weights along one table axis

		Parameters
		----------
		axis : np.ndarray
			Table coordinates, increasing
		values : np.ndarray
			Requested coordinates

		Returns
		-------
		lower, upper : np.ndarray
			Indices of the bracketing nodes
		weight : np.ndarray
			Weight of the upper node
		"""
		if len(axis) == 1:
			zeros = np.zeros(len(values), dtype=int)
			return zeros, zeros, np.zeros(len(values))
		if np.any(values < axis[0] - 1e-6) or np.any(values > axis[-1] + 1e-6):
			raise ValueError('Requested receivers lie outside the Green\'s function table')
		upper = np.clip(np.searchsorted(axis, values, side='left'), 1, len(axis) - 1)
		lower = upper - 1
		weight = np.clip((values - axis[lower]) / (axis[upper] - axis[lower]), 0.0, 1.0)
		return lower, upper, weight

	def receivers(self, z, offsets):
		"""
		Elementary seismograms of receivers, interpolated and rotated

		Parameters
		----------
		z : float
			Source depth [km]
		offsets : np.ndarray
			Receiver offsets (north, east) [km] from the source, shape (n, 2)

		Returns
		-------
		g : np.ndarray
			float32, shape (n, 6, npts, 3), components N, E, Z
		"""
		offsets = np.asarray(offsets, dtype=float)
		distance = np.hypot(offsets[:, 0], offsets[:, 1])
		azimuth = np.arctan2(offsets[:, 1], offsets[:, 0])
		z0, z1, wz = self._brackets(self.depths, np.full(1, float(z)))
		r0, r1, wr = self._brackets(self.distances, distance)
		g0 = np.zeros((len(distance),) + self.data.shape[2:])
		for zi, zw in ((z0[0], 1.0 - wz[0]), (z1[0], wz[0])):
			if zw == 0.0:
				continue
			g0 += zw * ((1.0 - wr)[:, None, None, None] * self.data[zi, r0] + wr[:, None, None, None] * self.data[zi, r1])
		return rotate_from_azimuth_zero(g0, azimuth).astype(np.float32)

	def elemse_records(self, z, offsets):
		"""
		Axitra elemse records of receivers (same layout as an elemse file)

		Parameters
		----------
		z : float
			Source depth [km]
		offsets : np.ndarray
			Receiver offsets (north, east) [km] from the source, shape (n, 2)

		Returns
		-------
		records : np.ndarray
			``ELEMSE_RECORD`` array, receivers in the given order
		"""
		g = self.receivers(z, offsets)
		records = np.empty(g.shape[:3], dtype=ELEMSE_RECORD)
		records['head'] = 16
		records['tail'] = 16
		records['t'] = self.times
		records['N'] = g[..., 0]
		records['E'] = g[..., 1]
		records['Z'] = g[..., 2]
		return records.reshape(-1)


def table_axes(depths_km, max_distance_km, distance_step_km, min_distance_km=0.0):
	"""
	Distance axis covering the receivers of a run

	Parameters
	----------
	depths_km : list of float
		Source depths [km]
	max_distance_km : float
		Largest source-receiver distance needed [km]
	distance_step_km : float
		Table distance spacing [km]
	min_distance_km : float, optional
		Smallest distance needed [km], by default 0

	Returns
	-------
	depths : list of str
		Depths formatted for gr_xyz
	distances : np.ndarray
		Evenly spaced distances covering [min, max]
	"""
	start = math.floor(max(min_distance_km, 0.0) / distance_step_km) * distance_step_km
	stop = math.ceil(max_distance_km / distance_step_km) * distance_step_km
	distances = np.round(np.arange(start, stop + distance_step_km / 2, distance_step_km), 6)
	# Axitra cannot put a receiver exactly on the source axis
	distances[distances < 1e-3] = 1e-3
	return ['{0:1.3f}'.format(z) for z in depths_km], distances
