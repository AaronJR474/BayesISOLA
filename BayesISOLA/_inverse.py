#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
from contextlib import nullcontext
import os
import numpy as np
from pyproj import Geod

try:
	from tqdm.auto import tqdm
except ImportError:
	tqdm = None

from BayesISOLA.inverse_problem import invert, whiten_covariance_array

_INVERSION_STATE = None
_INVERSION_BLAS_LIMITER = None

def _init_inversion_worker(state, blas_threads=1):
	"""Initialize one inversion worker with shared read-only state."""
	global _INVERSION_STATE, _INVERSION_BLAS_LIMITER
	_INVERSION_STATE = state

	# Avoid nested BLAS oversubscription when BayesISOLA already parallelizes
	# over grid points with multiprocessing. Environment variables protect any
	# libraries loaded later; threadpoolctl also limits BLAS already loaded in
	# the worker process.
	for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
		os.environ[name] = str(int(blas_threads))
	try:
		from threadpoolctl import threadpool_limits
		_INVERSION_BLAS_LIMITER = threadpool_limits(limits=int(blas_threads), user_api="blas")
	except ImportError:
		_INVERSION_BLAS_LIMITER = None


def _invert_worker(task):
	"""Invert one grid point using state initialized once per worker."""
	point_id, elemse_path = task
	s = _INVERSION_STATE
	if s is None:
		raise RuntimeError("BayesISOLA inversion worker was not initialized.")
	return invert(
		point_id, s["d_shifts"], s["norm_d"], s["Cd_inv"], s["Cd_inv_shifts"],
		s["nr"], s["components"], s["stations"], s["npts_elemse"],
		s["npts_slice"], s["elemse_start_origin"], s["origin_time"],
		s["samprate"], s["deviatoric"], s["decompose"],
		s["invert_displacement"], elemse_path,
		covariance_factors=s["covariance_factors"],
		data_whitened=s["data_whitened"],
	)
 
def run_inversion(self):
	"""
	Runs function :func:`invert` in parallel.
	
	Module :class:`multiprocessing` does not allow running function of the same class in parallel, so the function :func:`invert` cannot be method of class :class:`ISOLA` and this wrapper is needed.
	"""
	grid = self.grid
	d_shifts = self.d.d_shifts
	Cd_inv = self.cova.Cd_inv
	Cd_inv_shifts = self.cova.Cd_inv_shifts
	
	todo = []
	for i in range (len(grid)):
		point_id = str(i).zfill(4)
		grid[i]['id'] = point_id
		if not grid[i]['err']:
			todo.append(i)
	
	# Create the data norm once for every source-time shift. Noise covariance
	# uses the stored whitening factors, so the shifted data are whitened once
	# here and reused by every grid-point worker. ACF keeps the legacy
	# shift-dependent inverse-covariance path.
	factorized_noise = bool(getattr(self.cova, 'factorized_noise', False)) and not Cd_inv_shifts
	covariance_factors = None
	data_whitened = False
	if factorized_noise:
		covariance_factors = {"LT": self.cova.LT, "LT3": self.cova.LT3}
		d_shifts = [
			whiten_covariance_array(d_shift, covariance_factors, self.inp.stations, self.d.npts_slice)
			for d_shift in d_shifts
		]
		norm_d = [float(np.dot(d_shift.T, d_shift)[0, 0]) for d_shift in d_shifts]
		data_whitened = True
	else:
		norm_d = []
		for shift in range(len(d_shifts)):
			d_shift = d_shifts[shift]
			if Cd_inv_shifts:  # ACF
				Cd_inv = Cd_inv_shifts[shift]
			if Cd_inv:
				idx = 0
				dCd_blocks = []
				for C in Cd_inv:
					size = len(C)
					dCd_blocks.append(np.dot(d_shift[idx:idx+size, :].T, C))
					idx += size
				dCd = np.concatenate(dCd_blocks, axis=1)
				norm_d.append(np.dot(dCd, d_shift)[0, 0])
			else:
				norm_d.append(float(np.dot(d_shift.T, d_shift)[0, 0]))

	show_progress = bool(getattr(self.d, 'progress', True)) and tqdm is not None
	if self.threads > 1: # parallel
		# Large inversion inputs are invariant across grid points. Initialize them
		# once per worker rather than serializing them again for every task.
		state = {
			"d_shifts": d_shifts,
			"norm_d": norm_d,
			"Cd_inv": Cd_inv,
			"Cd_inv_shifts": Cd_inv_shifts,
			"nr": self.inp.nr,
			"components": self.d.components,
			"stations": self.inp.stations,
			"npts_elemse": self.d.npts_elemse,
			"npts_slice": self.d.npts_slice,
			"elemse_start_origin": self.d.elemse_start_origin,
			"origin_time": self.inp.event['t'],
			"samprate": self.d.samprate,
			"deviatoric": self.deviatoric,
			"decompose": self.decompose,
			"invert_displacement": self.d.invert_displacement,
			"covariance_factors": covariance_factors,
			"data_whitened": data_whitened,
		}
		with mp.Pool(processes=self.threads, initializer=_init_inversion_worker, initargs=(state, 1)) as pool:
			progress_context = tqdm(total=len(todo), desc='Moment-tensor inversion', unit='pt') if show_progress else nullcontext()
			with progress_context as bar:
				callback = (lambda _: bar.update(1)) if bar is not None else None
				results = [
					pool.apply_async(_invert_worker, args=((grid[i]['id'], grid[i]['path']),), callback=callback)
					for i in todo
				]
				output = [p.get() for p in results]
	else: # serial
		output = []
		indices = tqdm(todo, desc='Moment-tensor inversion', unit='pt') if show_progress else todo
		for i in indices:
			res = invert(grid[i]['id'], d_shifts, norm_d, Cd_inv, Cd_inv_shifts, self.inp.nr, self.d.components, self.inp.stations, self.d.npts_elemse, self.d.npts_slice, self.d.elemse_start_origin, self.inp.event['t'], self.d.samprate, self.deviatoric, self.decompose, self.d.invert_displacement, grid[i]['path'], covariance_factors=covariance_factors, data_whitened=data_whitened)
			output.append(res)
	min_misfit = output[0]['misfit']
	for i in todo:
		grid[i].update(output[todo.index(i)])
		grid[i]['shift_idx'] = grid[i]['shift']
		#grid[i]['shift'] = self.g.shift_min + grid[i]['shift']*self.g.SHIFT_step/self.d.max_samprate
		grid[i]['shift'] = self.d.shifts[grid[i]['shift']]
		min_misfit = min(min_misfit, grid[i]['misfit'])
	self.max_sum_c = self.max_c = self.sum_c = 0
	for i in todo:
		gp = grid[i]
		gp['sum_c'] = 0
		for idx in gp['shifts']:
			GP = gp['shifts'][idx]
			if gp['det_Ca'] == np.inf:
				GP['c'] = 0
			else:
				GP['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * (GP['misfit']-min_misfit))
			gp['sum_c'] += GP['c']
		gp['c'] = gp['shifts'][gp['shift_idx']]['c']
		self.sum_c += gp['sum_c']
		self.max_c = max(self.max_c, gp['c'])
		self.max_sum_c = max(self.max_sum_c, gp['sum_c'])

def find_best_grid_point(self):
	"""
	Set ``self.centroid`` to a grid point with higher variance reduction -- the best solution of the inverse problem.
	"""
	self.centroid = max(self.grid, key=lambda g: g['VR']) # best grid point
