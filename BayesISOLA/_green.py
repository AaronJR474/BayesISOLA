#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import multiprocessing as mp
import numpy as np
import os.path
import hashlib
import re
import shutil
from pathlib import Path
from contextlib import nullcontext

try:
	from tqdm.auto import tqdm
except ImportError:
	tqdm = None

from BayesISOLA.axitra import AXITRA_MAX_RECEIVERS, Axitra_depth_chunk, Axitra_wrapper
from BayesISOLA.gf_table import GreenTable, table_axes
from BayesISOLA.gf_store import GreenStore, axitra_parameter_hash, format_receiver, receiver_key
import BayesISOLA.syngine
from BayesISOLA._paths import axitra_executable, green_path

def set_Greens_parameters(self):
	"""
	Sets parameters for Green's function calculation:
		- time window length ``self.tl``
		- number of frequencies ``self.freq``
		- spatial periodicity ``self.xl``
		
	Writes used parameters to the log file.
	"""
	self.tl = self.npts_elemse/self.samprate
	#freq = int(math.ceil(fmax*tl))
	#self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2) # pocitame 2x vic frekvenci, nez pak proleze filtrem, je to pak lepe srovnatelne se signalem, ktery je kauzalne filtrovany
	self.freq = int(self.npts_elemse/2)+1
	# optionally stop at gf_fmax_factor * fmax instead of Nyquist (data are prefiltered to the same band)
	gf_fmax_factor = getattr(self, 'gf_fmax_factor', None)
	if gf_fmax_factor:
		self.freq = min(int(np.ceil(gf_fmax_factor*self.fmax*self.tl))+1, self.freq)
	self.xl = max(np.ceil(self.d.stations[self.d.nr-1]['dist']/1000), 100)*1e3*20 # `xl` 20x vetsi nez nejvetsi epicentralni vzdalenost, zaokrouhlena nahoru na kilometry, minimalne 2000 km
	# optional event-wide xl, so runs with fewer (closer) stations can reuse stored Green's functions
	gf_xl_km = getattr(self, 'gf_xl_km', None)
	if gf_xl_km:
		self.xl = max(self.xl, float(gf_xl_km) * 1e3)
	self.log("\nGreen's function calculation:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n  npts for inversion: {3:4d}\n  source time function: {4:s}".format(self.npts_elemse, self.tl, self.freq, self.npts_slice, self.d.stf_description))

def write_Greens_parameters(self):
	"""
	Writes file grdat.hed - parameters for gr_xyz (Axitra)
	"""
	for model in self.d.models:
		if model:
			f = green_path(self.d.green_dir, 'grdat-' + model + '.hed')
		else:
			f = green_path(self.d.green_dir, 'grdat.hed')
		grdat = open(f, 'w', encoding='utf-8', newline='\n')
		grdat.write("&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.models[model], xl=self.xl)) # 'nc' is probably ignored in the current version of gr_xyz???
		grdat.close()

def verify_Greens_parameters(self):
	"""
	Check whetrer parameters in file grdat.hed (probably used in Green's function calculation) are the same as used now.
	If it agrees, return True, otherwise returns False, print error description, and writes it into log.
	"""
	try:
		grdat = open(green_path(self.d.green_dir, 'grdat.hed'), 'r')
	except:
		readable = False
	else:
		readable = True
	if not readable or grdat.read() != "&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.nr, xl=self.xl):
		desc = 'Pre-calculated Green\'s functions calculated with different parameters (e.g. sampling) than used now, calculate Green\'s functions again.'
		self.log(desc)
		print(desc)
		print ("Expected content of Axitra grdat.hed:\n&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.nr, xl=self.xl))
		return False
	grdat.close()
	return True

def verify_Greens_headers(self):
	"""Verify cached Axitra payloads and metadata for every grid point."""
	md5_crustal = hashlib.md5(Path(green_path(self.d.green_dir, 'crustal.dat')).read_bytes()).hexdigest()
	md5_station = hashlib.md5(Path(green_path(self.d.green_dir, 'station.dat')).read_bytes()).hexdigest()
	txt_soutype = Path(green_path(self.d.green_dir, 'soutype.dat')).read_text(encoding='utf-8').strip().replace('\n', '_')

	for g, gp in enumerate(self.grid.grid):
		point_id = str(g).zfill(4)
		payload = Path(green_path(self.d.green_dir, 'elemse' + point_id + '.dat'))
		metadata = Path(green_path(self.d.green_dir, 'elemse' + point_id + '.txt'))

		if not payload.is_file() or payload.stat().st_size == 0:
			desc = 'Elementary-seismogram payload for grid point {0:d} was not found or is empty. '.format(g)
			self.log(desc)
			print(desc)
			return False

		try:
			text = metadata.read_text(encoding='utf-8').strip()
		except (OSError, UnicodeError):
			desc = 'Meta-data file for grid point {0:d} was not found or could not be read. '.format(g)
			self.log(desc)
			print(desc)
			return False

		if not text:
			desc = 'Meta-data file for grid point {0:d} is empty. '.format(g)
			self.log(desc)
			print(desc)
			return False

		expected = '{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(
			gp['x']/1e3, gp['y']/1e3, gp['z']/1e3, md5_crustal, md5_station, txt_soutype
		)
		if text != expected:
			desc = 'Pre-calculated grid point {0:d} was calculated with different parameters. '.format(g)
			fields = text.split()
			if len(fields) < 6:
				desc += 'Its metadata record is incomplete. '
			else:
				if fields[0:3] != expected.split()[0:3]:
					desc += 'Its coordinates differ, probably the shape of the grid was changed. '
				if fields[3] != md5_crustal:
					desc += 'The Axitra crustal.dat file has a different hash, probably the crustal model was changed. '
				if fields[4] != md5_station:
					desc += 'The Axitra station.dat file has a different hash, probably the station set was different. '
				if fields[5] != txt_soutype:
					desc += 'Source time function (file soutype.dat) was different. '
			self.log(desc)
			print(desc)
			return False
	return True

# fewest receivers worth one Axitra run (each run has a fixed cost)
MIN_RECEIVERS_PER_RUN = 200

# subdirectory of the Axitra workspace holding elemse files keyed by coordinates
GREEN_CACHE_DIR = 'elemse_cache'

def _grdat_path(self, model):
	"""
	Path of the Axitra grdat parameter file of a crustal model

	Parameters
	----------
	model : str
		Crustal model identifier, '' for the single-model case

	Returns
	-------
	path : Path
		Path to ``grdat.hed`` or ``grdat-<model>.hed``
	"""
	return Path(green_path(self.d.green_dir, 'grdat-' + model + '.hed' if model else 'grdat.hed'))

def _green_signature(self):
	"""
	Hashes of the Axitra inputs shared by all grid points

	Returns
	-------
	md5_crustal : str
		md5 of ``crustal.dat``
	md5_station : str
		md5 of ``station.dat``
	txt_soutype : str
		Source time function description from ``soutype.dat``
	"""
	md5_crustal = hashlib.md5(Path(green_path(self.d.green_dir, 'crustal.dat')).read_bytes()).hexdigest()
	md5_station = hashlib.md5(Path(green_path(self.d.green_dir, 'station.dat')).read_bytes()).hexdigest()
	txt_soutype = Path(green_path(self.d.green_dir, 'soutype.dat')).read_text(encoding='utf-8').strip().replace('\n', '_')
	return md5_crustal, md5_station, txt_soutype

def _green_cache_key(grdat_text, metadata_text):
	"""
	Cache key of one elemse file: its Axitra parameters and point metadata

	Parameters
	----------
	grdat_text : str
		Content of the grdat file used for the computation
	metadata_text : str
		Content of the elemse metadata file (coordinates and input hashes)

	Returns
	-------
	key : str
		sha1 hex digest
	"""
	return hashlib.sha1((grdat_text + '|' + metadata_text).encode('utf-8')).hexdigest()

def stash_reusable_Green(self):
	"""
	Move valid elemse files into a cache keyed by coordinates and parameters.

	Grid point files are named by index, so a changed grid (e.g. an adaptive
	expansion) would otherwise recompute every point. Files are only stashed when
	they were computed with the current ``crustal.dat``, ``station.dat``,
	``soutype.dat`` and after the current grdat file was written, so the key is
	exact. :func:`calculate_Green` restores matching points instead of computing them.

	Returns
	-------
	n_stashed : int
		Number of elemse files moved into the cache
	"""
	green_dir = Path(self.d.green_dir)
	try:
		md5_crustal, md5_station, txt_soutype = _green_signature(self)
	except OSError:
		return 0
	cache_dir = green_dir / GREEN_CACHE_DIR
	n_stashed = 0
	for model in self.d.models:
		grdat = _grdat_path(self, model)
		try:
			grdat_text = grdat.read_text(encoding='utf-8')
			grdat_mtime = grdat.stat().st_mtime
		except OSError:
			continue
		pattern = re.compile(r'elemse\d+' + (re.escape('-' + model) if model else '') + r'\.txt')
		for metadata in green_dir.iterdir():
			if not pattern.fullmatch(metadata.name):
				continue
			payload = metadata.with_suffix('.dat')
			try:
				text = metadata.read_text(encoding='utf-8').strip()
				fresh = payload.stat().st_size > 0 and payload.stat().st_mtime >= grdat_mtime
			except (OSError, UnicodeError):
				continue
			fields = text.split()
			if not fresh or len(fields) < 6 or fields[3:6] != [md5_crustal, md5_station, txt_soutype]:
				continue
			cache_dir.mkdir(exist_ok=True)
			key = _green_cache_key(grdat_text, text)
			os.replace(payload, cache_dir / (key + '.dat'))
			os.replace(metadata, cache_dir / (key + '.txt'))
			n_stashed += 1
	return n_stashed

def _restore_cached_Green(self, model):
	"""
	Restore cached elemse files for grid points whose coordinates match.

	Parameters
	----------
	model : str
		Crustal model identifier, '' for the single-model case

	Returns
	-------
	todo : list of int
		Indices of grid points that still have to be computed
	"""
	grid = self.grid.grid
	cache_dir = Path(self.d.green_dir) / GREEN_CACHE_DIR
	if not getattr(self, 'use_precalculated_Green', False) or not cache_dir.is_dir():
		return list(range(len(grid)))
	md5_crustal, md5_station, txt_soutype = _green_signature(self)
	grdat_text = _grdat_path(self, model).read_text(encoding='utf-8')
	suffix = '-' + model if model else ''
	todo = []
	for i, gp in enumerate(grid):
		text = '{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(
			gp['x']/1e3, gp['y']/1e3, gp['z']/1e3, md5_crustal, md5_station, txt_soutype
		)
		cached = cache_dir / (_green_cache_key(grdat_text, text) + '.dat')
		if not cached.is_file():
			todo.append(i)
			continue
		point_id = str(i).zfill(4) + suffix
		for ext in ('.dat', '.txt'):
			target = green_path(self.d.green_dir, 'elemse' + point_id + ext)
			os.replace(cached.with_suffix(ext), target)
			# valid for the current grdat, so mark as fresh for later stages
			os.utime(target)
	n_reused = len(grid) - len(todo)
	if n_reused:
		self.log("Reused Green's functions of {0:d} of {1:d} grid points{2:s} from earlier stages".format(
			n_reused, len(grid), ' (model ' + model + ')' if model else ''))
	return todo

def calculate_or_verify_Green(self):
	"""
	If ``self.use_precalculated_Green`` is True, verifies whether the pre-calculated Green's functions were calculated on the same grid and with the same parameters (:func:`verify_Greens_headers` and :func:`verify_Greens_parameters`)
	Otherwise calculates Green's function (:func:`write_Greens_parameters` and :func:`calculate_Green`).
	"""
	
	if not self.use_precalculated_Green: # calculate Green's functions in all grid points
		self.write_Greens_parameters()
		self.calculate_Green()
	else: # verify whether the pre-calculated Green's functions are calculated on the same grid and with the same parameters
		differs = False
		if not self.verify_Greens_parameters():
			differs = True
		if not self.verify_Greens_headers():
			differs = True
		if differs:
			if self.use_precalculated_Green == 'auto':
				self.log('Shape or the grid or some parameters changed, calculating Gren\'s functions again...')
				# keep points whose coordinates and parameters are unchanged
				stash_reusable_Green(self)
				self.write_Greens_parameters()
				self.calculate_Green()
			else:
				raise ValueError('Metadata of pre-calculated Green\'s functions differs from actual calculation. More details are shown above and in the log file.')

def calculate_Green(self):
	"""
	Calculate Green's functions (elementary seismograms) for all grid points.

	By default (``self.gf_bundle_depths``) Axitra runs once per source depth,
	with every (grid point, station) offset as a receiver, and the output is
	split into the usual per-point elemse files (see :func:`_calculate_Green_by_depth`).
	Otherwise :func:`Axitra_wrapper` runs once per grid point. Both run in
	parallel over ``self.threads`` processes and give the same files.

	When ``self.progress`` is true and :mod:`tqdm` is available, progress is
	reported per completed spatial grid point.
	"""
	grid = self.grid.grid
	logfile = self.d.outdir+'/log_green.txt'
	green_dir = str(self.d.green_dir)
	gr_xyz_executable = str(axitra_executable('gr_xyz'))
	elemse_executable = str(axitra_executable('elemse'))
	open(logfile, "w", encoding="utf-8", newline="\n").close() # erase file contents
	show_progress = bool(getattr(self, 'progress', True)) and tqdm is not None
	# run `gr_xyz` aand `elemse`
	for model in self.d.models:
		desc = "Green's functions" + (f" ({model})" if model else "")
		todo = _restore_cached_Green(self, model)
		if not todo:
			continue
		if getattr(self, 'gf_table', None) is not None:
			_calculate_Green_from_table(self, model, todo, gr_xyz_executable, elemse_executable)
			continue
		if getattr(self, 'gf_bundle_depths', True) and self.d.models[model] <= AXITRA_MAX_RECEIVERS:
			_calculate_Green_by_depth(self, model, todo, logfile, gr_xyz_executable, elemse_executable, desc, show_progress)
			continue
		if self.threads > 1: # parallel
			with mp.Pool(processes=self.threads) as pool:
				progress_context = tqdm(total=len(todo), desc=desc, unit='pt') if show_progress else nullcontext()
				with progress_context as bar:
					callback = (lambda _: bar.update(1)) if bar is not None else None
					results = [pool.apply_async(Axitra_wrapper, args=(i, model, grid[i]['x'], grid[i]['y'], grid[i]['z'], self.npts_exp, self.elemse_start_origin, logfile, green_dir, gr_xyz_executable, elemse_executable), callback=callback) for i in todo]
					output = [p.get() for p in results]
			for i, ok in zip(todo, output):
				if ok == False:
					grid[i]['err'] = 1
					grid[i]['VR'] = -10
		else: # serial
			indices = tqdm(todo, desc=desc, unit='pt') if show_progress else todo
			for i in indices:
				gp = grid[i]
				ok = Axitra_wrapper(i, model, gp['x'], gp['y'], gp['z'], self.npts_exp, self.elemse_start_origin, logfile, green_dir, gr_xyz_executable, elemse_executable)
				if not ok:
					gp['err'] = 1
					gp['VR'] = -10

def _receiver_chunks(missing, threads):
	"""
	Split receivers to compute into balanced Axitra runs, one source depth per run.

	Parameters
	----------
	missing : dict
		Source depth string -> list of (key, formatted offset) to compute
	threads : int
		Number of parallel processes

	Returns
	-------
	chunks : list of tuple
		(depth string, list of (key, offset)), shallow depths first as they are
		the most expensive
	"""
	total = sum(len(receivers) for receivers in missing.values())
	# enough runs to keep every process busy, but each run has a fixed cost
	per_run = min(AXITRA_MAX_RECEIVERS, max(MIN_RECEIVERS_PER_RUN, math.ceil(total / (2 * max(int(threads), 1)))))
	chunks = []
	for z in sorted(missing, key=float):
		receivers = missing[z]
		size = math.ceil(len(receivers) / math.ceil(len(receivers) / per_run))
		for start in range(0, len(receivers), size):
			chunks.append((z, receivers[start:start+size]))
	return chunks

def _station_coordinates(self, model):
	"""
	Read the receiver coordinates Axitra uses for a crustal model.

	Parameters
	----------
	model : str
		Crustal model identifier, '' for the single-model case

	Returns
	-------
	coordinates : list of tuple
		(north, east, depth) [km] of each station, in Axitra receiver order
	"""
	name = 'station-' + model + '.dat' if model else 'station.dat'
	lines = Path(green_path(self.d.green_dir, name)).read_text(encoding='utf-8').splitlines()[2:]
	nr = self.d.models[model]
	return [tuple(float(v) for v in line.split()[:3]) for line in lines[:nr]]

def _green_store(self):
	"""
	Receiver-level Green's function store of this run

	Returns
	-------
	store : GreenStore
		``self.gf_store_dir`` when set (e.g. shared by all runs of an event),
		otherwise ``<green_dir>/gf_store``
	"""
	root = getattr(self, 'gf_store_dir', None) or Path(self.d.green_dir) / 'gf_store'
	return GreenStore(root)

def _calculate_Green_by_depth(self, model, todo, logfile, gr_xyz_executable, elemse_executable, desc, show_progress):
	"""
	Compute elemse files with one Axitra run per source depth (or chunk of it).

	Each grid point's receivers are its station offsets, so the records written
	for it are those of the per-point calculation. Receivers already in the
	Green's function store (same model, parameters, depth and offset, e.g. from
	an earlier adaptive stage or another run of the same event) are reused, only
	the missing ones are computed. The per-point elemse files and metadata are
	written exactly as :func:`Axitra_wrapper` writes them.

	Parameters
	----------
	model : str
		Crustal model identifier, '' for the single-model case
	todo : list of int
		Indices of grid points to compute
	logfile : str
		Path to the Green's function log
	gr_xyz_executable, elemse_executable : str
		Paths to the Axitra executables
	desc : str
		Progress bar label
	show_progress : bool
		Show a tqdm progress bar
	"""
	grid = self.grid.grid
	green_dir = Path(self.d.green_dir)
	suffix = '-' + model if model else ''
	stations = _station_coordinates(self, model)
	npts = 2 ** self.npts_exp
	md5_crustal, md5_station, txt_soutype = _green_signature(self)
	store = _green_store(self)
	param_hash = axitra_parameter_hash(green_dir, model, self.npts_exp, self.elemse_start_origin)
	index = store.index()

	# receivers of every grid point, and those not in the store yet
	point_keys = {}
	missing = {}
	queued = set()
	for i in todo:
		# source position as written for the per-point gr_xyz run
		xs = float('{0:1.3f}'.format(grid[i]['x']/1e3))
		ys = float('{0:1.3f}'.format(grid[i]['y']/1e3))
		z = '{0:1.3f}'.format(grid[i]['z']/1e3)
		keys = []
		for sx, sy, sz in stations:
			offset = format_receiver(sx - xs, sy - ys, sz)
			key = receiver_key(param_hash, z, offset)
			keys.append(key)
			if key not in index and key not in queued:
				queued.add(key)
				missing.setdefault(z, []).append((key, offset))
		point_keys[i] = keys
	n_receivers = len(todo) * len(stations)
	self.log("Green's functions{0:s}: {1:d} of {2:d} receivers found in the store, {3:d} to compute".format(
		' (model ' + model + ')' if model else '', n_receivers - len(queued), n_receivers, len(queued)))

	chunks = _receiver_chunks(missing, self.threads)
	tasks = []
	for c, (z, receivers) in enumerate(chunks):
		chunk_id = 'd' + str(c).zfill(4)
		offsets = [tuple(float(v) for v in offset) for _, offset in receivers]
		tasks.append((chunk_id, model, z, offsets, self.npts_exp, self.elemse_start_origin, str(green_dir), gr_xyz_executable, elemse_executable))

	log = open(logfile, 'a', encoding='utf-8', newline='\n')
	failed = set()
	progress_context = tqdm(total=len(queued), desc=desc, unit='receiver') if show_progress and queued else nullcontext()
	pool_context = mp.Pool(processes=int(self.threads)) if self.threads > 1 and len(tasks) > 1 else nullcontext()
	with progress_context as bar, pool_context as pool:
		pending = [pool.apply_async(Axitra_depth_chunk, args=task) for task in tasks] if pool is not None else None
		for k, ((z, receivers), task) in enumerate(zip(chunks, tasks)):
			elemse_file, message = pending[k].get() if pending is not None else Axitra_depth_chunk(*task)
			if message:
				log.write(message)
			keys = [key for key, _ in receivers]
			if elemse_file is not None and store.add(elemse_file, keys, npts):
				log.write('depth {0:s} km ({1:s}): {2:d} receivers\n'.format(z, task[0], len(keys)))
			else:
				log.write('depth {0:s} km ({1:s}): no usable elemse output for {2:d} receivers\n'.format(z, task[0], len(keys)))
				failed.update(keys)
			shutil.rmtree(green_dir / 'depth_runs' / task[0], ignore_errors=True)
			if bar is not None:
				bar.update(len(keys))
	log.close()

	# per-point files, byte-identical record layout to a per-point elemse run
	index = store.index()
	for i in todo:
		keys = point_keys[i]
		if any(key in failed or key not in index for key in keys):
			grid[i]['err'] = 1
			grid[i]['VR'] = -10
			continue
		point_id = str(i).zfill(4) + suffix
		store.elemse_records(keys).tofile(green_path(green_dir, 'elemse' + point_id + '.dat'))
		with open(green_path(green_dir, 'elemse' + point_id + '.txt'), 'w', encoding='utf-8', newline='\n') as meta:
			meta.write('{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(
				grid[i]['x']/1e3, grid[i]['y']/1e3, grid[i]['z']/1e3, md5_crustal, md5_station, txt_soutype))

def _calculate_Green_from_table(self, model, todo, gr_xyz_executable, elemse_executable):
	"""
	Write elemse files interpolated from a (depth, distance) Green's function table.

	``self.gf_table`` is a :class:`~BayesISOLA.gf_table.GreenTable`, a path to a
	saved table, or ``{'distance_step_km': step}`` to build a table for this run
	(its depths, distances up to the farthest receiver) before interpolating.
	The table must have been computed with the same Axitra inputs. Distance
	(and depth) interpolation is an approximation; the azimuth is exact.

	Parameters
	----------
	model : str
		Crustal model identifier, '' for the single-model case
	todo : list of int
		Indices of grid points to compute
	gr_xyz_executable, elemse_executable : str
		Paths to the Axitra executables (to build a table)
	"""
	grid = self.grid.grid
	green_dir = Path(self.d.green_dir)
	suffix = '-' + model if model else ''
	stations = np.array(_station_coordinates(self, model))
	md5_crustal, md5_station, txt_soutype = _green_signature(self)
	param_hash = axitra_parameter_hash(green_dir, model, self.npts_exp, self.elemse_start_origin)

	sources = {i: (float('{0:1.3f}'.format(grid[i]['x']/1e3)), float('{0:1.3f}'.format(grid[i]['y']/1e3)), grid[i]['z']/1e3) for i in todo}
	table = self.gf_table
	if isinstance(table, dict):
		depths = sorted({z for _, _, z in sources.values()})
		distances = [np.hypot(stations[:, 0] - xs, stations[:, 1] - ys) for xs, ys, _ in sources.values()]
		depth_strings, distance_axis = table_axes(
			depths, max(d.max() for d in distances), float(table['distance_step_km']), min(d.min() for d in distances))
		self.log("Green's function table: {0:d} depths x {1:d} distances ({2:.3g} km spacing)".format(
			len(depth_strings), len(distance_axis), float(table['distance_step_km'])))
		table = GreenTable.build(green_dir, model, depth_strings, distance_axis, self.npts_exp, self.elemse_start_origin,
								 param_hash, threads=self.threads, gr_xyz_executable=gr_xyz_executable,
								 elemse_executable=elemse_executable)
	elif not isinstance(table, GreenTable):
		table = GreenTable.load(table)
	if table.param_hash != param_hash:
		raise ValueError("The Green's function table was computed with different Axitra inputs (model, parameters or sampling)")

	for i, (xs, ys, z) in sources.items():
		offsets = np.column_stack([stations[:, 0] - xs, stations[:, 1] - ys])
		point_id = str(i).zfill(4) + suffix
		table.elemse_records(z, offsets).tofile(green_path(green_dir, 'elemse' + point_id + '.dat'))
		with open(green_path(green_dir, 'elemse' + point_id + '.txt'), 'w', encoding='utf-8', newline='\n') as meta:
			meta.write('{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(
				grid[i]['x']/1e3, grid[i]['y']/1e3, grid[i]['z']/1e3, md5_crustal, md5_station, txt_soutype))

def use_elemse_from_files(self, path):
	"""
	Add a path to elementary seismograms to grid points. It enables using external software for calculating elementary seismograms / Green's functions.
	
	:param path: path to a directory containing subdirectories with elementary seismograms for different grid points
	:type path: string
	"""
	self.log('\nUsing elementary seismograms from: '+path)
	self.d.stf_description = "Source time function is contained in GFs from elementary seismograms from external source."
	for gp in self.grid.grid:
		gp['path'] = os.path.join(path, gp['z_id'], gp['x_id']+gp['y_id'])

def use_elemse_from_syngine(self, model="ak135f_5s", output_root_path="input/GFs"):
	"""
	Run a query to IRIS Syngine web service, save synthetic seismograms to files and add the path to the elementary seismograms to coresponding grid point.
	
	:param model: Earth model for synthetic seismograms. The list of available models: http://ds.iris.edu/ds/products/syngine/#earth
	:type model: string, optional
	:param output_root_path: path where the elementary seismograms are saved to
	:type output_root_path: string, optional
	"""
	self.log('\nDownloading elementary seismograms from Syngine.\n\tEarth model: '+model)
	self.d.stf_description = "Source time function is contained in GFs from elementary seismograms from Syngine. By default Instaseis returns the response to a very narrow Gaussian source time function with a full-width at half-max of approximately two-thirds the shortest period resolved, which is as close to a delta function as possible with AxiSEM."
	query = BayesISOLA.syngine.generate_query()
	for stn in self.d.stations:
		query.bulk.append({"networkcode": stn['network'], "stationcode": stn['code'], "locationcode": stn['location'], "latitude": stn['lat'], "longitude": stn['lon']})
	for gp in self.grid.grid:
		path = os.path.join(output_root_path, gp['z_id'], gp['x_id']+gp['y_id'])
		query.do_query_simple(model, gp['lat'], gp['lon'], gp['z'], self.d.event['t'], self.d.event['t']+self.t_min, self.d.event['t']+self.t_max, path)
		gp['path'] = path
