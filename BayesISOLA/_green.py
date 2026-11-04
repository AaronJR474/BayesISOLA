#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import os.path
import hashlib
import re
from pathlib import Path
from contextlib import nullcontext

try:
	from tqdm.auto import tqdm
except ImportError:
	tqdm = None

from BayesISOLA.axitra import Axitra_wrapper
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
	Runs :func:`Axitra_wrapper` (Green's function calculation) in parallel.

	When ``self.progress`` is true and :mod:`tqdm` is available, progress is
	reported per completed spatial grid point. The numerical work and result
	ordering are unchanged.
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
