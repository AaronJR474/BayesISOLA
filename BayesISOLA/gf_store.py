#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append-only store of Axitra elementary seismograms, one entry per receiver.

In a 1-D model the elementary seismograms of a receiver depend only on the
crustal model, the Axitra parameters, the source time function, the sampling,
the source depth and the source-receiver offset. Entries are keyed by exactly
those inputs, so any run, adaptive stage or run variant (station subset) that
needs the same receiver reuses it instead of running Axitra again. Several
processes can share one store: chunks are written to new files and only become
visible once complete.
"""

import hashlib
import os
import re
import uuid
from pathlib import Path

import numpy as np

from BayesISOLA.fileformats import ELEMSE_RECORD

# Bump when the Axitra numerics change (code, compiler fixes) so old entries are not reused
STORE_VERSION = '1'


def axitra_parameter_hash(green_dir, model, npts_exp, elemse_start_origin):
	"""
	Hash of every Axitra input shared by the receivers of a run.

	Parameters
	----------
	green_dir : str or path-like
		Axitra workspace holding the crustal, grdat and soutype files
	model : str
		Crustal model identifier, '' for the single-model case
	npts_exp : int
		Number of samples is ``2**npts_exp``
	elemse_start_origin : float
		Time between elementary seismogram start and origin time [s]

	Returns
	-------
	param_hash : str
		sha1 hex digest
	"""
	green_dir = Path(green_dir)
	suffix = '-' + model if model else ''
	# the receiver count is the only grdat value that does not change the numbers
	grdat = re.sub(r'\bnr=\d+', 'nr=?', (green_dir / ('grdat' + suffix + '.hed')).read_text(encoding='utf-8'))
	digest = hashlib.sha1()
	for part in (
		STORE_VERSION.encode(),
		(green_dir / ('crustal' + suffix + '.dat')).read_bytes(),
		grdat.encode(),
		(green_dir / 'soutype.dat').read_bytes(),
		'{0:d}|{1:8.3f}'.format(int(npts_exp), float(elemse_start_origin)).encode(),
	):
		digest.update(part)
		digest.update(b'\0')
	return digest.hexdigest()


def format_receiver(dx, dy, dz):
	"""
	Format a receiver offset exactly as it is written to the Axitra station file

	Parameters
	----------
	dx, dy, dz : float
		Receiver position relative to the source (north, east, depth) [km]

	Returns
	-------
	offset : tuple of str
		Formatted north, east, depth
	"""
	return '{0:.4f}'.format(dx), '{0:.4f}'.format(dy), '{0:.4f}'.format(dz)


def receiver_key(param_hash, z, offset):
	"""
	Store key of one receiver

	Parameters
	----------
	param_hash : str
		Result of :func:`axitra_parameter_hash`
	z : str
		Source depth [km] as passed to gr_xyz
	offset : tuple of str
		Result of :func:`format_receiver`

	Returns
	-------
	key : str
		Key of the receiver
	"""
	return '|'.join((param_hash, z, *offset))


class GreenStore:
	"""
	Elementary seismograms of computed receivers, stored in chunks.

	Each chunk is ``<id>.npy`` (float32, shape (receivers, 6, npts, 3) with
	components N, E, Z), ``<id>.t.npy`` (time of each sample) and ``<id>.keys``
	(one receiver key per line, written last).

	Parameters
	----------
	root : str or path-like
		Store directory, created if missing
	"""

	def __init__(self, root):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self._index = {}
		self._seen = set()
		self._arrays = {}

	def index(self):
		"""
		Map of every stored receiver key to its chunk and row (refreshed from disk)

		Returns
		-------
		index : dict
			key -> (chunk id, row)
		"""
		for keys_file in sorted(self.root.glob('*.keys')):
			chunk = keys_file.name[:-len('.keys')]
			if chunk in self._seen:
				continue
			self._seen.add(chunk)
			for row, key in enumerate(keys_file.read_text(encoding='utf-8').split()):
				self._index.setdefault(key, (chunk, row))
		return self._index

	def add(self, elemse_file, keys, npts):
		"""
		Add the output of one elemse run to the store

		Parameters
		----------
		elemse_file : str or path-like
			elemse output, receivers in the order of ``keys``
		keys : list of str
			Receiver keys
		npts : int
			Samples per elementary seismogram

		Returns
		-------
		ok : bool
			False when the file does not hold the expected number of records
		"""
		records = np.fromfile(elemse_file, dtype=ELEMSE_RECORD)
		if len(records) != len(keys) * 6 * npts:
			return False
		records = records.reshape(len(keys), 6, npts)
		times = records['t'][0, 0]
		if not np.array_equal(records['t'], np.broadcast_to(times, records.shape)):
			return False
		data = np.stack([records['N'], records['E'], records['Z']], axis=-1)
		chunk = uuid.uuid4().hex
		# data first, keys last: a chunk is only visible once it is complete
		for name, array in ((chunk + '.npy', data), (chunk + '.t.npy', times)):
			tmp = self.root / (name + '.tmp')
			with open(tmp, 'wb') as f:
				np.save(f, array)
			os.replace(tmp, self.root / name)
		tmp = self.root / (chunk + '.keys.tmp')
		tmp.write_text('\n'.join(keys) + '\n', encoding='utf-8')
		os.replace(tmp, self.root / (chunk + '.keys'))
		return True

	def _chunk(self, chunk):
		"""
		Memory-mapped data and times of a chunk

		Parameters
		----------
		chunk : str
			Chunk id

		Returns
		-------
		data : np.ndarray
			Shape (receivers, 6, npts, 3)
		times : np.ndarray
			Shape (npts,)
		"""
		if chunk not in self._arrays:
			self._arrays[chunk] = (
				np.load(self.root / (chunk + '.npy'), mmap_mode='r'),
				np.load(self.root / (chunk + '.t.npy')),
			)
		return self._arrays[chunk]

	def elemse_records(self, keys):
		"""
		Assemble an Axitra elemse file (records of these receivers, in order)

		Parameters
		----------
		keys : list of str
			Receiver keys, all present in the store

		Returns
		-------
		records : np.ndarray
			``ELEMSE_RECORD`` array, byte-identical to what elemse writes
		"""
		index = self._index
		blocks = []
		for key in keys:
			chunk, row = index[key]
			data, times = self._chunk(chunk)
			block = np.empty((6, len(times)), dtype=ELEMSE_RECORD)
			block['head'] = 16
			block['tail'] = 16
			block['t'] = times
			block['N'] = data[row, :, :, 0]
			block['E'] = data[row, :, :, 1]
			block['Z'] = data[row, :, :, 2]
			blocks.append(block.reshape(-1))
		return np.concatenate(blocks)
