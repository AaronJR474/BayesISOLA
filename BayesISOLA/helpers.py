#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various data manipulation / arithmetic / waveform filtering functions.

"""

import numpy as np
from functools import lru_cache
from math import gcd
from scipy.signal import iirfilter, sosfilt

def rename_keys(somedict, prefix='', suffix=''):
	"""
	Returns a dictionary with keys renamed by adding some prefix and/or suffix
	:param somedict: dictionary, whose keys will be remapped
	:type somedict: dictionary
	:param prefix: new keys starts with
	:type prefix: string, optional
	:param suffix: new keys ends with
	:type suffix: string, optional
	:returns : dictionary with keys renamed
	"""
	return dict(map(lambda key, value: (prefix+str(key)+suffix, value), somedict.items()))

def next_power_of_2(n):
	"""
	Return next power of 2 greater than or equal to ``n``
	
	:type n: integer
	"""
	return 2**(n-1).bit_length()

def lcmm(b, *args):
	"""
	Returns generelized least common multiple.
	
	:param b,args: numbers to compute least common multiple of them 
	:type b,args: float, which is a multiple of 0.00033333
	:returns: the least multiple of ``a`` and ``b``
	"""
	b = 3/b
	if b - round(b) < 1e6:
		b = round(b)
	for a in args:
		a = 3/a
		if a - round(a) < 1e6:
			a = round(a)
		b = gcd(a, b)
	return 3/b

@lru_cache(maxsize=256)
def _butter_sos(btype, freq, df, corners):
	"""
	Butterworth SOS coefficients designed exactly as ObsPy's lowpass/highpass

	Parameters
	----------
	btype : str
		'lowpass' or 'highpass'
	freq : float
		Corner frequency [Hz]
	df : float
		Sampling rate [Hz]
	corners : int
		Filter order

	Returns
	-------
	sos : np.ndarray
		Second-order sections, shared through the cache so must not be modified
	"""
	fe = 0.5 * df
	if btype == 'highpass' and freq / fe > 1:
		raise ValueError("Selected corner frequency is above Nyquist.")
	return iirfilter(corners, freq / fe, btype=btype, ftype='butter', output='sos')

def filter_array(data, df, fmin, fmax):
	"""
	Apply the :func:`my_filter` filter chain to an array along its last axis

	Bit-identical to filtering every row with ObsPy, without the per-call
	overhead of ObsPy's entry-point lookup and processing history.

	Parameters
	----------
	data : np.ndarray
		Waveforms, time along the last axis
	df : float
		Sampling rate [Hz]
	fmin : float
		Highpass corner [Hz], skipped if 0 or None
	fmax : float
		Lowpass corner [Hz], skipped if 0 or None

	Returns
	-------
	data : np.ndarray
		Filtered waveforms (float64)
	"""
	if fmax:
		data = sosfilt(_butter_sos('lowpass', float(fmax), float(df), 4), data, axis=-1)
	if fmin:
		sos = _butter_sos('highpass', float(fmin), float(df), 2)
		data = sosfilt(sos, data, axis=-1)
		data = sosfilt(sos, data, axis=-1)
	return data

def my_filter(data, fmin, fmax):
	"""
	Filter used for filtering both elementary and observed seismograms

	Lowpass (4 corners) at ``fmax`` then twice highpass (2 corners) at ``fmin``,
	all causal Butterworth, applied in place to every trace of a Stream.
	Traces sharing sampling rate and length are filtered in one batched call.
	"""
	groups = {}
	for tr in data:
		if np.ma.isMaskedArray(tr.data):
			# keep ObsPy's handling (it raises for masked data)
			if fmax:
				tr.filter('lowpass', freq=fmax)
			if fmin:
				tr.filter('highpass', freq=fmin, corners=2)
				tr.filter('highpass', freq=fmin, corners=2)
			continue
		groups.setdefault((tr.stats.sampling_rate, tr.stats.npts), []).append(tr)
	for (df, _), traces in groups.items():
		filtered = filter_array(np.vstack([tr.data for tr in traces]), df, fmin, fmax)
		for tr, row in zip(traces, filtered):
			tr.data = row

def prefilter_data(st, f):
	"""
	Drop frequencies above Green's function computation high limit using :func:`numpy.fft.fft`.
	
	:param st: stream to be filtered
	:type st: :class:`~obspy.core.stream`
	:param f: the highest frequency which will be kept in st, the frequencies above will be erased
	:type f: float
	"""
	for tr in st:
		npts = tr.stats.npts
		NPTS = next_power_of_2(npts)
		TR = np.fft.fft(tr.data,NPTS)
		df = tr.stats.sampling_rate / NPTS
		flim = int(np.ceil(f/df))
		TR[flim:NPTS-flim+1] = 0+0j
		tr_filt = np.fft.ifft(TR)
		tr.data = np.real(tr_filt[0:npts])

def decimate(a, n=2):
	"""
	Decimates given sequence.
	
	:param data: data
	:type data: 1-D array
	:param n: decimation factor
	:type n: integer, optional
	
	Before decimating, filter out frequencies over Nyquist frequency using :func:`numpy.fft.fft`
	"""
	npts = len(a)
	#NPTS = npts # next_power_of_2(npts)
	NPTS = npts
	A = np.fft.fft(a, NPTS)
	idx = int(np.round(npts/n/2))
	A[idx:NPTS-idx+1] = 0+0j
	a = np.fft.ifft(A)
	if npts % (2*n) == 1 or n!=2: # keep odd length for decimation factor 2
		return a[:npts:n].real
	else:
		return a[1:npts:n].real
