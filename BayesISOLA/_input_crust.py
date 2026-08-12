#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

from BayesISOLA._paths import green_path

def read_crust(self, source, output=None):
	"""
	Copy one or more crustal-model files into the active Axitra workspace
	
	:param source: path to crust file
	:type source: string
	:param output: path to copy target
	:type output: string, optional
	"""
	source = str(source)
	if output is None:
		output = str(green_path(self.green_dir, 'crustal.dat'))
	else:
		output = str(output)
	inputs = []
	for model in self.models:
		if model:
			inp  = source[0:source.rfind('.')] + '-' + model + source[source.rfind('.'):]
			outp = output[0:output.rfind('.')] + '-' + model + output[output.rfind('.'):]
		else:
			inp  = source
			outp = output
		shutil.copyfile(inp, outp)
		inputs.append(inp)
	self.log('Crustal model(s): '+', '.join(inputs))
	self.logtext['crust'] = ', '.join(inputs)
