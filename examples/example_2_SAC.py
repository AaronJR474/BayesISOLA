#! /usr/bin/env python

from pathlib import Path

import BayesISOLA

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input" / "example_2_SAC"
OUTPUT = HERE / "output" / "example_2_SAC"

inputs = BayesISOLA.load_data(outdir=OUTPUT)
inputs.read_event_info(str(INPUT / "event.isl"))
inputs.set_source_time_function("triangle", 2.0)
inputs.read_network_coordinates(str(INPUT / "network.stn"))
inputs.read_crust(str(INPUT / "crustal.dat"))
inputs.load_files(str(INPUT / "sac"), separator="", pz_dir=str(INPUT / "pzfiles"), pz_separator="", pz_suffix=".pz")
inputs.detect_mouse(figures=True)

grid = BayesISOLA.grid(
    inputs,
    location_unc=1000,
    depth_unc=3000,
    time_unc=1,
    step_x=200,
    step_z=200,
    max_points=500,
    circle_shape=True,
    rupture_velocity=1000,
)

data = BayesISOLA.process_data(
    inputs,
    grid,
    threads=8,
    use_precalculated_Green="auto",
    fmax=0.15,
    fmin=0.02,
)

cova = BayesISOLA.covariance_matrix(data)
cova.covariance_matrix_noise(crosscovariance=True, save_non_inverted=True)

solution = BayesISOLA.resolve_MT(data, cova, deviatoric=False)
plot = BayesISOLA.plot(solution)
plot.html_log(h1="Example 2 (2013-12-12 00:59:18 Sargans)", mouse_figures="mouse/")
