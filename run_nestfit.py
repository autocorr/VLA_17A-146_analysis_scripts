#!/usr/bin/env python3

import sys
import shutil
from pathlib import Path

import numpy as np

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
from nestfit import core as nf

from . import (PDir, read_cube, TARGETS, VELOS)


TARGETS_NOMOS = [t for t in TARGETS if t != 'G285_mosaic']


def get_cubestack(target, rms=0.33):
    im_11_cube = read_cube(PDir.vla_map_name(target, 'nh3_11',
            modif='jfeather', ext='image.fits'))
    pb_11_cube = read_cube(PDir.vla_map_name(target, 'nh3_11',
            modif='joint', ext='pb.fits'))
    im_22_cube = read_cube(PDir.vla_map_name(target, 'nh3_22',
            modif='jfeather', ext='image.fits'))
    pb_22_cube = read_cube(PDir.vla_map_name(target, 'nh3_22',
            modif='joint', ext='pb.fits'))
    nm_11 = nf.NoiseMap.from_pbimg(rms, pb_11_cube._data)
    nm_22 = nf.NoiseMap.from_pbimg(rms, pb_22_cube._data)
    cubes = (
            nf.DataCube(im_11_cube, noise_map=nm_11),
            nf.DataCube(im_22_cube, noise_map=nm_22),
    )
    stack = nf.CubeStack(cubes)
    # first channel in some of the datasets are NaNs
    stack.cubes[0].data[:,:,0] = 0
    stack.cubes[1].data[:,:,0] = 0
    return stack


def get_bins(vsys):
    bin_minmax = [
            (vsys-4.0, vsys+4.0),  # vcen
            ( 7.0, 30.0),  # trot
            ( 2.8, 12.0),  # tex
            (12.5, 16.5),  # ncol
            ( 0.0,  2.0),  # sigm
    ]
    bins = np.array([
            np.linspace(lo, hi, 200)
            for (lo, hi) in bin_minmax
    ])
    return bins


def run_nestfit(store_prefix):
    for target in ['G24051']:  # XXX
    #for target in TARGETS_NOMOS:
        store_name = f'data/run/{store_prefix}_{target}'
        store_filen = f'{store_name}.store'
        if Path(store_filen).exists():
            shutil.rmtree(store_filen)
        stack = get_cubestack(target)
        utrans = nf.get_irdc_priors(vsys=VELOS[target])
        fitter = nf.CubeFitter(stack, utrans, ncomp_max=2)
        fitter.fit_cube(store_name=store_name, nproc=8)


def postprocess_run(store_prefix):
    for target in ['G24051']:  # XXX
    #for target in TARGETS_NOMOS:
        par_bins = get_bins(VELOS[target])
        store_name = f'data/run/{store_prefix}_{target}'
        store = nf.HdfStore(store_name)
        #nf.aggregate_run_attributes(store)
        nf.convolve_evidence(store, std_pix=1.5)
        nf.aggregate_run_products(store)
        nf.aggregate_run_pdfs(store, par_bins=par_bins)
        nf.convolve_post_pdfs(store, std_pix=1.5)
        nf.quantize_conv_marginals(store)
        stack = get_cubestack(target)
        nf.deblend_hf_intensity(store, stack)
        store.close()


