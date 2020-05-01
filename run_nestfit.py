#!/usr/bin/env python3

import sys
import shutil
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import spectral_cube
from astropy import convolution

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
import nestfit as nf
from nestfit.main import get_irdc_priors

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
            nf.DataCube(im_11_cube, noise_map=nm_11, trans_id=1),
            nf.DataCube(im_22_cube, noise_map=nm_22, trans_id=2),
    )
    stack = nf.CubeStack(cubes)
    # first channel in some of the datasets are NaNs
    stack.cubes[0].data[:,:,0] = 0
    stack.cubes[1].data[:,:,0] = 0
    return stack


def get_runner(stack, utrans, ncomp=1):
    spec_data, has_nans = stack.get_spec_data(190, 190)
    assert not has_nans
    runner = nf.AmmoniaRunner.from_data(spec_data, utrans, ncomp=ncomp)
    return runner


def get_bins(vsys):
    bin_minmax = [
            (vsys-4.0, vsys+4.0),  # vcen
            ( 7.0, 30.0),  # trot
            ( 2.8, 12.0),  # tex
            (12.5, 16.5),  # ncol
            ( 0.0,  2.0),  # sigm
            ( 0.0,  1.0),  # orth
    ]
    bins = np.array([
            np.linspace(lo, hi, 200)
            for (lo, hi) in bin_minmax
    ])
    return bins


def if_exists_delete_store(name):
    filen = f'{name}.store'
    if Path(filen).exists():
        print(f'-- Deleting {filen}')
        shutil.rmtree(filen)


def run_nested(target, store_prefix, nproc=8):
    store_name = f'data/run/{store_prefix}_{target}'
    if_exists_delete_store(store_name)
    utrans = get_irdc_priors(vsys=VELOS[target])
    runner_cls = nf.AmmoniaRunner
    stack = get_cubestack(target)
    fitter = nf.CubeFitter(stack, utrans, runner_cls, ncomp_max=2,
            nlive_snr_fact=5)
    fitter.fit_cube(store_name=store_name, nproc=nproc)


def run_nested_all(store_prefix, nproc=8):
    for target in TARGETS_NOMOS:
        run_nested(target, store_prefix, nproc=nproc)


def postprocess_run(target, store_prefix):
    evid_kernel = convolution.Gaussian2DKernel(1.5)  # std-dev in pixels
    s2 = np.sqrt(2) / 2
    k_arr = np.array([
            [s2**2, s2**1, s2**2],
            [s2**1, s2**0, s2**1],
            [s2**2, s2**1, s2**2],
    ])
    post_kernel = convolution.CustomKernel(k_arr)
    utrans = get_irdc_priors(vsys=VELOS[target])
    par_bins = get_bins(VELOS[target])
    store_name = f'data/run/{store_prefix}_{target}'
    store = nf.HdfStore(store_name)
    stack = get_cubestack(target)
    runner = get_runner(stack, utrans, ncomp=1)
    # begin post-processing steps
    nf.aggregate_run_attributes(store)
    nf.convolve_evidence(store, evid_kernel)
    nf.aggregate_run_products(store)
    nf.aggregate_run_pdfs(store, par_bins=par_bins)
    nf.convolve_post_pdfs(store, post_kernel, evid_weight=False)
    nf.quantize_conv_marginals(store)
    nf.deblend_hf_intensity(store, stack, runner)
    store.close()


def parallel_postprocess(store_prefix, nproc=12):
    args = zip(
            TARGETS_NOMOS,
            [store_prefix] * len(TARGETS_NOMOS),
    )
    with Pool(nproc) as pool:
        pool.starmap(postprocess_run, args)


if __name__ == '__main__':
    prefix = 'nested'
    args = sys.argv[1:]
    assert len(args) > 0
    assert args[0] in ('--run-nested', '--post-proc')
    flag = args[0]
    if flag == '--run-nested':
        run_nested_all(prefix, nproc=16)
    elif flag == '--post-proc':
        parallel_postprocess(prefix, nproc=12)


