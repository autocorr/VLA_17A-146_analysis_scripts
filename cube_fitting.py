#!/usr/bin/env python3
"""
============
Cube Fitting
============

Fit the NH3 (1,1) and (2,2) transitions using for the VLA SCC data.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from skimage import morphology
from matplotlib import pyplot as plt
from matplotlib import (patheffects, colors)

import aplpy
import pyspeckit
from pyspeckit.spectrum.models import ammonia
import radio_beam
from astropy.io import fits
from astropy import units as u
from astropy import (convolution, coordinates, wcs)

from . import (PDir, read_cube, MOL_NAMES, RESTFREQS, TARGETS, VELOS)


# Percent point function, inverse of normal cumulative distribution function
MAD_K = 1 / stats.norm.ppf(0.75)


def calc_errmap(cube, pbeam_cube, chans=10):
    """
    Calculate the image cube RMS using the (MAD * 1.4826) over the specified
    number of channels at the start of the cube.
    """
    values = np.ravel(cube.to(u.K)[:chans,:,:])
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(med - values))
    rms = MAD_K * mad
    errmap = np.nan_to_num(rms / pbeam_cube[0,:,:].value)
    return errmap


def default_mask(hdul, threshold=0.001):
    """
    Parameters
    ----------
    hdul : astropy.io.fits.hdu.hdulist.HDUList
    threshold : number, default 0.001
        Threshold chosen to be approximately 4-sigma from matched-filter cube
        (mf) of G28539 NH3 (1,1) cube
    """
    data = hdul[0].data.copy()
    mask = np.nan_to_num(data) > threshold
    mask = morphology.remove_small_objects(mask, min_size=40)
    mask = morphology.opening(mask, morphology.disk(1))
    return mask


def calc_peak_index(hdul):
    data = hdul[0].data
    ymax, xmax = np.unravel_index(np.nanargmax(data), data.shape)
    return xmax, ymax


def calc_snr_peak(cube, ix, errmap):
    """
    Calculate peak SNR from cube. Cube and error map should be consistent
    depending on whether they have been corrected for primary beam attenuation.
    The index ordering should match `calc_peak_index`.

    Parameters
    ----------
    cube : spectral_cube.SpectralCube
    ix : tuple
    errmap : np.ndarray
    """
    spec = cube.to(u.K)[:,ix[1],ix[0]]
    rms = errmap[ix[1],ix[0]]
    snr = spec.max().value / rms
    return snr


def make_pyspeckit_cubestack(cubes, mask):
    """
    Converts the cubes to use a frequency axis and brightness temperature
    units, then initializes the `pyspeckit.CubeStack`

    Parameters
    ----------
    cubes : list
    mask : np.ndarray
    """
    pysk_cubes = []
    for cube in cubes:
        cube = cube.with_spectral_unit(u.GHz).to(u.K)
        pysk_cubes.append(pyspeckit.Cube(cube=cube, maskmap=mask))
    stack = pyspeckit.CubeStack(pysk_cubes, maskmap=mask)
    if not 'cold_ammonia' in stack.specfit.Registry.multifitters:
        stack.specfit.Registry.add_fitter(
                'cold_ammonia', ammonia.cold_ammonia_model(), 6)
    return stack


def insert_parcube_header_keywords(hdu):
    header_params = [
            ('PLANE1',  'TKIN'),
            ('PLANE2',  'TEX'),
            ('PLANE3',  'COLUMN'),
            ('PLANE4',  'SIGMA'),
            ('PLANE5',  'VELOCITY'),
            ('PLANE6',  'FORTHO'),
            ('PLANE7',  'eTKIN'),
            ('PLANE8',  'eTEX'),
            ('PLANE9',  'eCOLUMN'),
            ('PLANE10', 'eSIGMA'),
            ('PLANE11', 'eVELOCITY'),
            ('PLANE12', 'eFORTHO'),
            ('CDELT3',   1),
            ('CTYPE3',  'FITPAR'),
            ('CRVAL3',   0),
            ('CRPIX3',   1),
    ]
    for key, val in header_params:
        hdu.header.set(key, val)


def cubefit(target, multicore=1):
    """
    Fit the NH3 (1,1) and (2,2) cubes for the requested target region.

    Parameters
    ----------
    target : str
        Target name, ex 'G24051'
    multicore : number, default 1
        Number of cores to use in parallel spectral fitting
    ext : str
        Extra file extension
    """
    def get_path(mol, imtype='image', ext=None):
        img_ext = f'{imtype}.{ext}.fits' if ext is not None else f'{imtype}.fits'
        modif = None if imtype == 'pb' else 'jfeather'
        return PDir.vla_map_name(target, mol, modif=modif, ext=img_ext)
    print(':: Reading in data')
    image_11_cube = read_cube(get_path('nh3_11'))
    pbcor_11_cube = read_cube(get_path('nh3_11', imtype='pbcor'))
    image_mf_cube = read_cube(get_path('nh3_11', ext='mf'))
    image_m0_hdul = fits.open(get_path('nh3_11', ext='mf0'))
    image_m1_hdul = fits.open(get_path('nh3_11', ext='mf1'))
    pbeam_11_cube = read_cube(get_path('nh3_11', imtype='pb'))
    image_22_cube = read_cube(get_path('nh3_22'))
    pbcor_22_cube = read_cube(get_path('nh3_22', imtype='pbcor'))
    errmap = calc_errmap(image_11_cube, pbeam_11_cube)
    # make mask and compute centroid guess from matched-filter-peak
    mask = default_mask(image_m0_hdul)
    ix_peak = calc_peak_index(image_m0_hdul)
    snr_peak = calc_snr_peak(pbcor_11_cube, ix_peak, errmap)
    vcen = np.squeeze(image_m1_hdul[0].data)
    vmid = np.nanmedian(vcen[mask])
    vmin = vmid - 5  # km/s
    vmax = vmid + 5  # km/s
    # set fit property initial guesses
    stack = make_pyspeckit_cubestack([pbcor_11_cube, pbcor_22_cube], mask)
    guesses = np.zeros((6,) + stack.cube.shape[1:], dtype=float)
    guesses[0,:,:] = 12    # Kinetic temperature
    guesses[1,:,:] =  3    # Excitation temperature
    guesses[2,:,:] = 14.5  # log(Column density)
    guesses[3,:,:] =  0.4  # Velocity dispersion
    guesses[4,:,:] = vcen  # Velocity centroid
    guesses[5,:,:] =  0.0  # ortho-NH3 fraction
    # perform fit
    print(':: Beginning line fitting')
    stack.fiteach(
            fittype='cold_ammonia',
            guesses=guesses,
            integral=False,
            verbose_level=3,
            signal_cut=2,
            fixed=[False, False, False, False, False, True],
            limitedmin=[True, True, True, True, True, True],
            minpars=[5.0, 2.8, 12.0, 0.04, vmin, 0.0],
            maxpars=[0.0, 0.0, 17.0, 0.00, vmax, 1.0],
            start_from_point=ix_peak,
            use_neighbor_as_guess=False,
            position_order=1/snr_peak,
            errmap=errmap,
            multicore=multicore,
    )
    # fix header and write out property cube
    print(':: Writing results to file')
    fitdata = np.concatenate([stack.parcube, stack.errcube])
    fithdu = fits.PrimaryHDU(fitdata, header=stack.header)
    insert_parcube_header_keywords(fithdu)
    out_path = PDir.ROOT / Path(f'property_maps/{target}/{target}_nh3_parmap.fits')
    fithdu.writeto(str(out_path), overwrite=True)


def cubefit_all():
    for target in TARGETS:
        if target in ('G285_mosaic',):
            continue
        cubefit(target, multicore=16)


def inspect_fit_results(target):
    """
    Use the pyspeckit map fit viewer tool to inspect fits from the parameter
    maps. Note that this has to be run with an ipython that is using a gui for
    matplot, not agg.
    """
    def get_cube(mol):
        path = PDir.vla_map_name(target, mol, modif='jfeather',
                ext='pbcor.fits')
        cube = read_cube(path)
        return cube
    pbcor_11_cube = get_cube('nh3_11')
    pbcor_22_cube = get_cube('nh3_22')
    parmap_filen = f'property_maps/{target}/{target}_nh3_parmap.fits'
    parmap_path = PDir.ROOT / Path(parmap_filen)
    stack = make_pyspeckit_cubestack([pbcor_11_cube, pbcor_22_cube], None)
    stack.xarr.velocity_convention = 'radio'
    stack.load_model_fit(str(parmap_path), 6, fittype='cold_ammonia')
    # Init special stacked ammonia spectrum viewer
    vmid = VELOS[target]
    stack.plot_special = pyspeckit.wrappers.fitnh3.plotter_override
    stack.plot_special_kwargs = {
            'fignum': 3,
            'vrange': [vmid-50, vmid+50],  # km/s
            'show_hyperfine_components': False,
    }
    stack.has_fit[:,:] = stack.parcube[2,:,:] > 0
    # Start the interactive fit viewer
    plt.ion()
    plt.close('all')
    stack.mapplot(estimator=2, vmin=14.2, vmax=15.5)
    plt.figure(3)
    plt.show()
    return stack


