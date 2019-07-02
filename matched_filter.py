#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from matplotlib import ticker
from matplotlib import pyplot as plt

import pyspeckit
import spectral_cube
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from astropy import (convolution, coordinates, wcs)

from . import (PDir, read_cube, MOL_NAMES, RESTFREQS, TARGETS, VELOS)


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)


#def make_freq_axis(cube, nchan=260):  # XXX, for plotting
def make_freq_axis(cube, nchan=190):
    restfreq = cube.header['RESTFRQ'] * u.Hz
    dv = cube.spectral_axis[1] - cube.spectral_axis[0]
    df = (dv / c.c * restfreq).to('GHz')
    faxis = np.arange(-nchan, nchan+1) * df + restfreq
    spaxis = pyspeckit.spectrum.units.SpectroscopicAxis(faxis)
    return spaxis


def make_ammonia_kernel(cube, normalize=True):
    spaxis = make_freq_axis(cube)
    model = pyspeckit.spectrum.models.ammonia.ammonia(
            spaxis,
            trot=12,  # K
            ntot=15,  # log cm**-2
            width=0.3,  # km/s, dispersion
    )
    kernel = convolution.CustomKernel(model)
    if normalize:
        kernel.normalize()
    return kernel


def make_ammonia_delta_kernel(cube, normalize=True):
    spaxis = make_freq_axis(cube)
    components = pyspeckit.spectrum.models.ammonia.ammonia(
            spaxis,
            trot=12,  # K
            ntot=15,  # log cm**-2
            width=0.05,  # km/s, 1 chan ~ 0.156 km/s
            return_components=True,
    )
    if spaxis[0] > 23.7 * u.GHz:
        # HF for the (2,2)
        #hf_set1 = components[18:24,:].sum(axis=0)
        #hf_set2 = components[33:39,:].sum(axis=0)
        #model = hf_set1 + hf_set2
        model = components[18:39,:].sum(axis=0)
    else:
        # HF for the (1,1)
        #hf_set1 = components[0:5,:].sum(axis=0)
        #hf_set2 = components[13:18,:].sum(axis=0)
        #model = hf_set1 + hf_set2
        model = components[0:18,:].sum(axis=0)
    kernel = convolution.CustomKernel(model)
    if normalize:
        kernel.normalize()
    return kernel


def match_filter_cube(path, delta=False):
    pathbase = str(path).rstrip('.fits')
    cube = read_cube(path)
    if delta:
        kernel = make_ammonia_delta_kernel(cube)
        dext = 'd'
    else:
        kernel = make_ammonia_kernel(cube)
        dext = ''
    mf_cube = cube.spectral_smooth(kernel, num_cores=4)
    mf_cube.write(pathbase+f'.{dext}mf.fits', overwrite=True)
    # limit velo range
    vc = VELOS[path.stem[:6]] * u.km/u.s
    slab = mf_cube.spectral_slab(vc-3*u.km/u.s, vc+3*u.km/u.s)
    # moment-zero / peak-intensity
    mf0_cube = slab.max(axis=0)
    mf0_cube.write(pathbase+f'.{dext}mf0.fits', overwrite=True)
    # apply noise threshold
    thresh = 2 * 6e-4 * cube.unit
    ixarr = slab.with_mask(slab > thresh).argmax(axis=0)
    data = slab.spectral_axis[ixarr.ravel()].reshape(1, *ixarr.shape)
    # moment-one / intensity weighted velocity
    mf1_cube = spectral_cube.SpectralCube(data, slab.wcs, header=slab.header)
    mf1_cube.write(pathbase+f'.{dext}mf1.fits', overwrite=True)


def match_filter_cube_all(delta=False):
    for target in TARGETS:
        if target == 'G285_mosaic':
            continue
        for mol in ('nh3_11', 'nh3_22'):
            print('::', target, mol)
            path = PDir.vla_map_name(target, mol, modif='jfeather',
                    ext='image.fits')
            match_filter_cube(path, delta=delta)


def test_match_filter_cube():
    path = PDir.vla_map_name('G24051', 'nh3_11', modif='jfeather',
            ext='image.fits')
    match_filter_cube(path, delta=False)


def test_get_vaxis_kernel(mol):
    path = PDir.vla_map_name('G24051', mol, modif='jfeather',
            ext='image.fits')
    cube = read_cube(path)
    spaxis = make_freq_axis(cube, nchan=260)
    restfreq = cube.header['RESTFRQ'] * u.Hz
    vaxis = ((spaxis - restfreq) / restfreq * c.c).to('km/s')
    #kernel = make_ammonia_kernel(cube, normalize=False)
    kernel = make_ammonia_delta_kernel(cube, normalize=False)
    return vaxis, kernel


def test_plot_ammonia_kernel():
    vaxis, kernel11 = test_get_vaxis_kernel('nh3_11')
    _, kernel22 = test_get_vaxis_kernel('nh3_22')
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.plot(vaxis, kernel11.array[::-1] / kernel11.array.max(),
            color='orangered', label=r'$\mathrm{NH_3} \ (1,1)$',
            drawstyle='steps-mid', linewidth=0.7)
    #v_offset = 0 * u.km / u.s
    v_offset = 2 * u.km / u.s  # XXX
    ax.plot(vaxis+v_offset, kernel22.array[::-1] / kernel22.array.max(),
            color='dodgerblue', label=r'$\mathrm{NH_3} \ (2,2)$',
            drawstyle='steps-mid', linewidth=0.7)
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r'$v_\mathrm{lsr} \ \ [\mathrm{km \, s^{-1}}]$')
    ax.set_ylabel(r'Relative Intensity')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    plt.tight_layout()
    #plt.savefig(PDir.PLT / Path('test_nh3_kernel.pdf'))
    plt.savefig(PDir.PLT / Path('test_nh3_kernel_delta.pdf'))  # XXX
    plt.close('all')


