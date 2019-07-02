#!/usr/bin/env python3
"""
================
Analysis Scripts
================

Analyse the GBT/VLA data for the SCC NH3 survey.
"""

from pathlib import Path

import numpy as np

import spectral_cube
from astropy import units as u


FWHM = np.sqrt(8 * np.log(2))

MOL_NAMES = [
    'h2o_6_5', 'CCS_2_1', 'hc7n_20_19', 'hc7n_21_20',
    'nh3_11', 'nh3_22', 'nh3_33', 'hc5n_9_8', 'nh3_44',
]

RESTFREQS = {
    'h2o_6_5':    22.2350798,
    'CCS_2_1':    22.3440308,
    'hc7n_20_19': 22.5599155,
    'hc7n_21_20': 23.6878974,
    'nh3_11':     23.6944955,
    'nh3_22':     23.7226333,
    'nh3_33':     23.8701292,
    'hc5n_9_8':   23.9639007,
    'nh3_44':     24.1394163,
}

TARGETS = [
    'G22695', 'G23297', 'G23481', 'G23605', 'G24051', 'G28539', 'G28565',
    'G29558', 'G29601', 'G30120', 'G30660', 'G30912', 'G285_mosaic',
]

VELOS = {
    'G30912': 50.74, 'G30660': 80.20, 'G30120': 65.31, 'G29601': 75.78,
    'G29558': 79.72, 'G28539': 88.60, 'G28565': 87.46, 'G22695': 77.80,
    'G23297': 55.00, 'G23481': 63.80, 'G23605': 87.00, 'G24051': 81.51,
    'G285_mosaic': 88.6,
}


class PathDirectory:
    ROOT = Path('/users/bsvoboda/lustre/17A-146/data')
    CAT = ROOT / Path('catalogs')
    GBT = ROOT / Path('gbt_cubes')
    IMG = ROOT / Path('images')
    PLT = ROOT / Path('plots')

    def map_name(self, name, mol, modif=None, ext='fits'):
        if modif is None:
            return Path(f'{name}/{name}_{mol}.{ext}')
        else:
            return Path(f'{name}/{name}_{mol}_{modif}.{ext}')

    def gbt_map_name(self, *args, **kwargs):
        return self.GBT / self.map_name(*args, **kwargs)

    def vla_map_name(self, *args, **kwargs):
        return self.IMG / self.map_name(*args, **kwargs)

PDir = PathDirectory()


def read_cube(path, velo=True):
    cube = spectral_cube.SpectralCube.read(str(path))
    if velo:
        return cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    else:
        return cube


def write_cube(cube, path, overwrite=True):
    cube.write(str(path), overwrite=overwrite)


