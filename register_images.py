#!/usr/bin/env python3
"""
Register VLA and GBT images to see if there are significant pointing offsets.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import (patheffects, colors)

import spectral_cube
import image_registration
from astropy import units as u

from . import (TARGETS, PDir, read_cube)


warnings.simplefilter('ignore')  # NOTE stop that matplotlib deprecation warning
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


class ImagePair:
    nthreads = 4

    def __init__(self, name, mol):
        self.name = name
        self.mol = mol
        self.gbt = self.read_gbt()
        self.vla = self.read_vla()
        self.nchan = self.gbt.shape[0]

    @property
    def gbt_rms(self):
        return self.gbt[0:50,190,190].std()

    @property
    def vla_rms(self):
        return self.vla[0:50,190,190].std()

    @property
    def gbt_errmap(self):
        return np.ones(self.gbt[0].shape) * self.gbt_rms

    @property
    def vla_errmap(self):
        return np.ones(self.vla[0].shape) * self.vla_rms

    def read_gbt(self):
        path = PDir.GBT / Path(f'for_joint/{self.name}_{self.mol}.jypix.fits')
        cube = read_cube(path)
        cube = cube.with_fill_value(0)
        return cube

    def read_vla(self):
        path = PDir.vla_map_name(self.name, self.mol, ext='image.fits')
        cube = read_cube(path)
        cube = cube.with_fill_value(0)
        cube = cube.convolve_to(self.gbt.beam)
        cube *= u.beam / u.pix
        return cube


def calc_offset(imgpair, thresh=4):
    columns = ('chan', 'gbt_snr', 'vla_snr', 'xoff', 'yoff', 'e_xoff',
            'e_yoff')
    df = pd.DataFrame(columns=columns)
    df = df.set_index('chan')
    gbt_thresh = thresh * imgpair.gbt_rms
    vla_thresh = thresh * imgpair.vla_rms
    for ii in range(imgpair.nchan):
        gbt_slice = imgpair.gbt[ii]
        vla_slice = imgpair.vla[ii]
        gbt_snr = gbt_slice.max() / imgpair.gbt_rms
        vla_snr = vla_slice.max() / imgpair.vla_rms
        if gbt_snr < thresh or vla_snr < thresh:
            continue
        df.loc[ii, ['gbt_snr', 'vla_snr']] = gbt_snr, vla_snr
        offset_pars = image_registration.chi2_shift(
                vla_slice.value, gbt_slice.value,
                err=imgpair.gbt_errmap.value,
                nthreads=imgpair.nthreads, return_error=True,
                upsample_factor='auto')
        df.loc[ii, ['xoff', 'yoff', 'e_xoff', 'e_yoff']] = offset_pars
    df['w_xoff'] = 1 / df.e_xoff**2
    df['w_yoff'] = 1 / df.e_yoff**2
    return df


def calc_weighted_offset(df):
    if len(df) == 0:
        return [np.nan] * 4
    xoff_mean = (df.w_xoff * df.xoff).sum() / df.w_xoff.sum()
    yoff_mean = (df.w_yoff * df.yoff).sum() / df.w_yoff.sum()
    xoff_err  = 1 / (np.sqrt(np.sum(df.w_xoff)))
    yoff_err  = 1 / (np.sqrt(np.sum(df.w_yoff)))
    return xoff_mean, yoff_mean, xoff_err, yoff_err


def apply_image_registration(imgpair, offset):
    image_registration.fft_tools.shift.shiftnd()


def test_calc_offset(imgpair=None):
    if imgpair is None:
        imgpair = ImagePair('G24051', 'nh3_11')
    print(f':: source: {imgpair.name} {imgpair.mol}')
    df = calc_offset(imgpair)
    out_path = PDir.CAT / Path(f'offsets_{imgpair.name}_{imgpair.mol}.csv')
    df.to_csv(out_path)
    print(f'-- good comparisons: {df.shape[0]}')
    offset = calc_weighted_offset(df)
    print(f'-- weighted offset: {offset}')
    return offset


def test_calc_offset_all():
    weighted_offsets = {}
    for name in TARGETS:
        if name == 'G285_mosaic':
            continue
        for mol in ('nh3_11', 'nh3_22'):
            key = f'{name}_{mol}'
            imgpair = ImagePair(name, mol)
            offset = test_calc_offset(imgpair)
            weighted_offsets[key] = offset
    return weighted_offsets


def plot_offset_hists():
    #targets = [s for s in TARGETS if not s.endswith('mosaic')]
    targets = ['G28539', 'G30660', 'G22695', 'G23605', 'G24051', 'G23297',
            'G23481', 'G29558', 'G30120', 'G28565', 'G29601', 'G30912']
    fig, axes = plt.subplots(nrows=12, ncols=1, sharex=True, sharey=True,
            figsize=(4, 9))
    pix_size = 0.5  # arcsec
    x_lo, x_hi = -30, 30  # arcsec
    y_lo, y_hi =   0, 35
    bins = np.linspace(x_lo, x_hi, 30)
    for ax, name in zip(axes, targets):
        df11 = pd.read_csv(PDir.CAT/Path(f'offsets_{name}_nh3_11.csv'),
                index_col='chan')
        df22 = pd.read_csv(PDir.CAT/Path(f'offsets_{name}_nh3_22.csv'),
                index_col='chan')
        ax.vlines(0, y_lo, y_hi, color='0.5', linestyle='dashed', linewidth=0.5)
        if df11.xoff.notnull().sum() > 0:
            ax.hist(df11.xoff*pix_size, bins=bins,
                    histtype='step', color='dodgerblue')
            ax.vlines(df11.xoff.median()*pix_size, y_lo, y_hi,
                    linestyle='dashed', linewidth=1.0, color='dodgerblue')
            ax.annotate(df11.xoff.notnull().sum(), xy=(0.915, 0.7),
                    xycoords='axes fraction', fontsize=9)
        if df11.yoff.notnull().sum() > 0:
            ax.hist(df11.yoff*pix_size, bins=bins,
                    histtype='step', color='red')
            ax.vlines(df11.yoff.median()*pix_size, y_lo, y_hi,
                    linestyle='dashed', linewidth=1.0, color='red')
        if df22.xoff.notnull().sum() > 0:
            ax.hist(df22.xoff*pix_size, bins=bins,
                    histtype='stepfilled', color='royalblue')
        if df22.xoff.notnull().sum() > 0:
            ax.hist(df22.yoff*pix_size, bins=bins,
                    histtype='stepfilled', color='darkred')
        ax.annotate(name, xy=(0.025, 0.7), xycoords='axes fraction',
                fontsize=9)
        ax.minorticks_on()
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\delta\theta \ [\mathrm{arcsec}]$')
    ax.set_ylabel(r'$N$')
    plt.tight_layout(h_pad=0.1)
    outname = f'gbt_vla_pos_offsets.pdf'
    plt.savefig(PDir.PLT/Path(outname), dpi=300)
    plt.close('all')


