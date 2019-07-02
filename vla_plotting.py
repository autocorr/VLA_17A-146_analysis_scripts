#!/usr/bin/env python3

import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from skimage import morphology
from matplotlib import pyplot as plt
from matplotlib import (patheffects, colors)

import aplpy
import radio_beam
from astropy.io import fits
from astropy import units as u
from astropy import (convolution, coordinates, wcs)

from . import (PDir, read_cube, MOL_NAMES, RESTFREQS, TARGETS, VELOS)


warnings.simplefilter('ignore')  # NOTE stop that matplotlib deprecation warning
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out')
plt.rc('ytick', direction='out')


CLR_CMAP = plt.cm.Spectral_r
CLR_CMAP.set_bad('0.5', 1.0)
HOT_CMAP = plt.cm.afmhot
HOT_CMAP.set_bad('0.5', 1.0)  # 0.2 for darker
#RDB_CMAP = plt.cm.coolwarm_r  # also seismic_r
RDB_CMAP = plt.cm.RdBu
RDB_CMAP.set_bad('0.5', 1.0)
TAU_CMAP = plt.cm.YlGnBu_r
TAU_CMAP.set_bad('0.5', 1.0)
VIR_CMAP = plt.cm.viridis
VIR_CMAP.set_bad('0.5', 1.0)


def open_fits(filen):
    hdul = fits.open(PDir.ROOT / Path(filen))
    return hdul


def default_mask(hdul):
    data = deepcopy(hdul[0].data)
    mask = np.nan_to_num(data) > 0
    mask = morphology.remove_small_objects(mask, min_size=64)
    kernel = morphology.disk(1)
    mask = morphology.opening(mask.squeeze(), kernel)
    return mask


def get_map_center(hdul):
    shape = hdul[0].data.shape
    im_wcs = wcs.WCS(hdul[0].header)
    decim_deg = im_wcs.all_pix2world(shape[1]/2, shape[2]/2, 0, 0)
    ra, dec = decim_deg[0].item(), decim_deg[1].item()
    co = coordinates.SkyCoord(ra, dec, frame='fk5', unit=(u.deg, u.deg))
    return co


def convert_jy_to_k(hdul, mol):
    beam = radio_beam.Beam.from_fits_header(hdul[0].header)
    freq = RESTFREQS[mol] * u.GHz
    conv = (1 * u.Jy).to(u.K, u.brightness_temperature(beam, freq))
    hdul[0].data = hdul[0].data * conv.value
    return conv


def add_label(label, xy, fontsize=10):
    txt = plt.annotate(label, xy=xy, xycoords='axes fraction',
        fontsize=fontsize)
    txt.set_path_effects([patheffects.withStroke(linewidth=4.5,
        foreground='w')])
    return txt


def save_figure(filen):
    for ext in ('png', 'pdf', 'eps'):
        path = PDir.PLT / Path('{0}.{1}'.format(filen, ext))
        plt.savefig(str(path), dpi=300)
        print('-- {0} saved'.format(ext))
    plt.close('all'); plt.cla(); plt.clf()


def fix_label_ticks(gc, co, radius=1.5/60):
    gc.recenter(co.fk5.ra.value, co.fk5.dec.value, radius=radius)
    gc.axis_labels.set_ypad(-7)
    gc.ticks.set_color('black')
    gc.tick_labels.set_xformat('hh:mm:ss')
    gc.tick_labels.set_yformat('dd:mm:ss')
    gc.axis_labels.set_font(size='small')
    gc.tick_labels.set_font(size='x-small')


def add_beam(gc, corner='top left', facecolor='black',
        edgecolor='none'):
    gc.add_beam()
    gc.beam.set_corner(corner)
    gc.beam.set_color(facecolor)
    gc.beam.set_edgecolor(edgecolor)
    gc.beam.set_linewidth(1.0)


def add_colorbar(gc, box, label=None, top=False):
    gc.add_colorbar(box=box, pad=0.2, width=0.1, axis_label_text=label)
    gc.colorbar.set_font(size='x-small')
    gc.colorbar._colorbar.ax.tick_params(direction='in')
    if top:
        gc.colorbar.set_location('top')
        gc.colorbar.set_width(0.1)
        gc.colorbar.set_pad(0.12)
        gc.colorbar._colorbar.ax.tick_params(direction='in')
        gc.colorbar._colorbar.ax.xaxis.set_ticks_position('top')


def add_primary_beam_contour(gc, hdul, power=0.205):
    gc.show_contour(hdul, dimensions=[0,1], slices=[100,0], levels=[power],
            linewidths=0.5, linestyles='dashed', colors='0.25',
            convention='calabretta')


def zero_out_corners(hdu):
    hdu[0].data[ 0, 0] = 0
    hdu[0].data[ 0,-1] = 0
    hdu[0].data[-1, 0] = 0
    hdu[0].data[-1,-1] = 0


def plot_four_moments(source, mol, matched=True):
    filen = 'moments/{0}/{0}_{1}'.format(source, mol)
    img0 = open_fits(filen + '_snr2_smooth0.integrated.pbcor.fits')
    pb = open_fits('images/{0}/{0}_{1}_joint.pb.fits'.format(source, mol))
    if matched:
        img1 = open_fits('images/{0}/{0}_{1}_jfeather.image.mf0.fits'.format(source, mol))
        pbdata = pb[0].data[0,0,:,:].reshape(img1[0].data.shape)
        img1[0].data = img1[0].data / pbdata
    else:
        img1 = open_fits(filen + '_snr2_smooth0.maximum.fits')
    img2 = open_fits(filen + '_snr3_smooth0.weighted_coord.fits')
    img3 = open_fits(filen + '_snr3_smooth0.weighted_dispersion_coord.fits')
    convert_jy_to_k(img0, mol)
    convert_jy_to_k(img1, mol)
    for hdul in (img0, img1, img2, img3):
        data = hdul[0].data
        mask = default_mask(hdul).reshape(data.shape)
        data[~mask] = np.nan
        zero_out_corners(hdul)
    for hdul in (img0, img1):
        data = hdul[0].data
        data[~np.isfinite(data)] = 0.0
    fig = plt.figure(figsize=(8, 6.5))
    split_size = 0.02
    x_margin_size = 0.15
    y_margin_size = 0.075
    delta = 0.06
    x0 = x_margin_size - delta
    y0 = y_margin_size
    dx = (1 - 2 * x_margin_size - split_size) / 2
    dy = (1 - 2 * y_margin_size - split_size) / 2
    x1 = x0 + dx + split_size + 1.5 * delta
    y1 = y0 + dy + split_size + 0.015
    labelx = 0.035
    labely = 0.035
    xlabel = r'$\alpha \ (\mathrm{J2000})$'
    ylabel = r'$\delta \ (\mathrm{J2000})$'
    #alma = open_fits('other_data/alma_b6/calibrated_{0}_joint.image.fits'.format(source))
    #def add_alma_contours(gc):
    #    levels = np.array([3, 20]) * 5e-5  # in Jy/beam
    #    colors = ['0.400', '0.000']
    #    gc.show_contour(alma, levels=levels, colors=colors,
    #            linewidths=0.5, convention='calabretta')
    # Moment 0, integrated intensity
    gc0 = aplpy.FITSFigure(img0, figure=fig, subplot=[x0, y1, dx, dy])
    gc0.show_colorscale(vmin=0, vmax=15, cmap=HOT_CMAP)
    gc0.tick_labels.hide_x()
    gc0.axis_labels.hide_x()
    gc0.axis_labels.set_ytext(ylabel)
    add_label(r'Moment 0 $(\int I \, dv)$', (labelx, labely))
    add_colorbar(gc0, [x0+dx+0.01, y1, 0.02, dy],
        label=r'$\int I \, dv \ \ [\mathrm{K \, km \, s^{-1}}]$')
    add_beam(gc0, facecolor='white')
    # Maximum, peak intensity
    gc1 = aplpy.FITSFigure(img1, figure=fig, subplot=[x1, y1, dx, dy])
    if matched:
        gc1.show_colorscale(vmin=-0.1, vmax=5, cmap=HOT_CMAP, stretch='power',
                exponent=1)
        add_label(r'Matched Peak', (labelx, labely))
        add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
            label=r'$\mathrm{max}(\phi * I) \ \ [\mathrm{K}]$')
    else:
        gc1.show_colorscale(vmin=0, vmax=10, cmap=HOT_CMAP, stretch='power',
                exponent=1)
        add_label(r'Maximum $(I_\mathrm{pk})$', (labelx, labely))
        add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
            label=r'$T_\mathrm{pk} \ \ [\mathrm{K}]$')
    gc1.axis_labels.hide()
    gc1.tick_labels.hide()
    add_beam(gc1, facecolor='white')
    # Moment 1, intensity weighted velocity
    gc2 = aplpy.FITSFigure(img2, figure=fig, subplot=[x0, y0, dx, dy])
    vcen = VELOS[source]
    gc2.show_colorscale(vmin=vcen-2, vmax=vcen+2, cmap=RDB_CMAP)
    gc2.axis_labels.set_xtext(xlabel)
    gc2.axis_labels.set_ytext(ylabel)
    add_label(r'Moment 1 $(v_\mathrm{lsr})$', (labelx, labely))
    add_colorbar(gc2, [x0+dx+0.01, y0, 0.02, dy],
        label=r'$v_\mathrm{lsr} \ \ [\mathrm{km \, s^{-1}}]$')
    add_beam(gc2)
    # Moment 2, velocity dispersion
    gc3 = aplpy.FITSFigure(img3, figure=fig, subplot=[x1, y0, dx, dy])
    gc3.show_colorscale(vmin=0.15, vmax=0.8, cmap=CLR_CMAP)
    add_label(r'Moment 2 $(\sigma_v)$', (labelx, labely))
    add_colorbar(gc3, [x1+dx+0.01, y0, 0.02, dy],
        label=r'$\sigma_v \ \ [\mathrm{km \, s^{-1}}]$')
    gc3.tick_labels.hide_y()
    gc3.axis_labels.hide_y()
    gc3.axis_labels.set_xtext(xlabel)
    add_beam(gc3)
    # Common axis options
    center = get_map_center(img0)
    for gc in (gc0, gc1, gc2, gc3):
        fix_label_ticks(gc, center, radius=1.45/60)
        add_primary_beam_contour(gc, pb, power=0.205)
        add_primary_beam_contour(gc, pb, power=0.5)
    out_filen = '{0}_{1}_moment_four_panel'.format(source, mol)
    out_filen = out_filen + '_mf0' if matched else out_filen
    save_figure(out_filen)


def plot_four_moments_all():
    for source in TARGETS:
        print(':: ', source)
        #for mol in MOL_NAMES:
        for mol in ('nh3_11', 'nh3_22'):
            if source == 'G285_mosaic' or mol not in ('nh3_11', 'nh3_22'):
                continue
            print('-- ', mol)
            plot_four_moments(source, mol)


class ParmapHandler:
    filen = 'property_maps/{0}/{0}_nh3_parmap.fits'
    props = {
            'tkin': 0, 'e_tkin':  6,
            'texc': 1, 'e_texc':  7,
            'ncol': 2, 'e_ncol':  8,
            'sigm': 3, 'e_sigm':  9,
            'vcen': 4, 'e_vcen': 10,
    }
    err_offset = 6
    bad_thresh = 1e3

    def __init__(self, source):
        self.source = source
        self.hdul = open_fits(self.filen.format(source))
        self.header = self.hdul[0].header
        self.wcs = wcs.WCS(self.hdul[0].header)
        self.clean_hdul()

    def clean_hdul(self):
        ncol = self.get_hdu('ncol').data.copy()
        e_ncol = self.get_hdu('e_ncol').data.copy()
        ncol[(e_ncol > 0.5) | (ncol > 16.9)] = np.nan
        mask = np.isnan(ncol).squeeze()
        data = self.hdul[0].data
        for ii in range(5):
            data[ii][mask] = np.nan

    def get_hdu(self, prop, snr_thresh=3):
        ix = self.props[prop]
        data = self.hdul[0].data[ix:ix+1,:,:]
        data[data == 0] = np.nan
        data[data > self.bad_thresh] = np.nan
        if not prop.startswith('e_'):
            ix_e = ix + self.err_offset
            err = self.hdul[0].data[ix_e:ix_e+1,:,:]
            data[data / err < snr_thresh] = np.nan
        header = self.header.copy()
        hdu = fits.PrimaryHDU(data=data, header=self.header)
        return hdu


def plot_four_nh3_properties(source):
    pmh = ParmapHandler(source)
    pb = fits.open(PDir.vla_map_name(source, 'nh3_11', ext='pb.fits'))
    fig = plt.figure(figsize=(8, 6.5))
    split_size = 0.02
    x_margin_size = 0.15
    y_margin_size = 0.075
    delta = 0.06
    x0 = x_margin_size - delta
    y0 = y_margin_size
    dx = (1 - 2 * x_margin_size - split_size) / 2
    dy = (1 - 2 * y_margin_size - split_size) / 2
    x1 = x0 + dx + split_size + 1.5 * delta
    y1 = y0 + dy + split_size + 0.015
    labelx = 0.035
    labely = 0.035
    xlabel = r'$\alpha \ (\mathrm{J2000})$'
    ylabel = r'$\delta \ (\mathrm{J2000})$'
    def add_ncol_contours(gc):
        levels = np.array([14.75, 15.25])  # log(N / cm^2)
        colors = ['0.20', '0.00']
        gc.show_contour(data=pmh.get_hdu('ncol'), levels=levels,
                colors=colors, linewidths=0.5, convention='calabretta')
    # NH3 column density
    gc0 = aplpy.FITSFigure(pmh.get_hdu('ncol'), figure=fig, subplot=[x0, y1, dx, dy])
    gc0.show_colorscale(vmin=14.0, vmax=15.5, cmap=VIR_CMAP)
    gc0.tick_labels.hide_x()
    gc0.axis_labels.hide_x()
    gc0.axis_labels.set_ytext(ylabel)
    add_label(r'Column Density', (labelx, labely))
    add_colorbar(gc0, [x0+dx+0.01, y1, 0.02, dy],
        label=r'$\log(N(\mathrm{NH_3}) \, [\mathrm{cm^{-2}}])$')
    # Kinetic temperature
    gc1 = aplpy.FITSFigure(pmh.get_hdu('tkin'), figure=fig, subplot=[x1, y1, dx, dy])
    gc1.show_colorscale(vmin=10, vmax=18, cmap=VIR_CMAP)
    gc1.axis_labels.hide()
    gc1.tick_labels.hide()
    add_label(r'Kinetic Temp.', (labelx, labely))
    add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
        label=r'$T_\mathrm{K} \, [\mathrm{K}]$')
    # Velocity centroid
    gc2 = aplpy.FITSFigure(pmh.get_hdu('vcen'), figure=fig, subplot=[x0, y0, dx, dy])
    vcen = VELOS[source]
    gc2.show_colorscale(vmin=vcen-2, vmax=vcen+2, cmap=RDB_CMAP)
    gc2.axis_labels.set_xtext(xlabel)
    gc2.axis_labels.set_ytext(ylabel)
    add_label(r'Velo. Centroid', (labelx, labely))
    add_colorbar(gc2, [x0+dx+0.01, y0, 0.02, dy],
        label=r'$v_\mathrm{lsr} \ \ [\mathrm{km \, s^{-1}}]$')
    # Velocity dispersion
    gc3 = aplpy.FITSFigure(pmh.get_hdu('sigm'), figure=fig, subplot=[x1, y0, dx, dy])
    gc3.show_colorscale(vmin=0.15, vmax=0.8, cmap=CLR_CMAP)
    add_label(r'Velo. Dispersion', (labelx, labely))
    add_colorbar(gc3, [x1+dx+0.01, y0, 0.02, dy],
        label=r'$\sigma_v \ \ [\mathrm{km \, s^{-1}}]$')
    gc3.tick_labels.hide_y()
    gc3.axis_labels.hide_y()
    gc3.axis_labels.set_xtext(xlabel)
    # Common axis options
    center = get_map_center([pmh.get_hdu('ncol')])
    for gc in (gc0, gc1, gc2, gc3):
        fix_label_ticks(gc, center, radius=1.45/60)
        add_beam(gc)
        add_primary_beam_contour(gc, pb, power=0.205)
        add_primary_beam_contour(gc, pb, power=0.5)
    save_figure(f'{source}_parmap_four_panel')


def plot_four_nh3_properties_all():
    for source in TARGETS:
        if source == 'G285_mosaic':
            continue
        print(':: ', source)
        plot_four_nh3_properties(source)


