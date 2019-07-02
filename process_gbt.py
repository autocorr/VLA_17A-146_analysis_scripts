#!/usr/bin/env python3
"""
Process the GBT KFPA data cubes.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

import aplpy
import radio_beam
import spectral_cube
from astropy import units as u
from astropy import (coordinates, convolution)
from astropy.io import fits

from . import (FWHM, MOL_NAMES, TARGETS, PDir, read_cube, write_cube)


def calc_velo_res(cube):
    spax = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio').spectral_axis
    return abs(spax[0] - spax[1])


def baseline_poly_cube(filep):
    # FIXME
    cube = read_cube(filep)
    data = cube._data.copy()
    spax = cube.spectral_axis.value
    chans = np.arange(cube.shape[0])
    mask = np.argwhere(
        ((60.5 < spax) & (spax <= 74.0)) |
        ((88.4 < spax) & (spax <= 99.9))).flatten()
    window_chans = chans[mask]
    unit = cube[:,0,0].unit
    for ii in range(cube.shape[1]):
        for jj in range(cube.shape[2]):
            spec = cube[:,ii,jj].value
            if np.isnan(spec).sum() == len(spec):
                continue
            pars = np.polyfit(window_chans, spec[mask], deg=3)
            poly = np.poly1d(pars)
            data[:,ii,jj] -= poly(chans)
    cube._data = data
    write_cube(cube,
        PDir.DAT/Path('bgps_5021_{0}_cube_base.fits'.format(mol)))


def calc_moments(mol='NH3_11'):
    if mol == 'CS':
        rms = 320 * u.mK
    else:
        rms = 120 * u.mK
    v_lo = 83.0 * u.km/u.s
    v_hi = 90.0 * u.km/u.s
    # FIXME
    filep = PDir.map_name('G285_mosaic', 'NH3_11')
    cube = read_cube(filep)
    beam = radio_beam.Beam.from_fits_header(cube.header)
    cube = cube.spectral_slab(v_lo, v_hi)
    cube = cube.spectral_smooth(
        convolution.Gaussian1DKernel(4 / 2.355))
    spax = cube.spectral_axis
    # smooth in area and velocity, create mask
    bigger_beam = radio_beam.Beam(
            major=2*beam.major, minor=2*beam.minor, pa=beam.pa)
    cube_s = cube.convolve_to(bigger_beam)
    # calculate moments
    filen_fmt = 'data/test_imaging/test_gbt_moments/G285_mosaic_gbt_NH3_11_{0}.fits'
    mom0 = cube.with_mask(cube_s > 1 * rms).moment(order=0)
    write_cube(mom0, PDir.IMG/Path(filen_fmt.format('mom0')))
    mom1 = cube.with_mask(cube_s > 2 * rms).moment(order=1)
    write_cube(mom1, PDir.IMG/Path(filen_fmt.format('mom1')))
    mom2 = cube.with_mask(cube_s > 2 * rms).moment(order=2)
    write_cube(mom2, PDir.IMG/Path(filen_fmt.format('mom2')))
    mom3 = cube.with_mask(cube_s > 2 * rms).moment(order=3)
    write_cube(mom3, PDir.IMG/Path(filen_fmt.format('mom3')))
    momS = cube.with_mask(cube_s > 2 * rms).linewidth_sigma()
    write_cube(momS, PDir.IMG/Path(filen_fmt.format('sigma')))
    momM = cube.with_mask(cube_s > 1 * rms).max(axis=0)
    write_cube(momM, PDir.IMG/Path(filen_fmt.format('max')))
    momV = cube.with_mask(cube_s > 2 * rms).argmax(axis=0).astype(float)
    momV[momV == 0] = np.nan
    chans = np.unique(momV)
    chans = chans[~np.isnan(chans)]
    for ix in chans:
        momV[momV == ix] = spax[int(ix)].value
    momV = spectral_cube.lower_dimensional_structures.Projection(momV, wcs=mom0.wcs)
    write_cube(momV, PDir.IMG/Path(filen_fmt.format('vmax')))


def mask_erode_edges(mol='hcop'):
    # FIXME
    filen = 'bgps_5021_{0}'.format(mol) + '_{0}.fits'
    cube = fits.open(PDir.DAT / Path(filen.format('cube_crop')))
    images = [
        fits.open(PDir.IMG/Path(filen.format(s)))
        for s in
        ('mom0', 'mom1', 'mom2', 'mom3', 'sigma', 'max', 'vmax')
    ]
    mask = (~np.isnan(cube[0].data[0])).astype(int)
    mask = ndimage.binary_erosion(mask, iterations=10).astype(bool)
    for hdu in images:
        filen = hdu.filename()
        print('-- Masking : {0}'.format(filen))
        hdu[0].data[~mask] = np.nan
        hdu.writeto(filen, overwrite=True)


def calc_delta_v_maps():
    # FIXME
    filen = 'bgps_5021_{0}.fits'
    sigm = fits.open(PDir.IMG / Path(filen.format('htcop_sigma')))
    m112 = fits.open(PDir.IMG / Path(filen.format('hcop_mom1')))
    m113 = fits.open(PDir.IMG / Path(filen.format('htcop_mom1')))
    mx12 = fits.open(PDir.IMG / Path(filen.format('hcop_vmax')))
    mx13 = fits.open(PDir.IMG / Path(filen.format('htcop_vmax')))
    for hdu in (sigm, m113, mx13):
        hdu[0].data = hdu[0].data[1:-1,1:-1]
    header = m112[0].header.copy()
    deltav_mom1 = (m112[0].data - m113[0].data) / (2.355 * sigm[0].data)
    deltav_mxm1 = (mx12[0].data - m113[0].data) / (2.355 * sigm[0].data)
    deltav_vmax = (mx12[0].data - mx13[0].data) / (2.355 * sigm[0].data)
    fits.PrimaryHDU(deltav_mom1, header=header).writeto(
            str(PDir.IMG/Path(filen.format('deltav_mom1'))), overwrite=True)
    fits.PrimaryHDU(deltav_mxm1, header=header).writeto(
            str(PDir.IMG/Path(filen.format('deltav_mxm1'))), overwrite=True)
    fits.PrimaryHDU(deltav_vmax, header=header).writeto(
            str(PDir.IMG/Path(filen.format('deltav_vmax'))), overwrite=True)


def extract_spectra_at_pos(cube, pos, ap_r=5*u.arcsec):
    # FIXME
    dpix = abs(cube.header['CDELT1']) * u.deg.to('arcsec')
    if isinstance(ap_r, u.Quantity):
        rpix = ap_r.to('arcsec').value / dpix
    else:
        rpix = ap_r / dpix
    lon = pos.fk5.ra.value
    lat = pos.fk5.dec.value
    pix0, pix1, _ = cube.wcs.all_world2pix(lon, lat, 0, 0)  # ra, dec, freq
    _, nx1, nx0 = cube.shape  # freq, dec, ra
    x1, x0 = np.indices([nx1, nx0], dtype='float')
    mask = (pix0.item() - x0)**2 + (pix1.item() - x1)**2 < rpix**2
    mcube = cube.with_mask(mask)
    return mcube.mean(axis=(1,2))


def grid_positions(pos, h_step_size=5*u.arcsec, w_step_size=5*u.arcsec,
        num_h_steps=16, num_w_steps=16):
    # FIXME
    dlon = coordinates.Angle(w_step_size)
    dlat = coordinates.Angle(h_step_size)
    for ih in range(-num_h_steps, num_h_steps+1):
        for iw in range(-num_w_steps, num_w_steps+1):
            lat_off = ih * dlat
            row_pos = coordinates.SkyCoord(pos.ra, pos.dec+lat_off)
            # correct for longitude offset as a function of latitude
            lon_off = iw * dlon / np.cos(row_pos.dec.to('radian'))
            new_pos = coordinates.SkyCoord(row_pos.ra+lon_off, row_pos.dec)
            yield ih, iw, new_pos


def extract_grid_spectra():
    # FIXME
    cb_h12 = read_cube(PDir.DAT/Path('bgps_5021_hcop_cube_base.fits'))
    cb_h13 = read_cube(PDir.DAT/Path('bgps_5021_htcop_cube_base.fits'))
    #cb_n11 = read_cube(PDir.DAT/Path('B29558_NH3_11_all.fits'))
    #cb_n22 = read_cube(PDir.DAT/Path('B29558_NH3_22_all.fits'))
    cubes = (
        ('hcop',   cb_h12),
        ('htcop',  cb_h13),
        #('nh3_11', cb_n11),
        #('nh3_22', cb_n22),
    )
    cen_pos = coordinates.SkyCoord('18:44:37.1', '-2:55:04.3', frame='fk5',
        unit=(u.hourangle, u.deg))
    # 2.75' x 2.75' region with 33 by 33 spectra
    grid = grid_positions(cen_pos,
        h_step_size=5*u.arcsec, w_step_size=5*u.arcsec,
        num_h_steps=16, num_w_steps=16)
    with h5py.File(PDir.IMG/Path('spectra.hdf5'), 'w') as f:
        f.attrs['source'] = 'BGPS 5021'
        f.attrs['ra'] = cen_pos.fk5.ra.value
        f.attrs['dec'] = cen_pos.fk5.dec.value
        # velocity axes
        v_grp = f.create_group('velocity_axes')
        for name, cube in cubes:
            v_grp.create_dataset(name, data=cube.spectral_axis.value)
        # extract spectra at each position, for each line, and store
        for ih, iw, pos in grid:
            print('-- {0:4d}, {1:4d}'.format(ih, iw))
            grp = f.create_group('/{0}/{1}'.format(ih, iw))
            grp.attrs['ra'] = pos.fk5.ra.value
            grp.attrs['dec'] = pos.fk5.dec.value
            for name, cube in cubes:
                spec = extract_spectra_at_pos(cube, pos, ap_r=5*u.arcsec)
                dset = grp.create_dataset(name, data=spec.value)


def specsmooth_to_vla():
    #for target in ('G28539',):
    #    for mol in ('NH3_11',):
    for target in TARGETS:
        for mol in ('NH3_11', 'NH3_22'):
            print(f'-- Smoothing cube for {target}_{mol}')
            gbt_filep = PDir.gbt_map_name(target, mol, modif='all_conv')
            gbt_c = read_cube(gbt_filep)
            # need the VLA cube to get the precise spectral axis to interpolate onto
            vla_filep = PDir.vla_map_name(target, mol.lower(), ext='image.fits')
            vla_c = read_cube(vla_filep)
            gbt_res = calc_velo_res(gbt_c)
            vla_res = calc_velo_res(vla_c)
            kernel_width = ((vla_res**2 - gbt_res**2)**(0.5) / gbt_res / FWHM).to('')
            kernel = convolution.Gaussian1DKernel(kernel_width)
            smooth_c = gbt_c.spectral_smooth(kernel)
            interp_c = smooth_c.spectral_interpolate(
                    vla_c.spectral_axis, suppress_smooth_warning=True)
            outp = PDir.gbt_map_name(target, mol, modif='vavg')
            write_cube(interp_c, outp)


