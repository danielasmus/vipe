#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-23: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

from .crop_image import crop_image as _crop_image
from .find_source import find_source as _find_source


def merge_fits_files(infolder=None, infiles=None, suffix='', fout=None,
                     method='mean', prefix='', sigma=3, target=None,
                     filt=None, rotangle=None, northup=False, refbox=20,
                     fitmeth='fastgauss', refsign=1, imgoffsetangle=None,
                     verbose=False, tolrot=0.01, pfov=None,
                     replace_nan_with='median', instrument=None):

    """
    Take either a folder full of single (chop/nod) frames or a given list of
    files and simply average them into one merged file
    """

    if infolder is not None and infiles is None:
       infiles = [ff for ff in os.listdir(infolder)
         if (ff.endswith(suffix+'.fits')) & (ff.startswith(prefix))]

    if infolder is None:
        infolder = ''

    nfiles = len(infiles)
    ima = []
    shapes = []
    rots = []

    ffull = infolder + "/" + infiles[0]
    hdu = fits.open(ffull)
    head0 = hdu[0].header
    hdu.close()

    if instrument is None:
        instrument = head0['INSTRUME']
        print('Found instrument: ',instrument)


    if instrument == 'VISIR':
        if imgoffsetangle is None:
            imgoffsetangle = 92.5
        rotsense = 1  # for VISIR the ADA.POSANG is 360-PA

    elif (instrument == 'NAOS+CONICA') | (instrument == 'NACO'):
        imgoffsetangle = 0  # apparently correct?
        rotsense = -1  # for NACO the ADA.POSANG really gives the PA

    print('Found n potential files: ',nfiles)

    if verbose:
        print("MERGE_FITS_FILES: fitmeth: ", fitmeth)
        print("MERGE_FITS_FILES: refbox: ", refbox)
        print("MERGE_FITS_FILES: refsign: ", refsign)


    for i in range(nfiles):

        ffull = infolder + "/" + infiles[i]
        hdu = fits.open(ffull)

        if verbose:
            print("MERGE_FITS_FILES: i, infiles[i]: ", i, infiles[i])

        head = hdu[0].header
        rot = head['HIERARCH ESO ADA POSANG']

        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])
        else:
            pupiltrack = False

        # --- if pupil tracking is on then the parang in the VISIR fits-header is
        #     not normalised by the offset angle of the VISIR imager with respect
        #     to the adapter/rotator
        if pupiltrack:
            rot = rot - imgoffsetangle

        if verbose:
            print("MERGE_FITS_FILES: rot: ", rot)


        if target is not None:
            targ = head['HIERARCH ESO OBS TARG NAME']
            if str(targ) != target:
                continue
        if filt is not None:
            filtf = head['HIERARCH ESO INS FILT1 NAME']
            if str(filtf) != filt:
                continue
        if rotangle is not None:
            if rot != rotangle:
                continue

        im = hdu[0].data
        hdu.close()

        # --- rotate images if Northup is requested to be up
        if np.abs(rot) > tolrot and northup:

            # --- ndimage is not compaticble with nans, so we need to replace them
            print("MERGE_FITS_FILES: Encountered NaNs not compatible with "
                   + "rotation. Replacing with ", replace_nan_with)

            if ((type(replace_nan_with) == float) |
               (type(replace_nan_with) == int)):
                im[np.isnan(im)] = replace_nan_with

            elif replace_nan_with == 'median':
                im[np.isnan(im)] = np.nanmedian(im)

            # --- WARNING: does not work with large NaN areas at the borders
            elif replace_nan_with == 'interpol':
                kernel = Gaussian2DKernel(stddev=1)
                im = interpolate_replace_nans(im, kernel)

#            print(np.isnan(im).any())
#
#            plt.imshow(im, origin='bottom', interpolation='nearest',
#                   norm=LogNorm())
#            plt.title(str(i)+' - after replacing NaNs, before rotation')
#            plt.show()

            im = ndimage.interpolation.rotate(im, rotsense*rot, order=3)

#            plt.imshow(im, origin='bottom', interpolation='nearest',
#                       norm=LogNorm())
#            plt.show()

        ima.append(im)
        shapes.append(np.shape(im))
        rots.append(rot)

    shapes = np.array(shapes)
    n = len(ima)

    if verbose:
        print("MERGE_FITS_FILES: n: ", n)

    # --- after rotating the images probably have different sizes
    if northup:
        minsize = [np.min(shapes[:,0]),np.min(shapes[:,1])]
        for i in range(n):

            if verbose:
                print("MERGE_FITS_FILES: i: ", i)
                print("MERGE_FITS_FILES: shapes: ", shapes[i])
                print("MERGE_FITS_FILES: argmax: ", np.unravel_index(np.nanargmax(ima[i]),
                                                 shapes[i]))

                plt.imshow(ima[i], origin='bottom', interpolation='nearest',
                       norm=LogNorm())
                plt.title(str(i)+' - rot '+str(rots[i])+' - before centered crop')
                plt.show()

            fit, _, _ = _find_source(ima[i], method=fitmeth,
                                          searchbox=refbox, fitbox=refbox,
                                          sign=refsign, verbose=verbose,
                                          plot=verbose)

            ima[i] = _crop_image(ima[i], box=minsize, cenpos=fit[2:4])

            if verbose:
                plt.imshow(ima[i], origin='bottom', interpolation='nearest',
                           norm=LogNorm())

                plt.title(str(i)+' - rot '+str(rots[i]))
                plt.show()
                print(i,fit)

        # --- if WCS is present in the header, it needs to be updated to
        if "CTYPE1" in head0:
            if head0["CTYPE1"] == "RA---TAN":
                if pfov is None:
                    if "HIERARCH ESO INS PFOV" in head0:
                        pfov = float(head["HIERARCH ESO INS PFOV"])  # VISIR
                    else:
                        pfov =  float(head["HIERARCH ESO INS PIXSCALE"])  # NACO

                ra = head0["HIERARCH ESO TEL TARG ALPHA"]
                sec = ra % 100
                min = (ra % 10000 - sec)/100
                hour = (int(ra)/10000)
                ra_deg = 15*(hour + min/60.0 + sec/3600.0)
                # print(ra_deg)

                dec = head0["HIERARCH ESO TEL TARG DELTA"]
                sec = dec % 100
                min = (dec % 10000 - sec)/100
                deg = (int(dec)/10000)
                if deg < 0:
                    dec_deg = deg - min/60.0 - sec/3600.0
                else:
                    dec_deg = deg + min/60.0 + sec/3600.0
                # print(ra_deg)

                head0["CRPIX1"] = minsize[1] * 0.5
                head0["CRPIX2"] = minsize[0] * 0.5
                head0["CRVAL1"] = ra_deg
                head0["CRVAL2"] = dec_deg
                if "CD1_1" in head0:
                    del head0["CD1_1"]
                    del head0["CD1_2"]
                    del head0["CD2_1"]
                    del head0["CD2_2"]
                head0["CDELT1"] = -pfov/3600.0
                head0["CDELT2"] = pfov/3600.0




    ima = np.array(ima)
    #print(np.shape(ima))

    if method == 'mean':
        outim = np.nanmean(ima, axis=0)

    if method == 'median':
        outim = np.nanmedian(ima, axis=0)

    if method == 'sigmaclip':
        outim = np.array(np.mean(sigma_clip(ima, sigma=sigma, axis=0,
                                            maxiters=2),
                         axis=0))

    print(" - " + str(n) + " images combined.")

    if fout is not None:
        fits.writeto(fout, outim, head0, overwrite=True)

    return(outim)
