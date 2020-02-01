#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus
    - 2020-01-23: rename from get_background to measure_bkg


NOTES:
    -

TO-DO:
    -
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'



def measure_bkg(im, ignore_aper=None, method='median', binsize=0.1,
                   show_plot=False, stdonly=False):

    """
    estimate the background and its variation for a given image with optionally
    ignoring a certain aperture in the center of the image for the estimate
    """

    s = np.shape(im)
    mask = np.ones(s)

    if ignore_aper:

        xcen = 0.5 * s[1]
        ycen = 0.5 * s[0]

        for x in range(s[1]):
            for y in range(s[0]):
                rdist = np.sqrt((x - xcen)**2 + (y - ycen)**2)
                if rdist < 0.5*ignore_aper:
                    mask[y,x] = 0

    bg = im[mask != 0].flatten()
    # print(len(bg))
    bgstd = np.nanstd(bg)

    if stdonly:
        return(bgstd)

    if method == 'median':
        bgval = np.nanmedian(bg)

    elif method == 'distmax':
        # value where the pixel distribution is peaking
        immin = np.nanmin(bg)
        immax = np.nanmax(bg)
#        bins = np.arange(immin, immax, binsize)
#        nbins = len(bins)
#        hist = np.zeros(nbins)
#
#        for i in range(nbins-1):
#            id = np.where((bg >= bins[i]) & (bg < bins[i+1]))[0]
#            hist[i] = len(id)
        nbins = int(np.round((immax - immin)/binsize))
        print(immax,immin, nbins, len(bg), np.shape(bg))
        hist, bins = np.histogram(bg, nbins, range=[immin, immax])
        bgval = bins[np.argmax(hist)]


        if show_plot:
            # n, bins, patches = plt.hist(bg, nbins)
            plt.bar(bins[0:-1], hist, fill=True, color='grey', linewidth=0)
            #plt.yscale('log')
            plt.xlim(immin,immax)
            #print(np.argmax(n), np.max(n), bins[np.argmax(n)])
            plt.show()


    return(bgval, bgstd)

