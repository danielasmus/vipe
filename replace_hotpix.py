#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
USED BY:
    - find_beam_pos
    - reduce_exposure

HISTORY:
    - 2020-07-11: created by Daniel Asmus



NOTES:
    -

TO-DO:
    -
"""
import numpy as np


def replace_hotpix(im, sigmathres=3, niters=1, maxneighbor=1, bgstd=None,
                   verbose=False):

    """
    replace hot pixels in an image with the median of the surrounding values
    if less than maxneighbor (default=1) px in the direct 3x3 neighborhood
    are also above the threshold of sigmathres (default=3) x bgstd
    (default: stddev of whole image). Niter (default=1) iterations are done.
    Returned are the number of replaced pixels and the resulting image
    """

    funname = "REPLACE_HOTPIX"

    cim = np.copy(im)

    nrep = 0

    if bgstd is None:
        bgstd = np.nanstd(im)

    totmed = np.nanmedian(im)

    if verbose:
        print(funname + ": bgstd: " + str(bgstd))
        print(funname + ": totmed: " + str(totmed))

    for i in range(niters):

        # ---- find all hot pix candidated
        ids = np.where(np.abs(cim) > np.abs(totmed+sigmathres*bgstd))

        ncand = len(ids[0])
        if verbose:
            print(funname + ": ncand: " + str(ncand))

        for j in range(ncand):

            y = ids[0][j]
            x = ids[1][j]

            mim = cim[np.max([y-1,0]):y+2, np.max([0,x-1]):x+2]
            med = np.nanmedian(mim)

            hot = np.abs(mim) > np.abs(med+sigmathres*bgstd)
            if np.sum(hot) <= maxneighbor+1:
                if hot[1,1]:
                    cim[y,x] = med
                    nrep += 1

    if verbose:
        print(funname + ": nrep: " + str(nrep))

    return(nrep, cim)

