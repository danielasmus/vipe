#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


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



def subtract_source(im, siglim=3, niter=3, plot=False, verbose=False,
                    method="sigclip", fill_value=float('nan')):

    """
    Do a simple subtraction of the source from an image without any knowledge.
    This routine is intended for background estimates.
    Default is sigma-clipping.
    """

    rim = np.copy(im)

    std = np.nanstd(rim)

    if plot:
        plt.figure(1, figsize=(8,8))
        plt.imshow(rim, origin='bottom',
                   interpolation='nearest')

        plt.show()

    if plot or verbose:
        print("SUBTRACT_SOURCE: Original STDDEV, MIN, MAX: ",
              std, np.nanmax(rim)-np.nanmin(rim))


    for i in range(niter):

        med = np.nanmedian(rim)

        if method == "sigclip":
            ids = np.nonzero((rim > siglim * std + med) |
                             (rim < med - siglim * std))
            rim[ids] = fill_value


        std = np.nanstd(rim)

        if plot:
            plt.figure(1, figsize=(8,8))
            plt.imshow(rim, origin='bottom',
                   interpolation='nearest')

            plt.show()

        if plot or verbose:
            print("SUBTRACT_SOURCE: Iteration, STDDEV, MIN, MAX: ",
                  i, std, np.nanmax(rim)-np.nanmin(rim))


    return(rim)
