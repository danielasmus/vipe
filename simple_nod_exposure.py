#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-23: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits




from .read_raw import read_raw as _read_raw
from .fits_get_info import fits_get_info as _fits_get_info
from .simple_image_plot import simple_image_plot as _simple_image_plot

def simple_nod_exposure(ima=None, fin=None, ndit=None, superdit=None,
                        datatype=None, plot=False, writeplot=False,
                        firstchoppos=None,
                        fout=None, outim=None, head=None, verbose=False):
    """
    Create a simple, blind merge of the chop position frames of a VISIR raw
    file
    """
    if ima is None:
        ima, head, datatype, firstchoppos = _read_raw(fin)

    if datatype is None:
        datatype = _fits_get_info(head, "datatype")

    if firstchoppos is None:
        if datatype == "burst":
            firstchoppos = "B"
        else:
            if head["HIERARCH ESO DET FRAM TYPE"] == "HCYCLE2":
                firstchoppos = "B"

            else:
                firstchoppos = "A"

    # --- for burst mode we need to compute the nod exposure by combining the
    #    individual DITs correctly
    if datatype == "burst":
        if ndit is None:
            ndit = head["HIERARCH ESO DET NDIT"]

        # --- dit averaging or superdit
        if superdit is None:
            superdit = head["HIERARCH ESO DET NAVRG"]

    else:
        ndit = 1
        superdit = 1


    s = np.shape(ima)

    ndit = int(ndit/superdit)

    # --- find all frames belonging to chop A and chop B
    chopa = np.array(range(s[0])) % (2 * ndit) < ndit
    chopb = np.array(range(s[0])) % (2 * ndit) >= ndit

    mimchopa = np.mean(ima[chopa, :, :], axis=0)
    mimchopb = np.mean(ima[chopb, :, :], axis=0)

    if firstchoppos == 'B':
        sign = -1
    else:
        sign = 1

    outim = sign * (mimchopa - mimchopb)

    # --- optional output writting
    if plot:
        plt.imshow(outim, origin='bottom', interpolation='nearest')
        plt.show()

    if fout is not None:

        fits.writeto(fout, outim, head, overwrite=True)
        if verbose:
            print("SIMPLE_NOD_EXPOSURE: Output written to: ",fout)

    if writeplot:
        if fout is not None:
            fout = fout.replace(".fits", ".png")

        else:
            fout = fin.replace(".fits", ".png")
            _simple_image_plot(outim, fout, log=True)


    return(outim)

