#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-21: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
from astropy.io import fits

from .fits_get_info import fits_get_info as _fits_get_info


def read_raw(fin):
    """
    Read a VISIR raw fits file and return the data as array, the header, the
    datatype, and the chopper position of the first frame
    """

    hdu = fits.open(fin)
    head = hdu[0].header

    datatype = _fits_get_info(head, "datatype")

    # --- burst mode data
    if datatype == "burst":

        ima = hdu[1].data

        # --- in burst mode, the first chopping position always seems to be B
        firstchoppos = "B"


    # --- Half-cycle and cycsum data
    else:

        n_ext = len(hdu) - 2  # exclude the last frame
        im = hdu[1].data
        head1 = hdu[1].header

        # --- copy the information of the first frame chop position into the
        #     main header for later use
        head["HIERARCH ESO DET FRAM TYPE"] = head1["HIERARCH ESO DET FRAM TYPE"]

        if head1["HIERARCH ESO DET FRAM TYPE"] == "HCYCLE2":
            firstchoppos = "B"
        else:
            firstchoppos = "A"

        n_dim = np.array(np.shape(im))

        ima = np.zeros([n_ext, n_dim[0], n_dim[1]])

        for i in range(n_ext):

            ima[i, :, :] = np.array(hdu[i+1].data)


    hdu.close()

    return(ima, head, datatype, firstchoppos)
