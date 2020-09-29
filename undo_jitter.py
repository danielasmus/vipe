#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-21: created by Daniel Asmus
    - 2020-06-11: support for ISAAC added & logfile & instrument added


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
from scipy import ndimage
from astropy.io import fits

from .calc_jitter import calc_jitter as _calc_jitter
from .read_raw import read_raw as _read_raw
from .print_log_info import print_log_info as _print_log_info


def undo_jitter(im=None, fin=None, head=None, fout=None,
                fillvalue=float('nan'), instrument=None, logfile=None):

    """
    compute and undo the jitter for a given VISIR or ISAAC data frame for
    an individual nod (single frame or cube)
    """

    funname = "UNDO_JITTER"

    if im is None:
        im, h, _, _ = _read_raw(fin)

    if head is None:
        head = h

    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")
        else:
            msg = (funname + ": ERROR: Could not determine instrument!")

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)

    jitter = _calc_jitter(head=head, instrument=instrument, logfile=logfile)

    # only do the jitter correction if the difference is larger than 0.1 pixel
    if (np.abs(jitter[0]) < 0.1) & (np.abs(jitter[1]) < 0.1):
        return(im)

    # print(jitter)

    # --- determine whether single frame or cube
    s = np.shape(im)

    if len(s) == 3:

        nx = s[2]
        ny = s[1]
        nz = s[0]

    if len(s) == 2:

        nx = s[1]
        ny = s[0]
        nz = 1

    x_i = np.array(range(nx)) + jitter[1]
    y_i = np.array(range(ny)) + jitter[0]
    ygrid, xgrid = np.meshgrid(x_i, y_i)

    if nz == 1:
        outim = ndimage.map_coordinates(im, [xgrid, ygrid], cval=fillvalue)

    else:

        outim = np.zeros([nz, ny, nx], dtype='f')

        for i in range(nz):  # this step takes a lot of time

            outim[i, :, :] = ndimage.map_coordinates(im[i, :, :],
                                                      [xgrid, ygrid],
                                                      cval=fillvalue)

    if fout is not None:
        fits.writeto(fout, outim, head, overwrite=True)
        print(funname + ": Output written to: ",fout)

    return(outim)


