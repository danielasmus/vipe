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
from scipy import ndimage
from astropy.io import fits

from .calc_jitter import calc_jitter as _calc_jitter
from .read_raw import read_raw as _read_raw


def undo_jitter(ima=None, fin=None, head=None, fout=None,
                fillvalue=float('nan')):

    """
    compute and undo the jitter for a given VISIR cube
    """

    if ima is None:
        ima, h, _, _ = _read_raw(fin)

    if head is None:
        head = h

    jitter = _calc_jitter(head=head)

    # only do the jitter correction if the difference is larger than 0.1 pixel
    if (np.abs(jitter[0]) < 0.1) & (np.abs(jitter[1]) < 0.1):
        return(ima)

    # print(jitter)

    s = np.shape(ima)

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
        outima = ndimage.map_coordinates(ima, [xgrid, ygrid], cval=fillvalue)

    else:

        outima = np.zeros([nz, ny, nx], dtype='f')

        for i in range(nz):  # this step takes a lot of time

            outima[i, :, :] = ndimage.map_coordinates(ima[i, :, :],
                                                      [xgrid, ygrid],
                                                      cval=fillvalue)

    if fout is not None:
        fits.writeto(fout, outima, head, overwrite=True)
        print("    - Output written to: ",fout)

    return(outima)


