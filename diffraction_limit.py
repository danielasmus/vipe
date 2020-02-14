#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.1"

"""
USED BY:
    - find_beam_pos

HISTORY:
    - 2020-01-21: created by Daniel Asmus
    - 2020-02-22: change from "== str" to isinstance which is more robust


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
from astropy.io import fits

from .filt_get_wlen import filt_get_wlen as _filt_get_wlen
from .fits_get_info import fits_get_info as _fits_get_info
from .print_log_info import print_log_info as _print_log_info

def diffraction_limit(any_input, unit=None, pfov=None, instrument=None,
                      insmode=None, teldiam=8.2, logfile=None):
    """
    compute the diffraction limit for a given input which could either be a
    wavelength in [micron] or a filtername (str) or a fits header and return
    the diffraction limit for this wavelength and a given telescope diameter
    (default = 8.2m) in arcsec (default) or pixel if unit is set to 'px' and
    either a pfov is directly provided or extractable from the fits header
    """

    funname = "DIFFRACTION_LIMIT"


    # --- wavelength directly provided?
    if isinstance(any_input, float) or isinstance(any_input, int):
        wlen = any_input

    # --- filtername provided?
    elif isinstance(any_input, str):
        wlen = _filt_get_wlen(any_input, instrument=instrument,
                              insmode=insmode)

    # --- fits header provide?
    elif type(any_input) == fits.header.Header:

        head = any_input
        wlen = _fits_get_info(head, 'wlen')

        if instrument == None:
            instrument = _fits_get_info(head, "INSTRUME")

        if insmode is None:
            insmode = _fits_get_info(head, "insmode")

        if unit == "px" or "pix" in unit:
            if pfov is None:
                pfov = _fits_get_info(head, 'pfov')

    else:
        wlen = None

    # print("funname, any_input, type(any_input), wlen: ",funname, any_input, type(any_input), wlen)

    if wlen is None:
        msg = (funname + ": ERROR: no valid wavelength provided: any_input|type|instrument|insmode|pfov= "
               + any_input + "|" + str(type(any_input)) + "|" + str(instrument)
               + "|" + str(insmode)
               + "|" + str(pfov)
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    diflim = np.arcsin(1.028*wlen*0.000001/teldiam)*206264.806247  # arcsec

    if unit == "px" or "pix" in unit:
        diflim = diflim / pfov

    return(diflim)
