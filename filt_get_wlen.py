#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-22: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""

from . import visir_params as _vp

def filt_get_wlen(filtname, instrument="VISIR", insmode="imaging"):
    """
    return the central wavelength in micron for a given filter/instrument
    """
    # print(instrument, insmode)
    if instrument == "VISIR":

        if "ima" in insmode or "img" in insmode or "spc" in filtname:
            # print("is ima", filtname, _vp.filtwlens[filtname])
            return(_vp.filtwlens[filtname])

        elif "spec" in insmode or "spc" in insmode :
            return(_vp.filtwlens[filtname+"_spc"])