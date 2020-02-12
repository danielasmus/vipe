#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.1.0"

"""
USED BY:
    - diffraction_limit

HISTORY:
    - 2020-01-22: created by Daniel Asmus
    - 2020-02-10: implement upper case for all parameters


NOTES:
    -

TO-DO:
    -
"""

from . import visir_params as _vp

def filt_get_wlen(filtname, instrument="VISIR", insmode="IMG"):
    """
    return the central wavelength in micron for a given filter/instrument
    """

    filtname = filtname.upper()
    instrument = instrument.upper()
    insmode = insmode.upper()

    # print(instrument, insmode)
    if instrument == "VISIR":

        if "IMA" in insmode or "IMG" in insmode or "SPC" in filtname:
            # print("is ima", filtname, _vp.filtwlens[filtname])
            return(_vp.filtwlens[filtname])

        elif "SPEC" in insmode or "SPC" in insmode :
            return(_vp.filtwlens[filtname+"_SPC"])