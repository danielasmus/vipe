#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.1.1"

"""
USED BY:
    - calc_beam_pos


HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-11: change variable name nodmode to nodpos
    - 2020-06-11: use fits_get_info for the PFOV for ISAAC compability
    - 2020-06-18: use visir and isaac param files
    - 2020-09-29: make sure that pfov is a float


NOTES:
    -

TO-DO:
    -
"""
import numpy as np

from .fits_get_info import fits_get_info as _fits_get_info
from . import visir_params as _vp
#from . import isaac_params as _ip


def calc_chopoffset(head=None, chopang=None, chopthrow=None, pfov=None,
                       rotang=None, pupiltrack=False, imgoffsetangle=None,
                       insmode=None, verbose=False, instrument=None):

    """
    compute the chopping offset for chop position B with respect to chop
    position A
    """

    if instrument is None:
        instrument = _fits_get_info(head, "INSTRUME")

    if chopang is None:
        chopang = float(head["HIERARCH ESO TEL CHOP POSANG"])

    if chopthrow is None:
        chopthrow = float(head["HIERARCH ESO TEL CHOP THROW"])

    if pfov is None:
        pfov = float(_fits_get_info(head, "PFOV"))

    if rotang is None:
        rotang = float(head["HIERARCH ESO ADA POSANG"])

    if insmode is None:
        insmode = _fits_get_info(head, "insmode")

    if head is not None:
        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])


    if verbose:
        print(" - COMPUTE_CHOPOFFSET: insmode: ", insmode)
        print(" - COMPUTE_CHOPOFFSET: pupiltrack: ", pupiltrack)
        print(" - COMPUTE_CHOPOFFSET: chop angle in header: ", chopang)
        print(" - COMPUTE_CHOPOFFSET: position angle in header: ", rotang)

    if instrument == "VISIR":
        # --- if pupil tracking is on then the parang in the VISIR fits-header is
        #     not normalised by the offset angle of the VISIR imager with respect
        #     to the adapter/rotator
        if pupiltrack:
            rotang = rotang - _vp.imgoffsetangle


        # --- is we do acquisition for the spectro on the imager then the imager is
        #     ~90 deg rotated, i.e., the rotang has to be modified
        if insmode == "acq-img-spc":
            rotang = rotang + 90.13   # angle between imager and spectro

    throw_pix = chopthrow / pfov

    # what matters is the angular difference between rotang and chopang
    ang_diff = (chopang - rotang) / 360.0 * 2.0 * np.pi

    coffset = np.zeros(2)
    coffset[1] = - np.sin(ang_diff) * throw_pix  # x
    coffset[0] = - np.cos(ang_diff) * throw_pix  # y

    if verbose:
        print(" - COMPUTE_CHOPOFFSET: real position angle: ", rotang)
        print(" - COMPUTE_CHOPOFFSET: throw length [px]: ", throw_pix)


    return(coffset)

