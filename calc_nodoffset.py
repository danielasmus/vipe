#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.1"

"""
USED BY:
    - calc_beam_pos
    - calc_jitter

HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-11: change variable name nodmode to nodpos

NOTES:
    -

TO-DO:
    -
"""
import numpy as np

from .calc_chopoffset import calc_chopoffset as _calc_chopoffset
from .fits_get_info import fits_get_info as _fits_get_info

def calc_nodoffset(head=None, chopang=None, chopthrow=None, pfov=None,
                      rotang=None, noddir=None, coffset=None,
                      pupiltrack=False, imgoffsetangle=None, insmode=None,
                      instrument=None):

    """
    compute the nodding offset for nod position B with respect to nod
    position A
    """

    if instrument is None:
        instrument = _fits_get_info(head, "INSTRUME")

    if insmode is None:
        insmode = _fits_get_info(head, "insmode")

    if head is not None:
        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])

    if rotang is None:
        rotang = float(head["HIERARCH ESO ADA POSANG"])

    if coffset is None:
        coffset = _calc_chopoffset(head=head, chopang=chopang,
                                     chopthrow=chopthrow, pfov=pfov,
                                     rotang=rotang, pupiltrack=pupiltrack,
                                     imgoffsetangle=imgoffsetangle,
                                     insmode=insmode, instrument=instrument)

    if noddir is None:
        noddir = _fits_get_info(head, "CHOPNOD DIR")

    noffset = np.zeros(2)

    if noddir == 'PARALLEL':
        noffset = - coffset

    # still needs to be verified to work (probably wrong for rotangle != 0)!!
    if noddir == 'PERPENDICULAR':
        noffset = _calc_chopoffset(head=head, chopang=chopang,
                                     chopthrow=chopthrow, pfov=pfov,
                                     rotang=rotang+90.0, pupiltrack=pupiltrack,
                                     imgoffsetangle=imgoffsetangle,
                                     insmode=insmode)

    return(noffset)

