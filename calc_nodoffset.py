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

from .calc_chopoffset import calc_chopoffset as _calc_chopoffset
from .fits_get_info import fits_get_info as _fits_get_info

def calc_nodoffset(head=None, chopang=None, chopthrow=None, pfov=None,
                      rotang=None, nodmode=None, coffset=None,
                      pupiltrack=False, imgoffsetangle=92.5, insmode=None,
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

    if nodmode is None:
        nodmode = head["HIERARCH ESO SEQ CHOPNOD DIR"]

    noffset = np.zeros(2)

    if nodmode == 'PARALLEL':
        noffset = - coffset

    # still needs to be verified to work (probably wrong for rotangle != 0)!!
    if nodmode == 'PERPENDICULAR':
        noffset = _calc_chopoffset(head=head, chopang=chopang,
                                     chopthrow=chopthrow, pfov=pfov,
                                     rotang=rotang+90.0, pupiltrack=pupiltrack,
                                     imgoffsetangle=imgoffsetangle,
                                     insmode=insmode)

    return(noffset)

