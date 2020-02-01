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

from .fits_get_info import fits_get_info as _fits_get_info
from .calc_chopoffset import calc_chopoffset as _calc_chopoffset
from .calc_nodoffset import calc_nodoffset as _calc_nodoffset

def calc_beampos(head=None, chopang=None, chopthrow=None, nodmode=None,
                    refpix=None, pfov=None, rotang=None,
                    winx=None, winy=None, verbose=False, pupiltrack=False,
                    imgoffsetangle=92.5, insmode=None, instrument=None):
    """
    compute the beam positions according to the chopping parameters of
    the observation
    """


    if head is not None:
        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])


    if instrument is None:
        instrument = _fits_get_info(head, "INSTRUME")

    if insmode is None:
        insmode = _fits_get_info(head, "insmode")

    insmode = insmode.upper()

    if refpix is None:
        if (insmode == "ACQ-SPC-SPC") | (insmode == "SPC"):
            refpix = [512, 529]
        else:
            refpix = [530, 430]

    coffset = _calc_chopoffset(head=head, chopang=chopang,
                                 chopthrow=chopthrow, pfov=pfov,
                                 rotang=rotang, pupiltrack=pupiltrack,
                                 imgoffsetangle=imgoffsetangle,
                                 insmode=insmode, verbose=verbose,
                                 instrument=instrument)

    if verbose:
        #print(" - COMPUTE_BEAMPOS: insmode: ", insmode)
        #print(" - COMPUTE_BEAMPOS: pupiltrack: ", pupiltrack)
        print(" - COMPUTE_BEAMPOS: refpix: ", refpix)
        print(" - COMPUTE_BEAMPOS: coffset: ", coffset)

    if nodmode is None:
        nodmode = head["HIERARCH ESO SEQ CHOPNOD DIR"]

    noffset = _calc_nodoffset(head=head, coffset=coffset, nodmode=nodmode,
                                pupiltrack=pupiltrack,
                                imgoffsetangle=imgoffsetangle)

    if verbose:
        print(" - COMPUTE_BEAMPOS: noffset: ", noffset)

    if winx is None:
        if head is not None:

            # --- sometimes only one of the following keyword sets is present
            #     and sometimes both are but the values are not consistent
            #     which is why the larger value should be used.
            if 'HIERARCH ESO DET SEQ1 WIN STRY' in head:
                sx = head['HIERARCH ESO DET SEQ1 WIN STRX'] - 1
                sy = head['HIERARCH ESO DET SEQ1 WIN STRY'] - 1
            else:
                sx = 0
                sy = 0

            if 'HIERARCH ESO DET WIN STRY' in head:
                x = head['HIERARCH ESO DET WIN STRX'] - 1
                y = head['HIERARCH ESO DET WIN STRY'] - 1
            else:
                x = 0
                y = 0

            startx = np.max([sx, x])
            starty = np.max([sy, y])

        else:
            startx = 0
            starty = 0
    else:
        startx = int(512 - 0.5*winx)
        starty = int(512 - 0.5*winy)

    if verbose:
        print(" - COMPUTE_BEAMPOS: startx, starty, winx, winy: ", startx, starty, winx, winy)

    if nodmode == 'PARALLEL':

        beampos = np.zeros([3, 2])
        beampos[0, 0] = refpix[0] - starty
        beampos[0, 1] = refpix[1] - startx

        beampos[1, :] = beampos[0, :] + coffset
        beampos[2, :] = beampos[0, :] - coffset

    # still needs to be verified to work (probably wrong for rotangle != 0)!!
    if nodmode == 'PERPENDICULAR':

        beampos = np.zeros([4, 2])

        beampos[0, 0] = refpix[0] - starty - 0.5 * (coffset[0] + noffset[0])
        beampos[0, 1] = refpix[1] - startx - 0.5 * (coffset[1] + noffset[1])

        beampos[1, :] = beampos[0, :] + coffset

        beampos[2, :] = beampos[0, :] + noffset

        beampos[3, :] = beampos[2, :] + coffset

    # print('startx, starty, nodmode, refpix, coffset, noffset',startx,starty,nodmode, refpix, coffset, noffset)

    return(beampos)
