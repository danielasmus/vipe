#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.1"

"""
USED BY:
    - reduce_exposure
    - find_beam_pos

HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-11: changed variable name nodmode to noddir


NOTES:
    -

TO-DO:
    -
"""
import numpy as np

from .fits_get_info import fits_get_info as _fits_get_info
from .calc_chopoffset import calc_chopoffset as _calc_chopoffset
from .calc_nodoffset import calc_nodoffset as _calc_nodoffset

from . import visir_params as _vp
from . import isaac_params as _ip

def calc_beampos(head=None, chopang=None, chopthrow=None, noddir=None,
                    refpix=None, pfov=None, rotang=None,
                    winx=None, winy=None, verbose=False, pupiltrack=False,
                    imgoffsetangle=None, insmode=None, instrument=None):
    """
    compute the beam positions according to the chopping parameters of
    the observation
    """


    if head is not None:
        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])


    if instrument is None:
        instrument = _fits_get_info(head, "INSTRUME")

    if noddir is None:
        noddir = _fits_get_info(head, keys="CHOPNOD DIR")

    if insmode is None:
        insmode = _fits_get_info(head, "insmode")

    insmode = insmode.upper()

    if refpix is None:
        if instrument == "VISIR":
            if (insmode == "ACQ-SPC-SPC") | (insmode == "SPC"):
                refpix = _vp.refpix_spc
            else:
                refpix = _vp.refpix_img
        elif instrument == "ISAAC":
            refpix = _ip.refpix_img

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



    noffset = _calc_nodoffset(head=head, coffset=coffset, noddir=noddir,
                                pupiltrack=pupiltrack, pfov=pfov,
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

    if noddir == 'PARALLEL':

        beampos = np.zeros([3, 2])
        beampos[0, 0] = refpix[0] - starty
        beampos[0, 1] = refpix[1] - startx

        beampos[1, :] = beampos[0, :] + coffset
        beampos[2, :] = beampos[0, :] - coffset

    # still needs to be verified to work (probably wrong for rotangle != 0)!!
    if noddir == 'PERPENDICULAR':

        beampos = np.zeros([4, 2])

        beampos[0, 0] = refpix[0] - starty - 0.5 * (coffset[0] + noffset[0])
        beampos[0, 1] = refpix[1] - startx - 0.5 * (coffset[1] + noffset[1])

        beampos[1, :] = beampos[0, :] + coffset

        beampos[2, :] = beampos[0, :] + noffset

        beampos[3, :] = beampos[2, :] + coffset

    # print('startx, starty, nodmode, refpix, coffset, noffset',startx,starty,nodmode, refpix, coffset, noffset)

    return(beampos)

