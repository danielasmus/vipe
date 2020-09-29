#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.0.0"

"""
USED BY:
    - undo_jitter
    - reduce_burst_file


HISTORY:
    - 2020-01-21: created by Daniel Asmus
    - 2020-02-11: variable nodmode renamed to noddir
    - 2020-06-11: support for ISAAC added & some cleaning up


NOTES:
    -

TO-DO:
    -
"""

from .calc_nodoffset import calc_nodoffset as _calc_nodoffset
from .print_log_info import print_log_info as _print_log_info
from .fits_get_info import fits_get_info as _fits_get_info


def calc_jitter(head=None, chopang=None, chopthrow=None, pfov=None,
                   rotang=None, noddir=None, noffset=None, nodpos=None,
                   cumoffsetx=None, cumoffsety=None, verbose=False,
                   silent=False, pupiltrack=False, imgoffsetangle=92.5,
                   logfile=None, instrument=None):

    """
    compute the jitter offset (in px)
    TO DO: implement correct solution for pupiltrack. For this the offset
    values probably also have to be rotatedby the imager offset angle
    (currently returns zero offset)
    """

    funname = "CALC_JITTER"


    # --- determine instrument:
    if instrument == None:
        try:
            instrument = head["INSTRUME"].replace(" ", "")
        except:
            msg = (funname + ": ERROR: Could not determine instrument!")

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)


    # --- are we in pupil-tracking mode (ALTAZTRACK=True)?
    if head is not None:
        if "HIERARCH ESO TEL ROT ALTAZTRACK" in head:
            pupiltrack = (head["HIERARCH ESO TEL ROT ALTAZTRACK"])


    # --- temporary dirty solution for now (hopefully nobody does pupiltack
    #     with jittering!)
    if pupiltrack:
        return([0,0])

    if cumoffsetx is None:
        if 'HIERARCH ESO SEQ CUMOFFSETX' in head:
            cumoffsetx = head["HIERARCH ESO SEQ CUMOFFSETX"]
        else:
            cumoffsetx = 0
            if not silent:
                print(funname + ": WARNING: CUMOFFSETX not in header! Assume 0...")

    if cumoffsety is None:
        if 'HIERARCH ESO SEQ CUMOFFSETY' in head:
            cumoffsety = head["HIERARCH ESO SEQ CUMOFFSETY"]
        else:
            cumoffsety = 0
            if not silent:
                print(funname + ": WARNING: CUMOFFSETY not in header! Assume 0...")

    if nodpos is None:
        nodpos = _fits_get_info(head, keys="NODPOS", logfile=logfile,
                                instrument=instrument)
        # if "HIERARCH ESO SEQ NODPOS" in head:
        #     nodpos = head["HIERARCH ESO SEQ NODPOS"]
        # else:
        #     nodpos = 'A'
        #     if not silent:
        #         print(funname + ": WARNING: NODPOS not in header! Assume A...")

    if verbose:
        print("COMPUTE_JITTER: cumoffsety/x: ", cumoffsety, cumoffsetx)
        print("COMPUTE_JITTER: nodpos: ", nodpos)
        print("COMPUTE_JITTER: pupiltrack: ", pupiltrack)

    if nodpos == 'A':
        return([cumoffsety, cumoffsetx])

    if nodpos == 'B':

        if noddir is None:
            noddir = _fits_get_info(head, keys="NODDIR", logfile=logfile,
                                instrument=instrument)

        else:
            if instrument == "VISIR":
                noddir = 'PERPENDICULAR'
                if not silent:
                    print(funname + ": WARNING: CHOPNOD DIR not in header! Assume PERPENDICULAR...")
            elif instrument == "ISAAC":
                noddir = 'PARALLEL'
                if not silent:
                    print(funname + ": WARNING: CHOPNOD DIR can not be determined! Assume PARALLEL...")


        if noffset is None:
            noffset = _calc_nodoffset(head=head, chopang=chopang,
                                        chopthrow=chopthrow, pfov=pfov,
                                        rotang=rotang, noddir=noddir,
                                        pupiltrack=pupiltrack,
                                        imgoffsetangle=imgoffsetangle)

        jx = cumoffsetx - noffset[1]
        jy = cumoffsety - noffset[0]

        if verbose:
            print("COMPUTE_JITTER: nod offset: ", noffset)
            print("COMPUTE_JITTER: jy, jx: ", jy, jx)

        return([jy, jx])

