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
from astropy.time import Time


def select_rows(table, keys):

    """
    Select specific rows in a given table depending on given specifications for
    its different columns, provided through keys as a dictionary with  each
    entry being a list of three objects: [minval, selval, maxval]
    """

    dsel = np.copy(table)

    # --- select the files fulfilling the selected criteria

    for k in keys:

        # --- for a date range convert to mjd
        if k == 'DATE-OBS' or k == 'dateobs':
            # --- lower date limit
            if keys[k][0] is not None:
                keys['MJD-OBS'][0] = Time(keys[k][0],
                                          format='isot', scale='utc').mjd

                if keys['MJD-OBS'][0] % 1 == 0:
                    keys['MJD-OBS'][0] += 0.5

            # --- list of specific dates
            if keys[k][1] is not None:
                if isinstance(keys[k][1], list):
                    mjds = []
                    for i in range(len(keys[k][1])):
                        mjds.append(Time(keys[k][1][i],
                                         format='isot', scale='utc').mjd + 0.5)

                    mjds = np.array(mjds)
                    keys['MJD-OBS'][1] = mjds


                else:
                    keys['MJD-OBS'][1] = Time(keys[k][1], format='isot',
                                              scale='utc').mjd + 0.5

            # --- upper date limit
            if keys[k][2] is not None:
                keys['MJD-OBS'][2] = Time(keys[k][2], format='isot',
                                          scale='utc').mjd

                if keys['MJD-OBS'][2] % 1 == 0:
                    keys['MJD-OBS'][2] += 0.5

            k = 'MJD-OBS'

        if keys[k][0] is not None:
            ids = np.where(dsel[k] >= keys[k][0])[0]
            if len(ids) > 0:
                dsel = dsel[ids]
            else:
                print("ERROR: NO rows fulfill selection! Aborting...")
                break

        if keys[k][1] is not None:

            # --- do something special for MJDs (only one )
            if k == "MJD-OBS" or k == 'mjd':
                    ids = np.where((dsel[k] >= keys[k][1])
                                   & (dsel[k] < keys[k][1]+1))[0]

            else:
                if isinstance(keys[k][1], list):
                    ids = [i for x, i in enumerate(dsel[k]) if x in keys[k][1]]

                else:
                    ids = np.where(dsel[k] == keys[k][1])[0]

            if len(ids) > 0:
                dsel = dsel[ids]
            else:
                print("ERROR: NO rows fulfill selection! Aborting...")
                break

        if keys[k][2] is not None:
            ids = np.where(dsel[k] <= keys[k][2])[0]
            if len(ids) > 0:
                dsel = dsel[ids]
            else:
                print("ERROR: NO rows fulfill selection! Aborting...")
                break

    print("Number of selected rows: ",len(dsel))

    return(dsel)

