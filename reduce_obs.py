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
import os
import sys
import traceback
import shutil
import inspect
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Column

from .duplicates_in_column import duplicates_in_column as _duplicates_in_column
from .get_std_flux import get_std_flux as _get_std_flux
from .print_log_info import print_log_info as _print_log_info
from .reduce_exposure import reduce_exposure as _reduce_exposure



def reduce_obs(ftabraw, ftabpro, infolder, outfolder, logfile=None,
                       maxshift=10, debug=False, plot=False,
                       selobj=None, instrument=None,
                       selexp=None, selsetup=None, selobid=None, selmode=None,
                       seldtype=None, extract=True, verbose=False,
                       overwrite=False, statcalfolder=None,
                       obstype=None, mindate=None, seldate=None, maxdate=None,
                       box=None, chopsubmeth='averchop',
                       alignmethod='fastgauss', searcharea='chopthrow',
                       searchsmooth=3, crossrefim=None,
                       AA_pos=None, refpos=None, findbeams=True,
                       ditsoftaver=1, sky_xrange=None,
                       sky_yrange=None):
    """
    wrapper reduction loop routine: go over the selected observations and
    reduce them by combining the different nod files
    PARAMETERS:

     - outfolder: (default='.') folder to write output into
     - overwrite: (default=False) overwrite existing data?
     - selobj = (optional) reduce data only for the selected object
     - selexp = (optional) reduce data only for the selected exposure number
     - selsetup = (optional) reduce data only for the selected setup
     - selobid = (optional) reduce data only for the selected OB ID
     - selmode = (optional) reduce data only for the selected mode
     - seldtype = (optional) reduce data only for the selected data type
     - obstype = (optional) reduce data only for the selected observation type
     - mindate = (optional) reduce data only for a date larger than providied
     - seldate = (optional) reduce data only for the selected data
     - maxdate = (optional) reduce data only for a data smaller than provided
     - maxshift: (default=10) maximum allowed discrepany between expected and found beam
                 position in a chop/nod pattern in pixel
     - extract: (default=True) Try detect a source in the combined nod image and extract subimage
     - searcharea: (default='chopthrow') specify the area to search for the sub beam
     - box: (optional) by default the subimage will have the size of the chopthrow
     - chopsubmeth: (default='averchop') method for the subtraction of the chops in burst mode. By default the average of the exposures in the offset position of the corresponding pair is used.
     - AA_pos: (optional) specify position of the beam in chop/nod position AA for burst alignment
     - refpos: (optional) instead of searching for the beam position, provide one directly for the alignment
     - findbeams: (default: True) Should the code try to find the beam positions or just extract at the computed position?
     - alignmethod: (default='fastgauss') specify the fitting algorithm for the frame alignment in burst mode data
     - verbose = False

    TO BE ADDED LATER:
        - correct treatment of HR and HRX SPC
        - correct treatment of SPC ACQ data
        - proper modification of the headers of the products
    """

    if debug:
        verbose = True

    if logfile is None:
        logfile = ftabpro.replace("_pro.csv", ".log")

    if not os.path.isfile(logfile):
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info('\n\nNew execution of REDUCE_OBSERVATION\n',
                   logfile, mode=mode)

    msg = ("Input parameters: \n")
    _print_log_info(msg, logfile)

#    noparams = reduce_observation.__code__.co_argcount
#    paranames = reduce_observation.__code__.co_varnames
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    msg = ""
    for i in args:
        msg = msg + (str(i) + ": " + str(values[i]) + "\n")

    _print_log_info(msg, logfile, logtime=False)

    dpro = ascii.read(ftabpro, header_start=0, delimiter=',',
                              guess=False)

    # --- data table integrity checks
    dupl = _duplicates_in_column(table=dpro, colname='mjd')
    if dupl is None:
        dupl = _duplicates_in_column(table=dpro, colname='expid')

    if dupl is not None:
        msg = ("ERROR: Duplicated files in table! expids: "
               + ', '.join([str(e) for e in dupl])
               + '. Exiting...')

        _print_log_info(msg, logfile)
        exit

    ntot = len(dpro)

    # --- add columns to product tabe for some basic object properties
    newcols = ['BGmed',
               'BGstd',
               'GF_medBG',
               'GF_peakampl',
               'GF_totcnts',
               'GF_majFWHM_as',
               'GF_minFWHM_as',
               'GF_posang',
               'STDflux_Jy',
               'GF_mJy/ADU/DIT',
               'bestSN',
               'sensit_mJy'
               ]

    for newcol in newcols:
        if newcol not in dpro.colnames:
            dpro.add_column(Column(np.zeros(ntot), name=newcol))
            print("New Col added.")

    if not "nowarn" in dpro.colnames:
        nowarn = np.zeros(ntot, dtype=int)
        newCol = Column(nowarn, name='nowarn')
        dpro.add_column(newCol)

    if not "noerr" in dpro.colnames:
        noerr = np.zeros(ntot, dtype=int)
        newCol = Column(noerr, name='noerr')
        dpro.add_column(newCol)

    # --- decide which exposures should be reduced
    #     and whether together or separated
    if selexp is not None:
        if not hasattr(selexp, "__len__"):
            selexp = [selexp]
            print("selexp", selexp)
        tored = selexp
    else:
        tored = dpro['expid']

        if (mindate is not None):

            t = Time(mindate, format='isot', scale='utc')
            minmjd = t.mjd + 0.5

            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['mjd'][i] >= minmjd]

            tored = list(set(tored).intersection(ids))

        if (maxdate is not None):

            t = Time(maxdate, format='isot', scale='utc')
            maxmjd = t.mjd + 0.5

            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['mjd'][i] < maxmjd]

            tored = list(set(tored).intersection(ids))

        if (seldate is not None):

            t = Time(seldate, format='isot', scale='utc')
            selmjd = t.mjd + 0.5

            ids = [x for i, x in enumerate(dpro['expid'])
                   if (dpro['mjd'][i] >= selmjd) & (dpro['mjd'][i] < selmjd+1)]

            tored = list(set(tored).intersection(ids))

        if (obstype is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if obstype in dpro['tempname'][i]]

            tored = list(set(tored).intersection(ids))

        if (selobj is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['targname'][i] in selobj]

            tored = list(set(tored).intersection(ids))

        if (selsetup is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['setup'][i] in selsetup]

            tored = list(set(tored).intersection(ids))

        if (selobid is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['obsid'][i] in selobid]

            tored = list(set(tored).intersection(ids))

        if (selmode is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['mode'][i] in selmode]

            tored = list(set(tored).intersection(ids))

        if (seldtype is not None):
            ids = [x for i, x in enumerate(dpro['expid'])
                   if dpro['datatype'][i] in seldtype]

            tored = list(set(tored).intersection(ids))

    nred = len(tored)

    if nred > 1:
        tored = np.sort(tored)

    msg = (" - Number of selected seperate exposures: " + str(nred))
    _print_log_info(msg, logfile)

    draw = ascii.read(ftabraw, header_start=0, delimiter=',', guess=False)

    # --- data table integrity check
    dupl = _duplicates_in_column(table=draw, colname='MJD-OBS')
    if dupl is not None:
        msg = ("ERROR: Duplicated files in table! MJDs: "
               + ', '.join([str(e) for e in dupl])
               + '. Exiting...')

        _print_log_info(msg, logfile)
        exit

    # --- test whether output folder exists
    expfolder = outfolder + '/exposures'
    if not os.path.exists(expfolder):
        os.makedirs(expfolder)

    # --- loop over all the different exposures
    for i in range(nred):

        targ = dpro['targname'][tored[i]].replace(" ","").lstrip().rstrip()
        setup = dpro['setup'][tored[i]]
        date = dpro['dateobs'][tored[i]][0:19]
        insmode = dpro['mode'][tored[i]]
        datatype = dpro['datatype'][tored[i]]
        nof = dpro['nof'][tored[i]]
        tempname = dpro['tempname'][tored[i]]


        noe = 0

        msg = ("\n"
               + "\n-------------------------------------------")
        _print_log_info(msg, logfile, logtime=False)

        msg = ("Reduction of Exp: " + str(tored[i])
               + "\n - Target name: " + str(targ)
               + "\n - Instrument Mode: " + str(insmode)
               + "\n - Setup: " + str(setup)
               + "\n - Datatype: " + str(datatype)
               + "\n - Template name: " + str(tempname)
               + "\n - Date: " + str(date)
               + "\n - Number of raw files: " + str(nof)
               )

        _print_log_info(msg, logfile)

        # --- if the target is a calibrator put results in a different subfolder
        cal = False
        if insmode == "SPC":
            if "cal" in tempname:
                cal = True

        else:
            try:
               std =  _get_std_flux(targ, filtname=setup, instrument=instrument) > 0
               if std & ("obs" not in tempname):
                   cal = True
            except:
                msg = "Target apparently no calibrator."
                _print_log_info(msg, logfile, logtime=False)


        if cal:
            msg = "Target identified as calibrator."
            _print_log_info(msg, logfile, logtime=False)

            outname = ("CAL_"+
                       insmode + '_' +
                       setup + '_' +
                       date + '_' +
                       targ)

            # --- create subfolders for the objects
            subfold = expfolder +  '/CAL/' + setup

        # --- name string for SCI observation
        else:
            outname = (targ + '_' +
                       insmode + '_' +
                       setup + '_' +
                       date)

            # --- create subfolders for the objects
            subfold = expfolder +  '/' + targ

        if not os.path.isdir(subfold):
            os.makedirs(subfold)

        try:
            es = _reduce_exposure(draw=draw, dpro=dpro, expid=tored[i],
                                 temprofolder=outfolder,
                                 outname=outname, rawfolder=infolder,
                                 outfolder=subfold, overwrite=overwrite,
                                 maxshift=maxshift, extract=extract,
                                 box=box, searchsmooth=searchsmooth,
                                 verbose=verbose, chopsubmeth=chopsubmeth,
                                 alignmethod=alignmethod, searcharea=searcharea,
                                 AA_pos=AA_pos, refpos=refpos,
                                 findbeams=findbeams, crossrefim=crossrefim,
                                 statcalfolder=statcalfolder,
                                 ditsoftaver=ditsoftaver,sky_xrange=sky_xrange,
                                 sky_yrange=sky_yrange, debug=debug, plot=plot,
                                 instrument=instrument, insmode=insmode)

            noe = noe + es
            msg = ("REDUCE_OBSERVATION: Number of errors: " + str(noe))
            _print_log_info(msg, logfile)

        except:
            e = sys.exc_info()
            msg = ("REDUCE_OBSERVATION: ERROR: Exposure failed to reduce. \n"
                   + str(e[1]) + ' ' + str(traceback.print_tb(e[2]))
                   + "\nContinue with next...")
            _print_log_info(msg, logfile)

        # --- update the database file
        formats = {
                   'GF_medBG' : "%.1f",
                   'GF_peakampl' : "%.1f",
                   'GF_totcnts' : "%.0f",
                   'GF_majFWHM_as' : "%.2f",
                   'GF_minFWHM_as' : "%.2f",
                   'GF_posang' : "%.1f",
                   'STDflux_Jy' : "%.2f",
                   'GF_mJy/ADU/DIT' : "%.3f",
                   'bestSN' : "%.1f",
                   'sensit_mJy' : "%.1f",
                   'BGmed' : "%.2f",
                   'BGstd' : "%.2f"
                   }

        shutil.copy(ftabpro, ftabpro.replace(".csv", "_backup.csv"))

        dpro.write(ftabpro, delimiter=',', format='ascii',
                   fill_values=[(ascii.masked, '')], formats=formats)
