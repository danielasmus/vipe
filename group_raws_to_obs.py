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
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy.stats import circmean, circvar
from astropy import units as u

from .duplicates_in_column import duplicates_in_column as _duplicates_in_column
from .print_log_info import print_log_info as _print_log_info


def check_masked(d):
    """
    check if table contains any masked values and if yes return cols and rows
    """
    masked = []
    for c in d.colnames:
        if type(d[c]) == MaskedColumn:
            nmask = np.sum(d[c].mask)
            if nmask > 0:
                masked.append([c, np.where(d[c].mask)[0]])

    return(masked)


def check_none(d):
    """
    check if table contains any None values and if yes return cols and rows
    """
    nones = []
    for c in d.colnames:
            nnone = np.sum(d[c] == None)
            if nnone > 0:
                nones.append([c, np.where(d[c] == None)[0]])

    return(nones)


#%%
# --- a little helper routine to determine the mean and variation params
def fill_values(draw, dpro, rawkey, prokey, i, j, nof, ktype='tel',
                deg=False, lowlim=None, uplim=None):

    if lowlim is None:
        lowlim = np.nanmin(draw[rawkey][j:j+nof])

    if uplim is None:
        uplim = np.nanmax(draw[rawkey][j:j+nof])

    ids = np.where((draw[rawkey][j:j+nof] >= lowlim)
                   & (draw[rawkey][j:j+nof] <= uplim))[0]

    if len(ids) == 0:
        if ktype == 'tel':
             dpro[prokey+"_mean"][i] = -999
             dpro[prokey+"_start"][i] = -999
             dpro[prokey+"_end"][i] = -999

        else:
           dpro[prokey+"_median"][i] = -999
           dpro[prokey+"_stddev"][i] = -999

        return

    else:
        vals = draw[rawkey][j:j+nof][ids]

    if not any(draw[rawkey].mask[j:j+nof]):

        if ktype == 'tel':
            if deg:
               dpro[prokey+"_mean"][i] = circmean(vals * u.deg).value

               if dpro[prokey+"_mean"][i] < 0:
                   dpro[prokey+"_mean"][i] += 360
               elif dpro[prokey+"_mean"][i] > 360:
                   dpro[prokey+"_mean"][i] -= 360
            else:
                dpro[prokey+"_mean"][i] = np.nanmean(vals)

            dpro[prokey+"_start"][i] = draw[rawkey][j]
            dpro[prokey+"_end"][i] = draw[rawkey][j+nof-1]

        else:

            if deg:
                dpro[prokey+"_median"][i] = circmean(vals * u.deg).value

                if dpro[prokey+"_median"][i] < 0:
                    dpro[prokey+"_median"][i] += 360

                elif dpro[prokey+"_median"][i] > 360:
                    dpro[prokey+"_median"][i] -= 360

                dpro[prokey+"_stddev"][i] = np.sqrt(circvar(vals * u.deg).value)

            else:
                dpro[prokey+"_median"][i] = np.nanmedian(vals)
                dpro[prokey+"_stddev"][i] = np.std(vals)

    else:
        if ktype == 'tel':
             dpro[prokey+"_mean"][i] = -999
             dpro[prokey+"_start"][i] = -999
             dpro[prokey+"_end"][i] = -999

        else:
           dpro[prokey+"_median"][i] = -999
           dpro[prokey+"_stddev"][i] = -999


#%%
# --- MAIN ROUTINE
def group_raws_to_obs(ftabraw, ftablog, ftabpro, maxgap=5,
                                logfile=None, overwrite=False):
    """
    Look into the raw file table and identify individual observations/exposures
    and print out a new table containing the information of the latter
    PARAMETERS:
     - maxgap: (default 5) provide the maximum time gap between two files in
               min
    """

    funname = "GROUP_RAWS_TO_OBS"

    if logfile is None:
        logfile = ftabraw.replace("_raw.csv", ".log")

    if not os.path.isfile(logfile):
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info('\n\nNew execution of GROUP_FILES_TO_OBSERVATIONS\n',
                   logfile, mode=mode)


    draw = ascii.read(ftabraw, header_start=0, delimiter=',',
                              guess=False)

    # --- data table integrity check
    dupl = _duplicates_in_column(table=draw, colname='MJD-OBS')
    if dupl is not None:
        msg = ("ERROR: Duplicated files in table! MJDs: "
               + ', '.join([str(e) for e in dupl])
               + '. Exiting...')

        _print_log_info(msg, logfile)
        exit

    dg = ascii.read(ftablog, header_start=0, delimiter=',',
                              guess=False)


    # --- sort files by date time
    id = np.argsort(draw['MJD-OBS'])
    draw = draw[id]

    # --- throw out incomplete entries
    id = np.where(np.array(draw['nodexptime'], dtype=float) > 0)[0]
    draw = draw[id]

    n = len(draw)

    # --- group the raw files into exposures
    expid = np.zeros(n, dtype=int)
    counter = 0
    nof = 1

    for i in np.arange(1, n):
        if (draw['OBS_ID'][i-1] != draw['OBS_ID'][i]):
            counter = counter + 1
        elif (draw['TPLNO'][i-1] != draw['TPLNO'][i]):
            counter = counter + 1
        elif (draw['TPL_EXPNO'][i-1] == draw['TPL_NEXP'][i-1]):
            counter = counter + 1
        elif (draw['MJD-OBS'][i] - draw['MJD-OBS'][i-1] > maxgap/60.0/24.0):
            counter = counter + 1

        expid[i] = counter

    # --- get the start MJDs of each exposure as unique identifier
    nexp = counter + 1
    mjds = np.zeros(nexp)

    for i in range(nexp):

        id = np.where(expid == i)[0]
        mjds[i] = draw['MJD-OBS'][id[0]]


    # --- in case the file exists update and append information
    if os.path.isfile(ftabpro) and not overwrite:

        new = False
        dpro = ascii.read(ftabpro, header_start=0, delimiter=',',
                              guess=False)

        # --- change all string columns to object type so that their string
        #     length becomes variable
        for col in dpro.itercols():
            if col.dtype.kind in 'SU':
                dpro.replace_column(col.name, col.astype('object'))

    # --- if the file not exists generate the empty table structure
    else:
      new = True
      dpro = Table({
                "expid": [None]*nexp,
                "targname": [None]*nexp,
                "ra": [None]*nexp,
                "dec": [None]*nexp,
                "progid": [None]*nexp,
                "obsid": [None]*nexp,
                "dateobs": [None]*nexp,
                "mjd": [None]*nexp,
                "tempno": [None]*nexp,
                "insmode": [None]*nexp,
                "tempname": [None]*nexp,
                "datatype": [None]*nexp,
                "setup": [None]*nexp,
                "pfov": [None]*nexp,
                "dit": [None]*nexp,
                "nodtime": [None]*nexp,
                "noddir": [None]*nexp,
                "jitter": [None]*nexp,
                "chopthrow": [None]*nexp,
                "chopangle": [None]*nexp,
                "chopfreq": [None]*nexp,
                "expnof": [None]*nexp,
                "nof": [None]*nexp,
                "exptime": [None]*nexp,
                "grade": [None]*nexp,
                "posang": [None]*nexp,
                "absrot_mean": [None]*nexp,
                "absrot_start": [None]*nexp,
                "absrot_end": [None]*nexp,
                "parang_mean": [None]*nexp,
                "parang_start": [None]*nexp,
                "parang_end": [None]*nexp,
                "alt_mean" : [None]*nexp,
                "alt_start" : [None]*nexp,
                "alt_end" : [None]*nexp,
                "az_mean" : [None]*nexp,
                "az_start" : [None]*nexp,
                "az_end" : [None]*nexp,
                "airm_mean" : [None]*nexp,
                "airm_start" : [None]*nexp,
                "airm_end" : [None]*nexp,
                "pwv_median" : [None]*nexp,
                "pwv_stddev" : [None]*nexp,
                "skytemp_median" : [None]*nexp,
                "skytemp_stddev" : [None]*nexp,
                "pressure_median" : [None]*nexp,
                "pressure_stddev" : [None]*nexp,
                "humidity_median" : [None]*nexp,
                "humidity_stddev" : [None]*nexp,
                "temperature_median" : [None]*nexp,
                "temperature_stddev" : [None]*nexp,
                "m1temp_median" : [None]*nexp,
                "m1temp_stddev" : [None]*nexp,
                "winddir_median" : [None]*nexp,
                "winddir_stddev" : [None]*nexp,
                "windspeed_median" : [None]*nexp,
                "windspeed_stddev" : [None]*nexp,
                "cohertime_median" : [None]*nexp,
                "cohertime_stddev" : [None]*nexp,
                "asmfwhm_median" : [None]*nexp,
                "asmfwhm_stddev" : [None]*nexp,
                "iafwhm_median" : [None]*nexp,
                "iafwhm_stddev" : [None]*nexp,
                "iafwhmlo_median" : [None]*nexp,
                "iafwhmlo_stddev" : [None]*nexp,
                "chopamed_median" : [-999]*nexp,
                "chopamed_stddev" : [-999]*nexp,
                "chopastd_median" : [-999]*nexp,
                "chopastd_stddev" : [-999]*nexp,
                "chopbmed_median" : [-999]*nexp,
                "chopbmed_stddev" : [-999]*nexp,
                "chopbstd_median" : [-999]*nexp,
                "chopbstd_stddev" : [-999]*nexp,
                "cdifmed_median" : [-999]*nexp,
                "cdifmed_stddev" : [-999]*nexp,
                "cdifstd_median" : [-999]*nexp,
                "cdifstd_stddev" : [-999]*nexp,
               },
               names = (    # with this array the order is fixed
                        "expid",
                        "targname",
                        "ra",
                        "dec",
                        "progid",
                        "obsid",
                        "dateobs",
                        "mjd",
                        "tempno",
                        "insmode",
                        "tempname",
                        "datatype",
                        "setup",
                        "pfov",
                        "dit",
                        "nodtime",
                        "noddir",
                        "jitter",
                        "chopthrow",
                        "chopangle",
                        "chopfreq",
                        "expnof",
                        "nof",
                        "exptime",
                        "grade",
                        "posang",
                        "absrot_mean",
                        "absrot_start",
                        "absrot_end",
                        "parang_mean",
                        "parang_start",
                        "parang_end",
                        "alt_mean",
                        "alt_start",
                        "alt_end",
                        "az_mean",
                        "az_start",
                        "az_end",
                        "airm_mean",
                        "airm_start",
                        "airm_end",
                        "pwv_median",
                        "pwv_stddev",
                        "skytemp_median",
                        "skytemp_stddev",
                        "pressure_median",
                        "pressure_stddev",
                        "humidity_median",
                        "humidity_stddev",
                        "temperature_median",
                        "temperature_stddev",
                        "m1temp_median",
                        "m1temp_stddev",
                        "winddir_median",
                        "winddir_stddev",
                        "windspeed_median",
                        "windspeed_stddev",
                        "cohertime_median",
                        "cohertime_stddev",
                        "asmfwhm_median",
                        "asmfwhm_stddev",
                        "iafwhm_median",
                        "iafwhm_stddev",
                        "iafwhmlo_median",
                        "iafwhmlo_stddev",
                        "chopamed_median",
                        "chopamed_stddev",
                        "chopastd_median",
                        "chopastd_stddev",
                        "chopbmed_median",
                        "chopbmed_stddev",
                        "chopbstd_median",
                        "chopbstd_stddev",
                        "cdifmed_median",
                        "cdifmed_stddev",
                        "cdifstd_median",
                        "cdifstd_stddev"
                        )
                   )


    # --- loop over all the exposures by MJD:
    # for i in tqdm(range(nexp)):
    for i in range(nexp):

        if not new:
            # --- check if already an entry in the table for that exposure
            id = np.where(dpro["mjd"] == mjds[i])[0]

            # --- otherwise add a row at the end of the table
            if len(id) == 0:
                dpro.add_row()
                id = len(dpro)-1
            else:
                id = id[0]



        # --- for new table just take the i-th row
        else:
            id = i

        # --- now gather all files belonging to that exposure
        ids = np.where(expid == i)[0]

        j = ids[0]


        #print(i, mjds[i], id, dpro["mjd"][id], j)

        # print(i,id,j)

        nof = len(ids)
        dpro["nof"][id] = nof

        dpro["expid"][id] = i

        if nof == 1 and "MoveToSlit" not in draw['TPL_ID'][j]:
            msg = ("WARNING: Only one raw file found for exposure. "
                   + "Exp ID: " + str(i)
                   + "; DATE-OBS: " + draw['DATE-OBS'][j]
                   + "; TPL_ID: " + draw['TPL_ID'][j]

                   )

            _print_log_info(msg, logfile)



        # print(i,j, draw['CYCSUM'][j])
        if "Burst" in draw['TPL_ID'][j]:
            dpro["datatype"][id] = 'burst'
        elif draw['CYCSUM'][j] == 'False':
            dpro["datatype"][id] = 'halfcyc'
        else:
            dpro["datatype"][id] = 'cycsum'

        # --- fetch the grade and add it
        dpro["dateobs"][id] = draw['DATE-OBS'][j][0:19]

        idg = [g for g, s in enumerate(dg['filename'])
               if dpro["dateobs"][id] in s]
        if idg:
            dpro["grade"][id] = dg['grade'][idg[0]]
        else:
            dpro["grade"][id] = '-'

        # --- fill in the other values
        if draw['insmode'][j] == 'IMG':
            dpro["setup"][id] = draw['INS_FILT1_NAME'][j]
        elif ((draw['insmode'][j] == 'ACQ-SPC-SPC')
              | (draw['insmode'][j] == 'ACQ-IMG-SPC')):
            dpro["setup"][id] = draw['INS_FILT2_NAME'][j]
        elif draw['insmode'][j] == 'SPC':
            dpro["setup"][id] = str(draw['INS_SLIT1_WID'][j])
        else:
            print(draw['insmode'][j])
            raise ValueError()

        # --- if pupil tracking was used add that info to the mode
        if draw['ALTAZTRACK'][j] == 'True':
            dpro["insmode"][id] = str(draw['insmode'][j]) + "-PT"
        else:
            dpro["insmode"][id] = str(draw['insmode'][j])

        try:
            dpro["exptime"][id] = float(draw['nodexptime'][j])*nof
        except:
            dpro["exptime"][id] = -1

        dpro["targname"][id] = draw['TARG_NAME'][j]
        dpro["ra"][id] = draw['RA'][j]
        dpro["dec"][id] = draw['DEC'][j]
        dpro["progid"][id] = draw['PROG_ID'][j]
        dpro["obsid"][id] = draw['OBS_ID'][j]
        dpro["mjd"][id] = mjds[i]
        dpro["tempno"][id] = draw['TPLNO'][j]
        dpro["tempname"][id] = draw['TPL_ID'][j]
        dpro["expnof"][id] = draw['TPL_NEXP'][j]

        dpro["pfov"][id] = draw['PFOV'][j]

        if not draw['SEQ1_DIT'].mask[j]:
            dpro["dit"][id] = draw['SEQ1_DIT'][j]
        else:
            dpro["dit"][id] = -1

        # print(draw['CHOPNOD_DIR'][j], type(draw['CHOPNOD_DIR'][j]))

        if not draw['CHOPNOD_DIR'].mask[j]:
            dpro["noddir"][id] = draw['CHOPNOD_DIR'][j]
        else:
            dpro["noddir"][id] = "None?"

        if not draw['JITTER_WIDTH'].mask[j]:
            dpro["jitter"][id] = draw['JITTER_WIDTH'][j]
        else:
            dpro["jitter"][id] = -1

        if not draw['CHOP_THROW'].mask[j]:
            dpro["chopthrow"][id] = draw['CHOP_THROW'][j]
        else:
            dpro["chopthrow"][id] = -1

        if not draw['CHOP_POSANG'].mask[j]:
            dpro["chopangle"][id] = draw['CHOP_POSANG'][j]
        else:
            dpro["chopthrow"][id] = -1

        if not draw['CHOP_FREQ'].mask[j]:
            dpro["chopfreq"][id] = draw['CHOP_FREQ'][j]
        else:
            dpro["chopfreq"][id] = -1

        # --- actual time between two nods
        # print(i, id,j, nof)
        try:
            dpro["nodtime"][id] = (draw['MJD-OBS'][j+nof-1]
                                   - draw['MJD-OBS'][j]) / (nof -1) * 24 * 3600
        except:
            dpro["nodtime"][id] = -1

        # --- now the changing telescope parameters
        if not any(draw['POSANG'].mask[j:j+nof]):
            dpro["posang"][id] = np.nanmean(draw['POSANG'][j:j+nof])

        fill_values(draw, dpro, 'ABSROT', 'absrot', id, j, nof,
                    ktype='tel', deg=True)
        fill_values(draw, dpro, 'PARANG', 'parang', id, j, nof,
                    ktype='tel')
        fill_values(draw, dpro, 'TEL_ALT', 'alt', id, j, nof, ktype='tel')
        fill_values(draw, dpro, 'TEL_AZ', 'az', id, j, nof, ktype='tel', deg=True)
        fill_values(draw, dpro, 'TEL_AIRM', 'airm', id, j, nof, ktype='tel')

        # --- now the ambient parameters
        fill_values(draw, dpro, 'IWV', 'pwv', id, j, nof,
                    ktype='ambi', lowlim=0, uplim=10)
        fill_values(draw, dpro, 'IRSKY_TEMP', 'skytemp', id, j, nof,
                    ktype='ambi', lowlim=-150, uplim=-35)
        fill_values(draw, dpro, 'AMBI_PRES', 'pressure', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'RHUM', 'humidity', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'AMBI_TEMP', 'temperature', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'M1_TEMP', 'm1temp', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'WINDDIR', 'winddir', id, j, nof,
                    ktype='ambi', deg=True)
        fill_values(draw, dpro, 'WINDSP', 'windspeed', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'TAU0', 'cohertime', id, j, nof,
                    ktype='ambi')
        fill_values(draw, dpro, 'TEL_AMBI_FWHM', 'asmfwhm', id, j, nof,
                    ktype='ambi', lowlim=0.1, uplim=3)
        fill_values(draw, dpro, 'IA_FWHM', 'iafwhm', id, j, nof, ktype='ambi',
                    lowlim=0.1, uplim=3)
        fill_values(draw, dpro, 'IA_FWHMLINOBS', 'iafwhmlo', id, j, nof,
                    lowlim=0.1, uplim=3, ktype='ambi')

        # --- now the background parameters
        fill_values(draw, dpro, 'chopamed', 'chopamed', id, j, nof, ktype='bg')
        fill_values(draw, dpro, 'chopastd', 'chopastd', id, j, nof, ktype='bg')
        fill_values(draw, dpro, 'chopbmed', 'chopbmed', id, j, nof, ktype='bg')
        fill_values(draw, dpro, 'chopbstd', 'chopbstd', id, j, nof, ktype='bg')
        fill_values(draw, dpro, 'cdifmed', 'cdifmed', id, j, nof, ktype='bg')
        fill_values(draw, dpro, 'cdifstd', 'cdifstd', id, j, nof, ktype='bg')

        # --- check is the PWV STD was larger during a file than in between
        if not any(draw['IWVSTD'].mask[j:j+nof]):
            iwvstd = np.nanmean(draw['IWVSTD'][j:j+nof])

            if dpro['pwv_stddev'][id] is not None:
                if (iwvstd > dpro['pwv_stddev'][id]) & (iwvstd > 0):
                    dpro['pwv_stddev'][id] = iwvstd

            else:
                dpro['pwv_stddev'][id] = iwvstd


        # --- clean some bad values:
        if dpro['exptime'][id] == 0:
            dpro['exptime'][id] = None



    # --- write the finished table sorted by MJD
    dpro.sort('mjd')


    # --- write out results

    formats = {"exptime" : "%.0f",
               "nodtime" : "%.0f",
               "absrot_mean" : "%.2f",
               "absrot_start" : "%.2f",
               "absrot_end" : "%.2f",
               "parang_mean" : "%.2f",
               "parang_start" : "%.2f",
               "parang_end" : "%.2f",
               "alt_mean" : "%.2f",
               "alt_start" : "%.2f",
               "alt_end" : "%.2f",
               "az_mean" : "%.2f",
               "az_start" : "%.2f",
               "az_end" : "%.2f",
               "airm_mean" : "%.3f",
               "airm_start" : "%.3f",
               "airm_end" : "%.3f",
               "pwv_median" : "%.2f",
               "pwv_stddev" : "%.2f",
               "skytemp_median" : "%.0f",
               "skytemp_stddev" : "%.0f",
               "pressure_median" : "%.2f",
               "pressure_stddev" : "%.2f",
               "humidity_median" : "%.1f",
               "humidity_stddev" : "%.1f",
               "temperature_median" : "%.2f",
               "temperature_stddev" : "%.2f",
               "m1temp_median" : "%.2f",
               "m1temp_stddev" : "%.2f",
               "winddir_median" : "%.0f",
               "winddir_stddev" : "%.1f",
               "windspeed_median" : "%.2f",
               "windspeed_stddev" : "%.2f",
               "cohertime_median" : "%.4f",
               "cohertime_stddev" : "%.4f",
               "asmfwhm_median" : "%.2f",
               "asmfwhm_stddev" : "%.2f",
               "iafwhm_median" : "%.2f",
               "iafwhm_stddev" : "%.2f",
               "iafwhmlo_median" : "%.2f",
               "iafwhmlo_stddev" : "%.2f",
               "chopamed_median" : "%.0f",
               "chopamed_stddev" : "%.1f",
               "chopastd_median" : "%.1f",
               "chopastd_stddev" : "%.2f",
               "chopbmed_median" : "%.0f",
               "chopbmed_stddev" : "%.1f",
               "chopbstd_median" : "%.1f",
               "chopbstd_stddev" : "%.2f",
               "cdifmed_median" : "%.2f",
               "cdifmed_stddev" : "%.2f",
               "cdifstd_median" : "%.2f",
               "cdifstd_stddev" : "%.2f",
               }

    # print(len(dpro), ftabpro)
    # return(dpro)

    masked = check_masked(dpro)

    if len(masked) > 0:
        msg = (funname + ": ERROR: DPRO contains masked values: ", masked)

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    nones = check_none(dpro)

    if len(nones) > 0:
        msg = (funname + ": ERROR: DPRO contains None's: ", nones)

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    dpro.write(ftabpro, delimiter=',', format='ascii',
               fill_values=[(ascii.masked, '')], formats=formats,
               overwrite=True)


    # --- update the draw table with the expid column
    if 'EXPID' in draw.colnames:
        draw['EXPID'] = expid
    else:
        newCol = Column(expid, name='EXPID')
        draw.add_column(newCol)

    draw.write(ftabraw, delimiter=',', format='ascii',
               fill_values=[(ascii.masked, '')])

    msg = (" - Number of separate exposures found: " + str(counter+1))
    _print_log_info(msg, logfile)

    return(dpro)

