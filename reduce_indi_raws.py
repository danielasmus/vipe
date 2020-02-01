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
import shutil
from tqdm import tqdm
from astropy.io import fits


from .fits_get_info import fits_get_info as _fits_get_info
from .print_log_info import print_log_info as _print_log_info
from .simple_image_plot import simple_image_plot as _simple_image_plot
from .simple_nod_exposure import simple_nod_exposure as _simple_nod_exposure
from .subtract_source import subtract_source as _subtract_source
from . import visir_params as _vp


def reduce_indi_raws(infolder, outfolder, ftabraw, overwrite=True,
                          verbose=False, logfile=None, justtable=False,
                          sky_xrange=None, sky_yrange=None):
    """
    Go through all the raw files in the input folder
    """

    outfolder = outfolder + "/hcycles/blindsums"

    # --- test whether output folder exists
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if logfile is None:
        logfile = ftabraw.replace("_raw.csv", ".log")

    if not os.path.isfile(logfile):
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info('\n\nNew execution of REDUCE_INDI_RAW_FILES\n',
                   logfile, mode=mode)

    files = [ff for ff in os.listdir(infolder)
             if (ff.endswith('.fits') & ff.startswith('VISIR'))]

    nfiles = len(files)

    msg = ('REDUCE_INDI_RAW_FILES: Number of raw files found: ' + str(nfiles))
    _print_log_info(msg, logfile)

    if os.path.isfile(ftabraw):
        newfile = False
    else:
        newfile = True

    if overwrite and not newfile:
        shutil.copy(ftabraw, ftabraw.replace(".csv", "_backup.csv"))

    if newfile or overwrite:
        f = open(ftabraw, 'w')
    else:
        f = open(ftabraw, 'a')
        f.write('\n')

    keys = (
            'DATE-OBS'+', '+
            'MJD-OBS'+', '+
            'TARG NAME'+', '+
            'RA'+ ', ' +
            'DEC'+ ', ' +
            'TEL AIRM'+ ', ' +
            'TEL ALT' + ', ' +
            'TEL AZ' + ', ' +
            'PARANG'+', '+
            'ABSROT' + ', ' +
            'POSANG'+', '+
            'ALTAZTRACK' + ', ' +
            'OBS ID'+', '+
            'PROG ID'+', '+
            'TPLNO'+', '+
            'TPL ID'+', '+
            'TPL EXPNO'+', '+
            'TPL NEXP'+', '+
            'CHOP FREQ'+', '+
            'CHOP POSANG'+', '+
            'CHOP THROW'+', '+
            'CHOPNOD DIR'+', '+
            'NODPOS'+', '+
            'JITTER WIDTH'+', '+
            'CUMOFFSETX'+', '+
            'CUMOFFSETY'+', '+
            'INSMODE'+', '+
            'PFOV'+', '+
            'INS FILT1 NAME'+', '+
            'INS FILT2 NAME'+', '+
            'INS RESOL'+', '+
            'INS GRAT1 WLEN'+', '+
            'INS SLIT1 WID'+', '+
            'CYCSUM'+', '+
            'FRAM FORMAT'+', '+
            'SEQ1 DIT' + ', ' +
            'DET WIN STRX'+', '+
            'DET WIN STRY'+', '+
            'M1 TEMP' +', '+
            'AMBI TEMP' +', '+
            'AMBI PRES' +', '+
            'RHUM' +', '+
            'WINDDIR'+', '+
            'WINDSP'+', '+
            'IWV'+', '+
            'IWVSTD' + ', ' +
            'IRSKY TEMP'+', '+
            'TAU0' +', '+
            'TEL AMBI FWHM'+', '+
            'IA FWHM'+', '+
            'IA FWHMLINOBS'+', '+
            'DPR TECH' + ', ' +
            'DPR CATG'
           )

    nkeys = len(keys.split(','))

    #f.write(keys+'\n')


    files = np.sort(files)

    for i in tqdm(range(nfiles)):

        #    if not os.path.isfile(fout):
    #        if i < nfiles-1:

        fin = infolder + "/" + files[i]
        msg = ("REDUCE_INDI_RAW_FILES: File No, Name: " + str(i) + ', '
               + files[i])
        _print_log_info(msg, logfile)

        fout = outfolder + "/" + files[i].replace(".fits", "_blindsum.fits")

        if overwrite is False and os.path.isfile(fout):

            msg = ("REDUCE_INDI_RAW_FILES: File already reduced. Continue...")
            _print_log_info(msg, logfile)
            continue

        # --- take care of corrupt files
        corrupt = False
        try:
            hdu = fits.open(fin)
            head = hdu[0].header
        except:
            corrupt = True

        if not corrupt and not justtable:
            #n_ext = len(hdu)
            fram_format = _fits_get_info(head, "FRAM FORMAT",
                                              fill_value='')
            try:
                if fram_format == "extension":

                    chopasky = hdu[1].data
                    chopbsky = hdu[2].data
                    data = hdu[-1].data

                elif fram_format == "cube-ext":  # burst mode
                    data = hdu[1].data
                else:
                    corrupt = True
                    msg = ('REDUCE_INDI_RAW_FILES: WARNING: File format not \
                            recognised. Found FRAM FORMAT is '
                            + str(fram_format))

                    _print_log_info(msg, logfile)
                    hdu.close()
            except:
                corrupt = True
                hdu.close()

        if corrupt:
            msg = ('REDUCE_INDI_RAW_FILES: File corrupt:' + str(i) + ', '
                   + fin)
            _print_log_info(msg, logfile)

            f.write(files[i]+', CORRUPT')
            for j in range(nkeys):
                f.write(',')
            f.write(',-1\n')
            continue



        # --- read fits header info
        params = _fits_get_info(head, keys=keys, fill_value='')

        # if len(params) != 35: print(len(params))
        if (i == 0) & (newfile | overwrite):
            f.write("filename,")
            # for j in range(len(params)):
            for p in params.keys():
                # f.write(params.keys()[j]+', ')
                f.write(str(p).replace(' ','_')+',')
            f.write('nodexptime,chopamed,chopastd,chopbmed,chopbstd,cdifmed,cdifstd,EXPID\n')

        f.write(files[i]+',')
        # for j in range(len(params)):
        for p in params:
            f.write(str(params[p]).replace(',',' ')+',')
            # f.write(str(params.values()[j]).replace(',',' ')+', ')

        nodexp = _fits_get_info(head, "nodexptime")
        msg = ("REDUCE_INDI_RAW_FILES: Nod exp time: " + str(nodexp))
        _print_log_info(msg, logfile)
        f.write("{:.0f}".format(nodexp))

        # --- compute the background characteristics
        if fram_format == "cube-ext":
            # --- for burst mode simply take the first frame of each chop position
            chopasky = data[0]
            chopbsky = data[head['HIERARCH ESO DET NDIT']/
                            head["HIERARCH ESO DET NAVRG"]]

        # --- if no specfic range is supplied to measure the sky use the full
        #     illuminated area of the VISIR detector
        if sky_xrange is None:
            sky_xrange = _vp.max_illum_xrange

        if sky_yrange is None:
            sky_yrange = _vp.max_illum_yrange

        chopamed = np.nanmedian(chopasky[sky_yrange[0]:sky_yrange[1],
                                         sky_xrange[0]:sky_xrange[1]])

        chopastd = np.nanstd(chopasky[sky_yrange[0]:sky_yrange[1],
                                      sky_xrange[0]:sky_xrange[1]])

        chopbmed = np.nanmedian(chopbsky[sky_yrange[0]:sky_yrange[1],
                                         sky_xrange[0]:sky_xrange[1]])

        chopbstd = np.nanstd(chopbsky[sky_yrange[0]:sky_yrange[1],
                                      sky_xrange[0]:sky_xrange[1]])

        f.write(',' + "{:.0f}".format(chopamed)
                + ',' + "{:.1f}".format(chopastd)
                + ',' + "{:.0f}".format(chopbmed)
                + ',' + "{:.1f}".format(chopbstd)
                )

        # --- write out a simple combination of the fits
        if not justtable:
            if fram_format == "extension":  # for extension data type simply take the last extension
                outim = data
                fits.writeto(fout, data, head, overwrite=True)
            elif fram_format == "cube-ext":  # for burst mode do a simple combination of the frames
                outim = _simple_nod_exposure(ima=data, fout=fout, head=head)


            # --- background estimation:
            bgim = _subtract_source(outim[sky_yrange[0]:sky_yrange[1],
                                           sky_xrange[0]:sky_xrange[1]])

            difmed = np.nanmedian(bgim)

            difstd = np.nanstd(bgim)

            f.write(',' + "{:.2f}".format(difmed)
                    + ',' + "{:.2f}".format(difstd)
                    )

            hdu.close()
            fout = fout.replace(".fits", ".png")
            _simple_image_plot(outim, fout, log=True)


        f.write(',-1\n')

    f.close()
