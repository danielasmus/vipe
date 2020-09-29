#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "3.1.1"

"""
HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-12: change to simple_image_plot, replace subtract source with
                  better background estimate, fixed INSMODE-->insmode, funname
                  added
    - 2020-02-13: routine restructured, added keyword instrument
    - 2020-06-09: add ISAAC support
    - 2020-06-20: add exclusion of dark, flats, linearity files and so on.
    - 2020-09-29: for VISIR extension mode, copy WCS from ext 1 to head0


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
import os
import shutil
import sys
import traceback
from tqdm import tqdm
from astropy.io import fits
from astropy.stats import sigma_clip


from .fits_get_info import fits_get_info as _fits_get_info
from .print_log_info import print_log_info as _print_log_info
from .simple_image_plot import simple_image_plot as _simple_image_plot
from .simple_nod_exposure import simple_nod_exposure as _simple_nod_exposure
# from .subtract_source import subtract_source as _subtract_source
from . import visir_params as _vp
from . import isaac_params as _ip


def reduce_indi_raws(infolder, outfolder, ftabraw, instrument=None,
                     overwrite=True,
                          verbose=False, logfile=None, justtable=False,
                          sky_xrange=None, sky_yrange=None):
    """
    Go through all the raw files in the input folder
    """
    funname = "REDUCE_INDI_RAWS"

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

    _print_log_info('\n\nNew execution of ' + funname + '\n',
                   logfile, mode=mode)

    if instrument is None:
        prefix = ""
    else:
        instrument = instrument.upper()
        prefix = instrument


    files = [ff for ff in os.listdir(infolder)
             if (ff.endswith('.fits') & ff.startswith(prefix))]

    files = np.sort(files)

    nfiles = len(files)

    msg = (funname + ': Number of raw files found: ' + str(nfiles))
    _print_log_info(msg, logfile)

    if os.path.isfile(ftabraw):
        newfile = False
    else:
        newfile = True

    if not newfile:
        shutil.copy(ftabraw, ftabraw.replace(".csv", "_backup.csv"))

    keystr = (
            'INSTRUME'+', '+
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
            'insmode'+', '+
            'PFOV'+', '+
            'INS FILT1 NAME'+', '+
            'INS FILT2 NAME'+', '+
            'INS FILT3 NAME'+', '+
            'INS FILT4 NAME'+', '+
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

    keys = keystr.split(',')
    nkeys = len(keys)

    addcols = 'nodexptime,chopamed,chopastd,chopbmed,chopbstd,cdifmed,cdifstd,EXPID'
    ncols = 1 + nkeys + len(addcols.split(","))

    # --- if there is no raw table or we want to overwrite
    if newfile or overwrite:
        f = open(ftabraw, 'w')

        # --- write the table header
        f.write("filename,")
        # for j in range(len(params)):
        for p in keys:
            # f.write(params.keys()[j]+', ')
            f.write(str(p).rstrip().lstrip().replace(' ','_')+',')

        f.write(addcols + '\n')


    # --- otherwise open existing table to append
    else:
        f = open(ftabraw, 'a')
        # f.write('\n')



    # --- loop over all the files
    for i in tqdm(range(nfiles)):

        #    if not os.path.isfile(fout):
    #        if i < nfiles-1:

        fin = infolder + "/" + files[i]
        msg = (" - " + str(i) + ', '
               + files[i])
        _print_log_info(msg, logfile)

        fout = outfolder + "/" + files[i].replace(".fits", "_blindsum.fits")

        # --- has the file been reduced already?
        if os.path.isfile(fout):
            reduced = True
        else:
            reduced = False


        # --- if the file was already reduced and the table present, assume it
        #     is included
        if reduced and not overwrite and not newfile and not justtable:
            msg = (" - File already reduced. Continue...")
            _print_log_info(msg, logfile)
            continue

        # --- can the raw file be read?
        hdu = None
        try:
            hdu = fits.open(fin)
            head = hdu[0].header

            if instrument is None:
                instrument = head["INSTRUME"].replace(" ", "")



            tempname = _fits_get_info(head, "TPL ID")

            # --- check that this file has proper header infos
            if tempname is None:
                msg = (funname + ': ERROR: TPL ID keyword missing from fits header. File type cannot be determined. Ignore and moving to next...')

                _print_log_info(msg, logfile)
                hdu.close()

                # --- write an empty line in the table
                f.write(files[i]+', CANNOT DETERMINE')
                for j in range(ncols-3):
                    f.write(',')

                f.write(',-1\n')

                continue

            # --- check if file is an on-sky frame or a calib
            excludes = ["Dark", "Flat", "NoChop", "Linearity", "Arc"]

            hit = 0
            for e in excludes:
                if e in tempname:
                    hit = 1
                    print(files[i], tempname, e)
                    break

            if hit == 1:
                print("Not on-sky chop/nod file. Ignore and moving to next...")
                hdu.close()
                continue



            if instrument == "VISIR":

                # --- get the data format
                fram_format = _fits_get_info(head, "FRAM FORMAT",
                                                  fill_value='')

                # --- get a sky frame for each chop position
                if fram_format == "extension":
                        chopasky = hdu[1].data
                        chopbsky = hdu[2].data
                        head1 = hdu[1].header

                elif fram_format == "cube-ext":  # burst mode
                        # --- for burst mode simply take the first frame of each
                        #     chop position
                        chopasky = hdu[1].data[0]
                        chopbsky = hdu[1].data[head['HIERARCH ESO DET NDIT']/
                                head["HIERARCH ESO DET NAVRG"]]

                else:
                    msg = (funname + ': ERROR: File format not \
                                recognised. Found FRAM FORMAT is '
                                + str(fram_format))

                    _print_log_info(msg, logfile)
                    hdu.close()

                    # --- write an empty line in the table
                    f.write(files[i]+', ' + fram_format)
                    for j in range(ncols-3):
                        f.write(',')
                    f.write(',-1\n')

                    continue

            elif instrument == "ISAAC":

                # --- exlude short wavelength SW files
                if "ISAACSW" in tempname:
                    print(" - Short wavelength file. Ignoring and moving to next...")
                    hdu.close()
                    continue


                # --- check if we are dealing with new or old data
                # --- new format is a cube with one average frame per chop pos
                if "CUBE" in head["ORIGFILE"]:

                    if head["NAXIS3"] == 2:
                        chopasky = hdu[0].data[0]
                        chopbsky = hdu[0].data[1]
                        # data = chopasky - chopbsky
                        fram_format = "cube"


                # --- in the old format, half-cycle files were written in
                #     addition to....
                elif "HCYCLE" in head["ORIGFILE"]:
                    print(" - Hcycle file. Ignoring and moving to next...")
                    hdu.close()
                    continue

                # --- fits files containing the chop difference
                else:
                    # data = hdu[0].data
                    chopasky = None
                    chopbsky = None
                    fram_format = "chopdif"

            else:
                msg = (funname + ': ERROR: File not recognised as neither VISIR nor ISAAC:'
                       + str(i) + ', '
                       + fin
                       )

                _print_log_info(msg, logfile)

                f.write(files[i]+', CANNOT RECOGNIZE')
                for j in range(ncols-3):
                    f.write(',')

                f.write(',-1\n')

                continue

        # --- if the file can not be read write ERROR
        except:
            msg = (funname + ': ERROR: File can not be opened:' + str(i) + ', '
                   + fin)
            _print_log_info(msg, logfile)

            f.write(files[i]+', CANNOT OPEN')
            for j in range(ncols-3):
                f.write(',')
            f.write(',-1\n')

            if hdu is not None:
                hdu.close()

            continue


        # --- read keywords from the fits header
        params = _fits_get_info(head, keys=keystr, fill_value='')

        # --- prepare outstring for table
        outline = files[i]+','

        for p in params:
            outline = outline + str(params[p]).replace(',',' ')+','

        # --- add the exposure time of the nod
        nodexp = _fits_get_info(head, "nodexptime")
        # msg = (" - Nod exp time: " + str(nodexp))
        # _print_log_info(msg, logfile)
        outline += ("{:.0f}".format(nodexp))

        # --- compute the MIR background characteristics of the frames
        # --- if no specfic range is supplied to measure the sky use the full
        #     illuminated area of the detector
        if sky_xrange is None:
            if instrument == "VISIR":
                sky_xrange = _vp.max_illum_xrange
            elif instrument == "ISAAC":
                sky_xrange = _ip.max_illum_xrange

        if sky_yrange is None:
            if instrument == "VISIR":
                sky_yrange = _vp.max_illum_yrange
            elif instrument == "ISAAC":
                sky_yrange = _ip.max_illum_yrange

        if chopasky is not None:
            chopamed = np.nanmedian(chopasky[sky_yrange[0]:sky_yrange[1],
                                             sky_xrange[0]:sky_xrange[1]])

            chopastd = np.nanstd(chopasky[sky_yrange[0]:sky_yrange[1],
                                          sky_xrange[0]:sky_xrange[1]])

            chopbmed = np.nanmedian(chopbsky[sky_yrange[0]:sky_yrange[1],
                                             sky_xrange[0]:sky_xrange[1]])

            chopbstd = np.nanstd(chopbsky[sky_yrange[0]:sky_yrange[1],
                                          sky_xrange[0]:sky_xrange[1]])

            # --- write background values into the table
            outline += (',' + "{:.0f}".format(chopamed)
                         + ',' + "{:.1f}".format(chopastd)
                         + ',' + "{:.0f}".format(chopbmed)
                         + ',' + "{:.1f}".format(chopbstd)
                        )

        else:
            outline += (',,,,')

        # --- if we just want the table but no reduction, we are done here.
        if justtable:
            hdu.close()

            # --- if we do just the table fill empty the last background entries
            outline += (',,,\n')

            f.write(outline)

            continue



        # --- otherwise if file reduced and no overwrite then we are also done
        elif overwrite is False and reduced:

            hdu.close()

            # --- open the reduced file
            hdu = fits.open(fin)
            outim = hdu[1].data
            hdu.close()

            msg = (" - File already reduced. Continue...")
            _print_log_info(msg, logfile)

        # --- otherwise, do simple combination chops in the fits
        else:

            if instrument == "VISIR":

                # --- get the actual data and produce the outbut
                if fram_format == "extension":  # for extension data type simply take the last extension
                    outim =  hdu[-1].data

                    # --- copy the WCS info from head1 to head0
                    head["CTYPE1"] = head1["CTYPE1"]
                    head["CTYPE2"] = head1["CTYPE2"]
                    head["CRPIX1"] = head1["CRPIX1"]
                    head["CRPIX2"] = head1["CRPIX2"]
                    head["CRVAL1"] = head1["CRVAL1"]
                    head["CRVAL2"] = head1["CRVAL2"]
                    head["CD1_1"] = head1["CD1_1"]
                    head["CD1_2"] = head1["CD1_2"]
                    head["CD2_1"] = head1["CD2_1"]
                    head["CD2_2"] = head1["CD2_2"]
                    head["CUNIT1"] = head1["CUNIT1"]
                    head["CUNIT2"] = head1["CUNIT2"]

                    fits.writeto(fout, outim, head, overwrite=True)

                elif fram_format == "cube-ext":  # for burst mode do a simple combination of the frames
                    outim = _simple_nod_exposure(ima=hdu[1].data, fout=fout, head=head)


            elif instrument == "ISAAC":

                if fram_format == "cube":
                    outim = chopasky - chopbsky
                elif fram_format== "chopdif":
                    outim = hdu[0].data

                    # --- for unknown reasons, the fits header of the ISAAC
                    #     acquisition images contains the NAXIS3 keyword
                    if "NAXIS3" in head:
                        del head["NAXIS3"]

                # --- problem with invalid ESO header keywords
                if 'ESO DET CHIP PXSPACE' in head:
                    del head['ESO DET CHIP PXSPACE']

                try:
                    fits.writeto(fout, outim, head, overwrite=True)

                except:

                    e = sys.exc_info()

                    msg = (funname + ': ERROR: Could not write fits: ' + str(i) + ', '
                           + fin + ":\n" + str(e[1]) + ' '
                           + str(traceback.print_tb(e[2])))
                    _print_log_info(msg, logfile)


                    outline += (',,,\n')
                    outline = outline.replace(instrument+',', "CANNOT WRITE,")

                    f.write(outline)


                    if hdu is not None:
                       hdu.close()

                    continue


            fout = fout.replace(".fits", ".png")
            _simple_image_plot(outim, fout, percentile=1, pwidth=6)


        # --- background estimation on the final product:
        # bgim = _subtract_source(outim[sky_yrange[0]:sky_yrange[1],
        #                                sky_xrange[0]:sky_xrange[1]])

        bgim = sigma_clip(outim[sky_yrange[0]:sky_yrange[1],
                                sky_xrange[0]:sky_xrange[1]],
                          sigma=3, maxiters=3, masked=False)

        difmed = np.nanmedian(bgim)

        difstd = np.nanstd(bgim)

        outline += (',' + "{:.2f}".format(difmed)
                    + ',' + "{:.2f}".format(difstd)
                    +',\n'
                   )

        # --- write completel line into table
        f.write(outline)

    f.close()
