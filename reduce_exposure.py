#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.2.0"

"""
USED BY:
    - reduce_obs


HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-11: updates as result of changes in get_std_flux and dpro
                  becoming a masked array
    - 2020-02-12: change to simple_image_plot, fix truncated fits header
                  comments, bug in all_blind plot, updated error and warning
                  budget, added silent for get_std_flux


NOTES:
    -

TO-DO:
    - change the make_gallery calls to a simpler package internal routine and
      uncomment the corresponding section in the main routine below
    - implement WCS for the extracted images
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import sys
import traceback
from tqdm import tqdm
from astropy.io import fits
from astropy.stats import sigma_clip

from .calc_beampos import calc_beampos as _calc_beampos
from .crop_image import crop_image as _crop_image
from .find_beam_pos import find_beam_pos as _find_beam_pos
from .find_source import find_source as _find_source
from .fits_get_info import fits_get_info as _fits_get_info
from .get_std_flux import get_std_flux as _get_std_flux
from .measure_sensit import measure_sensit as _measure_sensit
from .merge_fits_files import merge_fits_files as _merge_fits_files
from .print_log_info import print_log_info as _print_log_info
from .simple_image_plot import simple_image_plot as _simple_image_plot
# from .subtract_source import subtract_source as _subtract_source
from .reduce_burst_file import reduce_burst_file as _reduce_burst_file
from .undo_jitter import undo_jitter as _undo_jitter
from . import visir_params as _vp

def update_base_params(im, head, dpro, pfov, expid, fitbox=50, onlyhead=False):

    # --- fit the total image with a Gaussian
    params, perrs, fitims = _find_source(im, method='mpfit', fitbox=fitbox)

    # --- total counts of the Gaussian
    tot = np.sum(fitims[1]-params[0])

    # --- exposure time
    exptime = dpro['exptime'][expid]
    head["HIERARCH ESO QC EXPTIME"] = exptime

    head["FIT-BG"] =  (params[0], "Central source fit background estimate [cnts]")
    head["FIT-AMP"] = (params[1], "Centr. sour. Gaussian amplitude [cnts]")
    head["FIT-TOT"] = (tot, "Centr. sour. Gauss. total flux [cnts]")
    head["FIT-MAF"] = (params[4]*pfov, "Centr. sour. Gauss. major axis FWHM [as]")
    head["FIT-MIF"] = (params[5]*pfov, "Centr. sour. Gauss. minor axis FWHM [as]")
    head["FIT-PA"] = (params[6], "Centr. sour. Gauss. position angle E of N [deg]")

    if not onlyhead:
        dpro['GF_medBG'][expid] = params[0]
        dpro['GF_peakampl'][expid] = params[1]
        dpro['GF_totcnts'][expid] = tot
        dpro['GF_majFWHM_as'][expid] = params[4]*pfov
        dpro['GF_minFWHM_as'][expid] = params[5]*pfov
        dpro['GF_posang'][expid] = params[6]

    # --- image belongs to a standard star?
    if dpro['STDflux_Jy'][expid] > 0:

        std = True

        # --- standard star flux density in Jy
        flux = dpro['STDflux_Jy'][expid]

        # --- conversion factor
        conv = 1.e3 * flux / tot
        head["FIT-CONV"] = (conv, "Centr. sour. Gauss. conversion [mJy/ADU/DIT]")

        if not onlyhead:
            dpro['GF_mJy/ADU/DIT'][expid] = conv

    else:
        std = False
        flux = None

    # --- compute the best S/N similar to  to ESO VISIR pipeline
    try:
        snbest, sens = _measure_sensit(im=im, flux=flux, showplot=False,
                                      std=std, exptime=exptime)
    except:
        snbest = np.ma.masked
        sens = np.ma.masked

    head["BEST-SN"] = (snbest, "Best S/N of ESO-like aper phot")
    if not onlyhead:
        dpro['bestSN'][expid] = snbest

    if not np.ma.is_masked(sens):
        head["SENSIT"] =(sens, "10 sigma in 1h sensitivity [mJy]")

        if not onlyhead:
            dpro["sensit_mJy"][expid] = sens


#%%
# --- Helper routine to reduce burst exposure
def reduce_burst_exp(logfile, filenames, rawfolder, burstfolder, ditsoftaver,
                     overwrite, noddir, bsumfolder, draw, rid, noe, box,
                     AA_pos, alignmethod, chopsubmeth, refpos, verbose,
                     searchsmooth, debug, plot, crossrefim, outfolder,
                     outnames, head0, dpro, pfov, expid):
    """
    Internal helper routine to reduce (VISIR) burst mode exposures.
    Used by reduce_exp
    """

    funname = "REDUCE_BURST_EXP"

    # --- 3.1 First reduce the individual files with beam aligning
    msg = (" - Reducing individual raw cubes...")
    _print_log_info(msg, logfile)

    nf = len(filenames)

    for i in range(nf):

        msg = (" - i, File: " + str(i)  + ", " + filenames[i] )
        _print_log_info(msg, logfile)

        fin = rawfolder + '/' + filenames[i]

        fout = burstfolder + "/" + filenames[i].replace(".fits", "_median.fits")
        if ditsoftaver > 1:
            fout = fout.replace(".fits", "_aver" + str(ditsoftaver) + ".fits")

        if overwrite == False and os.path.isfile(fout):

            msg = (funname + ": File already reduced. Continue...")
            _print_log_info(msg, logfile)
            continue

        # --- first find the off-nod image but only for perpendicular nodding
        if noddir == "PERPENDICULAR":
            k = i + (-1)**i
            offnodfile = bsumfolder + '/' + filenames[k].replace(".fits", "_blindsum.fits")

            # --- double check that the offnodfile really has a different nod position
            if draw['NODPOS'][rid[i]] == draw['NODPOS'][rid[k]]:
                msg = (funname + ": ERROR: wrong offnod file provided: \n"
                        + filenames[k] + "\nContinue with next file...")

                noe = noe + 1
                _print_log_info(msg, logfile)
                continue

        else:
            offnodfile = None

        reffile = bsumfolder + '/' + filenames[i].replace(".fits", "_blindsum.fits")

        try:
            _reduce_burst_file(fin, outfolder=burstfolder, logfile=logfile,
                              offnodfile=offnodfile, reffile=reffile, box=box,
                              chopsubmeth=chopsubmeth, refpos=refpos,
                              AA_pos=AA_pos, alignmethod=alignmethod,
                              verbose=verbose, searchsmooth=searchsmooth,
                              crossrefim=crossrefim, ditsoftaver=ditsoftaver,
                              debug=debug, plot=plot)

        except:
            e = sys.exc_info()
            msg = (funname + ": ERROR: Burst reduction failed: \n"
               + str(e[1]) + '' + str(traceback.print_tb(e[2]))
               + "\nContinue with next file...")

            noe = noe + 1

            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['noerr'][expid] = dpro['noerr'][expid] + 1

            continue


    # --- 3.2 Combine aligned cubes
    msg = funname + ": Combining cubes..."
    _print_log_info(msg, logfile, empty=1)

    medfins = []
    meanfins = []

    for i in range(nf):

       meanfins.append(burstfolder + "/" +
                       filenames[i].replace(".fits", "_mean.fits"))

       medfins.append(burstfolder + "/" +
                      filenames[i].replace(".fits", "_median.fits"))


    # --- 3.2.1 first do a blind merge of the individual nods
    msg = funname + ": Performing blind merge..."
    _print_log_info(msg, logfile)

    aim = _merge_fits_files(infiles=meanfins, method='mean', northup=True,
                           verbose=verbose, plot=plot, alignmethod=None)
    fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
            '_nonodal_meanall.fits')

    update_base_params(aim, head0, dpro, pfov, expid)

    fits.writeto(fout, aim, head0, overwrite=True)
    fout = fout.replace(".fits", ".png")
    _simple_image_plot(aim, fout, scale="log", cenax=True, pfov=pfov)

    msg = (" - Output written: " + fout)
    _print_log_info(msg, logfile)


    aim = _merge_fits_files(infiles=medfins, method='median', northup=True,
                           verbose=verbose, alignmethod=None)

    update_base_params(aim, head0, dpro, pfov, expid)

    fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
            '_nonodal_medall.fits')
    fits.writeto(fout, aim, head0, overwrite=True)
    fout = fout.replace(".fits", ".png")
    _simple_image_plot(aim, fout, scale="log", cenax=True, pfov=pfov)

    msg = (" - Output written: " + fout)
    _print_log_info(msg, logfile)

    # --- 3.2.2 then try the same alignment method that was used for the individual frames
    msg = funname + ": Performing aligned merge..."
    _print_log_info(msg, logfile)

    try:
        aim = _merge_fits_files(infiles=meanfins, method='mean', northup=True,
                           verbose=verbose, plot=plot, alignmethod=alignmethod)
        fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                '_meanall.fits')

        update_base_params(aim, head0, dpro, pfov, expid)

        fits.writeto(fout, aim, head0, overwrite=True)
        fout = fout.replace(".fits", ".png")
        _simple_image_plot(aim, fout, scale="log", cenax=True, pfov=pfov)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)


        aim = _merge_fits_files(infiles=medfins, method='median', northup=True,
                           verbose=verbose, alignmethod=alignmethod)

        update_base_params(aim, head0, dpro, pfov, expid)

        fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                '_medall.fits')
        fits.writeto(fout, aim, head0, overwrite=True)
        fout = fout.replace(".fits", ".png")
        _simple_image_plot(aim, fout, scale="log", cenax=True, pfov=pfov)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)

    except:

        e = sys.exc_info()
        msg = (funname + ":  - ERROR: Aligning of individual nods failed: \n"
               + str(e[1]) + '' + str(traceback.print_tb(e[2])))

        _print_log_info(msg, logfile)

        noe += 1

        if dpro is not None:
            dpro['noerr'][expid] = dpro['noerr'][expid] + 1

    return(noe)


#%%

# === 3. DO THE REDUCTION OF INDIVIDUAL EXPOSURES
def reduce_exposure(rawfiles=None, draw=None, dpro=None, expid=None, sof=None,
                    temprofolder=None, rawfolder=None, logfile=None,
                    outname='VISIR_OBS', outfolder='.', overwrite=False,
                    maxshift=10, extract=True, searcharea='chopthrow',
                    box=None, statcalfolder=None, crossrefim=None,
                    chopsubmeth='averchop', AA_pos=None, refpos=None,
                    alignmethod='fastgauss', findbeams=True,
                    verbose=False, searchsmooth=3, ditsoftaver=1,
                    sky_xrange=None, plot=False, insmode=None,
                    sky_yrange=None, debug=False, instrument=None):

    """
    main reduction routine: reduce a give exposure by combining the nods,
    perform alignment of DITs/chops in case of burst/half-cycle data and
    extract source images

    PARAMETERS:
     - rawfiles: (optional) list of raw fits files belonging to the exposure to reduce
     - draw: (optional) data structure containing all the raw file information as result of the routine reduce_indi_raw_files
     - dpro: (optional) data structure containg all the exposure information as a product of the routine group_files_into_observations
     - temprofolder: (optional) folder containing the temporary products, i.e., individually reduced files. If 'None', the output folder is taken
     - rawfolder: (optional) folder containing the raw fits files to be reduced
     - outname: (default='VISIR_OBS') name prefix for the logfile
     - outfolder: (default='.') folder to write output into
     - overwrite: (default=False) overwrite existing data?
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

    NOT FULLY IMPLEMENTED:
     - call as a stand-alone routine, i.e., without providing draw and dpro
       This would require implementation extracting files from the sof.txt as
       well as reduction of the raw files with reduce_indi_raws

    TO BE ADDED LATER:
        - correct treatment of HR and HRX SPC
        - correct treatment of SPC ACQ data
        - proper modification of the headers of the products
        - de-rotation for classical imaging with pupil tracking (burst should work)
    """

    funname = "REDUCE_EXPOSURE"

    if debug:
        verbose = True

    noe = 0  # number of error counter

    if temprofolder is None:
        temprofolder = outfolder

    # --- create output folders
    hcycfolder = temprofolder+'/hcycles'
    bsumfolder = hcycfolder+'/blindsums'
    nodfolder = temprofolder+'/nods'

    if not os.path.exists(hcycfolder):
        os.makedirs(hcycfolder)

    if not os.path.exists(bsumfolder):
        os.makedirs(bsumfolder)

    if not os.path.exists(nodfolder):
        os.makedirs(nodfolder)

    # --- test whether output folder exists
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if logfile is None:
        logfile = outfolder + '/' + outname + ".log"
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info(funname + ": Reduction of " + outname, logfile, mode=mode)


    # --- read some important exposure parameters:
    # --- if dpro is provided that is easy
    if dpro is not None:

        if instrument is None:
            instrument = dpro["instrument"][expid]

        if insmode is None:
            insmode = dpro["tempname"][expid].split("_")[0]

        targname = dpro["targname"][expid]
        datatype = dpro["datatype"][expid]
        tempname = dpro["tempname"][expid]
        setup = dpro["setup"][expid]
        pfov = dpro["pfov"][expid]
        chopthrow = dpro["chopthrow"][expid]
        noddir = dpro["noddir"][expid]
    else:
        targname = None
        datatype = None
        tempname = None
        setup = None
        pfov = None
        chopthrow = None
        noddir = None

    # --- if the table with the raw files is provided get them from there
    if expid is not None and draw is not None:
        rid = np.where(draw['EXPID'] == expid)[0]
        nf = len(rid)
        filenames = draw['filename'][rid]

        # --- abort reduction if we have any corrput files
        if 'CORRUPT' in draw['DATE-OBS'][rid]:
            msg = funname + ": ERROR: This observations contains corrupt files! Aborting..."
            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['noerr'][expid] = dpro['noerr'][expid] + 1

            return(1)

        # # --- in addition we can get some
        # if tempname is None:
        #     tempname = draw['TPL_ID'][rid[0]]
        # if insmode is None:
        #     insmode = draw['insmode'][rid[0]
        # #fram_format = draw['FRAM_FORMAT'][rid[0]]

    # --- or in case a sof file in esorex style is provided
    elif sof is not None:
        fsof = open(sof)
        # --- implement extractio of the fits files here
        # rawfiles =


    # --- if the raw files provided
    if rawfiles is not None:
        nf = len(rawfiles)
        rid = np.arange(0,nf)
        filenames = rawfiles.split('/')[-1]

        if rawfolder is None:
            rawfolder = rawfiles.split('/')[0:-1]

    # --- check that we actually have some files
    if nf == 0:
        msg = funname + ": ERROR: No corresponding raw files found! Aborting..."
        _print_log_info(msg, logfile)

        if dpro is not None:
            dpro['noerr'][expid] = dpro['noerr'][expid] + 1
        return(1)

    # --- TO BE IMPLEMENTED: check if blindsums are available or whether this needs to be done
    # ....

    # --- get the first head
    f0 = bsumfolder + '/' + filenames[0].replace(".fits", "_blindsum.fits")
    print(f0)
    hdu = fits.open(f0)
    head0 = hdu[0].header
    hdu.close()

    # --- in case we still don't have them, get some key parameters
    if targname is None:
        targname = _fits_get_info(head0, "target")

    if instrument is None:
        instrument = _fits_get_info(head0, "instrument")

    if insmode is None:
        insmode = _fits_get_info(head0, "insmode")

    if tempname is None:
        tempname = _fits_get_info(head0, "TPL ID")

    if datatype is None:
        datatype = _fits_get_info(head0, "datatype")

    if pfov is None:
        pfov = _fits_get_info(head0, "pfov")

    if setup is None:
        setup = _fits_get_info(head0, "setup")

    if chopthrow is None:
        chopthrow = _fits_get_info(head0, "CHOP THROW")

    if noddir is None:
        noddir = _fits_get_info(head0, "CHOPNOD DIR")


    # --- compute the maximum field of view from half of
    if not box:
        box = int(np.floor(chopthrow / pfov))

    # --- check if target is a calibrator
    if insmode != "SPC":

        if "cal" in tempname:
            should_be_cal = True
            silent = False
        else:
            should_be_cal = False
            silent = True

        # --- first just check if the target is found in the calibrator table
        #     even if it was observed with a science template
        try:
            flux = _get_std_flux(head0, logfile=logfile, silent=silent)
        except:
            flux = -1

        # --- if a flux was found update the table and header
        if flux != -1:
            head0["HIERARCH ESO QC JYVAL"] = (flux, "STD flux density in filter used [Jy]")
            if dpro is not None:
                dpro['STDflux_Jy'][expid] = flux

        # --- if it was not found but the observation was with a cal template
        #     raise en error
        elif should_be_cal:
            msg = (funname + ": WARNING: target not found in flux reference table")
            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

            flux = -1
    else:
        flux = -1


    msg =  " - Target name: " + str(targname) + "\n"

    if flux > 0:
        msg += " - STD flux [Jy]: " + "{:.2f}".format(flux) + "\n"

    msg += (
            " - Instrument: " + str(instrument) + "\n" +
            " - Instrument mode: " + str(insmode) + "\n" +
            " - Template name: " + str(tempname) + "\n" +
            " - PFOV [as]: " + str(pfov) + "\n" +
            " - Setup: " + str(setup) + "\n" +
            " - Data type: " + str(datatype) + "\n" +
            " - Nod direction: " + str(noddir) + "\n" +
            " - Chop throw [as]: " + str(chopthrow) + "\n" +
            " - Max box size [px]: "+str(box)
            )

    _print_log_info(msg, logfile, logtime=False)


    # === 1 Go over all files of the exposure and create nodding pairs
    msg = funname + ": Creating blind nodding pairs..."
    _print_log_info(msg, logfile, empty=1)

    nodims = []
    heads = []
    outnames = []

    ima = None
    imb = None
    heada = None
    # headb = None

    for i in range(nf):
        msg = (" - i, File: " + str(i)  + ", " + filenames[i] )
        _print_log_info(msg, logfile)


        fsingred = (bsumfolder + '/' +
                    filenames[i].replace(".fits", "_blindsum.fits"))

        # --- read in the data
        hdu = fits.open(fsingred)
        head = hdu[0].header
        im = hdu[0].data
        hdu.close()

        # --- nod pos A or B?
        if draw['NODPOS'][rid[i]] == 'A':
            ima = im
            heada = head
        else:
            imb = im

        # print(draw['NODPOS'][rid[i]], ima is not None, imb is not None)

        # === 1.1 combine files to nodding pairs
        if (i > 0) & (ima is not None) & (imb is not None):

            # --- build the nod subtracted image
            imc = ima - imb

            ind = int(np.floor(i/2))

            fout = (nodfolder + '/' + outname + '_nod'
                        + "{:02.0f}".format(ind) + '.fits')

            if overwrite or not os.path.isfile(fout):
                fits.writeto(fout, imc, heada, overwrite=True)
                pout = fout.replace(".fits", "_per1.png")
                _simple_image_plot(imc, pout, percentile=1, pwidth=6)
                pout = fout.replace(".fits", "_minmax.png")
                _simple_image_plot(imc, pout, percentile=0.0001, pwidth=6)

                msg = (" - Output written: " + fout)
                _print_log_info(msg, logfile)

            # --- collect for further reduction
            nodims.append(imc)
            outnames.append(outname)
            heads.append(heada)

            ima = None
            imb = None
            heada = None

    msg = funname + ": Number of nodding pairs reduced: " + str(len(nodims))
    _print_log_info(msg, logfile)


    # === 2 Combination of the nodding pairs
#    if len(heads) == 0:   # --- no idea why this was here???
#        continue
    msg = funname + ": Merging blind nodding pairs..."
    _print_log_info(msg, logfile, empty=1)

    nodims = np.array(nodims)
    nnods = len(nodims)

    # --- 2.1 do jitter correction
    if 'HIERARCH ESO SEQ JITTER WIDTH' in head0:
        if head0['HIERARCH ESO SEQ JITTER WIDTH'] > 0:
            for i in range(nnods):
                # joff = V.compute_jitter(head=heads[iddd[m]])
                # print("      - Nod no / jitter offs.: ",m,joff)
                nodims[i] = _undo_jitter(ima=nodims[i], head=heads[i])

    # msg = funname + ": 1. simple combination"
    # _print_log_info(msg, logfile)

    totim = np.mean(nodims, axis=0)
    fout = (outfolder + '/' + outnames[0] + '_all_blind.fits')

    # --- if no specfic range is supplied to measure the sky use the full
    #     illuminated area of the VISIR detector
    if sky_xrange is None:
        sky_xrange = _vp.max_illum_xrange

    if sky_yrange is None:
        sky_yrange = _vp.max_illum_yrange

    # --- background estimation:
    # bgim = _subtract_source(totim[sky_yrange[0]:sky_yrange[1],
    #                                sky_xrange[0]:sky_xrange[1]])

    bgim = sigma_clip(totim[sky_yrange[0]:sky_yrange[1],
                        sky_xrange[0]:sky_xrange[1]],
                  sigma=3, maxiters=3, masked=False)

    dpro["BGmed"][expid] = np.nanmedian(bgim)

    dpro["BGstd"][expid] = np.nanstd(bgim)

    msg = ( " - BG median [ADU/DIT]: " + str(dpro["BGmed"][expid]) + "\n" +
            " - BG STDDEV [ADU/DIT]: " + str(dpro["BGstd"][expid])
           )
    _print_log_info(msg, logfile, logtime=False)


    if overwrite or not os.path.isfile(fout):
        fits.writeto(fout, totim, head0, overwrite=True)
        pout = fout.replace(".fits", "_per1.png")
        _simple_image_plot(totim, pout, percentile=1, pwidth=6)
        pout = fout.replace(".fits", "_minmax.png")
        _simple_image_plot(totim, pout, percentile=0.0001, pwidth=6)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)


    # === 3 BURST & CYCSUM data ===
    if datatype == "halfcyc" or datatype == "burst":

        msg = (funname + ": Burst/halfcyc data encountered...")
        _print_log_info(msg, logfile, empty=1)

        burstfolder = hcycfolder+'/'+alignmethod

        if not os.path.exists(burstfolder):
            os.makedirs(burstfolder)

        try:
            noe = reduce_burst_exp(logfile, filenames, rawfolder, burstfolder,
                         ditsoftaver,overwrite, noddir, bsumfolder, draw,
                         rid, noe, box, AA_pos, alignmethod, chopsubmeth,
                         refpos, verbose, searchsmooth, debug, plot,
                         crossrefim, outfolder, outnames, head0, dpro, pfov,
                         expid)
        except:
            e = sys.exc_info()
            msg = (funname + ": ERROR: Burst reduction failed!")

            noe = noe + 1

            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['noerr'][expid] = dpro['noerr'][expid] + 1



    # === 4 Automatic source extraction for imaging
    if extract and insmode != 'SPC':

        # ---- WARNING: de-rotation for classical imaging with pupil tracking still to be implemented!

        # --- first find the source positions
        if findbeams:

            msg = funname + ": Trying to detect beams in combined image..."
            _print_log_info(msg, logfile, empty=2)

            beamsfound = True
            try:
                beampos, fitparams = _find_beam_pos(im=totim, head=head0,
                                                    searcharea=searcharea,
                                                    fitbox=0.5*box,
                                                    nodpos='both',
                                                    verbose=verbose,
                                                    AA_pos=AA_pos,
                                                    plot=plot,
                                                    instrument=instrument,
                                                    insmode=insmode,
                                                    logfile=logfile,
                                                    chopthrow=chopthrow,
                                                    noddir=noddir,
                                                    filt=setup,
                                                    pfov=pfov)

            except:
                e = sys.exc_info()
                msg = (funname + ": WARNING: Beam Position could not be found: \n"
                       + str(e[1]) + ' ' + str(traceback.print_tb(e[2]))
                       + "\nContinue assuming the positions...")
                _print_log_info(msg, logfile)

                if dpro is not None:
                        dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                beamsfound = False


            if beamsfound:

                nbeams = len(beampos)
                bg = fitparams[0]
                amp = np.abs(fitparams[1])
                fwhm = np.mean(fitparams[4:6])
                axrat = np.max(fitparams[4:6])/np.min(fitparams[4:6])
                total = 0.25 * (np.pi * fitparams[1] * fitparams[4] * fitparams[5])/np.log(2.0)
                angle = fitparams[6]
                msg = (" - Found source params:\n" +
                       "     - BG: "+  str(bg) + "\n" +
                       "     - Amplitude: "+  str(amp) + "\n" +
                       "     - Total: "+ str(total) + "\n" +
                       "     - Aver. FWHM [px]: "+  str(fwhm) + "\n" +
                       "     - Aver. FWHM [as]: "+  str(fwhm*pfov) + "\n" +
                       "     - Maj./min axis: "+  str(axrat) + "\n" +
                       "     - Angle: "+  str(angle)
                       )

                _print_log_info(msg, logfile)

                if -1 in beampos:
                    msg = (funname + ": ERROR: Not all expected beams were found! No source extraction possible...")
                    _print_log_info(msg, logfile)

                    if dpro is not None:
                        dpro['noerr'][expid] = dpro['noerr'][expid] + 1

                    return(noe+1)

        # --- if beams are not found or looked for estimate their position
        if not findbeams or not beamsfound:

            # --- compute the expected beam positions
            beampos = _calc_beampos(head=head0, verbose=verbose)
            nbeams = len(beampos)
            fwhm = 7
            bg = np.nanmedian(totim)
            amp = np.nanmax(gaussian_filter(totim, sigma=3))

            # --- Modify the calculated positions in case the user provided the beam
            #     position of chop A nod A
            if AA_pos is not None:
                xdif = AA_pos[1] - beampos[0,1]
                ydif = AA_pos[0] - beampos[0,0]

                beampos[:,1] = beampos[:,1] + xdif
                beampos[:,0] = beampos[:,0] + ydif

        subims = []
        titles = []

        beamposstr = ""
        for b in range(nbeams):
            beamposstr = (beamposstr + "{:3.0f}".format(beampos[b,0]) + ", "
                          + "{:3.0f}".format(beampos[b,1]) + " | ")

        msg = (
               " - No of beams: " + str(nbeams) + "\n" +
               " - Beampos: " + beamposstr + "\n"
               )

        _print_log_info(msg, logfile)

        # === 4.1 Blind addition
        # --- go over all nodding pairs and extract all the sub-images
        #     first blindly
        msg = (funname +
               ": Extracting beams from individual nods at fixed position ...")

        _print_log_info(msg, logfile, empty=1)

        for j in tqdm(range(nnods)):
            for k in range(nbeams):

                # --- check that the beam is indeed on the detector
                s = np.shape(nodims[j])
                if ((beampos[k,0] > s[0]) | (beampos[k,1] > s[1])
                    | (beampos[k,0] < 0) | (beampos[k,1] < 0)):

                    if verbose:
                        msg = (" - WARNING: Beampos not on frame: " +
                               str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
                               ". Exclude...")

                        _print_log_info(msg, logfile)

                    if dpro is not None:
                        dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    continue

                beamsign = (-1)**(np.ceil(0.5*k))

                subims.append(_crop_image(nodims[j]*beamsign, box=box,
                                           cenpos=beampos[k,:], silent=True)
                              )

                titles.append("Nod: " + str(j) + " Beam: " + str(k))

        # --- write out
        fout = (outfolder + '/' + outnames[0] + '_all_blind_extr_cube.fits')
        fits.writeto(fout, subims, head0, overwrite=True)

#        fout = fout.replace(".fits", "_log.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=True,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')
#
#        fout = fout.replace("_log.png", "_lin.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=False,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')

        # --- take into account that in the case of parallel nodding the
        #     the central image has to be counted double.
        #     The first beam is always the double.
        if noddir == "PARALLEL":
            subims[0] = 0.5 * subims[0]
            subims.append(subims[0])


        totim = np.mean(subims, axis=0)

        if datatype == "cycsum":
            onlyhead = False
        else:
            onlyhead = True

        try:
            update_base_params(totim, head0, dpro, pfov, expid, onlyhead=onlyhead)
        except:
            e = sys.exc_info()
            msg = (funname + ": ERROR: Failed to update base parameters: \n"
                   + str(e[1]) + '' + str(traceback.print_tb(e[2])))
            _print_log_info(msg, logfile)
            noe = noe + 1
            if dpro is not None:
                dpro['noerr'][expid] = dpro['noerr'][expid] + 1

        fout = (outfolder + '/' + outnames[0] + '_all_blind_extr.fits')
        fits.writeto(fout, totim, head0, overwrite=True)

        pout = fout.replace(".fits", "_log.png")
        _simple_image_plot(totim, pout, scale="log", pfov=pfov, cenax=True)

        pout = fout.replace(".fits", "_lin.png")
        _simple_image_plot(totim, pout, pfov=pfov, cenax=True)

        # --- to be uncommented once the make_gallery routine has been changed
        # try:
        #     fout = fout.replace(".fits", "_log.png")
        #     I.make_gallery(ims=[totim], outname=fout, pfovs=pfov, log=True,
        #                 papercol=1, ncols=1, cmap='gnuplot2',
        #                 inv=False, permin=40, permax=99.9, latex=False)

        #     fout = fout.replace("_log.png", "_lin.png")
        #     I.make_gallery(ims=[totim], outname=fout, pfovs=pfov, log=False,
        #                 papercol=1, ncols=1, cmap='gnuplot2',
        #                 inv=False, permin=40, permax=99.9, latex=False)

        # except:
        #     e = sys.exc_info()
        #     msg = (funname + ": ERROR:  Gallery plots failed to be created: \n"
        #            + str(e[1]) + '' + str(traceback.print_tb(e[2])))
        #     _print_log_info(msg, logfile)
        #     noe = noe + 1


        msg = " - Output written: " + fout
        _print_log_info(msg, logfile)


        # --- if no beams were found here we are done here
        if not findbeams or not beamsfound:
            return(noe)


        # === 4.2 Optional Fine-centered addition
        # --- go over all nodding pairs and extract all the sub-images
        msg = funname + ": Trying to detect and extract beams from individual nods..."
        _print_log_info(msg, logfile, empty=2)

        subims = []
        titles = []
        for j in tqdm(range(nnods)):
            for k in range(nbeams):


                # --- check that the beam is indeed on the detector
                s = np.shape(nodims[j])
                if ((beampos[k,0] > s[0]) | (beampos[k,1] > s[1])
                    | (beampos[k,0] < 0) | (beampos[k,1] < 0)):

                    if verbose :
                        msg = (" - WARNING: Beampos not on frame: " +
                               str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
                               ". Exclude...")

                        _print_log_info(msg, logfile)

                    # if dpro is not None:
                    #     dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    continue



                beamsign = (-1)**(np.ceil(0.5*k))

                try:
                    params, _, _ = _find_source(nodims[j]*beamsign,
                                                 guesspos=beampos[k,:],
                                                 searchbox=box,
                                                 fitbox=int(np.max([6*fwhm,0.25*box])),
                                                 method='mpfit',
                                                 guessbg=bg, guessamp=amp,
                                                 guessFWHM=fwhm,
                                                 minamp=0.3*amp,
                                                 minFWHM=0.5*fwhm,
                                                 maxFWHM=2*fwhm)

                except:

                    if verbose:
                        msg = (" - WARNING: Could not find source. Continuing...")

                        _print_log_info(msg, logfile)

                    # if dpro is not None:
                    #     dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    continue

#                print('params: ', params)
#                print('params[2:4]', params[2:4])
#                print('beamsign: ', beamsign)
#                print('box: ', box)
#                print('i,j: ',i,j)

                cim = _crop_image(nodims[j]*beamsign, box=box,
                                           cenpos=params[2:4], silent=True)

                dist = beampos[k,:] - params[2:4]

                if verbose:
                    msg = (" - Nod: " + str(j) + " Beam: " + str(k) +
                           " Detected shift: " + str(dist)
                           )

                    _print_log_info(msg, logfile)


                    bg = params[0]
                    amp = np.abs(params[1])
                    fwhm = np.mean(params[4:6])
                    axrat = np.max(params[4:6])/np.min(params[4:6])
                    total = 0.25 * (np.pi * params[1] * params[4] * params[5])/np.log(2.0)
                    angle = params[6]
                    msg = (" - Found source params:\n" +
                           "     - BG: "+  str(bg) + "\n" +
                           "     - Amplitude: "+  str(amp) + "\n" +
                           "     - Total: "+ str(total) + "\n" +
                           "     - Aver. FWHM [px]: "+  str(fwhm) + "\n" +
                           "     - Aver. FWHM [as]: "+  str(fwhm*pfov) + "\n" +
                           "     - Maj./min axis: "+  str(axrat) + "\n" +
                           "     - Angle: "+  str(angle)
                          )

                    _print_log_info(msg, logfile)


                if np.sqrt(dist[0]**2 + dist[1]**2) > maxshift:

                    if verbose:
                        msg = (" - WARNING: shift too large! Exclude frame")
                        _print_log_info(msg, logfile)
                    # if dpro is not None:
                    #     dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    # noe = noe+1
                    continue

                subims.append(cim)
                titles.append("Nod: " + str(j) + " Beam: " + str(k))

        # --- if object not detected in individual nods abort here
        if len(subims) == 0:
            msg = (funname +
                   ": WARNING: No source detected in the individual nods! Exiting... ")
            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

            return(noe)

        # -- also if <50% of the beams were found abort
        elif len(subims) < 0.5 * nnods * nbeams:
            msg = (funname +
                   ": WARNING: <50% of the beams detected in individual nods! Exiting... ")
            _print_log_info(msg, logfile)

            if dpro is not None:
                dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

            return(noe)

        # --- id something is detected continued with writing out results
        fout = (outfolder + '/' + outnames[0] + '_all_extr_cube.fits')
        fits.writeto(fout, subims, head0, overwrite=True)

#        fout = fout.replace(".fits", "_log.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=True,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')
#
#        fout = fout.replace("_log.png", "_lin.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=False,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')

        if noddir == "PARALLEL":
            subims[0] = 0.5 * subims[0]
            subims.append(subims[0])

        totim = np.mean(subims, axis=0)

        update_base_params(totim, head0, dpro, pfov, expid, onlyhead=onlyhead)

        fout = (outfolder + '/' + outnames[0] + '_all_extr.fits')
        fits.writeto(fout, totim, head0, overwrite=True)



        pout = fout.replace(".fits", "_log.png")
        _simple_image_plot(totim, pout, scale="log", pfov=pfov, cenax=True)

        pout = fout.replace(".fits", "_lin.png")
        _simple_image_plot(totim, pout, pfov=pfov, cenax=True)

        # --- to be uncommented once the make_gallery routine has been changed
        # try:
        #     fout = fout.replace(".fits", "_log.png")
        #     I.make_gallery(ims=[totim], outname=fout, pfovs=pfov, log=True,
        #                    papercol=1, ncols=1, cmap='gnuplot2',
        #                    inv=False, permin=40, permax=99.9, latex=False)

        #     fout = fout.replace("_log.png", "_lin.png")
        #     I.make_gallery(ims=[totim], outname=fout, pfovs=pfov, log=False,
        #                    papercol=1, ncols=1, cmap='gnuplot2',
        #                    inv=False, permin=40, permax=99.9, latex=False)

        #     msg = (" - Output written: " + fout)
        #     _print_log_info(msg, logfile)


        # except:
        #     e = sys.exc_info()
        #     msg = (funname + ": ERROR:  Gallery plots failed to be created: \n"
        #            + str(e[1]) + '' + str(traceback.print_tb(e[2])))
        #     _print_log_info(msg, logfile)
        #     noe = noe + 1

        msg = " - Output written: " + fout
        _print_log_info(msg, logfile)

    return(noe)

