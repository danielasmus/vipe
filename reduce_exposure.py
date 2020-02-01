#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-23: created by Daniel Asmus


NOTES:
    -

TO-DO:
    - change the make_gallery calls to a simpler package internal routine and
      uncomment the corresponding section in the main routine below
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import sys
import traceback
from astropy.io import fits

from .calc_beampos import calc_beampos as _calc_beampos
from .crop_image import crop_image as _crop_image
from .find_beam_pos import find_beam_pos as _find_beam_pos
from .find_source import find_source as _find_source
from .get_std_flux import get_std_flux as _get_std_flux
from .measure_sensit import measure_sensit as _measure_sensit
from .merge_fits_files import merge_fits_files as _merge_fits_files
from .print_log_info import print_log_info as _print_log_info
from .simple_image_plot import simple_image_plot as _simple_image_plot
from .subtract_source import subtract_source as _subtract_source
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

    head["FIT-BG"] =  (params[0], "Central Source Fit background estimate [cnts]")
    head["FIT-AMP"] = (params[1], "Central Source Fit amplitude of the Gaussian [cnts]")
    head["FIT-TOT"] = (tot, "Central Source Fit total flux of the Gaussian [cnts]")
    head["FIT-MAF"] = (params[4]*pfov, "Central Source Fit Gaussian major axis FWHM [as]")
    head["FIT-MIF"] = (params[5]*pfov, "Central Source Fit Gaussian minor axis FWHM [as]")
    head["FIT-PA"] = (params[6], "Central Source Fit Gaussian position angle E of N [deg]")

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
        head["FIT-CONV"] = (conv, "Central Source Fit Gaussian Conversion factor [mJy/ADU/DIT]")

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
        snbest = -1
        sens = -1

    head["BEST-SN"] = (snbest, "Best S/N from aperture phot following ESO pipeline")
    if not onlyhead:
        dpro['bestSN'][expid] = snbest

    if sens != -1:
        head["SENSIT"] =(sens, "10 sigma in 1h sensitivity [mJy]")

        if not onlyhead:
            dpro["sensit_mJy"][expid] = sens




#%%

# === 3. DO THE REDUCTION OF INDIVIDUAL EXPOSURES
def reduce_exposure(rawfiles=None, draw=None, dpro=None, expid=None,
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

    TO BE ADDED LATER:
        - correct treatment of HR and HRX SPC
        - correct treatment of SPC ACQ data
        - proper modification of the headers of the products
        - de-rotation for classical imaging with pupil tracking (burst should work)
    """

    if debug:
        verbose = True

    noe = 0  # number of errors encountered

    if temprofolder is None:
        temprofolder = outfolder

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

    if statcalfolder is None:
        home = os.path.expanduser("~")
        statcalfolder = home + '/work/common_routines/python'

    if logfile is None:
        logfile = outfolder + '/' + outname + ".log"
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info("Reduction of " + outname, logfile, mode=mode)


    # --- read some important exposure parameters:
    if dpro is not None:
        datatype = dpro["datatype"][expid]
        if insmode is None:
            insmode = dpro["insmode"][expid]
        tempname = dpro["tempname"][expid]

    if expid is not None and draw is not None:
        rid = np.where(draw['EXPID'] == expid)[0]
        nf = len(rid)
        filenames = draw['filename'][rid]
        if tempname is None:
            tempname = draw['TPL_ID'][rid[0]]
        if insmode is None:
            insmode = draw['insmode'][rid[0]]
        #fram_format = draw['FRAM_FORMAT'][rid[0]]

    else:
        nf = len(rawfiles)
        rid = np.arange(0,nf)
        filenames = rawfiles.split('/')[-1]
        rawfolder = rawfiles.split('/')[0:-1]

    if nf == 0:
        msg = " - ERROR: No corresponding raw files found! Continue..."
        _print_log_info(msg, logfile)

        if dpro is not None:
            dpro['noerr'][expid] = dpro['noerr'][expid] + 1
        return(1)

    if 'CORRUPT' in draw['DATE-OBS'][rid]:
        msg = " - ERROR: This observations contains corrupt files! Continue..."
        _print_log_info(msg, logfile)

        if dpro is not None:
            dpro['noerr'][expid] = dpro['noerr'][expid] + 1

        return(1)


    nodims = []
    heads = []
    outnames = []

    ima = None
    imb = None
    heada = None
    # headb = None

    # === 1 Go over all files of the exposure and create nodding pairs
    for i in range(nf):
        msg = (" - i, File: " + str(i)  + ", " + filenames[i] )
        _print_log_info(msg, logfile)


        # LATER: some source detection-based fine centering of individual
        #        files here
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

        # === 1.1 combine files to nodding pairs
        if (i > 0) & (ima is not None) & (imb is not None):

            # --- build the nod subtracted image
            imc = ima - imb

            fout = (nodfolder + '/' + outname + '_nod'
                        + "{:02.0f}".format(i/2) + '.fits')

            if overwrite or not os.path.isfile(fout):
                fits.writeto(fout, imc, heada, overwrite=True)
                fout = fout.replace(".fits", ".png")
                _simple_image_plot(imc, fout, log=True)

                msg = (" - Output written: " + fout)
                _print_log_info(msg, logfile)

            # --- collect for further reduction
            nodims.append(imc)
            outnames.append(outname)
            heads.append(heada)

            ima = None
            imb = None
            heada = None

    # === 2 Combination of the nodding pairs
#    if len(heads) == 0:   # --- no idea why this was here???
#        continue
    nodims = np.array(nodims)
    nnods = len(nodims)
    head0 = heads[0]

    # --- check if target is a calibrator
    fluxreffile = statcalfolder + '/' + 'ima_std_star_cat.fits'
    if not os.path.isfile(fluxreffile):
        msg = ("REDUCE EXPOSURE: ERROR: flux reference table file not found! "
               + fluxreffile)
        _print_log_info(msg, logfile)
        noe = noe + 1

    else:

        if insmode != "SPC":
            flux = _get_std_flux(head0, reffile=fluxreffile)
            if flux != -1:
                head0["HIERARCH ESO QC JYVAL"] = (flux, "STD flux density in filter used [Jy]")
                if dpro is not None:
                    dpro['STDflux_Jy'][expid] = flux

            elif "cal" in tempname:
                msg = ("GET_STD_FLUX: target not found in flux reference table")
                _print_log_info(msg, logfile)
                noe = noe + 1

    # --- do jitter correction
    if 'HIERARCH ESO SEQ JITTER WIDTH' in head0:
        if head0['HIERARCH ESO SEQ JITTER WIDTH'] > 0:
            for i in range(nnods):
                # joff = V.compute_jitter(head=heads[iddd[m]])
                # print("      - Nod no / jitter offs.: ",m,joff)
                nodims[i] = _undo_jitter(ima=nodims[i], head=heads[i])

    msg = "Do a simple combination of the individual nod cycles"
    _print_log_info(msg, logfile)

    totim = np.mean(nodims, axis=0)
    fout = (outfolder + '/' + outnames[0] + '_all_blind.fits')

    # --- if no specfic range is supplied to measure the sky use the full
    #     illuminated area of the VISIR detector
    if sky_xrange is None:
        sky_xrange = _vp.max_illum_xrange

    if sky_yrange is None:
        sky_yrange = _vp.max_illum_yrange

    # --- background estimation:
    bgim = _subtract_source(totim[sky_yrange[0]:sky_yrange[1],
                                   sky_xrange[0]:sky_xrange[1]])

    dpro["BGmed"][expid] = np.nanmedian(bgim)

    dpro["BGstd"][expid] = np.nanstd(bgim)

    msg = ( " - BG median [ADU/DIT]: " + str(dpro["BGmed"][expid]) + "\n" +
            " - BG STDDEV [ADU/DIT]: " + str(dpro["BGstd"][expid])
           )
    _print_log_info(msg, logfile)


    if overwrite or not os.path.isfile(fout):
        fits.writeto(fout, totim, head0, overwrite=True)
        fout = fout.replace(".fits", ".png")
        _simple_image_plot(totim, fout, log=True)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)


    # --- compute the maximum field of view from half of
    chopthrow = float(head0["HIERARCH ESO TEL CHOP THROW"])
    pfov = float(head0["HIERARCH ESO INS PFOV"])
    nodmode = head0["HIERARCH ESO SEQ CHOPNOD DIR"]

    if not box:
        box = int(np.floor(chopthrow / pfov))
    msg = ( " - PFOV [as]: " + str(pfov) + "\n" +
            " - Nod mode: " + nodmode + "\n" +
            " - Chop throw [as]: " + str(chopthrow) + "\n" +
            " - Max box size: "+str(box)
            )
    _print_log_info(msg, logfile)


    # === 3 BURST & CYCSUM data ===
    if datatype == "halfcyc" or datatype == "burst":

        msg = (" - Burst/halfcyc data encountered. Reduce individual cubes now...")
        _print_log_info(msg, logfile)

        burstfolder = hcycfolder+'/'+alignmethod

        if not os.path.exists(burstfolder):
            os.makedirs(burstfolder)

        for i in range(nf):

            msg = (" - i, File: " + str(i)  + ", " + filenames[i] )
            _print_log_info(msg, logfile)

            fin = rawfolder + '/' + filenames[i]

            fout = burstfolder + "/" + filenames[i].replace(".fits", "_median.fits")
            if ditsoftaver > 1:
                fout = fout.replace(".fits", "_aver" + str(ditsoftaver) + ".fits")

            if overwrite == False and os.path.isfile(fout):

                msg = ("REDUCE_EXPOSURE: File already reduced. Continue...")
                _print_log_info(msg, logfile)
                continue

            # --- first find the off-nod image but only for perpendicular nodding
            if nodmode == "PERPENDICULAR":
                k = i + (-1)**i
                offnodfile = bsumfolder + '/' + filenames[k].replace(".fits", "_blindsum.fits")

                # --- double check that the offnodfile really has a different nod position
                if draw['NODPOS'][rid[i]] == draw['NODPOS'][rid[k]]:
                    msg = ("REDUCE_EXPOSURE: ERROR: wrong offnod file provided: \n"
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
                msg = ("REDUCE_EXPOSURE: ERROR: Burst reduction failed: \n"
                   + str(e[1]) + '' + str(traceback.print_tb(e[2]))
                   + "\nContinue with next file...")

                noe = noe + 1
                _print_log_info(msg, logfile)
                continue

        msg = "REDUCE_EXPOSURE: combine all the merged cubes into one file..."
        _print_log_info(msg, logfile)

        medfins = []
        meanfins = []

        for i in range(nf):

           meanfins.append(burstfolder + "/" +
                           filenames[i].replace(".fits", "_mean.fits"))

           medfins.append(burstfolder + "/" +
                          filenames[i].replace(".fits", "_median.fits"))


        # --- first do a blind merge of the individual nods
        aim = _merge_fits_files(infiles=meanfins, method='mean', northup=True,
                               verbose=verbose, plot=plot, alignmethod=None)
        fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                '_nonodal_meanall.fits')

        update_base_params(aim, head0, dpro, pfov, expid)

        fits.writeto(fout, aim, head0, overwrite=True)
        fout = fout.replace(".fits", ".png")
        _simple_image_plot(aim, fout, log=True)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)


        aim = _merge_fits_files(infiles=medfins, method='median', northup=True,
                               verbose=verbose, alignmethod=None)

        update_base_params(aim, head0, dpro, pfov, expid)

        fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                '_nonodal_medall.fits')
        fits.writeto(fout, aim, head0, overwrite=True)
        fout = fout.replace(".fits", ".png")
        _simple_image_plot(aim, fout, log=True)

        msg = (" - Output written: " + fout)
        _print_log_info(msg, logfile)

        try:
            # ---then try the same alignment method that was used for the individual frames
            aim = _merge_fits_files(infiles=meanfins, method='mean', northup=True,
                               verbose=verbose, plot=plot, alignmethod=alignmethod)
            fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                    '_meanall.fits')

            update_base_params(aim, head0, dpro, pfov, expid)

            fits.writeto(fout, aim, head0, overwrite=True)
            fout = fout.replace(".fits", ".png")
            _simple_image_plot(aim, fout, log=True)

            msg = (" - Output written: " + fout)
            _print_log_info(msg, logfile)


            aim = _merge_fits_files(infiles=medfins, method='median', northup=True,
                               verbose=verbose, alignmethod=alignmethod)

            update_base_params(aim, head0, dpro, pfov, expid)

            fout = (outfolder + '/' + outnames[0] + '_' + alignmethod +
                    '_medall.fits')
            fits.writeto(fout, aim, head0, overwrite=True)
            fout = fout.replace(".fits", ".png")
            _simple_image_plot(aim, fout, log=True)

            msg = (" - Output written: " + fout)
            _print_log_info(msg, logfile)

        except:

            e = sys.exc_info()
            msg = ("REDUCE_EXPOSURE:  - ERROR: Aligning of individual nods failed: \n"
                   + str(e[1]) + '' + str(traceback.print_tb(e[2])))

            _print_log_info(msg, logfile)

            noe += 1



    # === 4 Automatic source extraction
    if extract and mode != 'SPC':

        # ---- WARNING: de-rotation for classical imaging with pupil tracking still to be implemented!

        # --- first find the source positions
        if findbeams:
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
                                                    logfile=logfile)

            except:
                e = sys.exc_info()
                msg = ("REDUCE_EXPOSURE: ERROR: Beam Position could not be found: \n"
                       + str(e[1]) + ' ' + str(traceback.print_tb(e[2]))
                       + "\nContinue assuming the positions...")
                _print_log_info(msg, logfile)
                beamsfound = False
                noe += 1

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
                msg = (" - ERROR: Not all expected beams were found! No source extraction possible...")
                _print_log_info(msg, logfile)

                if dpro is not None:
                    dpro['noerr'][expid] = dpro['noerr'][expid] + 1

                return(noe+1)

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

        # === 3.3.1 Blind addition
        # --- go over all nodding pairs and extract all the sub-images
        #     first blindly
        for j in range(nnods):
            for k in range(nbeams):

                # --- check that the beam is indeed on the detector
                s = np.shape(nodims[j])
                if ((beampos[k,0] > s[0]) | (beampos[k,1] > s[1])
                    | (beampos[k,0] < 0) | (beampos[k,1] < 0)):

                    msg = (" - WARNING: Beampos not on frame: " +
                           str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
                           ". Exclude...")

                    _print_log_info(msg, logfile)

                    if dpro is not None:
                        dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    continue

                beamsign = (-1)**(np.ceil(0.5*k))

                subims.append(_crop_image(nodims[j]*beamsign, box=box,
                                           cenpos=beampos[k,:]))

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
        if nodmode == "PARALLEL":
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
            msg = ("REDUCE_EXPOSURE: ERROR: Failed to update base parameters: \n"
                   + str(e[1]) + '' + str(traceback.print_tb(e[2])))
            _print_log_info(msg, logfile)
            noe = noe + 1

        fout = (outfolder + '/' + outnames[0] + '_all_blind_extr.fits')
        fits.writeto(fout, totim, head0, overwrite=True)

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
        #     msg = ("REDUCE_EXPOSURE: ERROR:  Gallery plots failed to be created: \n"
        #            + str(e[1]) + '' + str(traceback.print_tb(e[2])))
        #     _print_log_info(msg, logfile)
        #     noe = noe + 1


        msg = " - Output written: " + fout
        _print_log_info(msg, logfile)


        # === 3.3.2 Fine-centered addition
        # --- go over all nodding pairs and extract all the sub-images
        if not findbeams or not beamsfound:
            return(noe)

        subims = []
        titles = []
        for j in range(nnods):
            for k in range(nbeams):


                # --- check that the beam is indeed on the detector
                s = np.shape(nodims[j])
                if ((beampos[k,0] > s[0]) | (beampos[k,1] > s[1])
                    | (beampos[k,0] < 0) | (beampos[k,1] < 0)):

                    msg = (" - WARNING: Beampos not on frame: " +
                           str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
                           ". Exclude...")

                    _print_log_info(msg, logfile)

                    if dpro is not None:
                        dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    continue



                beamsign = (-1)**(np.ceil(0.5*k))

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

#                print('params: ', params)
#                print('params[2:4]', params[2:4])
#                print('beamsign: ', beamsign)
#                print('box: ', box)
#                print('i,j: ',i,j)

                cim = _crop_image(nodims[j]*beamsign, box=box,
                                           cenpos=params[2:4])

                dist = beampos[k,:] - params[2:4]
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
                    msg = (" - WARNING: shift too large! Exclude frame")
                    _print_log_info(msg, logfile)
                    if dpro is not None:
                        dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1

                    noe = noe+1
                    continue

                subims.append(cim)
                titles.append("Nod: " + str(j) + " Beam: " + str(k))

        # --- write out
        if len(subims) == 0:
            msg = ("REDUCE_EXPOSURE: WARNING: No source detected in the individual nods! ")
            _print_log_info(msg, logfile)
            if dpro is not None:
                dpro['nowarn'][expid] = dpro['nowarn'][expid] + 1
            return(noe+1)

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

        if nodmode == "PARALLEL":
            subims[0] = 0.5 * subims[0]
            subims.append(subims[0])

        totim = np.mean(subims, axis=0)

        update_base_params(totim, head0, dpro, pfov, expid, onlyhead=onlyhead)

        fout = (outfolder + '/' + outnames[0] + '_all_extr.fits')
        fits.writeto(fout, totim, head0, overwrite=True)

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
        #     msg = ("REDUCE_EXPOSURE: ERROR:  Gallery plots failed to be created: \n"
        #            + str(e[1]) + '' + str(traceback.print_tb(e[2])))
        #     _print_log_info(msg, logfile)
        #     noe = noe + 1

    return(noe)

