#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.1.0"

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
    - 2020-06-25: ISAAC support and correct WCS treatment for source extraction
    - 2020-07-09: fixed bug of wrong wheighting of double images. Creation of
                  extract_beams subroutine
    - 2020-09-29: Fix bug with bfound not defined for spectroscopy
    - 2020-10-05: provide maxshift as tolerance for beam find,
                  add custom searcharea in arcsec



NOTES:
    -

TO-DO:
    - change the make_gallery calls to a simpler package internal routine and
      uncomment the corresponding section in the main routine below
    - WCS for the extracted images
    - proper modification of the headers of the products
    - optional North-up alignment for classical (non-burst) imaging
    - fixed extraction for given WCS coordinates
    - correct treatment of SPC ACQ data
    - correct treatment of HR and HRX SPC
    - callability as stand-alone routine
    - replace RA,DEC in update_base_params to TEL TARG ALPHA/DELTA


"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import sys
import traceback
from copy import deepcopy
from tqdm import tqdm
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

from .angular_distance import angular_distance as _angular_distance
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
from .replace_hotpix import replace_hotpix as _replace_hotpix
from .undo_jitter import undo_jitter as _undo_jitter
from . import visir_params as _vp
from . import isaac_params as _ip

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
    # try:
    snbest, sens = _measure_sensit(im=im, flux=flux, showplot=False,
                                      std=std, exptime=exptime)
    # except:
        # snbest = np.ma.masked
        # sens = np.ma.masked

    print(snbest)
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
# --- Helper routine to update the WCS fits header to the new crop
def update_wcs(head, xpos_px, ypos_px, box):


    wcs = WCS(head)

    ra, dec = wcs.wcs_pix2world(xpos_px, ypos_px, 0)
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    newhead = deepcopy(head)

    if not hasattr(box, "__len__"):
        box = np.array([box, box], dtype=int)

    sy, sx = box
    # head["WCSAXES"] = 2
    # head["CTYPE1"]  = 'RA---TAN'
    # head["CTYPE2"]  = 'DEC--TAN'
    newhead["CRPIX1"] = 0.5 * sx + 0.5
    newhead["CRPIX2"] = 0.5 * sy + 0.5
    newhead["CRVAL1"] = coord.ra.deg
    newhead["CRVAL2"] = coord.dec.deg
    # head["CD1_1"] = -pfov/3600.0
    # head["CD1_2"] = 0
    # head["CD2_2"] = pfov/3600.0
    # head["CD2_1"] = 0

    return(newhead)

#%%
# --- Helper routine to extract beams

def extract_beams(nodims, beampos, box, noddir, head0,
                  logfile, verbose, debug, outfolder, outbasename, suffix,
                  nodalign=False, updatedb=False, dpro=None, fitparams=None,
                  pfov=None, expid=None, onlyhead=False, maxshift=None,
                  sigmaclip=None):

    # === 4.1 Blind addition
    # --- go over all nodding pairs and extract all the sub-images
    #     first blindly
    funname = "EXTRACT_BEAMS"
    # nowarn = 0
    noerr = 0


    subims = []
    # titles = []
    nbeams = len(beampos[:,0])
    nnods = len(nodims)


    beamposstr = ""
    for b in range(nbeams):
        beamposstr = (beamposstr + "{:3.0f}".format(beampos[b,0]) + ", "
                      + "{:3.0f}".format(beampos[b,1]) + " | ")

    msg = (
           " - No of beams: " + str(nbeams) + "\n" +
           " - Beampos: " + beamposstr + "\n"
           )

    _print_log_info(msg, logfile)


    # --- exclude beams that are not on the detector
    s = np.shape(nodims[0])
    use = np.full(nbeams, True)
    beamsign = np.zeros(nbeams)

    for k in range(nbeams):

        beamsign[k] = (-1)**(np.ceil(0.5*k))

        if ((beampos[k,0] >= s[0]) | (beampos[k,1] >= s[1])
            | (beampos[k,0] <= 0) | (beampos[k,1] <= 0)):

            msg = (" - Beampos not on frame: " +
                   str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
                   ". Exclude...")

            _print_log_info(msg, logfile)

            use[k] = False

    beampos = beampos[use,:]
    beamsign = beamsign[use]
    nbeams = len(beampos)

    # --- make sure that at least one beam is on the detector
    if nbeams == 0:

        msg = (" - Error: No beam on detector: " +
               str(beampos[k,0]) + ', ' + str(beampos[k,1]) +
               ". Exiting...")

        _print_log_info(msg, logfile)

        return(noerr+1)


    # --- for the case of fine-centred extraction
    if nodalign:

        # --- get the expected beam parameters from the fit
        guessbg = fitparams[0]
        guessamp = np.abs(fitparams[1])
        guessFWHM = np.mean(fitparams[4:6])

        minamp = 0.3*guessamp
        minFWHM = 0.5*guessFWHM
        maxFWHM = 2*guessFWHM

        maxshift /= pfov  # maxshift in px

        fitbox = int(np.max([6*guessFWHM,0.25*box]))

        nexp = nnods*nbeams
        nfound = 0


        for j in tqdm(range(nnods)):
            for k in range(nbeams):

                try:
                    params, _, _ = _find_source(nodims[j]*beamsign[k],
                                                 guesspos=beampos[k,:],
                                                 searchbox=box,
                                                 fitbox=fitbox,
                                                 method='mpfit',
                                                 guessbg=guessbg,
                                                 guessamp=guessamp,
                                                 guessFWHM=guessFWHM,
                                                 minamp=minamp,
                                                 minFWHM=minFWHM,
                                                 maxFWHM=maxFWHM)

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

                cim = _crop_image(nodims[j]*beamsign[k], box=box,
                                           cenpos=params[2:4], silent=True)

                dist = beampos[k,:] - params[2:4]

                if verbose:
                    msg = (" - Nod: " + str(j) + " Beam: " + str(k) +
                           " Detected shift: " + str(dist)
                           )

                    _print_log_info(msg, logfile)


                    bg_t = params[0]
                    amp_t = params[1]
                    fwhm_t = np.mean(params[4:6])
                    axrat_t = np.max(params[4:6])/np.min(params[4:6])
                    total_t = 0.25 * (np.pi * params[1] * params[4] * params[5])/np.log(2.0)
                    angle_t = params[6]
                    msg = (" - Found source params:\n" +
                           "     - BG: "+  str(bg_t) + "\n" +
                           "     - Amplitude: "+  str(amp_t) + "\n" +
                           "     - Total: "+ str(total_t) + "\n" +
                           "     - Aver. FWHM [px]: "+  str(fwhm_t) + "\n" +
                           "     - Aver. FWHM [as]: "+  str(fwhm_t*pfov) + "\n" +
                           "     - Maj./min axis: "+  str(axrat_t) + "\n" +
                           "     - Angle: "+  str(angle_t)
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

                # --- all double beams have to be halved but added twice
                if (noddir == "PARALLEL") & (beamsign[k] > 0):
                    subims.append(0.5*cim)
                    subims.append(0.5*cim)
                else:
                    subims.append(cim)

                nfound += 1
                # titles.append("Nod: " + str(j) + " Beam: " + str(k))


        msg = (funname +
               ": Number of detected beams/expected: " + str(nfound) + "/" + str(nexp))
        _print_log_info(msg, logfile)


        # --- if object not detected in individual nods give warning
        if nfound == 0:
            msg = (funname +
                   ": No source detected in the individual nods! Source probably too faint...")
            _print_log_info(msg, logfile)

            return(noerr+1)

        # -- also if <50% of the beams were found abort reject the fine
        #    tuned extraction and proceed with bling
        elif nfound < 0.5 * nexp:
            msg = (funname +
                   ": <50% of the beams detected in individual nods! Source probably too faint... ")
            _print_log_info(msg, logfile)

            return(noerr+1)


    # --- fixed extraction
    else:

        for j in tqdm(range(nnods)):
            for k in range(nbeams):

                cim = _crop_image(nodims[j]*beamsign[k], box=box,
                                           cenpos=beampos[k,:], silent=True)

                # --- all double beams have to be halved but added twice
                if (noddir == "PARALLEL") & (beamsign[k] > 0):
                    subims.append(0.5*cim)
                    subims.append(0.5*cim)
                else:
                    subims.append(cim)



    # --- update the WCS in the fits header
    newhead = update_wcs(head0, beampos[0,1], beampos[0,0], box)

    # --- write out
    if debug:
        fout = (outfolder + '/' + outbasename + '_all_extr_cube.fits')
        fits.writeto(fout, subims, newhead, overwrite=True)

#        fout = fout.replace(".fits", "_log.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=True,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')
#
#        fout = fout.replace("_log.png", "_lin.png")
#        I.make_gallery(ims=subims, outname=fout, pfovs=pfov, log=False,
#                       papercol=2, ncols=nbeams, cmap='gnuplot2', titles=titles,
#                       inv=False, permin=40, permax=99.9, titcols='white')

    # --- average all the sub-images
    totim = np.nanmean(subims, axis=0)



    # --- optional sigma clipping for the beam search
    if sigmaclip is not None:
        nrepl, totim = _replace_hotpix(totim, sigmathres=sigmaclip[0],
                                    niters=sigmaclip[1], verbose=verbose)

        msg = (funname + ": Number of replaced (hot) pixels: " + str(nrepl))
        _print_log_info(msg, logfile)


    if updatedb:

        # --- position of the found source
        tara = _fits_get_info(head0, keys="RA")
        tadec = _fits_get_info(head0, keys="DEC")

        wcs = WCS(head0)
        ra, dec = wcs.wcs_pix2world(beampos[0,1], beampos[0,0], 0)
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')


        dpro['GF_x_px'][expid] = beampos[0,1]
        dpro['GF_y_px'][expid] = beampos[0,0]

        dpro['GF_RA_hms'][expid] = coord.to_string('hmsdms').split()[0]
        dpro['GF_DEC_dms'][expid] = coord.to_string('hmsdms').split()[1]
    
        dpro['GF_angdist_as'][expid] = _angular_distance(ra, dec, tara, tadec)


        try:
            update_base_params(totim, newhead, dpro, pfov, expid, onlyhead=onlyhead)
        except:
            e = sys.exc_info()
            msg = (funname + ": ERROR: Failed to update base parameters: \n"
                   + str(e[1]) + '' + str(traceback.print_tb(e[2])))
            _print_log_info(msg, logfile)
            noerr = noerr + 1

    fout = (outfolder + '/' + outbasename + '_all_extr-' + suffix + '.fits')
    fits.writeto(fout, totim, newhead, overwrite=True)

    pout = fout.replace(".fits", "_log.png")
    _simple_image_plot(totim, pout, scale="log", pfov=pfov, cenax=True)

    pout = fout.replace(".fits", "_lin.png")

    # --- if the source is probably faint, use lower colour thresholds
    if suffix == "fixest" or suffix == "fix1det":
        percentile = 1
    else:
        percentile = None

    _simple_image_plot(totim, pout, pfov=pfov, cenax=True,
                       percentile=percentile)

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

    return(noerr)


#%%

# === 3. DO THE REDUCTION OF INDIVIDUAL EXPOSURES
def reduce_exposure(rawfiles=None, draw=None, dpro=None, expid=None, sof=None,
                    temprofolder=None, rawfolder=None, logfile=None,
                    outname='VISIR_OBS', outfolder='.', overwrite=False,
                    maxshift=0.5, extract=True, searcharea='chopthrow',
                    box=None, statcalfolder=None, crossrefim=None,
                    chopsubmeth='averchop', AA_pos=None, refpos=None,
                    alignmethod='fastgauss', findbeams=True,
                    verbose=False, ditsoftaver=1,
                    sky_xrange=None, plot=False, insmode=None,
                    sky_yrange=None, debug=False, instrument=None,
                    sourceext='unknown', searchsmooth=0.2,
                    sigmaclip=[3,1]):

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
     - maxshift: (default=0.5) maximum allowed discrepany between expected and found beam
                 position in a chop/nod pattern in arcsec
     - searchsmooth: (default=0.2) sigma for Gaussian smoothing for beam detection in arcsecc
     - extract: (default=True) Try detect a source in the combined nod image and extract subimage
     - searcharea: (default='chopthrow') specify the area to search for the sub beam as string in arcsec (e.g., "2 arcsec")
     - box: (optional) by default the subimage will have the size of the chopthrow
     - chopsubmeth: (default='averchop') method for the subtraction of the chops in burst mode. By default the average of the exposures in the offset position of the corresponding pair is used.
     - AA_pos: (optional) specify position of the beam in chop/nod position AA for burst alignment
     - refpos: (optional) instead of searching for the beam position, provide one directly for the alignment
     - findbeams: (default: True) Should the code try to find the beam positions or just extract at the computed position?
     - alignmethod: (default='fastgauss') specify the fitting algorithm for the frame alignment in burst mode data
     - verbose = False
     - sourceext: (default ='unknown') expected extent of the source ('compact', 'extended', 'unknown')

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
    now = 0  # number of warnings

    # --- flag whether only fits header or also table should be updated with
    #     new Gaussian fit paramaters (will be set false after reducing a burst)
    onlyhead = False

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
        chopangle = dpro["chopangle"][expid]
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
        if 'CANNOT' in draw['INSTRUME'][rid]:
            msg = funname + ": ERROR: This observations contains corrupt files! Aborting..."
            _print_log_info(msg, logfile)

            return(now, noe+1)

        # # --- in addition we can get some
        # if tempname is None:
        #     tempname = draw['TPL_ID'][rid[0]]
        # if insmode is None:
        #     insmode = draw['insmode'][rid[0]
        # #fram_format = draw['FRAM_FORMAT'][rid[0]]

    # --- or in case a sof file in esorex style is provided
    elif sof is not None:
        fsof = open(sof)
        # --- implement extraction of the fits files here
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

        return(now, noe+1)

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

    if chopangle is None:
        chopangle = _fits_get_info(head0, "CHOP ANGLE")

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
            flux = _get_std_flux(head0, logfile=logfile, instrument=instrument,
                                 silent=silent)
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
            msg = (funname + ": WARNING: target not found in flux reference table but should be calibrator")

            _print_log_info(msg, logfile)

            now += 1

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
            " - Chop angle [de]: " + str(chopangle) + "\n" +
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

    # --- check that we have at least one valid nodding pair
    if nnods == 0:
        msg = (funname + ": ERROR: No nodding pairs could be created! Aborting...")

        _print_log_info(msg, logfile)

        return(now, noe+1)


    # --- 2.1 do jitter correction
    if 'HIERARCH ESO SEQ JITTER WIDTH' in head0:
        if head0['HIERARCH ESO SEQ JITTER WIDTH'] > 0:
            for i in range(nnods):
                # joff = V.compute_jitter(head=heads[iddd[m]])
                # print("      - Nod no / jitter offs.: ",m,joff)
                nodims[i] = _undo_jitter(im=nodims[i], head=heads[i])

    # msg = funname + ": 1. simple combination"
    # _print_log_info(msg, logfile)

    totim = np.nanmean(nodims, axis=0)
    fout = (outfolder + '/' + outnames[0] + '_all_fullframe.fits')

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

        onlyhead = True



    # === 4 Automatic source extraction for imaging
    bfound = "N/A"
    if extract and insmode != 'SPC':

        # ---- WARNING: de-rotation for classical imaging with pupil tracking still to be implemented!

        # --- first find the source positions
        if findbeams:

            msg = funname + ": Trying to detect beams in combined image..."
            _print_log_info(msg, logfile, empty=2)

            try:
                bfound, nowarn, beampos, fitparams = _find_beam_pos(
                                  im=totim, head=head0, searcharea=searcharea,
                                  fitbox=0.5*box, nodpos='both',
                                  verbose=verbose, sourceext=sourceext,
                                  AA_pos=AA_pos, plot=plot, tol=maxshift/pfov,
                                  instrument=instrument, insmode=insmode,
                                  logfile=logfile, chopthrow=chopthrow,
                                  noddir=noddir, filt=setup, pfov=pfov,
                                  searchsmooth=searchsmooth,
                                  sigmaclip=sigmaclip)

                now += nowarn

            except:
                e = sys.exc_info()
                msg = (funname + ": WARNING: Beam Position could not be found: \n"
                        + str(e[1]) + ' ' + str(traceback.print_tb(e[2]))
                        + "\nContinue assuming the positions...")
                _print_log_info(msg, logfile)


                bfound = "fail"

            msg = funname + ": Result of beam search: " + bfound
            _print_log_info(msg, logfile)

            if bfound != "fail":

                # --- retrieve the coordinates of the found position
                wcs = WCS(head0)
                bra, bdec = wcs.wcs_pix2world(beampos[0,1], beampos[0,0], 0)

                bcoord = SkyCoord(ra=bra, dec=bdec, unit=(u.deg, u.deg),
                                  frame='icrs')

                tara = _fits_get_info(head0, keys="RA")
                tadec = _fits_get_info(head0, keys="DEC")

                angdist = _angular_distance(bra, bdec, tara, tadec)

                nbeams = len(beampos)
                bg = fitparams[0]
                amp = fitparams[1]
                fwhm = np.mean(fitparams[4:6])
                axrat = np.max(fitparams[4:6])/np.min(fitparams[4:6])
                total = 0.25 * (np.pi * fitparams[1] * fitparams[4] * fitparams[5])/np.log(2.0)
                angle = fitparams[6]
                msg = (funname + ":\n" +
                       " - Expected source position: " +
                       dpro["RA_hms"][expid] + " " + dpro["DEC_dms"][expid] + "\n" +
                       " - Found source params:\n" +
                       "     - Position x,y [px]: "+  '{:.1f}'.format(beampos[0,1])
                               + "," + '{:.1f}'.format(beampos[0,0]) + "\n" +
                       "     - Position [wcs]: "+ bcoord.to_string('hmsdms')  + "\n" +
                       "     - Angular distance [as]: " + '{:.2f}'.format(angdist) +"\n" +
                       "     - BG: "+  str(bg) + "\n" +
                       "     - Amplitude: "+  str(amp) + "\n" +
                       "     - Total: "+ str(total) + "\n" +
                       "     - Aver. FWHM [px]: "+  str(fwhm) + "\n" +
                       "     - Aver. FWHM (min 0.3) [as]: "+  str(fwhm*pfov) + "\n" +
                       "     - Maj./min axis: "+  str(axrat) + "\n" +
                       "     - Angle: "+  str(angle)
                       )

                _print_log_info(msg, logfile)


            else:
                now += 1

        else:
            bfound = "not tried"


        # === If bright enough, try Fine-centered addition
        # --- go over all nodding pairs and extract all the sub-images
        if bfound == "first attempt" or bfound == "global fit":


            msg = funname + ": Trying to detect and extract beams from individual nods..."
            _print_log_info(msg, logfile, empty=2)

            suffix = 'fine'

            es = extract_beams(nodims, beampos, box, noddir, head0, logfile,
                               verbose, debug, outfolder, outnames[0], suffix,
                               nodalign=True, updatedb=True, dpro=dpro,
                               pfov=pfov, fitparams=fitparams,
                               maxshift=maxshift, sigmaclip=sigmaclip,
                               expid=expid, onlyhead=onlyhead)

        else:
            es = 1

        # ================= Fixed extraction, if necessary

        # --- if a beam was detected do an extraction
        if findbeams and bfound != "fail" and es > 0:


            # --- first do some tests on fitted beams
            s = np.shape(nodims[0])
            use = np.full(nbeams, True)

            for k in range(nbeams):

                # --- first test that at least one of the beams is on the detector
                if ((beampos[k,0] >= s[0]) | (beampos[k,1] >= s[1])
                    | (beampos[k,0] <= 0) | (beampos[k,1] <= 0)):

                    use[k] = False


            # --- make sure that at least one beam is on the detector
            if np.sum(use) == 0:

                msg = (" - WARNING: None of fitted beams on detector! Using reference position...")

                _print_log_info(msg, logfile)

                now += 1

                bfound = "fail"

            else:

                msg = (funname +
                       ": Extracting beams from individual nods at found fixed position ...")

                _print_log_info(msg, logfile, empty=1)

            # --- label fits file depending on beam recovery mode/success:
            if bfound == "first attempt" or bfound == "global fit" or bfound == "global smooth":
                suffix = 'fixadet'
            elif bfound == "only one found":
                suffix = 'fix1det'


            es = extract_beams(nodims, beampos, box, noddir, head0, logfile,
                               verbose, debug, outfolder, outnames[0], suffix,
                               updatedb=True, dpro=dpro, pfov=pfov,
                               sigmaclip=sigmaclip,
                               expid=expid, onlyhead=onlyhead)

            noe += es


        # --- if none or only 1 beam was, or we did not search, estimate their
        #     position
        if not findbeams or bfound == "fail" or bfound == "only one found":

            # --- compute the expected beam positions
            msg = (funname + ": Extraction using reference positions for the beams...")

            _print_log_info(msg, logfile)


            beampos = _calc_beampos(head=head0, verbose=verbose)
            nbeams = len(beampos)
            # fwhm = 7
            # bg = np.nanmedian(totim)
            # amp = np.nanmax(gaussian_filter(totim, sigma=3))

            # --- Modify the calculated positions in case the user provided the beam
            #     position of chop A nod A
            if AA_pos is not None:
                xdif = AA_pos[1] - beampos[0,1]
                ydif = AA_pos[0] - beampos[0,0]

                beampos[:,1] = beampos[:,1] + xdif
                beampos[:,0] = beampos[:,0] + ydif


            suffix = 'fixest'

            es = extract_beams(nodims, beampos, box, noddir, head0, logfile,
                               verbose, debug, outfolder, outnames[0], suffix,
                               updatedb=False, dpro=dpro, pfov=pfov,
                               sigmaclip=sigmaclip,
                               expid=expid, onlyhead=onlyhead)

            noe += es


    if bfound == "first attempt" or bfound == "global fit" or bfound == "global smooth":
        dpro['all_beams_det'][expid] = "T"
    else:
        dpro['all_beams_det'][expid] = "F"

    msg = (funname + ": Number of warnings for exposure: "
           + str(now))
    _print_log_info(msg, logfile, empty=1, screen=False)

    msg = (funname + ": Number of errors for exposure: "
           + str(noe))
    _print_log_info(msg, logfile, screen=False)

    return(now, noe)


