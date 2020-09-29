#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.1.1"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-06-18: some adaptations for ISAAC support
    - 2020-06-19: add global smooth search and double-pos and bfound return
    - 2020-07-08: implement use of logfile and warning logging, add
                  minFWHM and maxFWHM as optional input parameter,
                  add optional sigmaclip
    - 2020-09-29: make sure that pfov is a float


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits



from .calc_beampos import calc_beampos as _calc_beampos
# from .diffraction_limit import diffraction_limit as _diffraction_limit
from .find_source import find_source as _find_source
from .fits_get_info import fits_get_info as _fits_get_info
from .print_log_info import print_log_info as _print_log_info
from .replace_hotpix import replace_hotpix as _replace_hotpix


# %%
# --- helper routine
def crosshair_marker(inner_r=0.5, pa = 0):
   ''' The path of an emtpy cross, useful for indicating targets without crowding the field.

      inner_r = empty inner radius. Default =0.5
   '''

   verts = [(-1, 0),
            (-inner_r, 0),
            (0, inner_r),
            (0, 1),
            (inner_r, 0),
            (1, 0),
            (0, -inner_r),
            (0, -1),
            (-1, 0),
            (-1, 0),
           ]

   pa = np.radians(pa)
   rot_mat = np.matrix([[np.cos(pa),-np.sin(pa)],[np.sin(pa),np.cos(pa)]])

   for (v, vert) in enumerate(verts):
      verts[v] = (vert*rot_mat).A[0]

   codes = [Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.CLOSEPOLY
            ]

   path = Path(verts, codes)

   return path


#%%
# --- helper routine to find other beam pos after finding one possible
def find_other_beams(im, oldpos, exppos_new, nb, noddir, nodpos, searchbox,
                     fitbox, maxFWHM, minFWHM, tol,
                     verbose, plot, fitpos, logfile, searchsmooth):


    funname = "FIND_OTHER_BEAMS"

    s = np.shape(im)

    dist = 0

    # --- find out which beam the found one belongs to

    # --- in case of PERPENDICULAR, it could be AA or BB:
    if (noddir == 'PERPENDICULAR') & (nodpos == "both"):

        # --- first assume that the beam is AA:
        if (exppos_new[3,0] < s[0]) & (exppos_new[3,1] < s[1]):

            if verbose:
                msg = (funname + ": Test whether found beam is AA...")
                _print_log_info(msg, logfile)

            params, _, _ = _find_source(im, sign=1, fitbox=fitbox,
                                             guesspos=exppos_new[3,:],
                                             searchbox=searchbox,
                                             plot=plot, verbose=verbose,
                                             maxFWHM=maxFWHM,
                                             method='mpfit',
                                             minFWHM=minFWHM,
                                             searchsmooth=searchsmooth
                                             )

            fitpos[3,:] = params[2:4]

            dist = np.sqrt((fitpos[3,0] - exppos_new[3,0])**2 +
                          (fitpos[3,1] - exppos_new[3,1])**2)

            if verbose:
               msg = (funname + ":  New expected pos: " + str(exppos_new[3,:])
                      + "\n" + funname + ":  Beam found at: " + str(fitpos[3,:])
                      + "\n" + funname + ":  Distance: " + str(dist)
                      )
               _print_log_info(msg, logfile)

        # --- if in that case BB would be off-chip or the above fit failed
        #     assume that BB was found
        if ((exppos_new[3,0] > s[0]) | (exppos_new[3,1] > s[1])
            | (dist > tol)):

            if verbose and dist > tol:
                msg = (funname +
                       ": No beam found at expected position. Now assume found beam was BB...")

                _print_log_info(msg, logfile)

            elif verbose:
                msg = (funname + ": Found beam seems to belong to BB. Try to find AA now...")

                _print_log_info(msg, logfile)

            exppos_new[3,:] = exppos_new[0,:]
            dif = exppos_new[3,:] - oldpos[3,:]
            exppos_new[0:3,:] = oldpos[0:3,:] + dif

            params, _, _ = _find_source(im, sign=1, fitbox=fitbox,
                                             guesspos=exppos_new[0,:],
                                             searchbox=searchbox,
                                             plot=plot, verbose=verbose,
                                             maxFWHM=maxFWHM,
                                             method='mpfit',
                                             minFWHM=minFWHM)

            fitpos[0,:] = params[2:4]

            dist = np.sqrt((fitpos[0,0] - exppos_new[0,0])**2 +
                          (fitpos[0,1] - exppos_new[0,1])**2)

    # --- if PARALLEL or PERPENDICULAR & the found positions not deviating too much
    if noddir == "PARALLEL" or dist <= tol:

        # --- try again to get the exact beam positions
        idond = np.where((exppos_new[:,0] > 0) & (exppos_new[:,0] < s[0]) & (exppos_new[:,1] > 0) & (exppos_new[:,1] < s[1]))[0]

        nbond = len(idond)

        idnond = []

        if verbose:
            msg =(funname +
                  ": Number of beams expected to be on the detector: " +
                  str(nbond)
                  )
            _print_log_info(msg, logfile)

        for i in range(nb):

            if i not in idond:
                idnond.append(i)

            else:
                beamsign = (-1)**(np.ceil(0.5*i))
    # print("beamsign: ",beamsign)

                if verbose:
                    msg = ("\n\n\n" + funname + ": Beam/sign: " + str(i)
                           + ", " + str(beamsign))
                    _print_log_info(msg, logfile)

                params, _, _ = _find_source(im, sign=beamsign, fitbox=fitbox,
                                     guesspos=exppos_new[i,:],
                                     searchbox=searchbox, guessmeth='max',
                                     method='mpfit',
                                     plot=plot, verbose=verbose,
                                     maxFWHM=maxFWHM, minFWHM=minFWHM)

                fitpos[i,:] = params[2:4]

                if i == 0:
                    fitparams = params


        if verbose:
            msg = (funname + ": Fitted beam positions: " +str(fitpos[idond]))
            _print_log_info(msg, logfile)

        if nbond < nb:
            avdif = np.mean(fitpos[idond] - oldpos[idond], axis=0)

            for i in range(len(idnond)):
                fitpos[idnond[i]] = oldpos[idnond[i]] + avdif


        # --- check that the values could make sens:
        difs = fitpos - exppos_new
        if (np.std(difs[:,0]) > tol) | (np.std(difs[:,1]) > tol):
            msg = (funname + ": ERROR: found positions not consistent with chop/nod pattern!")
            _print_log_info(msg, logfile)
            obfound  = "fail"

        else:

            if nbond == 1:
                obfound = "one"
            else:
                obfound = "all"

    return(obfound, fitpos, fitparams)

#%% --- helper routine to check whether found beam can be valid
# def validate_beams(im, beampos, amps, minamp):

#     s = np.shape(im)
#     nbeams = len(beampos[:,0])

#     valid = np.full(nbeams, True)

#     for k in range(nbeams):

#         # --- first test that at least one of the beams is on the detector
#         if ((beampos[k,0] >= s[0]) | (beampos[k,1] >= s[1])
#             | (beampos[k,0] <= 0) | (beampos[k,1] <= 0)):

#             valid[k] = False

#         # --- test that the amplitude is higher than the min
#         if np.abs(amps[k]) < minamp:
#             valid[k] = False

#     return(valid)



#%%
# --- main routine
def find_beam_pos(im=None, fin=None, ext=None, head=None, chopang=None,
                        chopthrow=None, noddir=None, refpix=None,
                        pfov=None, rotang=None, winx=None, winy=None,
                        fitbox=50, nodpos=None, AA_pos=None, verbose=False,
                        sourceext='unknown', filt=None, tol=10,
                        searcharea='chopthrow', pupiltrack=False,
                        imgoffsetangle=None, plot=False, tryglobal=True,
                        logfile=None, instrument=None, insmode=None,
                        minFWHM=None, maxFWHM=None, sigthres=1,
                        searchsmooth=0.2, sigmaclip=[3,1]):
    """
    Find the beam positions in a given chop (nod) file by first computing where
    they should be and then verifying by fit around these positions.
    Optional input parameters minFWHM and maxFWHM are in arcsec.
    """

    funname = "FIND_BEAM_POS"

    if head is None:
        hdu = fits.open(fin)
        head=hdu[0].header
        hdu.close()

    if im is None:
        hdu = fits.open(fin)
        im=hdu[ext].data
        hdu.close()

    if instrument is None:
        instrument = _fits_get_info(head, "instrument")

    if insmode is None:
        insmode = _fits_get_info(head, "insmode")

    if pfov is None:
        pfov = float(_fits_get_info(head, "pfov"))

    if filt is None:
        filt = _fits_get_info(head, "filt")

    if chopthrow is None:
        chopthrow = float(_fits_get_info(head, "CHOP THROW"))

    if noddir is None:
        noddir = _fits_get_info(head, "CHOPNOD DIR")

    if nodpos is None:
        nodpos = _fits_get_info(head, "NODPOS")

    # if dpro is not None and expid is not None:
    #     nowarn = dpro["nowarn"][expid]
    # else:
    nowarn = 0

    # --- sigma for the smoothing for beamsearch
    searchsmooth /= pfov   # convert to pixel


    # print("NOWARN: ", nowarn)

    # --- optional sigma clipping for the beam search
    if sigmaclip is not None:
        nrepl, im = _replace_hotpix(im, sigmathres=sigmaclip[0],
                                    niters=sigmaclip[1], verbose=verbose)

        msg = (funname + ": Number of replaced (hot) pixels: " + str(nrepl))
        _print_log_info(msg, logfile)


    # --- First compute the expected beam positions
    exppos = _calc_beampos(head=head, chopang=chopang, chopthrow=chopthrow,
                             noddir=noddir, refpix=refpix, pfov=pfov,
                             rotang=rotang, winx=winx, winy=winy,
                             verbose=verbose, pupiltrack=pupiltrack,
                             imgoffsetangle=imgoffsetangle,
                             instrument=instrument, insmode=insmode)

    # --- Modify the calculated positions in case the user provided the beam
    #     position of chop A nod A
    if AA_pos is not None:
        xdif = AA_pos[1] - exppos[0,1]
        ydif = AA_pos[0] - exppos[0,0]

        exppos[:,1] = exppos[:,1] + xdif
        exppos[:,0] = exppos[:,0] + ydif

    if minFWHM is not None:
        minFWHM = minFWHM/pfov   # convert from as to px
    else:

        # # --- determine minumum possible FWHM of astronomical source
        # diflim = _diffraction_limit(filt, instrument=instrument,
        #                             insmode=insmode, pfov=pfov, unit="px",
        #                             logfile=logfile)


        # minFWHM = 0.8*diflim

        # --- for finding beams use a relatively high minFWHM
        minFWHM = 0.3/pfov

    if maxFWHM is not None:
        maxFWHM = maxFWHM/pfov
    else:

        # --- determine the maximum FWHM depending on expected source extent
        if sourceext == 'extended':
            maxFWHM = None

        else:
            # --- if extent is compact or unknown assume a limit for now
            # --- VISIR is always closish to the diffraction limit
            if instrument == "VISIR":
                maxFWHM = 1.0/pfov

            # --- However, ISAAC is seeing limited
            elif instrument == "ISAAC":
                if "HIERARCH ESO TEL AMBI FWHM START" in head:
                    seeing = head["HIERARCH ESO TEL AMBI FWHM START"]
                else:
                    seeing = 1.0

                maxFWHM = seeing/pfov

                # --- make sure maxFWHM is larger than minFWHM (sometimes DIMM is -1!)
                if maxFWHM <= minFWHM:
                    maxFWHM = np.min([1.0/pfov, minFWHM*2])


    if searcharea == "chopthrow":
        searchbox = chopthrow / pfov

    else:
        searchbox = None

    if verbose:
           msg = (funname + ": Beam search parameters:\n " +
                  " - Searchbox [px]: " + str(searchbox) + "\n" +
                  " - Fit box size [px]: " + str(fitbox) + "\n" +
                  " - min & max FWHM [px]: " + str(minFWHM) + ", " +
                  str(maxFWHM)
                  )
           _print_log_info(msg, logfile)

    # --- decide which are the actual beams on the image to be found
    #     (for "both" all are used)
    if nodpos == 'A':
        exppos = exppos[:2,:]
    elif nodpos == 'B' and noddir == 'PARALLEL':
        exppos = exppos[[2,0],:]
    elif nodpos == 'B' and noddir == 'PERPENDICULAR':
        exppos = exppos[2:,:]


    if verbose:
        msg = (" - Expected beam positions: " + str(exppos))
        _print_log_info(msg, logfile)

    if plot:
        plt.imshow(im, origin="bottom", interpolation="nearest")
        marker = crosshair_marker(pa=45)
        plt.scatter(exppos[0,1], exppos[0,0], color="white", marker=marker, s=100)
        plt.text(exppos[0,1], exppos[0,0]-30, "AA", color="white")
        plt.scatter(exppos[1,1], exppos[1,0], color="black", marker=marker, s=100)
        plt.text(exppos[1,1], exppos[1,0]-30, "AB", color="black")

        if nodpos == "both":
            plt.scatter(exppos[2,1], exppos[2,0], color="black", marker=marker, s=100)
            plt.text(exppos[2,1], exppos[2,0]-30, "BA", color="black")

            if noddir == "PERPENDICULAR":
                plt.scatter(exppos[3,1], exppos[3,0], color="white", marker=marker, s=100)
                plt.text(exppos[3,1], exppos[3,0]-30, "BB", color="white")

        plt.show()
        # return(exppos, None)

    # --- Now do a fitting to get the positions accurately
    fitpos = np.copy(exppos)
    nb = len(fitpos[:,0])

    s = np.array(np.shape(im))

    # --- first only look for the beam positions expected to be on the detector:

    idond = np.where((exppos[:,0] > 0) & (exppos[:,0] < s[0]) & (exppos[:,1] > 0) & (exppos[:,1] < s[1]))[0]

    nbond = len(idond)

    idnond = []

    if nbond == 0:
        msg = (funname + ": WARNING: no beam expected to be on the detector!")
        _print_log_info(msg, logfile)

        nowarn +=1

        doglobal = True

    else:

        doglobal = False

        if verbose:
           msg = (funname + ": Number of beams expected to be on the detector: "
                  + str(nbond))
           _print_log_info(msg, logfile)

        for i in range(nb):

            if i not in idond:
                idnond.append(i)

            else:
                beamsign = (-1)**(np.ceil(0.5*i))
        # print("beamsign: ",beamsign)

                if verbose:
                    msg = ("\n\n\n" + funname + ": - Beam/sign: " + str(i)
                           + ", " + str(beamsign))
                    _print_log_info(msg, logfile)

                params, _, _ = _find_source(im, sign=beamsign, fitbox=fitbox,
                                         guesspos=exppos[i,:],
                                         searchbox=searchbox, guessmeth='max',
                                         method='mpfit',
                                         plot=plot, verbose=verbose,
                                         maxFWHM=maxFWHM, minFWHM=minFWHM,
                                         searchsmooth=searchsmooth)

        # plt.imshow(fit, origin='bottom')
        # plt.show()

        # print(params)
        # print(i, type(fitpos), np.shape(fitpos), type(exppos), np.shape(exppos))
                fitpos[i,:] = params[2:4]

                if i == 0:
                    fitparams = params


        if verbose:
            msg = (funname + ": Fitted beam positions: " + str(fitpos[idond]))
            _print_log_info(msg, logfile)

        if nbond < nb:
            avdif = np.mean(fitpos[idond] - exppos[idond], axis=0)

            for i in range(len(idnond)):
                fitpos[idnond[i]] = exppos[idnond[i]] + avdif

    # -- for debugging
    # tryglobal = False

        # --- check that the values could make sens:
        difs = fitpos - exppos

        if ((np.std(difs[:,0]) > tol) | (np.std(difs[:,1]) > tol)):
            msg = (funname + ": WARNING: found positions not consistent with chop/nod pattern!")
            _print_log_info(msg, logfile)

            nowarn += 1

            if tryglobal:
                doglobal = True

        else:
            bfound = "first attempt"

    # --- if beams not at expected postion try to find them by fitting the full
    #     frame
    if doglobal:
        # --- just fit the whole image to get the brightest beam
        msg = (funname + ": Attempting global fit...")
        _print_log_info(msg, logfile)

        # --- first rebin the image to increase signal and reduce
        #     fitting duration
        binsize = 4
        new_shape = [int(s[0]/binsize), int(s[1]/binsize)]
        compression_pairs = [(d, c//d) for d,c in zip(new_shape, im.shape)]
        flattened = [l for p in compression_pairs for l in p]
        rim = im.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(rim, 'sum')
            rim = op(-1*(i+1))

        if plot:
            plt.imshow(im, origin="bottom", interpolation="nearest")
            plt.title("Rebinned")
            plt.show()

        minFWHM_b = minFWHM/binsize

        if sourceext == "compact":
            maxFWHM_b = maxFWHM/binsize
            searchsmooth_b = 0

        else:
            searchsmooth_b = searchsmooth/binsize
            maxFWHM_b = maxFWHM

        params, _, _ = _find_source(rim, sign=1, plot=plot,
                                    searchsmooth=searchsmooth_b,
                                    verbose=verbose, method='mpfit',
                                    maxFWHM=maxFWHM_b, minFWHM=minFWHM_b)


        exppos_new = np.copy(exppos)
        exppos_new[0,:] = binsize*np.array(params[2:4])

        if ((exppos_new[0,0] >= s[0]) | (exppos_new[0,1] >= s[1])
                    | (exppos_new[0,0] <= 0) | (exppos_new[0,1] <= 0)):

            msg = (funname + ": WARNING: Beam candidate not on detetor: "
                   + str(exppos_new[0,:]))

            _print_log_info(msg, logfile)

            nowarn += 1

            bfound = "fail"

        else:

            dif = exppos_new[0,:] - exppos[0,:]
            msg = (funname + ": Possible beam found at: " + str(exppos_new[0,:]))
            _print_log_info(msg, logfile)

            exppos_new[1:,:] = exppos[1:,:] + dif

            # --- now check if other beams can be found
            obfound, fitpos, fitparams = find_other_beams(im, exppos, exppos_new,
                                                          nb, noddir, nodpos,
                                                          searchbox, fitbox,
                                                          maxFWHM, minFWHM, tol,
                                                          verbose, plot, fitpos,
                                                          logfile,
                                                          searchsmooth=searchsmooth)

            if obfound == "all":
                bfound = "global fit"
            elif obfound == "one":
                bfound = "only one found"
            else:
                bfound = "fail"

                msg = (funname
                       + ": WARNING: found positions not consistent with chop/nod pattern: "
                       + str(fitpos))
                _print_log_info(msg, logfile)

                nowarn += 1

        # --- in case this did not work, try again with smoothed and no maxFWHM
        if bfound == "fail":

            if searchsmooth > 0:

                msg = (funname + ": Attempting global smoothed fit...")
                _print_log_info(msg, logfile)

                if sourceext == "compact":
                    maxFWHM_b = 2*maxFWHM_b

                # --- smooth the image with a FWHM~0.5" Gauss kernel
                sim = gaussian_filter(im, sigma=searchsmooth, mode='nearest')
                srim = gaussian_filter(rim, sigma=searchsmooth/binsize, mode='nearest')


                fitparams, _, _ = _find_source(srim, sign=1, plot=plot,
                                                 verbose=verbose, method='mpfit',
                                                 maxFWHM=maxFWHM_b, minFWHM=minFWHM_b)


            # --- if no smoothed search is wished, use the solution from the
            #     binned global search above
            else:
                fitparams = params
                sim = im

            # --- revert values to unbinned
            fitparams[0] /= binsize
            fitparams[1] /= binsize
            fitparams[2] *= binsize
            fitparams[3] *= binsize
            fitparams[4] *= binsize
            fitparams[5] *= binsize

            exppos_new = np.copy(exppos)
            exppos_new[0,:] = fitparams[2:4]

            # --- test if the one found seems valid
            # --- first test that at least one of the beams is on the detector
            if ((exppos_new[0,0] >= s[0]) | (exppos_new[0,1] >= s[1])
                | (exppos_new[0,0] <= 0) | (exppos_new[0,1] <= 0)):

                msg = (funname + ": ERROR: Beam candidate not on detector: " + str(exppos_new[0,:]))
                _print_log_info(msg, logfile)

                bfound = "fail"

            # --- test that the amplitude is higher than the min
            elif fitparams[1] < sigthres * np.nanstd(im[int(0.25*s[0]):int(0.75*s[0]), int(0.25*s[1]):int(0.75*s[1])]):

                msg = (funname + ": ERROR: Beam amplitude too small: "
                       + str(fitparams[1]))
                _print_log_info(msg, logfile)

                bfound = "fail"

            else:
                dif = exppos_new[0,:] - exppos[0,:]
                msg = (funname + ": Possible beam found at: " +
                       str(exppos_new[0,:]))
                _print_log_info(msg, logfile)

                exppos_new[1:,:] = exppos[1:,:] + dif

                if sourceext == "compact":
                    searchsmooth = 0
                elif sourceext != "extended":
                    searchsmooth = 0.5*searchsmooth

                # --- now check if other beams can be found
                obfound, fitpos, params = find_other_beams(sim, exppos, exppos_new,
                                                              nb, noddir, nodpos,
                                                              searchbox, fitbox,
                                                              maxFWHM, minFWHM, tol,
                                                              verbose, plot, fitpos,
                                                              logfile,
                                                              searchsmooth=searchsmooth)

                if obfound == "all":
                    bfound = "global smooth"
                    fitparams = params
                else:
                    print(funname + ": WARNING: other beams not found! Using positions based on first...")
                    bfound = "only one found"
                    fitpos = exppos_new
                    nowarn += 1


    # if dpro is not None and expid is not None:
    #     dpro["nowarn"][expid] = nowarn

    # print("NOWARN: ", nowarn)

    return(bfound, nowarn, fitpos, fitparams)

