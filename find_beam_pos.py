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
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.io import fits

from .calc_beampos import calc_beampos as _calc_beampos
from .diffraction_limit import diffraction_limit as _diffraction_limit
from .find_source import find_source as _find_source
from .fits_get_info import fits_get_info as _fits_get_info


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
# --- main routine
def find_beam_pos(im=None, fin=None, ext=None, head=None, chopang=None,
                        chopthrow=None, nodmode=None, refpix=None,
                        pfov=None, rotang=None, winx=None, winy=None,
                        fitbox=50, nodpos=None, AA_pos=None, verbose=False,
                        sourceext='compact', filt=None, tol=10,
                        searcharea='chopthrow', pupiltrack=False,
                        imgoffsetangle=92.5, plot=False, tryglobal=True,
                        logfile=None, instrument=None, insmode=None):
    """
    Find the beam positions in a given chop (nod) file by first computing where
    they should be and then verifying by fit around these positions
    """

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

    # --- First compute the expected beam positions
    exppos = _calc_beampos(head=head, chopang=chopang, chopthrow=chopthrow,
                             nodmode=nodmode, refpix=refpix, pfov=pfov,
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

    # --- check with nodding position(s) is/are included in the image
    if nodpos is None:
        nodpos = head['HIERARCH ESO SEQ NODPOS']

    if nodmode is None:
        nodmode = head["HIERARCH ESO SEQ CHOPNOD DIR"]

    # --- determine the maximum source extent if the source is compact
    if sourceext == 'compact':
        if filt is None:
            if "HIERARCH ESO INS FILT1 NAME" in head:
                filt = head["HIERARCH ESO INS FILT1 NAME"]
            else:
                filt = head["HIERARCH ESO INS FILT2 NAME"]

        diflim = _diffraction_limit(filt, instrument=instrument,
                                    insmode=insmode, pfov=pfov, unit="px",
                                    logfile=logfile)
        maxFWHM = 2*diflim
        minFWHM = 0.8*diflim
    else:
        maxFWHM = None
        minFWHM = None


    if searcharea == "chopthrow":

        # --- compute the maximum field of view from half of
        if chopthrow is None:
            chopthrow = float(head["HIERARCH ESO TEL CHOP THROW"])

        if pfov is None:
            pfov = float(head["HIERARCH ESO INS PFOV"])

        searchbox = chopthrow / pfov

    else:
        searchbox = None

    if verbose:
            print(" - FIND_BEAM_POSITONS: Searchbox: ", searchbox)
            print(" - FIND_BEAM_POSITONS: Fit box size: ", fitbox)

    # --- decide which are the actual beams on the image to be found
    #     (for "both" all are used)
    if nodpos == 'A':
        exppos = exppos[:2,:]
    elif nodpos == 'B' and nodmode == 'PARALLEL':
        exppos = exppos[[2,0],:]
    elif nodpos == 'B' and nodmode == 'PERPENDICULAR':
        exppos = exppos[2:,:]


    if verbose:
        print(" - FIND_BEAM_POSITONS: Expected beam positions: ",exppos)

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

            if nodmode == "PERPENDICULAR":
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
        print(" - FIND_BEAM_POSITONS: WARNING: no beam expected to be on the detector!")
        doglobal = True

    else:

        doglobal = False

        if verbose:
           print(" - FIND_BEAM_POSITONS: Number of beams expected to be on the detector: ", nbond)

        for i in range(nb):

            if i not in idond:
                idnond.append(i)

            else:
                beamsign = (-1)**(np.ceil(0.5*i))
        # print("beamsign: ",beamsign)

                if verbose:
                    print("\n\n\n")
                    print(" - FIND_BEAM_POSITONS: Beam/sign: ", i, beamsign )

                params, _, _ = _find_source(im, sign=beamsign, fitbox=fitbox,
                                         guesspos=exppos[i,:],
                                         searchbox=searchbox, guessmeth='max',
                                         method='mpfit',
                                         plot=plot, verbose=verbose,
                                         maxFWHM=maxFWHM, minFWHM=minFWHM)

        # plt.imshow(fit, origin='bottom')
        # plt.show()

        # print(params)
        # print(i, type(fitpos), np.shape(fitpos), type(exppos), np.shape(exppos))
                fitpos[i,:] = params[2:4]

                #if i == 0:
                fitparams = params


        if verbose:
            print(" - FIND_BEAM_POSITONS: Fitted beam positions: ", fitpos[idond])

        if nbond < nb:
            avdif = np.mean(fitpos[idond] - exppos[idond], axis=0)

            for i in range(len(idnond)):
                fitpos[idnond[i]] = exppos[idnond[i]] + avdif

    # -- for debugging
    # tryglobal = False

        # --- check that the values could make sens:
        difs = fitpos - exppos
        if tryglobal & ((np.std(difs[:,0]) > tol) | (np.std(difs[:,1]) > tol)):
            print(" - FIND_BEAM_POSITONS: WARNING: found positions not consistent with chop/nod pattern!")

            doglobal = True

    if doglobal:
        # --- just fit the whole image to get the brightest beam
        print(" - FIND_BEAM_POSITONS: Attempt global fit...")

        # --- first rebin the image by 2x2 pix to increase signal and reduce
        #     fitting duration
        new_shape = [int(s[0]/4), int(s[1]/4)]
        compression_pairs = [(d, c//d) for d,c in zip(new_shape, im.shape)]
        flattened = [l for p in compression_pairs for l in p]
        rim = im.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(rim, 'sum')
            rim = op(-1*(i+1))


        params, _, _ = _find_source(rim, sign=1, plot=plot,
                                         verbose=verbose, method='mpfit',
                                         maxFWHM=maxFWHM, minFWHM=minFWHM)


        exppos_new = np.copy(exppos)
        exppos_new[0,:] = 4.0*np.array(params[2:4])
        dif = exppos_new[0,:] - exppos[0,:]
        print(" - FIND_BEAM_POSITONS: Beam found at: ", exppos_new[0,:])

        # --- assuming that the fit worked, find out which beam the fit
        #     belongs to
        # --- for parallel, it is easy
        exppos_new[1:,:] = exppos[1:,:] + dif
        dist = 0
        # --- if we are in perpendicular the situation is more complicated
        if (nodmode == 'PERPENDICULAR') & (nodpos == "both"):

            # --- first assume that the beam is AA:
            if (exppos_new[3,0] < s[0]) & (exppos_new[3,1] < s[1]):

                if verbose:
                    print(" -  FIND_BEAM_POSITONS: Test whether found beam is AA...")

                params, _, _ = _find_source(im, sign=1, fitbox=fitbox,
                                                 guesspos=exppos_new[3,:],
                                                 searchbox=searchbox,
                                                 plot=plot, verbose=verbose,
                                                 maxFWHM=maxFWHM,
                                                 method='mpfit',
                                                 minFWHM=minFWHM)

                fitpos[3,:] = params[2:4]

                dist = np.sqrt((fitpos[3,0] - exppos_new[3,0])**2 +
                              (fitpos[3,1] - exppos_new[3,1])**2)

                if verbose:
                   print(" -  FIND_BEAM_POSITONS:  New expected pos: ", exppos_new[3,:])
                   print(" -  FIND_BEAM_POSITONS:  Beam found at: ", fitpos[3,:])
                   print(" -  FIND_BEAM_POSITONS:  Distance: ", dist)

            # --- if in that case BB would be off-chip or the above fit failed
            #     assume that BB was found
            if ((exppos_new[3,0] > s[0]) | (exppos_new[3,1] > s[1])
                | (dist > tol)):

                if verbose and dist > tol:
                    print(" -  FIND_BEAM_POSITONS: No beam found at expected position. Now assume found beam was BB...")
                elif verbose:
                    print(" -  FIND_BEAM_POSITONS: Found beam seems to belong to BB. Try to find AA now...")


                exppos_new[3,:] = exppos_new[0,:]
                dif = exppos_new[3,:] - exppos[3,:]
                exppos_new[0:3,:] = exppos[0:3,:] + dif

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

        # --- if the found position is deviating too much,  look
        if dist > tol:
            print(" -  FIND_BEAM_POSITONS: ERROR beams not found! Return -1")
            fitpos[:,:] = -1
        else:

            # --- try again to get the exact beam positions
            idond = np.where((exppos_new[:,0] > 0) & (exppos_new[:,0] < s[0]) & (exppos_new[:,1] > 0) & (exppos_new[:,1] < s[1]))[0]

            nbond = len(idond)

            idnond = []

            if verbose:
                print(" - FIND_BEAM_POSITONS: Number of beams expected to be on the detector: ", nbond)

            for i in range(nb):

                if i not in idond:
                    idnond.append(i)

                else:
                    beamsign = (-1)**(np.ceil(0.5*i))
        # print("beamsign: ",beamsign)

                    if verbose:
                        print("\n\n\n")
                        print(" - FIND_BEAM_POSITONS: Beam/sign: ", i, beamsign )

                    params, _, _ = _find_source(im, sign=beamsign, fitbox=fitbox,
                                         guesspos=exppos_new[i,:],
                                         searchbox=searchbox, guessmeth='max',
                                         method='mpfit',
                                         plot=plot, verbose=verbose,
                                         maxFWHM=maxFWHM, minFWHM=minFWHM)

                    fitpos[i,:] = params[2:4]


                    fitparams = params


            if verbose:
                print(" - FIND_BEAM_POSITONS: Fitted beam positions: ", fitpos[idond])

            if nbond < nb:
                avdif = np.mean(fitpos[idond] - exppos[idond], axis=0)

                for i in range(len(idnond)):
                    fitpos[idnond[i]] = exppos[idnond[i]] + avdif


            # --- check that the values could make sens:
            difs = fitpos - exppos_new
            if (np.std(difs[:,0]) > tol) | (np.std(difs[:,1]) > tol):
                print(" - FIND_BEAM_POSITONS: ERROR: found positions still not consistent with chop/nod pattern! Returning -1")
                fitpos[:,:] = -1

    return(fitpos, fitparams)

