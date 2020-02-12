#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-23: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry


def measure_sensit(fin=None, ext=0, im=None, pos=None, std=True, rmax=11,
                   sky_rin=30, sky_rout=40, flux=None, exptime=None, head=None,
                   showplot=True, verbose=False
                   ):
    """
    comute the maximum S/N of an image in a similar way as the ESO pipeline does
    """

    if fin != None:

        hdu = fits.open(fin)
        im = hdu[ext].data

        if head == None:
            head = hdu[ext].header

        hdu.close()

    # --- position to measure the S/N
    if pos == None:
        pos = 0.5 * np.asarray(im.shape)

#    print(pos)
#    from aper import aper

    # --- define circular apertures from 1 px to rmax px
    aperrads = np.arange(1,rmax)
    apers = [CircularAperture((pos[1], pos[0]), r=a) for a in aperrads]

    # --- estimate the background level and standard deviation
    # --- define a sky annulus
    annul = CircularAnnulus((pos[1], pos[0]), r_in=sky_rin, r_out=sky_rout)
    # --- make a mask corresponding to the annulus
    annul_mask = annul.to_mask(method='center')[0]
    # --- apply the annulus mask to the image
    annul_data = annul_mask.multiply(im)
    # --- extract the data from that annulus
    bg = annul_data[annul_mask.data>0]
    bgstd = np.nanstd(bg)
    bgmed = np.nanmedian(bg)

    counts = [aperture_photometry(im, a, error=bgstd)['aperture_sum'][0] - bgmed * a.area() for a in apers]
    ecnts = [aperture_photometry(im, a, error=bgstd)['aperture_sum_err'][0] for a in apers]


    sn = np.array(counts)/np.array(ecnts)

    if verbose:
        print("Aper rad. | counts | error | S/N")

        for r in aperrads:
            print(r, counts[r-1], ecnts[r-1], sn[r-1])


    # sn = np.zeros(rmax)

    # for radius in range(1, rmax):

    #     (mag, magerr, flux, fluxerr, sky, skyerr, badflag,
    #      outstr) = aper(im, pos[0], pos[1], phpadu=1000.0, apr=radius,
    #                     skyrad=[30, 40], exact=True)

    #     print(flux, fluxerr, flux/fluxerr)
    #     sn[radius-1] = flux/fluxerr

    # --- best S/N and corresponding aperture radius
    snbest = np.max(sn)
    rbest = np.argmax(sn) + 1


    # --- check if the file was a standard which allow computation of
    # sensitivity
    sens = -1

    # --- in case the image is of a standard star try to determine the reached
    #     sensitivity
    if std:
        try:
            if flux == None:
                 flux = head["HIERARCH ESO QC JYVAL"]
            if exptime == None:
                if "HIERARCH ESO QC EXPTIME" in head:
                    exptime = head["HIERARCH ESO QC EXPTIME"]
                elif "HIERARCH ESO SEQ TIME" in head:
                    exptime = head["HIERARCH ESO SEQ TIME"]

            sens = flux * 1000.0 * 10.0 * np.sqrt(exptime / 3600.0) / snbest

        except:
            print("MEASURE_SENSITIVITY: ERROR: Can not compute sensitivity because flux/time information missing")


    if showplot:
        fig = plt.figure(num=10, figsize=(8,4))

        ax0 = fig.add_subplot(1, 2, 1)

        ax0.plot(aperrads, sn)
        ax0.set_xlabel("aperture radius [px]")
        ax0.set_ylabel("S/N")

        ax1 = fig.add_subplot(1, 2, 2)

        pim = im - np.nanmin(im)

        ax1.imshow(pim, origin='lower', norm=LogNorm(),
                        interpolation='nearest',  cmap='gnuplot2',
                        vmin=np.nanpercentile(pim, 0.1),
                        vmax=np.nanmax(pim)
                        )

        apers[rbest].plot(color='lime', lw=1)
        annul.plot(color='cyan', lw=1)

        ax1.set_xlabel('x [px]')
        ax1.set_ylabel('y [px]')

        plt.show()
        plt.close(10)



    if verbose:
        print("\n")
        print("MEASURE_SENSITIVITY:\n"
              + " - background level: " + "{:.2f}".format(bgmed) + "\n"
              + " - background STD: " + "{:.2f}".format(bgstd) + "\n"
              + " - best S/N: " + "{:.1f}".format(snbest) + "\n"
              + " - for radius [px]: " + str(rbest) + "\n"
              + " - counts: " + "{:.1f}".format(counts[rbest-1]) + "\n"
              + " - ecounts: " + "{:.1f}".format(ecnts[rbest-1])
              )
        if sens != -1:
            print(" - STD flux density [Jy]: " + "{:.2f}".format(flux) + "\n"
                  + " - exposure time [s]: " + "{:.0f}".format(exptime) + "\n"
                  + " - sensitivity: " + "{:.2f}".format(sens) + "\n"
                  )
        else:
            print(" - no sensitivty determined because flux/time info missing. ")

    return(snbest, sens)
