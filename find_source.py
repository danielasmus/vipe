#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.1.0"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-15: created by Daniel Asmus
    - 2020-01-15: explicitely import local gaussfit
    - 2020-01-23: rename from get_pointsource to find_source


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from astropy.modeling import models, fitting

from .gaussfitter import gaussfit as _gaussfit

from .crop_image import crop_image as _crop_image
from .measure_bkg import measure_bkg as _measure_bkg


def find_source(im, guesspos=None, searchbox=None, fitbox=None,
                    guessmeth='max', smooth=0, searchsmooth=3,
                    guessFWHM=None, guessamp=None, guessbg=None,
                    method='fast', sign=1, verbose=False, fixFWHM=False,
                    fixpos=False, minamp=0.01, maxamp=None, plot=False,
                    maxFWHM=None, minFWHM=None, posradius=None, silent=False):

    """
    find the point source in an image and provide its best fit parameters from
    a Gaussian fit
    Parameters:
        - im: image to be searched
        - guesspos: 2D array with the (y, x) guessed position for the point
                    source position. If None, use the middle of the whole image
        - searchbox: int or 2D array with (y, x) size, size of box to be
                     searched centered on the guesspos. If None, search the
                     whole image
        - fitbox: int or 2D array with (y, x) size, size of box used for the
                  fit. If None, use searchbox
        - guessmeth: method for guesstimating the point source position
                     (currently only 'max' and other). For 'max' use the
                     maximum brightness pixel in the searchbox. Other: use
                     centre of the searchbox
        - smooth: optional smoothing with a Gaussian. The value specifies the
                  width of the Gaussian used for the smoothing
        - searchsmooth: optional smoothing only for guesstimating source
                        position
        - guessFWHM: guess for the FWHM of the source. If None, use middle
                     between minFWHM and maxFWHM or 5 if the former are not
                     provided
        - guessamp: guess for the amplitude of the source. If None, use the
                    pixel brightness at the guess position
        - guessbg: guess for the background level in the image. If None, use
                   the median of the image.
        - method: method used for the fitting: 'fast': use LevMarLSQFitter
                  from astropy, 'mpfit': use the MPFIT package
        - sign:   sign of the point source to be searched
        - fixFWHM: fix the FWHM of the source to the guess value
        - fixpos: fix the position of the source to the guess value
        - minamp, maxamp: minimum and maximum allowed values for the amplitude
                          of the source
        - minFWHM, maxFWHM: minimum and maximum allowed values for the FWHM
                            of the source
        - posradius: maximum allowed radius in pix around the guess position
                     for the source location. If None, the whole searchbox is
                     allowed

    """

    s2f = (2.0 * np.sqrt(2.0*np.log(2.0)))
    f2s = 1.0/s2f

    s = np.shape(im)

    if sign < 0:
        im = -im

    if smooth > 0:
        im = gaussian_filter(im, sigma=smooth)

    if guesspos is None :
        guesspos = 0.5*np.array(s)

    if verbose:
        print("GET_POINTSOURCE: method: ", method)
        print("GET_POINTSOURCE: s: ", s)
        print("GET_POINTSOURCE: sign: ", sign)
        print("GET_POINTSOURCE: searchsmooth: ", searchsmooth)
        print("GET_POINTSOURCE: guessmeth: ", guessmeth)
        print("GET_POINTSOURCE: initial guessamp: ", guessamp)
        print("GET_POINTSOURCE: initial guessbg: ", guessbg)
        print("GET_POINTSOURCE: intial guesspos: ", guesspos)
        print("GET_POINTSOURCE: intial searchbox: ", searchbox)

    # --- define the search box
    if searchbox is not None :

        # test if the box provided is an integer, in which case blow up to array
        if not hasattr(searchbox, "__len__"):
            searchbox = np.array([searchbox, searchbox])

        searchbox = np.array(searchbox, dtype=int)

        sx0 = np.max([0, int(np.round(guesspos[1] - 0.5 * searchbox[1]))])
        sx1 = np.min([s[1], int(np.round(guesspos[1] + 0.5 * searchbox[1]))])
        sy0 = np.max([0, int(np.round(guesspos[0] - 0.5 * searchbox[0]))])
        sy1 = np.min([s[0], int(np.round(guesspos[0] + 0.5 * searchbox[0]))])
        searchim = im[sy0:sy1, sx0:sx1]

        if verbose:
            print("GET_POINTSOURCE: sy0, sy1, sx0, sx1 ", sy0, sy1, sx0, sx1)
        searchbox = np.array(np.shape(searchim))

        # print("GET_POINTSOURCE: ss: ", np.shape(searchim))

    else:
        searchbox = np.array(s, dtype=int)
        sx0 = 0
        sy0 = 0
        searchim = im

    if verbose:
        print("GET_POINTSOURCE: final searchbox: ", searchbox)

    # --- should the first guess be based on the max or on the position?
    if guessmeth == 'max':
        if searchsmooth > 0:
            # smoothing is quick and thus on by default
            ssim = gaussian_filter(searchim, sigma=searchsmooth,
                                   mode='nearest')

            guesspos = np.array(np.unravel_index(np.nanargmax(ssim),
                                                 searchbox))

            if plot is True:
                plt.figure(1, figsize=(3,3))
                plt.imshow(ssim, origin='bottom', interpolation='nearest')
                plt.title('Smoothed Search image')
                plt.show()

            # print("GET_POINTSOURCE: guesspos: ", guesspos)

        else:
            guesspos = np.array(np.unravel_index(np.nanargmax(searchim),
                                                 searchbox))

    else:
        guesspos = 0.5*searchbox

    if verbose:
        print("GET_POINTSOURCE: guesspos in searchbox: ", guesspos)
        print("GET_POINTSOURCE: guesspos in total image: ", guesspos[0] + sy0, guesspos[1] + sx0)
    # print("GET_POINTSOURCE: guesspos: ", guesspos)

    guesspos = np.array([guesspos[0] + sy0, guesspos[1] + sx0])

    if plot is True:
        plt.clf()
        plt.close(1)
        plt.figure(1, figsize=(3,3))
        plt.imshow(searchim, origin='bottom', interpolation='nearest')
        plt.title('Search image')
        plt.show()

    if verbose:
        print("GET_POINTSOURCE: intial fitbox: ", fitbox)

    # --- define the fit box
    if fitbox is not None:

        # test if the box provided is an integer, in which case blow up to array
        if not hasattr(fitbox, "__len__"):
            fitbox = np.array([fitbox, fitbox])

        fitbox = np.array(fitbox, dtype=int)
        fx0 = int(np.round(guesspos[1] - 0.5 * fitbox[1]))
        fx1 = int(np.round(guesspos[1] + 0.5 * fitbox[1]))
        fy0 = int(np.round(guesspos[0] - 0.5 * fitbox[0]))
        fy1 = int(np.round(guesspos[0] + 0.5 * fitbox[0]))

        guesspos = 0.5*fitbox
        # print('guesspos, fx0, fx1, fy0, fy1 ', guesspos, fx0, fx1, fy0, fy1)

        # for the new guess position, we have to take into account if the
        # fitbox is smaller than expected because being close to the edge

        if fx0 < 0:
            guesspos[1] = guesspos[1] + fx0
            fx0 = 0

#        if fx1 > s[1]:
#            guesspos[1] = guesspos[1] - (fx1 - s[1])
#            fx1 = s[1]

        if fy0 < 0:
            guesspos[0] = guesspos[0] + fy0
            fy0 = 0

#        if fy1 > s[0]:
#            guesspos[0] = guesspos[0] - (fy1 - s[0])
#            fy1 = s[0]

        fitim = im[fy0:fy1, fx0:fx1]
        fs = np.array(np.shape(fitim))

    else:
        fitim = im
        fx0 = 0
        fy0 = 0

    fs = np.shape(fitim)
    fitbox = fs

    if verbose:
        print("GET_POINTSOURCE: final fitbox: ", fitbox)
        print("GET_POINTSOURCE: final guesspos in fitbox: ", guesspos)


    if plot is True:
        plt.figure(1, figsize=(3,3))
        plt.imshow(fitim, origin='bottom', interpolation='nearest')
        plt.title('(Sub)image to be fitted')
        plt.show()

    if guessFWHM is None:
        if maxFWHM is not None and minFWHM is not None:
            guessFWHM = 0.5 * (maxFWHM + minFWHM)
        elif maxFWHM is not None:
            guessFWHM = 0.5 * maxFWHM
        elif minFWHM is not None:
            guessFWHM = 2 * minFWHM
        else:
            guessFWHM = 5

    # --- estimate the BG with ignoring central source (use either 3*FWHM or
    #     80% of image whatever is smaller). First generate a background image
    #     of sufficient size
    bgbox = int(np.round(6*guessFWHM))

    if verbose:
        print('bgbox:', bgbox)
        print("bgcenpos: ", [fy0 + 0.5*fitbox[0], fx0 + 0.5*fitbox[1]])

    bgim = _crop_image(im, box=bgbox,
                      cenpos=[fy0 + 0.5*fitbox[0], fx0 + 0.5*fitbox[1]],
                      exact=False)

    ignore_aper = np.min([3*guessFWHM, 0.8*np.max(s)])
    bgval, bgstd = _measure_bkg(bgim, ignore_aper=ignore_aper)

    if guessbg is None:
        guessbg = bgval

    if guessamp is None:
        guessamp = fitim[int(guesspos[0]), int(guesspos[1])] - guessbg

    if maxFWHM is None:
        maxFWHM = np.max(s)

    if minFWHM is None:
        minFWHM = 1

    maxsigma = maxFWHM * f2s
    minsigma = minFWHM * f2s

    if posradius is not None:
        minx = guesspos[1] - posradius
        maxx = guesspos[1] + posradius
        miny = guesspos[0] - posradius
        maxy = guesspos[0] + posradius
    else:
        minx = 0
        maxx = fs[1]
        miny = 0
        maxy = fs[0]

    sigma = guessFWHM * f2s
    guess = [guessbg, guessamp , guesspos[1],
             guesspos[0], sigma, sigma, 0]

    if verbose:
        print(' - GET_POINTSOURCE: Guess: ', guess)
        print(' - GET_POINTSOURCE: minFWHM: ', minFWHM)
        print(' - GET_POINTSOURCE: maxFWHM: ', maxFWHM)
        print(' - GET_POINTSOURCE: minsigma: ', minsigma)
        print(' - GET_POINTSOURCE: maxsigma: ', maxsigma)
        print(' - GET_POINTSOURCE: minamp: ', minamp)
        print(' - GET_POINTSOURCE: maxamp: ', maxamp)
        print(' - GET_POINTSOURCE: minx,maxx, miny,maxy: ', minx, maxx, miny, maxy)

    y, x = np.mgrid[:fs[0], :fs[1]]

    g_init = models.Gaussian2D(amplitude=guessamp, x_mean=guesspos[1],
                               y_mean=guesspos[0], x_stddev=sigma,
                               y_stddev=sigma)

    c_init = models.Const2D(amplitude=guessbg)

    init = g_init + c_init
    gim = init(x, y)

    if plot is True:
        plt.figure(1, figsize=(3,3))
        plt.imshow(gim, origin='bottom', interpolation='nearest')
        plt.title('Guess')
        plt.show()

    if np.isnan(fitim).any():
        if not silent:
            print("GET_POINTSOURCE: WARNING: image to be cropped contains NaNs!")
        fitim[np.isnan(fitim)] = guessbg  # set any NaNs to 0 for crop to work

    if ('mpfit' in method) :

        # params=[] - initial input parameters for Gaussian function.
        # (height, amplitude, x, y, width_x, width_y, rota)

        # parameter limits
        minpars = [0, minamp, minx, miny, minsigma, minsigma, 0]
        maxpars = [0, maxamp, maxx, maxy, maxsigma, maxsigma, 0]
        limitedmin = [False, True, True, True, True, True, False]
        limitedmax = [False, False, True, True, True, True, False]

        # ensure that the fit is positive if the sign is 1
        # (or negative if the sign is -1)

        if minamp is None:
            limitedmin[1] = False

        if maxamp:
            limitedmax[1] = True

        if fixFWHM:
            limitedmin[4] = True
            limitedmax[4] = True
            minpars[4] = sigma-0.001
            maxpars[4] = sigma+0.001
            limitedmin[5] = True
            limitedmax[5] = True
            minpars[5] = sigma-0.001
            maxpars[5] = sigma+0.001

        if fixpos:
            limitedmin[2] = True
            limitedmax[2] = True
            minpars[2] = guesspos[1]-0.001
            maxpars[2] = guesspos[1]+0.001
            limitedmin[3] = True
            limitedmax[3] = True
            minpars[3] = guesspos[0]-0.001
            maxpars[3] = guesspos[0]+0.001

        res = _gaussfit(fitim, err=None, params=guess, returnfitimage=True,
                       return_all=1, minpars=minpars, maxpars=maxpars,
                       limitedmin=limitedmin, limitedmax=limitedmax)

        params = res[0][0]

        perrs = res[0][1]
        if perrs is None:
            perrs = np.full(6,-1, dtype=float)
        fit = res[1]

    elif 'fast' in method:

        # ensure that the fit is positive
        init.amplitude_0.bounds = (minamp, maxamp)

        init.x_mean_0.bounds = (minx, maxx)
        init.y_mean_0.bounds = (miny, maxy)

        init.x_stddev_0.bounds = (minsigma, maxsigma)

        # --- ensure that angle stays in useful pounds
        #init.theta_0.bounds = (-2*np.pi, 2*np.pi)  # somehow fixing the angle does not work

        if fixFWHM:
            init.x_stddev_0.fixed = True
            init.y_stddev_0.fixed = True

        if fixpos:
            init.x_mean_0.fixed = True
            init.y_mean_0.fixed = True

        fit_meth = fitting.LevMarLSQFitter()
#        fit_meth = fitting.SimplexLSQFitter()  # very slow
#        fit_meth = fitting.SLSQPLSQFitter()  # not faster than LevMar

        g_fit = fit_meth(init, x, y, fitim, acc=1e-8)

        fit = g_fit(x, y)

        params = np.array([g_fit.amplitude_1.value, g_fit.amplitude_0.value,
                           g_fit.x_mean_0.value, g_fit.y_mean_0.value,
                           g_fit.x_stddev_0.value, g_fit.y_stddev_0.value,
                           g_fit.theta_0.value])
        perrs = params * 0  # this method does not provide uncertainty estimates

        # --- convert theta to deg:
        params[6] = params[6]/np.pi*180.0

        # --- theta measures the angle from the x-axis, so we need to add 90
        params[6] = params[6] + 90.0


        # print(init.x_stddev_0.fixed, init.x_stddev_0.bounds)
        # print(g_fit.x_stddev_0.fixed, g_fit.x_stddev_0.bounds)

    else:
        print("GET_POINTSOURCE: ERROR: non-valid method requested: " + method
              +"\n returning None")
        return(None, None, None)


    # --- use the STD of the BG for the BG level uncertainty if larger than
    #     error estimate
    if bgstd > perrs[0]:
        perrs[0] = bgstd

    if verbose:
        print("GET_POINTSOURCE: uncorrected fit params: ", params)
        print("GET_POINTSOURCE: uncorrected fit errs: ", perrs)

    # --- compute the position in the total image and switch x and y to agree
    #     with the numpy convention
    temp = np.copy(params)
    params[2] = temp[3] + fy0
    params[3] = temp[2] + fx0

    temp = np.copy(perrs)
    perrs[2] = temp[3]
    perrs[3] = temp[2]


    # --- if the y FWHM is larger than the one in x direction, switch them so
    #     that the first FWHM is the major axis one.
    if params[5] > params[4]:
        temp = params[4]
        params[4] = params[5]
        params[5] = temp

        temp = perrs[4]
        perrs[4] = perrs[5]
        perrs[5] = temp

        params[6] = params[6] + 90.0

    # --- normalise the angle
    params[6] = params[6] % 180
    if params[6] < 0:
        params[6] = params[6] + 180

    #
    if sign < 0:
        params[0] = - params[0]
        params[1] = - params[1]
        fit = - fit
        fitim = - fitim


    if verbose:
        print(" - GET_POINTSOURCE: fitted params: ", params)
    # convert sigma to FWHM for the output:
    params[4:6] = params[4:6] * s2f


    if plot is True:
        plt.figure(1, figsize=(3,3))
        plt.imshow(fit, origin='bottom', interpolation='nearest')
        plt.title('Fit with sign')
        plt.show()
        plt.close(1)

        plt.figure(1, figsize=(3,3))
        plt.imshow(fitim-fit, origin='bottom', interpolation='nearest')
        plt.title('Residual')
        plt.show()
        plt.close(1)

    ims = [fitim, fit, fitim-fit]

    return(params, perrs, ims)

