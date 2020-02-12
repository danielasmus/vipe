#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.1"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-11: variable noddir renamed to noddir


NOTES:
    -

TO-DO:
    -
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Column
from astropy.stats import sigma_clip


from .calc_jitter import calc_jitter as _calc_jitter
from .calc_beampos import calc_beampos as _calc_beampos
from .crop_image import crop_image as _crop_image
from .find_beam_pos import find_beam_pos as _find_beam_pos
from .fits_get_info import fits_get_info as _fits_get_info
from .find_source import find_source as _find_source
from .print_log_info import print_log_info as _print_log_info
from .read_raw import read_raw as _read_raw
from .simple_image_plot import simple_image_plot as _simple_image_plot
from .simple_nod_exposure import simple_nod_exposure as _simple_nod_exposure

#%%
# --- HELPER ROUTINE

def subtract_chops(ima=None, fin=None, ndit=None, superdit=None, fout=None,
                   head=None, ditsoftaver=1, method='averchop',
                   datatype=None, firstchoppos=None, choppos='both',
                   verbose=False, mediansubtract=False, debug=False):
    """
    takes a VISIR burst raw cube or half-cyc extension file and outputs a
    new cube with chop A - chop B for each cycle whereas the exact was of
    subtracting the DITs and which chopping position(s) to be returned can be
    selected.
    - ima: (optional) input cube
    - fin: (optional) input raw fits file
    - ndit: (optional) number of DITs per chopping halfcycle not taking into
            account the super-DIT
    - superdit: (optional) hardware DIT averaging
    - fout: (optional) output fits file for the chop-subtraced cube
    - head: (optional) header of the cube
    - ditsoftaver: (default=1) software averaging of DITs (only relevant for
                   burst mode) WARNING: works only in the 'plusndit' mode!!!
    - method: (default='averchop') method of which off-chop image to subtract
              (only relevant for burst):
                'averchop' : average of the next chopping position
                'plusndit' : the DIT that is NDITs away in the future
                'closest' : the closest off-chop DIT in time
                'averall' : the average of all off-chop DITs of the whole cube
                'medall' : the median of all off-chop DITs of the whole cube
    - datatype: (optional; 'burst', 'halfcyc'): halfcyc or burst data?
    - firstchoppos: (optional; 'A', 'B'): chopping position of the very first
                    frame in cube (only for halfcyc). In burst, always seems
                    to be 'B'
    - choppos: ('A', 'B', 'both'; default='both') which chopping positions
               should be returned. E.g., in case of chopping off-chip, only one
               position is relevant
    """

    if debug:
        verbose = True

    if ima is None:
        ima, head, datatype, firstchoppos = _read_raw(fin)

    if datatype is None:
        datatype = _fits_get_info(head, "datatype")

    if firstchoppos is None and datatype == "halfcyc":
        if head["HIERARCH ESO DET FRAM TYPE"] == "HCYCLE2":
            firstchoppos = "B"

        else:
            firstchoppos = "A"

    s = np.array(np.shape(ima))

    if verbose:
        print("SUBTRACT_CHOPS: datatpye: ", datatype)
        print("SUBTRACT_CHOPS: method: ", method)
        print("SUBTRACT_CHOPS: firstchoppos: ", firstchoppos)
        print("SUBTRACT_CHOPS: dimension of raw: ", s)


    if datatype == 'burst':

        if ndit is None:
            ndit = head["HIERARCH ESO DET NDIT"]


        if debug:
            ima = ima[0:2*ndit]
            s[0] = 2*ndit


        # --- super DIT = DIT averaging from the hardware side
        if superdit is None:
            superdit = head["HIERARCH ESO DET NAVRG"]

        # --- effective number of (super-)DITs per chopping half cycle
        ndit = int(ndit/superdit)

        # --- compute how many software-averaged DITs per chopping half cycle
        ndita = ndit / ditsoftaver

        # --- compute exact number of output frames
        nchops = s[0]/ndit

        z = nchops * ndita



        if verbose:
            print("SUBTRACT_CHOPS: ndit, superdit: ", ndit, superdit)
            print("SUBTRACT_CHOPS: no of chops: ", nchops)
            print("SUBTRACT_CHOPS: ditsoftaver, result. ndit: ", ditsoftaver, ndita)

        if ndit % ditsoftaver > 0:
            print("SUBTRACT_CHOPS: WARNING: selected DIT averaging does not" +
                  "match effective NDIT: ", ditsoftaver, ndit)
            print(" --> including remaining DITs into last average of " +
                  "sequence")

        if choppos != 'both':
            z = z/2

        # --- determine the indices to include into the output in case
        #     only one chop pos is to be returned
        #     for the case of software averaging only include the first indices
        #     of each group to be averaged. The second condition is to cut of
        #     those DITs that do not fit into the averaging
        if choppos == 'B':
            # ids = np.array(range(int(s[0]/ditsoftaver))) % (2 * ndit) < ndit
            ids = np.array([x for x in range(s[0]) if (x % (2 * ndit) < ndit) &
                            (x % ndit * 1.0/ ditsoftaver in range(ndita))])
        elif choppos == 'A':
            # ids = np.array(range(int(s[0]/ditsoftaver))) % (2 * ndit) >= ndit
            ids = np.array([x for x in range(s[0]) if (x % (2 * ndit) >= ndit)
                            & (x % ndit * 1.0/ ditsoftaver in range(ndita))])
        else:
            ids = np.array([x for x in range(s[0])
                            if (x % ndit * 1.0/ ditsoftaver in range(ndita))])

        if verbose:
            print("SUBTRACT_CHOPS: number selected indices: ", len(ids))
            print("SUBTRACT_CHOPS: number of output frames: ", z)

        outim = np.zeros([z, s[1], s[2]], dtype='float32')

        # use #+ndit as subtrahend:
        if method == 'plusndit':

            for j in tqdm(range(z)):
#            for j in range(10):


                i = ids[j]

                mod = i % ndit  # which DIT in this chop half cycle

                # --- for the last averaging group of a half cycle use all
                #     remaining DITs even if more than ditsoftaver
                if mod/ditsoftaver == ndita-1:
                    di = ndit - mod
                else:
                    di = ditsoftaver

#                print(j, i, mod, di, mod/ditsoftaver == ndita-1)

                if i < s[0] - ndit:
                    outim[j, :, :] = np.mean(1.0*ima[i : i + di, :, :]
                                             - 1.0* ima[i + ndit : i + ndit +
                                                   di, :, :],
                                             axis=0)

                # --- special treatment of last chopping half cycle
                else:
                    outim[j, :, :] = np.mean(1.0*ima[i : i + di, :, :]
                                             - 1.0*ima[i - ndit : i - ndit +
                                                   di, :, :],
                                             axis=0)

        # using always the closest single frame in time from the alternate
        # chopping position
        elif method == 'closest':

            for j, i in tqdm(enumerate(ids)):

                mod = i % ndit  # which DIT in this chop half cycle

                if (i < ndit):  # first half cycle
                    outim[j, :, :] = 1.0*ima[i, :, :] - 1.0*ima[ndit, :, :]

                elif (i > s[0] - ndit):  # last half cycle
                    outim[j, :, :] = 1.0*ima[i, :, :] - 1.0*ima[i - mod - 1, :, :]

                elif (mod < ndit / 2):
                    outim[j, :, :] = 1.0*ima[i, :, :] - 1.0*ima[i - mod - 1, :, :]

                else:
                    outim[j, :, :] = 1.0*ima[i, :, :] - 1.0*ima[i + ndit - mod, :, :]

        # subtract the average of the next chopping half cycle position
        elif method == 'averchop':

            for j, i in tqdm(enumerate(ids)):

                mod = i % ndit

                if i < s[0] - ndit:

                    outim[j, :, :] = (ima[i, :, :]
                                      - np.mean(ima[i + ndit - mod:i + ndit
                                                    - mod + ndit, :, :],
                                                axis=0))

                else:   # last half cycle
                    outim[j, :, :] = (ima[i, :, :]
                                      - np.mean(ima[i - mod - ndit : i - mod,
                                                    :, :],
                                                axis=0))

        # subtract the average of the all frames at the alternate chopping
        # position
        elif method == 'averall':

            # find all frames in chop position A
            chopa = np.array(range(s[0])) % (2 * ndit) >= ndit
            chopb = np.array(range(s[0])) % (2 * ndit) < ndit

            mimchopa = np.mean(ima[chopa, :, :], axis=0)
            mimchopb = np.mean(ima[chopb, :, :], axis=0)

            if choppos == 'both':
                outim[chopa, :, :] = ima[chopa, :, :] - mimchopb
                outim[chopb, :, :] = ima[chopb, :, :] - mimchopa
            elif choppos == 'A':
                outim[:,:,:] = ima[chopa, :, :] - mimchopb
            else:
                outim[:, :, :] = ima[chopb, :, :] - mimchopa


        elif method == 'medall':

            # find all frames in chop position A
            chopa = np.array(range(s[0])) % (2 * ndit) >= ndit
            chopb = np.array(range(s[0])) % (2 * ndit) < ndit

            mimchopa = np.nanmedian(ima[chopa, :, :], axis=0)
            mimchopb = np.nanmedian(ima[chopb, :, :], axis=0)

            if choppos == 'both':
                outim[chopa, :, :] = ima[chopa, :, :] - mimchopb
                outim[chopb, :, :] = ima[chopb, :, :] - mimchopa
            elif choppos == 'A':
                outim[:,:,:] = ima[chopa, :, :] - mimchopb
            else:
                outim[:, :, :] = ima[chopb, :, :] - mimchopa

        else:
            print("SUBTRACT_CHOPS: ERROR: no valid method specified: ", method)
            return(-1)

    if datatype == 'halfcyc':


        if debug:
            ima = ima[0:2]
            s[0] = 2

        z = s[0]/2
        outim = np.zeros([z, s[1], s[2]], dtype='float32')

        if firstchoppos == 'B':
            sign = -1
        else:
            sign = 1

        # print(sign)
        for i in tqdm(range(z)):
            outim[i, :, :] = sign * (ima[i * 2, :, :] - ima[i * 2 + 1, :, :])

    # --- it seems that there is a slight difference in the general background
    #     level between the two chopping positions in many observations
    #     this should get rid of this difference
    #     BUT: it does not make any difference because it is averaged out later
    if mediansubtract:
        med = np.nanmedian(outim, axis=[1,2])
        outim = outim - med[:, None, None]

    if fout is not None:

        fits.writeto(fout, outim, head, overwrite=True)
        if verbose:
            print("SUBTRACT_CHOPS: - Output written to: ",fout)

    return(outim)


#%%
# --- Helper routine
def find_shift(im, refpos=None, refim=None, guesspos=None, searchbox=None,
              fitbox=None, method='fast', guessmeth='max', searchsmooth=3,
              refsign=1, guessFWHM=None, guessamp=None, guessbg=None,
              fixFWHM=False, minamp=0.01, maxamp=None, maxFWHM=None,
              minFWHM=None, verbose=False, plot=False, silent=False):

    """
    detect and determine the shift or offset of one image either with respect
    to a reference image or a reference position. In the first case, the
    shift is determined with cross-correlation, while in the second a Gaussian
    fit is performed
    the convention of the shift is foundpos - refpos
    OPTIONAL INPUT
    - refpos: tuple, given (approx.) the reference position of a source that
    can be used for fine centering before cropping
    - method: string(max, mpfitgauss, fastgauss, skimage, ginsberg), giving the method that
    should be used for the fine centering using the reference source.
    'skimage' and 'ginsberg' do 2D cross-correlation
    - refim: 2D array, giving a reference image if the method for the
    centering is cross-correlation
    - fitbox: scalar, giving the box length in x and y for the fitting of the
    reference source

    """

    s = np.array(np.shape(im))

    if guesspos is None and refpos is not None:
        guesspos = refpos
    elif refpos is None and guesspos is not None:
        refpos = guesspos
    elif guesspos is None and refpos is None:
        guesspos = 0.5*np.array(s)
        refpos = guesspos

    if verbose:
        print("GET_SHIFT: guesspos: ", guesspos)
        print("GET_SHIFT: refpos: ", refpos)


    # --- if a reference image was provided then perform a cross-correlation
    if method in ['cross', 'skimage', 'ginsberg']:


        sr = np.array(np.shape(refim))
        if verbose:
            print("GET_SHIFT: input image dimension: ", s)
            print("GET_SHIFT: reference image dimension: ", sr)

        # --- for the cross-correlation, the image and reference must have the
        #     the same size
        if (s[0] > sr[0]) | (s[1] > sr[1]):
            cim = _crop_image(im, box=sr, cenpos=guesspos)

            cenpos = guesspos

#            # --- adjust the ref and guesspos
#            refpos = refpos - guesspos + 0.5 * sr
#            guesspos = 0.5 * sr

            if verbose:
                print("GET_SHIFT: refim smaller than im --> cut im")
                print("GET_SHIFT: adjusted guesspos: ", guesspos)
                print("GET_SHIFT: adjusted refpos: ", refpos)


        elif (sr[0] > s[0]) | (sr[1] > s[1]):
            cim = im
            refim = _crop_image(refim, box=s)

            cenpos = 0.5 * s

        else:
            cim = im

            cenpos = 0.5 * s


        # --- which cross-correlation algorithm should it be?
        if (method == 'cross') | (method == 'skimage'):
            from skimage.feature import register_translation

            shift, error, diffphase = register_translation(cim, refim,
                                                           upsample_factor=100)

            #print(shift,error)
            error = [error, error]  # apparently only on error value is returned?

        elif method == 'ginsberg':
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                import image_registration
            np.seterr(all='ignore') # silence warning messages about div-by-zero

            dx, dy, ex, ey = image_registration.chi2_shift(refim, cim,
                                                           return_error=True,
                                                           zeromean=True,
                                                           upsample_factor='auto')

            shift = [dy, dx]
            error = [ey, ex]

        else:
            print("GET_SHIFT: ERROR: requested method not available: ", method)
            sys.exit(0)

        fitim = None
        params = None
        perrs = None

        # --- correct shift for any difference between guesspos and refpos:
        shift = [shift[0] + cenpos[0] - refpos[0],
                 shift[1] + cenpos[1] - refpos[1]]


   # --- if not reference image is provided then perform a Gaussian fit
    else:

        # --- fit a Gaussian to find the center for the crop
        params, perrs, fitim = _find_source(im, searchbox=searchbox,
                                               fitbox=fitbox, method=method,
                                               verbose=verbose,
                                               guesspos=guesspos, sign=refsign,
                                               plot=plot, guessFWHM=guessFWHM,
                                               guessamp=guessamp,
                                               guessbg=guessbg,
                                               searchsmooth=searchsmooth,
                                               fixFWHM=fixFWHM, minamp=minamp,
                                               maxFWHM=maxFWHM,
                                               minFWHM=minFWHM,
                                               silent=silent)

        # --- compute the shift from the fit results:
        shift = np.array([params[2] - refpos[0], params[3] - refpos[1]])

        # --- eror on the shift from the fit results:
        error = np.array([perrs[2], perrs[3]])


    if verbose:
        print('GET_SHIFT: Found shift: ', shift)
        print('GET_SHIFT: Uncertainty: ', error)
        print('GET_SHIFT: Fit Params: ', params)

    return(shift, error, [params, perrs, fitim])




#%%
# --- Helper routine

def align_summary_plot(fin, fout, colours = ['red', 'blue'],
                            pwidth = 17.25/2.54,
                            logs = None,
                            ymins = None,
                            ymaxs = None,
                            boundaries=[0.08, 0.99, 0.05, 0.99, 0.03, 0.03],
                            method='fastgauss'):

    if type(fin) == list:
        nf = len(fin)
    else:
        nf = 1
        fin = [fin]


    if method in ['cross', 'skimage', 'ginsberg']:

        quants = [[[]], [[], []], [[]]]

        labels = [['med_frame'],['Xshift', 'Yshift'], ['SSIM']]


        if logs == None:
            logs = [False, False, False]

        if ymins == None:
            ymins = [None, None, None]

        if ymaxs == None:
            ymaxs = [None, None, None]


    else:

        quants = [[[], []], [[], []], [[], []], [[], []]]

        labels = [['med_frame', 'const.'], ['max.', 'SNpeak'], ['X', 'Y'],
                ['FWHM_X','FWHM_Y']]

        if logs == None:
            logs = [False, True, False, False]

        if ymins == None:
            ymins = [None, None, None, None]

        if ymaxs == None:
            ymaxs = [None, None, None, None]


    nrows = len(quants)
    ncols = 3

    nq = [len(q) for q in quants]

    for i in range(nf):
        d = ascii.read(fin[i], header_start=0, delimiter=',', guess=False)

        n = len(d)
        for j in range(n):
            for k in range(nrows):
                for l in range(nq[k]):
                    quants[k][l].append(d[labels[k][l]][j])


    #quants = np.array(quants)
    nd = len(quants[0][0])


    plt.clf()
    plt.close(1)

    figsize = (pwidth, (pwidth-1.0)/ncols * nrows + 1.0)

    plt.figure(1,figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(left=boundaries[0], right=boundaries[1], bottom=boundaries[2],
              top=boundaries[3], wspace=boundaries[4], hspace=boundaries[5])


    for y in range(nrows):

        # --- scatter plots
        ax = plt.subplot(gs[y,0])

        if labels[y][0] in ['Xshift']:
            submedian = True
        else:
            submedian = False

        for l in range(nq[y]):

            if submedian:

                med = np.nanmedian(quants[y][l])
                mquant = np.array(quants[y][l]) - med
                mlab =  labels[y][l] + ' - ' + '{:.1f}'.format(med)

            else:
                mquant = quants[y][l]
                mlab = labels[y][l]


            ax.scatter(range(nd), mquant, label = mlab, c=colours[l],
                       marker='.', alpha=0.5, linewidth=0)

            if l == 0:
                ymax = np.nanmax(mquant)
                ymin = np.nanmin(mquant)

            else:
                ymaxq = np.nanmax(mquant)
                yminq = np.nanmin(mquant)

                if ymaxq > ymax:
                    ymax = ymaxq

                if yminq < ymin:
                    ymin = yminq


        yrange = ymax - ymin

        if logs[y]:
            ax.set_yscale('log')

            if ymins[y] == None:
                ymins[y] = 10.0**(np.log10(ymin) - 0.1 * np.log10(yrange))

            if ymaxs[y] == None:
                ymaxs[y] = 10.0**(np.log10(ymax) + 0.1 * np.log10(yrange))

        else:
            if ymins[y] == None:
                ymins[y] = ymin - 0.1 * yrange

            if ymaxs[y] == None:
                ymaxs[y] = ymax + 0.1 * yrange


        # print(y, ymins[y], ymin)

        ax.set_ylim(ymins[y], ymaxs[y])
        ax.set_xlim(0, nd)

        if nq[y] > 1:
            ax.set_ylabel(labels[y][0] + ' | ' + labels[y][1])
        else:
            ax.set_ylabel(labels[y][0])

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        #ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax.legend(fontsize=6, loc='best')

        if y < nrows-1:
            ax.get_xaxis().set_ticklabels([])

        # --- histograms
        for l in range(nq[y]):
            ax = plt.subplot(gs[y,l+1])

            n, bins, patches = ax.hist(quants[y][l], edgecolor=colours[l],
                                       color=colours[l], linewidth=2, alpha=0.2,
                                       zorder=1)

            ax.get_yaxis().set_ticklabels([])

            if y < nrows-1:
                ax.get_xaxis().set_ticklabels([])

            med = np.nanmedian(quants[y][l])
            std = np.nanstd(quants[y][l])
            p25 = np.nanpercentile(quants[y][l], 25)
            p75 = np.nanpercentile(quants[y][l], 75)

            ymax = np.max(n)*1.1

            ax.plot([med, med], [0, ymax], c='black', linestyle='--',
                    linewidth=2, label='med: ' + "{:.2f}".format(med))

            ax.fill_betweenx([0, ymax], med-std, med+std, color='grey', alpha=0.3,
                             label='STD: ' + "{:.2f}".format(std), zorder=0)

            ax.plot([p25, p25], [0, ymax], c='black', linestyle='--',
                    linewidth=1, label='25%: ' + "{:.2f}".format(p25),
                    alpha=0.7)

            ax.plot([p75, p75], [0, ymax], c='black', linestyle='--',
                    linewidth=1, label='75%: ' + "{:.2f}".format(p75),
                    alpha=0.7)

            ax.set_ylim(0, ymax)
            ax.legend(fontsize=6, loc='best')


    plt.savefig(fout, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.close(1)




# %%
# --- Helper routine
def align_cube(box=None, ima=None, fin=None, ndit=None, superdit=None,
               fout=None, head=None, ext=0, summaryplot=True,
               method='mpfit', verbose=False, searchsmooth=3, plot=False,
               reffile=None, refim=None, refpos=None, cenpos=None, AA_pos=None,
               firstchoppos="B", ditsoftaver=1, crossrefim=None,
               searchbox=50, fitbox=50, fillval=float('nan'),
               offchip=None, guesspos=None, fitrestab=None, nodpos=None,
               noddir=None, debug=False,
               datatype=None, maxshift=None, onlychop=None, offnod=0,
               minFWHM=None, maxFWHM=None, maxamp=None, minamp=0.01,
               guessFWHM=None, guessamp=None, guessbg=None, logfile=None):
    """
    Takes a VISIR chop-subtracted cube (burst or halfcyc) and outputs a new
    cube with the source centered in each frame and optionally cropped
    The cross-correlation method will probably work at the moment only for the offchip chopping case, i.e., large objects
    Parameters:
        - ditsoftaver=1 :  number of DITs to be averaged before attempting to
                        to fit the reference source
        - method='fastgauss' : method to be used for the fitting
        - verbose=False : print output if True
        - searchsmooth=3  : optional smoothing before attempting to fit the reference
                      source
        - plot=False : show plots if True (much slower, so mostly for debugging)
        - refim=None : optionally provide a reference image for fitting of the
                       reference source
        - cenpos=None : center position for the crop of the output frames
        - firstchoppos : (default: "B") chopping position of the first frame,
                         should always be B for burst, and also for halfcyc
                         because this is matched in the subtract_chops routine
        - refpos=None  : optional, position of the  reference star for the
                         alignment, if not provided, cenpos is assumed
        - refbox=50 : box size for the fitting of the reference target
        - fillval=float('nan') : fill value for the borders of the aligned
                                 frames
        - offchip=None : specify whether the chopping offset position is off
                         the chip (for parallel chop/nodding, extended targets)
        - fitrestab=None : filename of the table to put the fitting values
        - nodpos=None : provide the nodding position of the frame (if not
                        given it will try to take it from the header)
        - datatype=None : provide the type of data, 'burst' or 'halfcy'
        - maxshift=None : optional set the maximum allowed shift for the
                          alingment of individual frames
        - onlychop=None : optionally extract only one chopping position and in case of offchip=TRUE it specifies that the burst cube contains only frames of that chop position
        - offnod=0  : optionally provide an alternate nodding position frame
                      for background subtraction
    """

    if debug:
        verbose = True

    # --- which method are we going to use for the alignment:
    if method in ['cross', 'skimage', 'ginsberg']:
       crosscor = True
    else:
       crosscor = False

    if crosscor:
        from skimage.measure import compare_ssim

    # --- get the basic values
    if ima is None:
        ima = fits.getdata(fin, ext=ext)

    if head is None:
        head = fits.getheader(fin)

    if datatype is None:
        if 'BURST' in head['HIERARCH ESO DET READ CURNAME']:
            datatype = 'burst'
        else:
            datatype = 'halfcyc'

    if ndit is None:
        ndit = head["HIERARCH ESO DET NDIT"]

    if superdit is None:
        superdit = head["HIERARCH ESO DET NAVRG"]

    ndit = int(ndit/superdit/ditsoftaver)

    if nodpos is None:
        nodpos = head["HIERARCH ESO SEQ NODPOS"]


    if noddir is None:
        noddir = head["HIERARCH ESO SEQ CHOPNOD DIR"]

    jitter = np.array(_calc_jitter(head=head, verbose=verbose))

    s = np.shape(ima)
    s = np.array(s)
#    s[0] = 1

    if box is None:
        box = np.array([s[1], s[2]])

    # --- test if the box provided is an integer, in which case blow up to
    #     array
    elif not hasattr(box, "__len__"):
        box = np.array([box, box])


    # --- in the case of cross-correlation, the refence position is simply the center of the image
    if crosscor and refpos is None:
        refpos = 0.5 * s[1:]

    msg = (" - ALIGN_CUBE: Input parameters:\n" +
           "     - method = " + str(method) + "\n" +
           "     - crosscor = " + str(crosscor) + "\n" +
           "     - datatype = " + str(datatype) + "\n" +
           "     - ndit = " + str(ndit) + "\n" +
           "     - superdit = " + str(superdit) + "\n" +
           "     - ditsoftaver = " + str(ditsoftaver) + "\n" +
           "     - noddir = " + str(noddir) + "\n" +
           "     - nodpos = " + str(nodpos) + "\n" +
           "     - jitter = " + str(jitter) + "\n" +
           "     - cube dimensions = " + str(s) + "\n" +
           "     - box = " + str(box) + "\n"  +
           "     - initial refpos = " + str(refpos) + "\n"
           )

    _print_log_info(msg, logfile, screen=verbose)



    # --- in order to see whether we chop off-chip, we need to compute where the second chop position would be
    if refpos is not None:
        if np.ndim(refpos) == 1:

            # --- calculate the expected beam positions, might be needed below
            beampos =  _calc_beampos(head=head)

            # --- select the right positions depending on the nodpos and noddir
            if nodpos == 'A':
                beampos = beampos[:2,:]
                diff = refpos - beampos[0,:]

            else:
                if noddir == 'PARALLEL':
                    beampos = beampos[[2,0],:]
                else:
                    beampos = beampos[2:,:]

                diff = refpos - beampos[1,:]

            refpos = beampos + diff

            msg = (" - ALIGN_CUBE: complement refpos with alternate beam position:\n" +
                   "     - computed beam positions  = " + str(beampos) + "\n" +
                   "     - complemented refpos = " + str(refpos) + "\n"
                   )

            _print_log_info(msg, logfile, screen=verbose)


    # --- if the alignment should be done with a reference source and the its position is not provided
    if refpos is None:

        msg = ("    - ALIGN_CUBE: No reference position(s) provided. Getting them from reference image...")
        _print_log_info(msg, logfile, screen=verbose)


        # --- check if a reference image (e.g., a blind addition of this cube is provided fro reference)
        if reffile is not None:
            hdu = fits.open(reffile)
            refim = hdu[0].data
            hdu.close()

        # --- if no reference image is provided, do a blind reduction of the cube to get the actual beam positions
        if refim is None:
            msg = ("No reference image provided. Do a blind reduction of provided cube...")
            _print_log_info(msg, logfile, screen=verbose)

            refim = _simple_nod_exposure(ima=ima, head=head, datatype=datatype)


        # --- look for the actual beam positions to use as a reference for the alignment
        refpos, _ = _find_beam_pos(im=refim, head=head, AA_pos=AA_pos,
                                        verbose=verbose, plot=plot)

        msg = ("    - Found positions: " + str(refpos))
        _print_log_info(msg, logfile, screen=verbose)


    # --- if no different extraction center is defined, assume that it is the
    #     same as the reference position
    if cenpos is None:
        cenpos = refpos

    # --- in case cenpos is given and we have multiple beams on the detector,
    #     we need to have a cenpos for each of the beams:
    else:
        if nodpos == 'A':
            diff = np.array(cenpos) - refpos[0,:]
        else:
            diff = np.array(cenpos) - refpos[1,:]

        cenpos =  beampos + diff


    # --- if no particular image is provided for the cross-correlation, use the reference image
    if crossrefim is None and refim is not None:
        crossrefim = refim


    # --- prepare the crossrefim image for the similarity check
    if crosscor:
        #sr = np.shape(crossrefim)

        #if sr[0] < box[0] or sr[1] < box[1]:

            #creftim = I.crop_image(crossrefim, box=box)

        #else:
            #creftim = np.copy(crossrefim)

        creftim = np.array(crossrefim, dtype='float32') - np.nanmedian(crossrefim)

        #  --- the homogenizing increased the similarity too much. Instead it is more important to compare only those pixels that have signal, i.e., the reference image should be not too big.
        #creftim = I.homogenize_image(creftim)



    # --- check whether we were chopping off chip, i.e. whether chop B can be
    #     used

    # --- check which beam is on the detector
    # --- first beam A
    if refpos[0,0]> s[1] or refpos[0,1] > s[2] or refpos[0,0] < 0 or refpos[0,1] < 0:
        if offchip is None:
            offchip = True

        if onlychop is None:
            onlychop = 'B'
        elif onlychop == 'A':
            msg = ("ALIGN_CUBE: ERROR: chop pos A is requested but apperently off-chip! Aborting...")
            _print_log_info(msg, logfile, screen=verbose)
            return(-1)

    # --- beam B
    if refpos[1,0]> s[1] or refpos[1,1] > s[2] or refpos[1,0] < 0 or refpos[1,1] < 0:
        if offchip is None:
            offchip = True
        if onlychop is None:
            onlychop = 'A'
        elif onlychop == 'B':
            msg = ("ALIGN_CUBE: ERROR: chop pos B is requested but apperently off-chip! Aborting...")
            _print_log_info(msg, logfile, screen=verbose)
            return(-1)

    # --- if none of the above conditions are triggered then offchip should be false (2019-05-29)
    if offchip is None:
        offchip = False


    #if offchip is None:
        #if ((np.max(refpos[:, 0]) > winy) | (np.min(refpos[:, 0]) < 0)
            #| (np.max(refpos[:, 1]) > winx) | (np.min(refpos[:, 1]) < 0)):
            #offchip = True
        #else:
            #offchip = False

    # --- in case of half-cyc data we have only one frame for chop A - B, and
    #     thus need to double the size of the cube for the aligned source
    #     images
    if datatype == 'halfcyc' and not offchip:
        nf = 2 * s[0]
#
#        if firstchoppos is None:
#            if head["HIERARCH ESO DET FRAM TYPE"] == "HCYCLE2":
#                firstchoppos = "A"
#
#            else:
#                firstchoppos = "B"

    else:
        nf = s[0]
#        firstchoppos = "B"


    msg = (' - ALIGN_CUBE: derived parameters:\n' +
           "      - final cenpos = " + str(cenpos) + "\n" +
           "      - off-chip = " + str(offchip) + "\n" +
           "      - onlychop = " + str(onlychop) + "\n" +
           "      - firstchoppos = " + firstchoppos + "\n" +
           "      - nf = " + str(nf) + "\n"
           )

    _print_log_info(msg, logfile, screen=verbose)


    if debug:
        nf = 2 * ndit

    # --- prepare output data cube, do not use double precision to safe space
    outim = np.zeros([nf, int(box[0]), int(box[1])], dtype='f')

    # --- output table for the alignment results
    if fitrestab is not None:
        f = open(fitrestab, 'w')

        if crosscor:  # for cross-correlation file is different
            f.write('frame_No, med_frame, mean_frame, Xshift, Yshift, Xerr, Yerr, SSIM\n')
        else:
            f.write('frame_No, med_frame, mean_frame, bgstd, const., max., '+
                    'X, Y, FWHM_X, FWHM_Y, rotangle, SNpeak\n')


    # --- go through all frames in the cube and crop a fine-centered image
    for i in tqdm(range(nf)):
    #for i in range(10):

        # --- establish the position and sign to look for the centering
        #     reference and the target position to be extracted

        # --- either on-chip chopping or off-chip chopping for burst with cube containing frames for both chop positions
        if offchip is False or (offchip and onlychop is not None):

            # --- WARNING: on-chip with half cycles is not tested!
            if (datatype == 'halfcyc'):
                j = int(i/2)
                chopsign = (-1)**(i+1)   # chopsign is positive for chop pos B
                offnodsign = -1*chopsign

                if firstchoppos == 'B':
                    posid = (i + 1) % 2
                else:
                    posid = i % 2

                userefpos = refpos[posid,:]
                usecenpos = cenpos[posid,:]

            if (datatype == 'burst'):
                j = i
                chopsign = 1

                if (((i % (2 * ndit) < ndit) and (firstchoppos == 'A'))
                    | ((i % (2 * ndit) >= ndit) and (firstchoppos == 'B'))):
                    posid = 0  # chop pos A
                    offnodsign = 1
                    if onlychop == 'B':
                        continue
                else:
                    posid = 1  # chop pos B
                    offnodsign = -1
                    if onlychop == 'A':
                        continue

                userefpos = refpos[posid,:]
                usecenpos = cenpos[posid,:]


        #  --- off-chip chopping for half-cycle or burst with cube containing frames from only one chop position
        else:

            j = i
            chopsign = 1
            offnodsign = 1

            if np.ndim(refpos) == 2:
                userefpos = refpos[0,:]
            else:
                userefpos = refpos

            if np.ndim(cenpos) == 2:
                usecenpos = cenpos[0,:]
            else:
                usecenpos = cenpos

        if verbose is True:
            print(" - i | j | sign: ", i, j, chopsign)
            print("userefpos: ",userefpos)
            print("usecenpos: ",usecenpos)


        #  --- construct image with right sign and optionally nod subtracted
        cim = chopsign*ima[j] - offnodsign * offnod

        # === Now determine the shift
        if guesspos is None:
            iguesspos = userefpos + jitter
        else:
            iguesspos = guesspos + jitter

        shift, error, fit_res = find_shift(cim, refpos=userefpos,
                                                   refim=crossrefim,
                                                   guesspos=iguesspos,
                                                   searchbox=searchbox,
                                                   searchsmooth=searchsmooth,
                                                   fitbox=fitbox,
                                                   method=method,
                                                   maxFWHM=maxFWHM,
                                                   minFWHM=minFWHM,
                                                   maxamp=maxamp,
                                                   minamp=minamp,
                                                   verbose=verbose, plot=plot,
                                                   guessFWHM=guessFWHM,
                                                   guessamp=guessamp,
                                                   guessbg=guessbg)

        outim[i, :, :] = _crop_image(cim, box=box, cenpos=usecenpos+shift)

        medframe = np.nanmedian(outim[i])
        meanframe = np.nanmean(outim[i])

        # --- for recording, subtract the jitter from the shift
        shift = shift - jitter

        # --- write fit results into a table
        if fitrestab is not None:

            if crosscor:

                # --- prepare the testimage for the similariy check
                testim = _crop_image(cim, box=np.shape(creftim), cenpos=usecenpos+shift)
                #testim = I.homogenize_image(testim)

                # --- subtract the median to be measure the SSIM independent of spatially constant background changes
                testim = testim - np.nanmedian(testim)

                # --- replace NaNs because the SSIM algorithm can not deal with them
                testim[np.isnan(testim)] = 0



                # --- we use here the SSIM instead of the correlation coefficient, e.g., np.corrcoef(a1.flat, a2.flat)[0,1] because the latter seems to be not sensitive enough to distortions owing to bad seeing.
                testim = np.array(testim, dtype='float32')
                ssim = compare_ssim(creftim, testim)

                f.write(
                        str(i) + ', ' +
                        str(medframe) + ', ' +
                        str(meanframe) + ', ' +
                        str(shift[1]) + ', ' +
                        str(shift[0]) + ', ' +
                        str(error[1]) + ', ' +
                        str(error[0]) + ', ' +
                        str(ssim) +
                        "\n")

            else:

                # --- correct the fit position for the expectaction
                fit_res[0][2] = fit_res[0][2] - userefpos[0] - jitter[0]  # x coord
                fit_res[0][3] = fit_res[0][3] - userefpos[1] - jitter[1]  # y coord

                f.write(
                        str(i) + ', ' +
                        str(medframe) + ', ' +
                        str(meanframe) + ', ' +
                        str(fit_res[1][0]) + ', ' +
                        str(fit_res[0][0]) + ', ' +
                        str(fit_res[0][1]) + ', ' +
                        str(fit_res[0][3]) + ', ' +
                        str(fit_res[0][2]) + ', ' +
                        str(fit_res[0][4]) + ', ' +
                        str(fit_res[0][5]) + ', ' +
                        str(fit_res[0][6]) + ', ' +
                        str(fit_res[0][1]/fit_res[1][0]) +
                        "\n")

        if plot is True:
            plt.imshow(outim[i, :, :], origin='bottom')
            plt.show()

    if fitrestab is not None:
        f.close()


    if summaryplot:

        fplot = fitrestab.replace(".csv", ".pdf")
        try:
            align_summary_plot(fitrestab, fplot, method=method, ymins=None, ymaxs=None)

        except:
           print("ALIGN_CUBE: ERROR: could not make summary plot!")

    # --- only print out non-empty frames
    id = np.where(np.nansum(np.nansum(outim, axis=1), axis=1) != 0)[0]
    outim = outim[id,:,:]

    if fout is not None:
        fits.writeto(fout, outim, head, overwrite=True)

    return(outim)

#%%
# --- HELPER ROUTINE
def merge_reduced_cube(ima=None, fin=None, fout=None, head=None, ext=0,
                       method='mean',fitrestab=None, maxshift=None,
                       maxFWHM=None, minFWHM=None, returnstats=False,
                       sigma=3, foutcube=None, maxbgdev=None,
                       maxframedev=None):

    """
    takes a VISIR chop-subtracted (and aligned) cube (burst or halfcyc)
    and merges all frames in the cube and outputs the resulting frame.
    Optionally, a frame selection can be done as well. For this a table with
    the fit parameters is needed.
    """

    if ima is None:
        hdu = fits.open(fin)
        ima = hdu[ext].data
        hdu.close()

    s = np.array(np.shape(ima))

    # --- check whether frame selection is requested
    if maxshift is not None or maxFWHM is not None or minFWHM is not None:

        if fitrestab is None:
            fitrestab = fin.replace("align.fits", "fitres.csv")

        good = np.ones(s[0], dtype='bool')

        data = ascii.read(fitrestab, header_start=0, delimiter=',',
                          guess=False)

        med_x = np.nanmedian(data["X"])
        med_y = np.nanmedian(data["Y"])
        med_FWHM_x = np.nanmedian(data["FWHM_X"])
        med_FWHM_y = np.nanmedian(data["FWHM_Y"])
        med_max = np.nanmedian(data["max."])
        med_const = np.nanmedian(data["const."])

        if "med_frame" in data.colnames:
            med_cube = np.nanmedian(data["med_frame"])

        else:
            med_cube = -999


        n_bad_frame = 0
        n_bad_const = 0
        n_bad_shift = 0
        n_bad_FWHM1 = 0
        n_bad_FWHM2 = 0

        print("Median X, Y, FWHM_X, FWHM_Y, max, const, cube: ", med_x, med_y, med_FWHM_x,
              med_FWHM_y, med_max, med_const, med_cube)

        # --- discard frames where the calculated source position deviates too
        #     much from the median
        if maxframedev is not None:

            fmax = med_cube + maxframedev
            fmin = med_cube - maxframedev

            bad = ((data["mean_frame"] < fmin) | (data["mean_frame"] > fmax))

            n_bad_frame = sum(bad)
            good[bad] = False



        if maxbgdev is not None:

            cmax = med_const + maxbgdev
            cmin = med_const - maxbgdev

            bad = ((data["const."] < cmin) | (data["const."] > cmax))

            n_bad_const = sum(bad)
            good[bad] = False



        if maxshift is not None:

            xmin = med_x - maxshift
            xmax = med_x + maxshift

            ymin = med_y - maxshift
            ymax = med_y + maxshift

            # print(xmin, xmax, ymin, ymax)

            bad = ((data["X"] < xmin) | (data["X"] > xmax) | (data["Y"] < ymin)
                   | (data["Y"] > ymax))

            n_bad_shift = sum(bad)
            good[bad] = False

        # --- discard frames where the measured FWHM of the source is
        #     unrealistically small
        if minFWHM is not None:

            bad = (data["FWHM_X"] < minFWHM) | (data["FWHM_Y"] < minFWHM)
            n_bad_FWHM1 = sum(bad)
            good[bad] = False

        # ---- discard frames where the measured FWHM of the source is too
        #      large
        if maxFWHM is not None:

            bad = (data["FWHM_X"] > maxFWHM) | (data["FWHM_Y"] > maxFWHM)
            n_bad_FWHM2 = sum(bad)
            good[bad] = False

        print(" - Number of frames:", s[0])
        print(" - Number of rejected frames for: mean(frame), const, shift, "
              " minFWHM, " + "maxFWHM, total:", n_bad_frame, n_bad_const,
              n_bad_shift, n_bad_FWHM1,
              n_bad_FWHM2, s[0]-sum(good)
              )

        stats = [s[0], med_cube, med_const, med_max, med_x, med_y, med_FWHM_x,
                 med_FWHM_y, n_bad_frame,
                 n_bad_const, n_bad_shift, n_bad_FWHM1, n_bad_FWHM2,
                 s[0]-sum(good)]

        ima = ima[good]

        if 'good' in data.colnames:
            data['good'] = good
        else:
            newCol = Column(good, name='good')
            data.add_column(newCol)



        data.write(fitrestab, delimiter=',', format='ascii',
                   fill_values='', overwrite=True)

    if method == 'mean':
        outim = np.nanmean(ima, axis=0)

    if method == 'median':
        outim = np.nanmedian(ima, axis=0)

    if method == 'sum':
        outim = np.nansum(ima, axis=0)

    if method == 'sigmaclip':
        # outim = np.array(np.mean(sigma_clip(ima, sigma=sigma, axis=0, iters=2),
        #                  axis=0))
        outim = np.array(np.mean(sigma_clip(ima, sigma=sigma, axis=0,
                                            maxiters=2, masked=False),
                         axis=0))

    if fout is not None:

        if head is None:
            head = fits.getheader(fin)

        fits.writeto(fout, outim, head, overwrite=True)


    if foutcube is not None:
        fits.writeto(foutcube, ima, head, overwrite=True)

    if returnstats:
        return(outim, stats)
    else:
        return(outim)






#%%
# --- MAIN ROUTINE
def reduce_burst_file(fin, outfolder='.', outname=None, logfile=None,
                      offnodim=None, offnodfile=None, searchsmooth=3,
                      refim=None, reffile=None, box=None,
                      chopsubmeth='averchop', refpos=None, crossrefim=None,
                      AA_pos=None, alignmethod='fastgauss', verbose=False,
                      ditsoftaver=1, debug=False, plot=False):

    """
    reduce a give burst mode raw file performing chop subtraction (optionally
    nod subtraction) and align the individual frames by fitting a source in
    each or perform crosscorrelation

    PARAMETERS:
     - fin: raw fits file with burst data
     - outfolder: (default='.') folder to write output into
     - outname: (optional) provide a name for the output files
     - logfile: (optional) provide a log file to write into
     - offnodim: (optional) provide an image for the alternate nod position to be subtracted
     - offnodfile: (optional) provide a fits file containing the image of the alternate nod position
     - refim: (optional) provide a reference image to find the beam position (normally a blindly added reduction of the burst cube)
     - reffile: (optional) provide a fits file containing the reference image
     - box: (optional) by default the subimage will have the size of the chopthrow
     - chopsubmeth: (default='averchop') method for the subtraction of the chops in burst mode. By default the average of the exposures in the offset position of the corresponding pair is used.
     - refpos: (optional) instead of searching for the beam position, provide one directly for the alignment
     - AA_pos: (optional) specify position of the beam in chop/nod position AA for burst alignment
     - alignmethod: (default='fastgauss') specify the fitting algorithm for the frame alignment in burst mode data
     - verbose = False

    TO BE ADDED LATER:
        - include half-cycle data?
        - proper modification of the headers of the products

    """

    if debug:
        verbose = True

    # --- read in data of the raw cube
    cube, head, datatype, firstchoppos = _read_raw(fin)

    # --- test whether output folder exists
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if outname is None:
        outname = fin.split('/')[-1]


    if logfile is None:
        logfile = outfolder + '/' + outname.replace(".fits", ".log")
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info("Burst reduction of " + outname, logfile, mode=mode)

    s = np.shape(cube)
    msg = (" Raw cube dimensions: " + str(s))
    _print_log_info(msg, logfile)

    # --- compute the maximum field of view from half of
    chopthrow = float(head["HIERARCH ESO TEL CHOP THROW"])
    pfov = float(head["HIERARCH ESO INS PFOV"])
    if not box:
        box = int(np.floor(chopthrow / pfov))
    msg = (" - Max box size: "+str(box))
    _print_log_info(msg, logfile)

    # --- chop subtraction ---
    msg = ("   - Subtract chops...")
    _print_log_info(msg, logfile)

    subcube = subtract_chops(ima=cube, method=chopsubmeth, head=head,
                             datatype=datatype, ditsoftaver=ditsoftaver,
                             verbose=verbose, debug=debug)

    s = np.shape(subcube)
    msg = (" Chop. sub. cube dimensions: " + str(s))
    _print_log_info(msg, logfile)

    # --- nod subtraction ---
    if offnodfile is not None:

        hdu = fits.open(offnodfile)
        offnodim = hdu[0].data
        hdu.close()

    if offnodim is not None:

        msg = ("    - Subtract nods...")
        _print_log_info(msg, logfile)

        for j in range(s[0]):
            subcube[j,:,:] = subcube[j,:,:] - offnodim[:,:]


    # --- now perform the alignment
    msg = ("    - Align cube...")
    _print_log_info(msg, logfile)

    atab = outfolder + "/" + outname.replace(".fits", "_fitres.csv")
    fout = outfolder + "/" + outname.replace(".fits", "_align.fits")

    if ditsoftaver > 1:
        atab = atab.replace(".fits", "_aver" + str(ditsoftaver) + ".csv")
        fout = fout.replace(".fits", "_aver" + str(ditsoftaver) + ".fits")

    ## --- for debugging
    if debug:
        dout = outfolder + "/" + outname.replace(".fits", "_subcube.fits")
        fits.writeto(dout, subcube, head, overwrite=True)

    acube = align_cube(box=box, ima=subcube, datatype=datatype,
                       head=head, method=alignmethod, ditsoftaver=ditsoftaver,
                       verbose=verbose, searchsmooth=searchsmooth, plot=plot,
                       fitrestab=atab, refim=refim, reffile=reffile, AA_pos=AA_pos,
                       fout=fout, crossrefim=crossrefim, logfile=logfile, debug=debug)

    msg = ("       - Output written to: " + fout)
    _print_log_info(msg, logfile)


    msg = ("    - Merge cube...")
    _print_log_info(msg, logfile)

    # --- do both a mean and a median image in parallel
    fout = outfolder + "/" + outname.replace(".fits", "_mean.fits")
    if ditsoftaver > 1:
        fout = fout.replace(".fits", "_aver" + str(ditsoftaver) + ".fits")


    mergeim = merge_reduced_cube(ima=acube, fout=fout,
                                 head=head, method='mean')

    msg = ("       - Output written to: " + fout)
    _print_log_info(msg, logfile)

    fout = fout.replace(".fits", ".png")
    _simple_image_plot(mergeim, fout, log=True)

    fout = outfolder + "/" + outname.replace(".fits", "_median.fits")
    if ditsoftaver > 1:
        fout = fout.replace(".fits", "_aver" + str(ditsoftaver) + ".fits")

    mergeim = merge_reduced_cube(ima=acube, fout=fout,
                                 head=head, method='median')

    msg = ("       - Output written to: " + fout)
    _print_log_info(msg, logfile)

    fout = fout.replace(".fits", ".png")
    _simple_image_plot(mergeim, fout, log=True)

