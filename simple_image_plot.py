#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.-1.0"

"""
USED BY:
    - reduce_exposure
    - reduce_indi_raws

HISTORY:
    - 2020-01-15: created by Daniel Asmus
    - 2020-01-23: remove the maxbg option for the vipe package


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


def simple_image_plot(im, fout, log=False, percentile=1,
                      binsize=None, cmap='gnuplot2', pwidth=14.17):

    if log:
        dim = np.log10(1000.0 * (im - np.nanmin(im)) /
                          (np.nanmax(im) - np.nanmin(im)) + 1)
        if binsize is None:
            binsize=0.003

    else:
        dim = im

    vmin = np.nanpercentile(dim, percentile)

    # print(vmin)
    plt.clf()

    s = np.shape(im)
    pheight = pwidth * s[0]/s[1]

    fig = plt.figure(figsize=(pwidth, pheight))


    plt.imshow(dim, origin='bottom',
               interpolation='nearest',
               vmin=vmin, cmap=cmap,
               vmax=np.nanpercentile(dim, 100-percentile))


    plt.savefig(fout, bbox_inches='tight', pad_inches=0.01)
#    plt.savefig(fout)

    plt.clf()
    plt.close(fig)

