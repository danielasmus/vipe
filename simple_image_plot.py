#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.0.0"

"""
USED BY:
    - reduce_exposure
    - reduce_indi_raws

HISTORY:
    - 2020-01-15: created by Daniel Asmus
    - 2020-01-23: remove the maxbg option for the vipe package
    - 2020-02-12: added parameters: scale, permin, permax, pfov, x/ylabel,
                  majtickinterval, tickcol, cenax


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
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
# from matplotlib import patheffects  # does not work for the axis


def simple_image_plot(im, fout, scale="lin", percentile=None, permin=None,
                      permax=None, vmin=None, vmax=None, cmap='gnuplot2',
                      pwidth=None, pfov=None, xlabel=None, ylabel=None,
                      majtickinterval=None, tickcol=None, cenax=False):
    """
    Make a simple plot of an image and write it to a file

    """

    ny, nx = im.shape

    # --- log scaling
    if scale == "log":

        # print("log")
        dim = np.log10(1000.0 * (im - np.nanmin(im)) /
                          (np.nanmax(im) - np.nanmin(im)) + 1)

        # --- set the min and max values as percentiles
        if percentile is None:
            if permin is None:
                permin = 1
            if permax is None:
                permax = 0.0001
        else:
            permin = percentile
            permax = percentile


    # --- linear scaling
    else:
        # print("lin")
        dim = im

        if percentile is None:
            if permin is None:
                permin = 1
            if permax is None:
                permax = 0.0001
        else:
            permin = percentile
            permax = percentile

    # --- set the absolute min max values for the plot
    if vmin is None:
        vmin = np.nanpercentile(dim, permin)

    if vmax is None:
        vmax = np.nanpercentile(dim, 100-permax)

    # print(vmin)
    plt.clf()

    if pwidth is None:
        pwidth = 16*nx/1024.0

    pheight = pwidth * ny/nx

    fig = plt.figure(figsize=(pwidth, pheight))

    # path_effects=[patheffects.withStroke(linewidth=3, foreground='red')]

    ax = plt.subplot(111)

    # ax.set_path_effects(path_effects)

    handle = ax.imshow(dim, origin='lower',
               interpolation='nearest',
               vmin=vmin, cmap=cmap,
               vmax=vmax)



    # --- set the image extent for the axis labeling
    if cenax:
        if pfov is not None:
            xmin =  (nx/2 -1) * pfov
            xmax =  -nx/2 * pfov
            ymin = -ny/2 * pfov
            ymax = (ny/2 -1) * pfov
            if xlabel is None:
                xlabel = 'x offset ["]'
            if ylabel is None:
                ylabel = 'y offset ["]'
        else:
            xmax =  (nx/2 -1)
            xmin =  -nx/2
            ymin = -ny/2
            ymax = (ny/2 -1)
            if xlabel is None:
                xlabel = 'x offset [px]'
            if ylabel is None:
                ylabel = 'y offset [px]'

        extent = [xmin, xmax, ymin, ymax]

        handle.set_extent(extent)


        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        xmin = 0
        ymin = 0
        xmax = nx
        ymax = ny

    # --- ticks
    # if majtickinterval is None:

    #     majorLocator = MultipleLocator(1)

    #     if ymax - ymin < 2:
    #         majorLocator = MultipleLocator(0.5)
    #     elif (ymax - ymin > 10) & (ymax - ymin <= 20):
    #         majorLocator = MultipleLocator(2)
    #     elif (ymax - ymin > 20) & (ymax - ymin <= 100):
    #         majorLocator = MultipleLocator(5)
    #     elif (ymax - ymin > 100) & (ymax - ymin <= 200):
    #         majorLocator = MultipleLocator(10)
    #     elif (ymax - ymin > 200) & (ymax - ymin <= 1000):
    #         majorLocator = MultipleLocator(20)
    #     elif ymax - ymin > 1000:
    #         majorLocator = MultipleLocator(100)


    # else:
    #     majorLocator = MultipleLocator(majtickinterval)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.xaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.yaxis.set_major_locator(majorLocator)

    if tickcol is None:
        if cmap=='gnuplot2':
            tickcol = "white"
        elif permin == permax:
            tickcol = "black"
        else:
            tickcol = "lime"

    ax.yaxis.set_tick_params(color=tickcol, which='both')
    ax.xaxis.set_tick_params(color=tickcol, which='both')

    plt.savefig(fout, bbox_inches='tight', pad_inches=0.01)
#    plt.savefig(fout)

    plt.clf()
    plt.close(fig)

