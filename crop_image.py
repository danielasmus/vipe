#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""

import numpy as np
from scipy import ndimage



def crop_image(im, box=None, cenpos=None, NaNto=0, silent=False, exact=True,
               fill_value=float('nan')):

    """
    crop a given image with several different options:
    OPTIONAL INPUT
    - box: either scalar: box total length in x and y each, or tuple: giving
    the length in (y, x)
    - cenpos: tuple, giving the center of the crop
    """

    s = np.array(np.shape(im))

    if box is None:
        box = s

    # test if the box provided is an integer, in which case blow up to array
    if not hasattr(box, "__len__"):

        box = np.array([box, box], dtype=int)

    if cenpos is None:
        cenpos = 0.5 * s


    # --- if the new center position is between two pixels:
    if ((cenpos[0] % 1 != 0) | (cenpos[1] % 1 != 0)) & exact:

        x_i = cenpos[1] - 0.5 * box[1] + np.arange(box[1])
        y_i = cenpos[0] - 0.5 * box[0] + np.arange(box[0])
        ygrid, xgrid = np.meshgrid(x_i, y_i)
            # print(box, ccpos, np.min(x_i), np.max(x_i), np.min(y_i), np.max(y_i))
    #    cim = interpolate.interp2d(x_i, y_i, im)
        # --- the best would probably be to use Lanczos interpolation instead of
        #     the spline kernel used below (and as done in the VISIR pipeline) but
        #     I can not find a corresponding routine ready to use in python...

        # --- ndimage can not deal with NaNs in the input, which is why they have
        #    to be set to something else here for now
        if np.isnan(im).any():

            if not silent:
                print("CROP_IMAGE: WARNING: image to be cropped contains NaNs!")

            if NaNto is not None:
                im[np.isnan(im)]=NaNto  # set any NaNs to 0 for crop to work
                if not silent:
                    print("CROP_IMAGE: Replacing NaNs with: ", NaNto)

        cim = ndimage.map_coordinates(im, [xgrid, ygrid], cval=float('nan'))
            # plt.imshow(im)
            # plt.show()

    # --- otherwise do a simple crop
    else:

        cim = np.full(box, fill_value)

        # --- limit indices within the input image (take int of cenpos to avoid
        #     rounding problems for non-integer center positions)
        x0 = int((int(cenpos[1]) - 0.5 * box[1]))
        x1 = int((int(cenpos[1]) + 0.5 * box[1]))
        y0 = int((int(cenpos[0]) - 0.5 * box[0]))
        y1 = int((int(cenpos[0]) + 0.5 * box[0]))

        # --- limit indices within the output image
        cx0 = 0
        cx1 = box[1]
        cy0 = 0
        cy1 = box[0]

        # --- now check if we are without bounds somewhere and adjust limits
        #     accordingly
#        print(cy0, cy1, y0, y1)
        if x0 < 0:
            cx0 = -x0
            x0 = 0

        if x1 > s[1]:
            cx1 = box[1] - (x1 - s[1])
            x1 = s[1]

        if y0 < 0:
            cy0 = -y0
            y0 = 0

        if y1 > s[0]:
            cy1 = box[0] - (y1 - s[0])
            y1 = s[0]


#        print(cy0, cy1, y0, y1)
        # --- finally do the crop
        cim[cy0:cy1, cx0:cx1] = im[y0:y1,x0:x1]

    return(cim)
