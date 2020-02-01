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
from astropy import units as u
from astropy.coordinates import SkyCoord


def angular_distance(r1, d1, r2, d2):
    """
    return the angular distance between two positions in the sky in arcsec
    positions cn be provided in degrees (float) or in sexagesimal units (string)
    """

    if np.isreal(r1):
        r1u = u.deg
    else:
        r1u = u.hourangle

    if np.isreal(r2):
        r2u = u.deg
    else:
        r2u = u.hourangle

    d1u = u.deg
    d2u = u.deg

    c1 = SkyCoord(r1, d1, unit=(r1u, d1u))
    c2 = SkyCoord(r2, d2, unit=(r2u, d2u))

#    print(r1u, d1u, r2u, d2u)
#    print(c1)
#    print(c2)

    d = c2.separation(c1)

    return(d.arcsec)


