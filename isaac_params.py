#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
USED BY:
    - filt_get_wlen
    - get_std_flux
    - reduce_exposure
    - reduce_indi_raws

HISTORY:
    - 2020-06-08: created by Daniel Asmus


NOTES:
    - Collection of parameters specific to ISAAC

TO-DO:
    -
"""

from collections import OrderedDict


# --- maximum illuminated area of detector
max_illum_xrange = (0, 1024)
max_illum_yrange = (0, 1024)


# --- filters
# --- ISAAC filter central wavelengths
filtwlens = OrderedDict()
filtwlens['Js'] = 1.244
filtwlens['H'] = 1.637
filtwlens['NB_2.07'] = 2.069
filtwlens['NB_2.13'] = 2.128
filtwlens['Ks'] = 2.152
filtwlens['NB_2.17'] = 2.168
filtwlens['NB_2.29'] = 2.286
filtwlens['NB_3.21'] = 3.21
filtwlens['NB_3.28'] = 3.275
filtwlens['L'] = 3.749
filtwlens['NB_3.80'] = 3.803
filtwlens['NB_4.07'] = 4.067
filtwlens['M_NB'] = 4.656

# --- zero points
zps = OrderedDict()
zps['Js'] = 1559
zps['H'] =  1025.4
zps['NB_2.07'] = 722.9
zps['NB_2.13'] = 686.3
zps['Ks'] = 665.8
zps['NB_2.17'] = 637.9
zps['NB_2.29'] = 606.5
zps['NB_3.21'] = 324.0  # estimated because not available on SVO filter service
zps['NB_3.28'] = 317.0
zps['L'] = 247.2
zps['NB_3.80'] = 243.7
zps['NB_4.07'] = 213.1
zps['M_NB'] = 164.5



# --- pixel scales
sf_pfov = 0.071
#spc_pfov = 0.076


# --- derotator angle for imaging -- no pupil tracking for ISAAC supported for now
imgoffsetangle = 0

# --- for ISAAC, PA of the rotator is in the direction of the PA on-sky
rotsense = -1

# --- for ISAAC, for a rotator posang = 0, North points down on the detector
rotoffset = 180

# --- reference pixel for acquisition
#refpix_spc = [512, 529]
refpix_img = [512, 512]