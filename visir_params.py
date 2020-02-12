#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.1.0"

"""
USED BY:
    - filt_get_wlen
    - reduce_exposure
    - reduce_indi_raws

HISTORY:
    - 2020-01-23: created by Daniel Asmus
    - 2020-02-10: changed spec filter suffix from _spc to _SPC


NOTES:
    - Collection of parameters specific to VISIR

TO-DO:
    -
"""

from collections import OrderedDict


# --- maximum illuminated area of detector
max_illum_xrange = (20, 880)
max_illum_yrange = (15, 880)


# --- filters
# --- VISIR filter central wavelengths
filtwlens = OrderedDict()
filtwlens['K-BAND'] = 2.2
filtwlens['M-BAND'] = 4.82
filtwlens['J7.9'] = 7.76
filtwlens['PAH1'] = 8.59
filtwlens['J8.9'] = 8.7
filtwlens['B8.7'] = 8.92
filtwlens['ARIII'] = 8.99
filtwlens['J9.8'] = 9.59
filtwlens['B9.7'] = 9.82
filtwlens['SIV_1'] = 9.82
filtwlens['SIV'] = 10.49
filtwlens['B10.7'] = 10.65
filtwlens['SIV_2'] = 10.77
filtwlens['PAH2'] = 11.25
filtwlens['B11.7'] = 11.52
filtwlens['PAH2_2'] =11.88
filtwlens['J12.2'] = 11.96
filtwlens['NEII_1'] = 12.27
filtwlens['B12.4'] = 12.47
filtwlens['NEII'] = 12.81
filtwlens['NEII_2'] = 13.04
filtwlens['NEII_2_SPC'] = 12.81
filtwlens['Q1'] = 17.65
filtwlens['Q2'] =  18.72
filtwlens['Q3'] = 19.5
filtwlens['10_5_4QP'] = 10.5
filtwlens['11_3_4QP'] = 11.3
filtwlens['12_4_AGP'] = 12.4
filtwlens['12_3_AGP'] = 12.4
filtwlens['10_5_SAM'] = 10.5
filtwlens['11_3_SAM'] = 11.3


# --- pixel scales
sf_pfov = 0.0453
spc_pfov = 0.076