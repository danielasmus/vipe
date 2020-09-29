#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    - Collection of python routines for handling astrophysical imaging data

TO-DO:
    -
"""


import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


from .angular_distance import angular_distance
from .calc_beampos import calc_beampos
from .calc_chopoffset import calc_chopoffset
from .calc_jitter import calc_jitter
from .calc_nodoffset import calc_nodoffset
from .crop_image import crop_image
from .diffraction_limit import diffraction_limit
from .duplicates_in_column import duplicates_in_column
from .filt_get_wlen import filt_get_wlen
from .find_beam_pos import find_beam_pos
from .find_source import find_source
from .fits_get_info import fits_get_info #,get_insmode, get_datatype, get_filtname
from .gather_SV_grades import gather_SV_grades
from .measure_bkg import measure_bkg
from .find_source import find_source
from .get_std_flux import get_std_flux
from .group_raws_to_obs import group_raws_to_obs
from .measure_sensit import measure_sensit
from .merge_fits_files import merge_fits_files
from .print_log_info import print_log_info
from .read_raw import read_raw
from .reduce_burst_file import reduce_burst_file
from .reduce_exposure import reduce_exposure
from .reduce_indi_raws import reduce_indi_raws
from .reduce_obs import reduce_obs
from .replace_hotpix import replace_hotpix
from .select_rows import select_rows
from .simple_image_plot import simple_image_plot
from .simple_nod_exposure import simple_nod_exposure
# from .subtract_source import subtract_source
from .undo_jitter import undo_jitter
from . import visir_params as vp





