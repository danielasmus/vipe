#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.1.1"

"""
USED BY:
    - calc_beam_pos
    - calc_jitter
    - get_std_flux
    - reduce_indi_raws
    - reduce_exposure
    - undo_jitter

HISTORY:
    - 2020-01-21: created by Daniel Asmus
    - 2020-02-11: get_setup added, changed type == str comp to more robust
                  isinstance
    - 2020-06-09: add ISAAC support
    - 2020-06-11: silent parameter & warning for ISAAC nodpos added & bugfix for multiple key hits
    - 2020-09-29: Enforce  pfov to be float (works only if not directly found)


NOTES:
    -

TO-DO:
    -
"""
# import numpy as np
# import pdb
from collections import OrderedDict
import os
import numpy as np
from astropy.io import fits

from .filt_get_wlen import filt_get_wlen as _filt_get_wlen
from .print_log_info import print_log_info as _print_log_info

#%%
# --- Helper routine to determine target name
def get_targname(any_input, logfile=None):
    """
    Helper routine to determine and return the target name for a given fits
    file or header and given instrument
    """
    funname = "GET_TARGNAME"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    # if instrument == None:
    #     if "INSTRUME" in head:
    #         instrument = head["INSTRUME"]

    # if instrument == "VISIR" or instrument == None:
    if "TARG NAME" in head:
        targname = head["HIERARCH ESO OBS TARG NAME"]
    else:
        targname = head["OBJECT"]



    return(targname)



#%%
# --- Helper routine to determine instrument mode
def get_insmode(any_input, instrument=None, logfile=None):
    """
    Helper routine to determine and return the instrument (imaging,
    spectroscopy, type of acquisition) mode for a given fits
    file or header and given instrument
    """
    funname = "GET_INSMODE"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    if instrument == "VISIR" or instrument == None:
        temp = head["HIERARCH ESO TPL ID"]
        optpath = head["HIERARCH ESO INS PATH"]

        if optpath == "spe_imag":
            insmode = "ACQ-SPC-SPC"

        elif optpath == "ima_imag":
            if "spec" in temp:
                insmode = "ACQ-IMG-SPC"
            else:
                insmode = "IMG"

        else:
            insmode = "SPC"

    elif instrument == "ISAAC":
        temp = head["HIERARCH ESO TPL ID"]
        # mode = head["HIERARCH ESO INS MODE"]

        if "img" in temp:
            insmode = "IMG"
        elif "spec" in temp:
            insmode = "SPC"

    return(insmode)


#%%
# --- Helper routine to determine instrument mode
def get_noddir(any_input, instrument=None, logfile=None):
    """
    Helper routine to determine and return the nodding direction with respect
    to the chopping direction for a given fits file or header and given
    instrument
    """
    funname = "GET_NODDIR"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    if instrument == "VISIR" or instrument == None:
        noddir = head["HIERARCH ESO SEQ CHOPNOD DIR"]

    elif instrument == "ISAAC":
        noddir = "PARALLEL"
        # mode = head["HIERARCH ESO INS MODE"]

    return(noddir)


#%%
# --- Helper routine to determine instrument mode
def get_nodpos(any_input, instrument=None, logfile=None):
    """
    Helper routine to determine and return the nodding position for a given
    fits file or header and given instrument
    """
    funname = "GET_NODPOS"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    if instrument == "VISIR" or instrument == None:
        nodpos = head["HIERARCH ESO SEQ NODPOS"]

    elif instrument == "ISAAC":
        # --- unfortunately, the nod position is not explicitely stated in the
        #     fits headers, so we assume it from the file number given that the
        #     nodding sequence should always be ABBA

        expno = head["HIERARCH ESO TPL EXPNO"]
        exptot = head["HIERARCH ESO TPL NEXP"]

        # --- is jittering on?
        if "HIERARCH ESO SEQ JITTER WIDTH" in head:
            jwidth = head["HIERARCH ESO SEQ JITTER WIDTH"]
        else:
            jwidth = 0

        if "HIERARCH ESO TEL CHOP THROW" in head:
            chopthrow = head["HIERARCH ESO TEL CHOP THROW"]
        else:
            chopthrow = 0

        #     so we have to estimate it considering the current
        #     offset in comparison to the jitter
        cumx = head["HIERARCH ESO SEQ CUMOFFSETX"]
        cumy = head["HIERARCH ESO SEQ CUMOFFSETY"]

        # --- pixel size
        if "HIERARCH ESO INS PIXSCALE" in head:
            pfov = float(head["HIERARCH ESO INS PIXSCALE"])
        elif "CD2_2" in head:
            pfov = 3600*0.5*(np.abs(head["CD2_2"]) + np.abs(head["CD1_1"]))

        #  --- if jitter is large compared to the chop throw we have a problem
        if 3 * jwidth > chopthrow and exptot > 2:
            msg = (funname + ": WARNING: 3*Jitter > Chop throw: Estimating nodpos from expno")

            if logfile is not None:
                _print_log_info(msg, logfile)

            # --- assume a ABBA nod pattern
            if expno % 4 < 2:
                nodpos = 'A'
            else:
                nodpos = 'B'

        # --- otherwise we can use the cumoffsets to estimate the nodpos
        else:
        # --- if we are close to the original position, we should be in A
            if (np.abs(cumx*pfov) <= 3 * jwidth) & (np.abs(cumy*pfov) <= 3 * jwidth):
                nodpos = "A"
            else:
                nodpos = "B"
        # # mode = head["HIERARCH ESO INS MODE"]

    return(nodpos)


#%%
# --- Helper routine to determine datatype
def get_datatype(any_input, instrument=None, logfile=None):

    """
    Helper routine to determine and return the datatype of a given fits file
    or header for a given instrument
    """

    funname = "GET_DATATYPE"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    if instrument == "VISIR" or instrument == None:
        cycsum = head['HIERARCH ESO DET CHOP CYCSUM']
        framformat = head['HIERARCH ESO DET FRAM FORMAT']

        if framformat == "cube-ext":
            return("burst")

        elif cycsum:
            return("cycsum")

        else:
            return("halfcyc")

    elif instrument == "ISAAC":
        return("cycsum")

        # if head["NAXIS3"] == 2:
        #     return("cube")
        # elif "HCYCLE" in head["ORIGFILE"]:
        #     return("hcycle")
        # else:
        #     return("chopdif")


#%%
# --- Helper routine to determine datatype
def get_cycsum(any_input, instrument=None, logfile=None):

    """
    Helper routine to determine and return whether cycsum was true or false
    for a given fits file or header for a given instrument
    """

    funname = "GET_CYCSUM"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    if instrument == "VISIR" or instrument == None:
        cycsum = head['HIERARCH ESO DET CHOP CYCSUM']

    elif instrument == "ISAAC":
        cycsum = "T"

    return(cycsum)



#%%
# --- helper routine to get filtername
def get_filtname(any_input, instrument=None, insmode=None, logfile=None):
    """
    Helper routine to get the filter name from any header or fits file for a
    given instrument
    """

    funname = "GET_FILTNAME"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    # --- instrument mode:
    if insmode is None:
        try:
            insmode = get_insmode(head, instrument=instrument)
        except:
            msg = (funname + ": WARNING: Could not determine instrument mode")
            _print_log_info(msg, logfile)

    # --- VISIR?
    if instrument == "VISIR":
        if insmode is not None:
            # --- spectroscopy?
            if "SPC" in insmode and "IMG" not in insmode:
                filtname = head["HIERARCH ESO INS FILT2 NAME"]
            else:  # or imaging?
                filtname = head["HIERARCH ESO INS FILT1 NAME"]
        else:
            if "HIERARCH ESO INS FILT1 NAME" in head:
                filtname = head["HIERARCH ESO INS FILT1 NAME"]
            else:
                filtname = head["HIERARCH ESO INS FILT2 NAME"]

    elif instrument == "ISAAC":  # implicitely assuming LWI4 mode
        filtname = head["HIERARCH ESO INS FILT3 NAME"]

    else:
        if "HIERARCH ESO INS FILT1 NAME" in head:
            filtname = head["HIERARCH ESO INS FILT1 NAME"]
        else:
            filtname = head["HIERARCH ESO INS FILT2 NAME"]

    return(filtname)


#%%
# --- Helper routine to compute exposure time of file
def compute_nod_exptime(any_input, instrument=None, logfile=None):
    """
    Helper routine to compute the exposure time of the nod, i.e., raw file from
    any header or fits file for a given instrument
    """

    funname = "COMPUTE_NOD_EXPTIME"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    # --- computation is different depending on instrument
    if instrument == "VISIR" or instrument == None:
        if 'HIERARCH ESO DET DIT' in head:
            dit = head['HIERARCH ESO DET DIT']
        else:
            dit = head['HIERARCH ESO DET SEQ1 DIT']

        ndit = head['HIERARCH ESO DET NDIT']
        nchop = head['HIERARCH ESO DET CHOP NCYCLES']

        res = dit * ndit * nchop * 2

    elif instrument == "ISAAC":
        dit = head['HIERARCH ESO DET DIT']
        ndit = head['HIERARCH ESO DET NDIT']
        nchop = head['HIERARCH ESO DET CHOP NCYCLES']
        res = dit * ndit * nchop * 2

    return(res)

#%%
# --- Helper routine to determine setup depending on instrument mode
def get_setup(any_input, instrument=None, insmode=None, logfile=None):
    """
    Helper routine to determine and return the setup (imaging filter or slit
    width) for a given fits file or header and optionally given instrument and
    mode
    """
    funname = "GET_SETUP"

    # --- determine input type
    # --- is it a fits file name?
    if isinstance(any_input, str):
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- otherwise assume it is a fits header
    elif type(any_input) == fits.header.Header:
        head = any_input
    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- determine instrument:
    if instrument == None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    instrument = instrument.upper()

    if insmode == None:
        insmode = get_insmode(head, instrument=instrument, logfile=logfile)

    insmode = insmode.upper()

    if instrument == "VISIR":
        # --- VISIR in imaging mode?
        if 'IMG' in insmode or "ACQ" in insmode:
            setup = get_filtname(head, instrument=instrument,
                                 insmode=insmode, logfile=logfile)

        # --- VISIR in spectro mode:
        elif insmode == 'SPC':
            setup = head['HIERARCH ESO INS SLIT1 WID']

        else:
            msg = (funname + ": ERROR: insmode not determined: "
                   + str(insmode)
                   )

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)


    elif instrument == "ISAAC":
        setup = get_filtname(head, instrument=instrument,
                            insmode=insmode, logfile=logfile)

    return(setup)


#%%
# --- main routine
def fits_get_info(any_input, keys=None, ext=None, instrument=None,
                  fill_value=None, logfile=None, silent=False):

    """
    take a fits header and extract a given list of keywords from it and return
    them as a structure
    """

    funname = "FITS_GET_INFO"

    # --- determine input type
    if isinstance(any_input, str):   # is it a file name?
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)

            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

            # --- if no keys nor extension are provided return structure of fits file
            if keys is None and ext is None:

                outstr = "Fits structure: \n "
                noe = len(hdu)
                for i in range(noe):

                    outstr = outstr + "Extension " + str(i) + ": "

                    if "EXTNAME" in hdu[i].header:
                        outstr = outstr + hdu[i].header["EXTNAME"] + " - "

                    if "NAXIS" in hdu[i].header:
                        naxis = int(hdu[i].header["NAXIS"])

                        if naxis == 0:
                            outstr = outstr + "NO DATA"
                        elif naxis == 1:
                           naxis1 = hdu[i].header["NAXIS1"]
                           outstr = outstr + "1D array of length " + str(naxis1)

                        elif naxis == 2:  # tabl or image
                             naxis1 =  hdu[i].header["NAXIS1"]
                             naxis2 =  hdu[i].header["NAXIS2"]
                             if "XTENSION" in hdu[i].header:
                                 if "TABLE" in hdu[i].header["XTENSION"]:
                                     outstr = outstr + "Table: " + str(naxis1) + " x " + str(naxis2)
                                 elif "IMAGE" in hdu[i].header["XTENSION"]:
                                     outstr = outstr + "Image: " + str(naxis1) + " x " + str(naxis2)
                             else:
                                 outstr = outstr + "Image: " + str(naxis1) + " x " + str(naxis2)

                        elif naxis == 3:  # -- cube
                             naxis1 =  hdu[i].header["NAXIS1"]
                             naxis2 =  hdu[i].header["NAXIS2"]
                             naxis3 =  hdu[i].header["NAXIS3"]
                             outstr = outstr + "Cube: " + str(naxis1) + " x " + str(naxis2) + " x " + str(naxis3)

                    outstr += " \n "

                hdu.close()
                return(outstr)

            # --- if an extension is specified but no keywords then return full
            #     header of this extension
            elif ext != None and keys is None:
                outstr = hdu[ext].header
                hdu.close()
                return(outstr)

            # --- if keywords are provided extract the header to search for them
            else:
                if ext is None:
                    ext = 0

                head = hdu[ext].header
                hdu.close()

    # --- fits header provided?
    elif type(any_input) == fits.header.Header:
        head = any_input

    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    # --- get the instrument if not provided
    if instrument is None:
        if "INSTRUME" in head:
            instrument = head["INSTRUME"].replace(" ", "")

    # --- if no keys are provided use a default list assuming VISIR data
    if keys is None:
        keys = (
                'DATE-OBS'+', '+
                'MJD-OBS'+', '+
                'TARG NAME'+', '+
                'RA'+', '+
                'DEC'+', '+
                'TEL AIRM'+', '+
                'PARANG'+', '+
                'POSANG'+', '+
                'OBS ID'+', '+
                'PROG ID'+', '+
                'TPLNO'+', '+
                'TPL ID'+', '+
                'TPL EXPNO'+', '+
                'TPL NEXP'+', '+
                'CHOP FREQ'+', '+
                'CHOP POSANG'+', '+
                'CHOP THROW'+', '+
                'CHOPNOD DIR'+', '+
                'NODPOS'+', '+
                'JITTER WIDTH'+', '+
                'CUMOFFSETX'+', '+
                'INS MODE'+', '+
                'PFOV'+', '+
                'INS FILT1 NAME'+', '+
                'INS FILT2 NAME'+', '+
                'INS RESOL'+', '+
                'INS GRAT1 WLEN'+', '+
                'INS SLIT1 WID'+', '+
                'CYCSUM'+', '+
                'FRAM FORMAT'+', '+
                'DET WIN STRX'+', '+
                'DET WIN STRY'+', '+
                'IWV'+', '+
                'IRSKY TEMP'+', '+
                'IA FWHM'+', '+
                'TEL AMBI FWHM'+', '+
                'DPR TECH'
                )

    # --- convert to upper case
    keys = keys.upper()

    # --- convert string into a list
    keys = keys.split(',')

    nkeys = len(keys)

    # --- create a structure with the keywords as items
    # params = {}
    params = OrderedDict()

    for i in range(nkeys):

        # --- remove the leading and trailing empty spaces
        key = keys[i].lstrip().rstrip()

        # --- first check if the key value is directly found
        ids = [j for j, s in enumerate(head) if key in s]
        if ids:

            # print(ids)

            # --- if more than one hit take the one that matches best (i.e.,
            #     is the shortest)
            if len(ids) > 1:
                lens = [len(head.cards[j][0]) for j in ids]
                idsel = np.argmin(lens)

                params[key] = head[ids[idsel]]

#                print(key+':')
#                for j in ids:
#                    print(' - '+str(head.keys()[j]))
            else:
                params[key] = head[ids[0]]

        # --- if not then maybe it is one of the following generals?
        # --- target name
        elif key in ['NAME', 'TARGNAME', 'TARGET', 'TARGET NAME']:
            try:
                params[key] = get_targname(head, logfile=logfile)

            except:

                msg = (funname
                        + ": WARNING: Target name could not be determined."
                        )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- Nodding direction
        elif key == 'CHOPNOD DIR':
            try:
                params[key] = get_noddir(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Chopnod direction could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- Nod position
        elif key == 'NODPOS':
            try:
                params[key] = get_nodpos(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Nod position could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- insmode
        elif key in ['INSMODE', 'INSTRUMENT MODE']:
            try:
                params[key] = get_insmode(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Instrument mode could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- cycsum
        elif key == 'CYCSUM':
            try:
                params[key] = get_cycsum(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: cycle sum flag could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- setup
        elif key in ['SETUP']:
            try:
                params[key] = get_setup(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Setup could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- filter name
        elif key in ['FILTER', 'FILTNAME', 'FILTERNAME', 'FILT NAME',
                     'FILTER NAME', "FILT"]:
            try:
                params[key] = get_filtname(head, instrument=instrument,
                                           logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Filtername could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value

        # --- wavelength
        elif key in ['WAVELENGTH', 'WLEN']:
            try:
                filtname = get_filtname(head, instrument=instrument,
                                    logfile=logfile)

            except:
                msg = (funname
                       + ": WARNING: Filtername could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)

            try:
                insmode = get_insmode(head, instrument=instrument,
                                    logfile=logfile)

            except:
                msg = (funname
                       + ": WARNING: Instrument mode could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)

            try:
                params[key] = _filt_get_wlen(filtname, instrument=instrument,
                                             insmode=insmode)

            except:
                msg = (funname
                       + ": WARNING: Wavelength could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value

        # --- exposure time of the nod, i.e., file
        elif key in ['NODEXPTIME', 'NODEXPOSURETIME']:
            try:
                params[key] = compute_nod_exptime(head, instrument=instrument,
                                                  logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Nod exposure time could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value

        # --- datatype
        elif key in ['DATATYPE']:
            try:
                params[key] = get_datatype(head, instrument=instrument,
                                                  logfile=logfile)
            except:
                msg = (funname
                       + ": WARNING: Datatype could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value

        elif key == "PFOV":
            if "HIERARCH ESO INS PFOV" in head:
                params[key] = float(head["HIERARCH ESO INS PFOV"])
            elif "HIERARCH ESO INS PIXSCALE" in head:
                params[key] = float(head["HIERARCH ESO INS PIXSCALE"])
            elif "CD2_2" in head:
                params[key] = 3600*0.5*(np.abs(head["CD2_2"]) + np.abs(head["CD1_1"]))
            else:
                msg = (funname
                       + ": WARNING: PFOV could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        elif key in ["SEQ1 DIT", "DIT", "DET DIT"]:
            if "HIERARCH ESO DET SEQ1 DIT" in head:
                params[key] = head["HIERARCH ESO DET SEQ1 DIT"]
            elif "HIERARCH ESO DET DIT" in head:
                params[key] = head["HIERARCH ESO DET DIT"]
            else:
                msg = (funname
                       + ": WARNING: DIT could not be determined."
                       )

                if not silent:
                    _print_log_info(msg, logfile)
                params[key] = fill_value


        # --- OK it is really not found so use a fill value
        else:
            params[key] = fill_value


    # --- if only one keyword was asked for simply return it rather than a
    #     structure
    if len(params) == 1:
        # params = params.values()[0]
        params = next(iter(params.values()))

    return(params)

