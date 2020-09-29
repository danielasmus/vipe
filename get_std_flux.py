#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "2.0.1"

"""
USED BY:
    - reduce_exposure

HISTORY:
    - 2020-01-21: created by Daniel Asmus
    - 2020-02-11: change from type==str to more robust isinstance
    - 2020-02-14: added parameter silent
    - 2020-06:10: add basic support for ISAAC
    - 2020-06-17: add more verbose output, debug ISAAC flux averaging


NOTES:
    -

TO-DO:
    - What to do with NB filters for ISAAC, which have no entries in the table
"""
import numpy as np
import os
import inspect
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord

from .print_log_info import print_log_info as _print_log_info
from .fits_get_info import fits_get_info as _fits_get_info
from .angular_distance import angular_distance as _angular_distance
from . import isaac_params as _ip



def get_std_flux(any_input, filtname=None, instrument=None, insmode=None,
                 reffile=None, maxdist=30, namecol='STARS', verbose=False,
                 logfile=None, silent=False):
    """
    Find and return the flux desnity in Jansky for either a given STD star name
    and filter, or from a fits file, or from a fits header.
    """

    funname = "GET_STD_FLUX"

    head = None
    starname = None
    ra = None
    dec = None

    # --- determine input type
    # --- is it a string?
    if isinstance(any_input, str):

        # --- is it a fits file name?
        if os.path.isfile(any_input):
            try:
                hdu = fits.open(any_input)
                head = hdu[0].header
                hdu.close()

                if verbose:
                    print(funname + ": detected input: FITS file")
            except:
                msg = (funname + ": ERROR: provided file is not a fits file: any_input= "
                       + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

        # --- is it a star name? I.e., is the a character in the string?
        elif any(c.isalpha() for c in any_input):
            starname = any_input

            if verbose:
                print(funname + ": detected input: star name")

        # --- otherwise assume that it is a coordinate string
        else:
            try:
                coord = SkyCoord(any_input, unit=(u.hourangle, u.deg),frame='icrs')
                ra = coord.ra.deg
                dec = coord.dec.deg

                if verbose:
                    print(funname + ": detected input: sky coordinates string")

            except:
                msg = (funname + ": ERROR: provided input not resolvable: any_input= "
                   + any_input)

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

    # --- is it a fits header?
    elif type(any_input) == fits.header.Header:
        head = any_input

        if verbose:
            print(funname + ": detected input: FITS header")

    # --- is it an array of coordinates?
    elif hasattr(any_input, "__len__"):

        # --- are these sexagesimal coordinates?
        if isinstance(any_input[0], str):
            unit = (u.hourangle, u.deg)

        # --- otherwise assume coordinates to be in degrees
        else:
            unit = (u.deg, u.deg)

        try:
            coord = SkyCoord(ra=any_input[0], dec=any_input[1], unit=unit,
                             frame='icrs')
            ra = coord.ra.deg
            dec = coord.dec.deg

            if verbose:
                print(funname + ": detected input: sky coordinates array")
        except:
            msg = (funname + ": ERROR: provided input not resolvable: any_input= "
                   + str(any_input[0]) + " " + str(any_input[1])
                   )

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)

    else:
        msg = (funname + ": ERROR: Invalid input provided: type(any_input): "
               + str(type(any_input))
               )

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

    # return(head,starname,coord)

    # --- if the head was provided the flux density might actually be in the
    #     header:
    if head is not None:

        flux = _fits_get_info(head, "JYVAL", silent=silent)

        if verbose:
            print(funname + ": flux found in fits header")

        # --- if we have the flux in the header we are done!
        if flux is not None:
            return(flux)

        # --- otherwise try to get the star name
        if starname is None:
            starname = _fits_get_info(head, "targname", silent=silent)

            if verbose:
                print(funname + ": star name from header: ", starname)

        # --- and the coordinates
        if ra is None:
            ra = _fits_get_info(head, "RA", silent=silent)
            dec = _fits_get_info(head, "DEC", silent=silent)

            if verbose:
                print(funname + ": coordinates from header: ", ra, dec)

        # --- which filter?
        if filtname is None:
            filtname = _fits_get_info(head, "filtname", silent=silent)

            if verbose:
                print(funname + ": star name from header: ", starname)


    # --- make sure that we have either the star name and/or its coordinates
    if starname is None and ra is None:
        msg = (funname + ": ERROR: No star name or coordinates provided or found!: starname, ra, dec: "
               + str(starname) + " " + str(ra) + " " + str(dec))

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- make sure that we know which filter by now!
    if filtname is None:
        msg = (funname + ": ERROR: No filter name provided or found!:")

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)


    # --- which instrument mode?
    if insmode is None:
       if head is not None:
           insmode = _fits_get_info(head, "insmode", silent=silent)

           if verbose:
               print(funname + ": instrument mode from header: ", insmode)

       else:
           if not silent:
               msg = (funname + ": WARNING: No instrument mode provided! Assuming imaging...")
               _print_log_info(msg, logfile)
           insmode = "IMG"

    # --- for SPCIMG the filter names have added "_spec" in the table
    insmode = insmode.upper()
    if "SPC" in insmode:
        filtname = filtname + "_spec"

    # --- some optional printout
    if verbose:
        msg = (funname + ": starname, ra | dec | filter | insmode: "
               + str(starname) + " | " + str(ra) + " | " + str(dec) + " | "
               + str(filtname) + " | " + str(insmode))

        if logfile is not None:
            _print_log_info(msg, logfile)
        else:
            print(msg)

    # --- we will have to look up the flux in the reference table
    # --- if no reftable is provided, we need to select the right one
    if reffile is None:
        # --- which instrument?
        if instrument is None:
            if head is not None:
                instrument = _fits_get_info(head, "INSTRUME", silent=silent)

                if verbose:
                    print(funname + ": instrument from header: ", starname)

            else:
                msg = (funname + ": ERROR: No instrument provided! Aborting...")

                if logfile is not None:
                    _print_log_info(msg, logfile)

                raise ValueError(msg)

        # --- get the path of this module for the reference files
        refpath = os.path.dirname(inspect.getfile(get_std_flux))

        if instrument == "VISIR":
            reffile = refpath + "/" + "ima_std_star_cat.fits"

        elif instrument == "ISAAC":
            reffile = refpath + "/" + "ISAAC_IMG_STD_cat_concat-all.csv"

    if verbose:
        print(funname + ": instrument: ", instrument)
        print(funname + ": ref file: ", reffile)


    # --- check that the file exists
    if not os.path.isfile(reffile):
        msg = (funname + ": ERROR: reference file with flux data not found: "
               + reffile)

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)

        print(msg)

    # --- read the file
    try:
        d = Table.read(reffile)

        if verbose:
            print(funname + ": ref file successfully read")

    except:
        msg = (funname + ": ERROR: reference file can not be read: "
               + reffile)

        if logfile is not None:
            _print_log_info(msg, logfile)

        raise ValueError(msg)
        print(msg)


    # --- first check for the name if match
    ids = []
    if starname is not None:

        ids = np.where(d[namecol] == starname)[0]

        # --- if no hit reformat the name to follow the convention of the ESO table
        if len(ids) == 0:
            starname = starname.replace(" ", "")

            ids = np.where(d[namecol] == starname)[0]

        # --- remove leading 0 in the number
        if len(ids) == 0:
            starname = starname.replace("HD0", "HD")
            starname = starname.replace("HD0", "HD")
            starname = starname.replace("HD0", "HD")
            starname = starname.replace("HD0", "HD")
            starname = starname.replace("HD0", "HD")

            ids = np.where(d[namecol] == starname)[0]

    # --- if no match with name check for match in coordinates
    if (len(ids) == 0) & (ra != None) & (dec != None):

        if verbose:
            print(funname + ": star name not found. Trying with coordinates...")

        ns = len(d)
        dist = np.zeros(ns)

        # --- compute the angular distances from the target
        for i in range(ns):
            dist[i] = _angular_distance(ra, dec, float(d['RA'][i]),
                                  float(d['DEC'][i]))

        ids = np.where(dist < maxdist)[0]

    if verbose:
        print(funname + ": matching rows in table: ", ids)

    # --- have we found a match?
    if len(ids) == 0:
        if not silent:
            msg = (funname + ": ERROR: No matching star found: starname, ra, dec: "
                   + str(starname) + " " + str(ra) + " " + str(dec))

            if logfile is not None:
                _print_log_info(msg, logfile)

        raise ValueError(msg)

    # --- take care of some inconsistencies in the VISIR ESO data
    if instrument == "VISIR":
        filt_alt1 = filtname.replace(".", "_")
        filt_alt2 = filtname.replace("-", "_")

        # --- in case of multiple candidates take closest
        if len(ids) > 1:
            ids = [np.argmin(dist)]

            if verbose:
                print(funname + ": Multiple hits found. Take closest.")

        # --- finally get the right value out of the table
        if filtname in d.colnames:
            return(float(d[filtname][ids[0]]))
        elif filt_alt1 in d.colnames:
            return(float(d[filt_alt1][ids[0]]))
        elif filt_alt2 in d.colnames:
            return(float(d[filt_alt2][ids[0]]))
        else:
            msg = (funname + ": ERROR: Filter not found in table: "
                   + str(filtname))

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)

    # --- for ISAAC, take care of the multiple entries and different filter
    #     names
    elif instrument == "ISAAC":

        # --- ISAAC's L filter is more similar to the general Lp filter,
        #     which thus should be preferred.
        if filtname == "L":
            # --- do a priority order list of filters
            filt_ord = ["Lp", "L", "Ks", "K"]  # --- if L is not available use Ks

        # --- ISAAC's L filter is more similar to the general Lp filter,
        #     which thus should be preferred.
        elif filtname == "M_NB":
            filt_ord = ["Mp", "M", "Lp", "L", "Ks", "K"]

        else:
            msg = (funname + ": ERROR: Filter not found in table: "
                   + str(filtname))

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)

        fd_Jy = None

        for f in filt_ord:
            if f in d.colnames:
                idf = np.where(d[f][ids] != 99)[0]

                if len(idf) == 0:
                    continue
                else:

                    if verbose:
                        print(funname + ": selected entries: ")
                        print("   - filter: ", f)
                        print("   - selected rows: ", idf)


                    mag = np.mean(d[f][ids[idf]])
                    # wlen = _ip.filtwlens[filtname]
                    zp = _ip.zps[filtname]
                    fd_Jy = zp * 10 ** (-mag/2.5)

                    # --- give a warning if we have to extrapolate from
                    #     another band
                    if (((filtname == "L") & (f != 'Lp') & (f != 'L'))
                        | ((filtname == "M_NB") & (f != 'Mp') & (f != 'M'))):
                        if not silent:
                            msg = (funname +
                                   ": WARNING: No flux available in this filter! Extrapolating from shorter band: "
                                   + f)
                            _print_log_info(msg, logfile)

                    return(fd_Jy)

        if fd_Jy is None:
            msg = (funname + ": ERROR: No flux available for this star and filter: "
                   + str(filtname))

            if logfile is not None:
                _print_log_info(msg, logfile)

            raise ValueError(msg)

