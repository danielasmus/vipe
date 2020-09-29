#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-23: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""

import os

from .print_log_info import print_log_info as _print_log_info


def gather_SV_grades(infolder, ftablog, logfile=None):
    """
    Look for night log files in a given folder and extract a list of all raw
    files with grades and write them into the given file name.
    """

    if logfile is None:
        logfile = ftablog.replace("_log.csv", ".log")

    if not os.path.isfile(logfile):
        mode = 'w'
    else:
        mode = 'a'

    _print_log_info('\n\nNew execution of GATHER_SV_GRADES\n',
                   logfile, mode=mode)

    logfiles = [ff for ff in os.listdir(infolder)
             if ff.endswith('.NL.txt')]

    nlogfiles = len(logfiles)

    msg = ('Number of log files found: ' + str(nlogfiles))
    _print_log_info(msg, logfile)

    f = open(ftablog, 'w')
    f.write("filename, grade\n")

    outlines = []

    for i in range(nlogfiles):

        fin = infolder + "/" + logfiles[i]
        ff = open(fin, 'r')

        # --- find the service mode grade
        grade = None
        for line in ff:
            if "Grade:" in line:
                grade = (line.split()[1].rstrip())
                # print(grade)
                # break
            if "VISIR." in line and grade != None:
                fname = line.split()[1].rstrip()
                outline = fname + ', ' + grade + '\n'
                if outline not in outlines:
                    outlines.append(outline)
                    f.write(outline)
            elif "ISAAC." in line and grade != None:
                fname = line.split()[1].rstrip()
                outline = fname + ', ' + grade + '\n'
                if outline not in outlines:
                    outlines.append(outline)
                    f.write(outline)

        ff.close()

    f.close()
