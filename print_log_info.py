#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.1.0"

"""
USED BY:
    - reduce_exposure
    - gather_SV_grades
    - group_raws_to_obs
    - reduce_indi_raws

HISTORY:
    - 2020-01-22: created by Daniel Asmus
    - 2020-02-10: added insert empty line feature


NOTES:
    -

TO-DO:
    -
"""
import os
import time
import datetime


def timestamp():
    return(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))


def print_log_info(msg, logfile=None, mode='a', screen=True, logtime=True,
                   empty=0):

    """
    print a message into a log file and on the screen
    """

    if logtime is True:
        t = timestamp() + ': '
    else:
        t = ''

    if screen:
        print(t + msg)

    if logfile is not None:
        if not os.path.isfile(logfile):
            flog = open(logfile, 'w')
        else:
            flog = open(logfile, mode)

        msg = t + msg + '\n'

        for i in range(empty):
            msg = "\n" + msg

        flog.write(msg)
        flog.close()
