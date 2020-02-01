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

from astropy.io import ascii


def duplicates_in_column(fin=None, table=None, column=None, colname=None):
    """
    Identify dublicates between the rows of one column. Input can either be a
    csv table file given in fin, an astropy table given in table or a column
    given in column. If the file or table are given colname is used to give the
    name string of the column to be searched.
    Returnes the values that appear duplicate.
    """

    if fin is not None:
        table = ascii.read(fin, header_start=0, delimiter=',', guess=False)

    if table is not None:
        column = table[colname]

    column = list(column)

    duplicates = set([x for x in column if column.count(x) > 1])

    if len(duplicates) < 1:
        return(None)
    else:
        return(duplicates)

