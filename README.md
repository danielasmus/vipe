Python package to reduce ESO VLT/VISIR and ISAAC imaging (and spectroscopic)
data.

Written for Python 3 (will not run in Python 2)

Requires: numpy, matplotlib, scipy, astropy, photutils.

Created by Daniel Asmus

New reduction steps would be as follows:

0. Downlad the data from the ESO archive and put into one folder.
   Uncompress *.Z if necessary

1. Look for the night logs and create a table of all raw files and grades
    Routine: gather_SV_grades

2. Reduce all the individual files amd create summary table
    Routine: reduce_indi_raw_files

3. Identify the exposure that the raw files belong to
    Routine: group_files_to_observations

4. Reduce the exposures
    Routine: reduce_observations

An example script with the name do_reduce_data.py is provided within the package.

