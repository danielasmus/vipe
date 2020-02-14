# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:42:44 2017

@author: dasmus
"""
import vipe as V


infolder = '/Users/dasmus/data/raw/VISIR/Neptune'
# infolder = '/Users/dasmus/data/raw/VISIR/Neptune/part1'
outfolder ='/Users/dasmus/data/processed/VISIR/vipe_test'

workfolder = '.'

tabstr = 'Neptune'

overwrite = False
justtable = False

maxgap = 5
maxshift = 10
instrument = "VISIR"
selobj = None
selexp = None
#selexp=[31,32,33]
#selexp = [49,50,51,52]
#selexp = [303,304]
# selexp = np.arange(6,22)
# selexp = [43]
selsetup = None
selobid = None
#selmode = ['IMG']
#seldtype = ['cycsum']
selmode = None
seldtype = None
#together = False
extract = True
verbose = False
#verbose = True
obstype = None
#obstype = "cal"
#mindate = '2018-10-09'
mindate = None
#mindate = '2018-05-21'
seldate = None
maxdate = None
box = None
alignmethod='fastgauss'
searcharea='chopthrow'
AA_pos=None
refpos=None
findbeams = True
debug = False
ignore_errors=False



print("Given Inputs:")
print(" - Infolder: "+ infolder)
print(" - Workfolder: "+ workfolder)
print(" - Outfolder: "+ outfolder)
print(" - Table-String: "+ tabstr)
print(" - Overwrite: "+ str(overwrite))

ftabraw = workfolder + '/' + tabstr + '_raw.csv'
ftabgrad = workfolder + '/' + tabstr + '_grades.csv'
ftabpro = workfolder + '/' + tabstr + '_pro.csv'

print(" === 0. Look for the night logs and create a table of all raw files and grades")
V.gather_SV_grades(infolder, ftabgrad)

print(" === 1. SCAN THE FILES GATHER INFO AND PLOT CONTENT OF RAW DATA...")
V.reduce_indi_raws(infolder, outfolder, ftabraw, overwrite=overwrite,
                        justtable=justtable)


print(" === 2. DIVIDE THE FILES INTO DIFFERENT EXPOSURES...")
dpro = V.group_raws_to_obs(ftabraw, ftabgrad, ftabpro)


print(" === 3. DO THE REDUCTION OF INDIVIDUAL EXPOSURES...")
V.reduce_obs(ftabraw, ftabpro, infolder, outfolder, maxshift=maxshift,
                      selobj=selobj, selexp=selexp, selsetup=selsetup,
                      selobid=selobid, selmode=selmode, seldtype=seldtype,
                      extract=extract, verbose=verbose, obstype=obstype,
                      mindate=mindate, seldate=seldate, maxdate=maxdate,
                      box=box, overwrite=overwrite, alignmethod=alignmethod,
                      searcharea=searcharea, AA_pos=AA_pos, refpos=refpos,
                      findbeams=findbeams, instrument=instrument,
                      ignore_errors=ignore_errors)


