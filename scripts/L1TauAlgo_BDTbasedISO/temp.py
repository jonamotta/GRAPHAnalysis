import os
import numpy as np
import pandas as pd
import root_pandas
import argparse
from sklearn.model_selection import train_test_split

outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_test/hdf5dataframes/matched'
name = 'threshold'

outFileTraining_dict = {
    'threshold'    : outdir+'/Training_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}

outFileValidation_dict = {
    'threshold'    : outdir+'/Validation_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}


outFileHH_dict = {
    'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}

outFileTenTau_dict = {
    'threshold'    : outdir+'/RelValTenTau_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}


outFileSingleTau_dict = {
    'threshold'    : outdir+'/RelValSingleTau_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}


outFileNu_dict = {
    'threshold'    : outdir+'/RelValNu_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}

outFileQCD_dict = {
    'threshold'    : outdir+'/QCD_PU200_th_matched.hdf5',
    'supertrigger' : outdir+'/',
    'bestchoice'   : outdir+'/',
    'bestcoarse'   : outdir+'/',
    'mixed'        : outdir+'/'
}



store = pd.HDFStore(outFileHH_dict[name], mode='r')
dfHH = store[name]
store.close()

store = pd.HDFStore(outFileQCD_dict[name], mode='r')
dfQCD = store[name]
store.close()

store = pd.HDFStore(outFileTenTau_dict[name], mode='r')
dfTenTau = store[name]
store.close()

store = pd.HDFStore(outFileSingleTau_dict[name], mode='r')
dfSingleTau = store[name]
store.close()

store = pd.HDFStore(outFileNu_dict[name], mode='r')
dfNu = store[name]
store.close()


print('\n** INFO: creating dataframes for Training/Validation')
print('** INFO: RelValTenTau Training/Validation')
dfTenTauTraining, dfTenTauValidation = train_test_split(dfTenTau, test_size=0.3)

print('** INFO: RelValSingleTau Training/Validation')
dfSingleTauTraining, dfSingleTauValidation = train_test_split(dfSingleTau, test_size=0.3)

print('** INFO: HH Training/Validation')
dfHHTraining, dfHHValidation = train_test_split(dfHH, test_size=0.3)

print('** INFO: QCD Training/Validation')
dfQCDTraining, dfQCDValidation = train_test_split(dfQCD, test_size=0.3)

print('** INFO: RelValNu Training/Validation')
dfNuTraining, dfNuValidation = train_test_split(dfNu, test_size=0.3)
dfNuTraining.set_index('event', inplace=True)
dfNuValidation.set_index('event', inplace=True)

# MERGE
print('** INFO: merging Training/Validation')
dfMergedTraining = pd.concat([dfTenTauTraining,dfSingleTauTraining,dfHHTraining,dfNuTraining,dfQCDTraining],sort=False)
dfMergedValidation = pd.concat([dfTenTauValidation,dfSingleTauValidation,dfHHValidation,dfNuValidation,dfQCDValidation],sort=False)
dfMergedTraining.fillna(0.0, inplace=True)
dfMergedValidation.fillna(0.0, inplace=True)

print('** INFO: saving file ' + outFileTraining_dict[name])
store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
store_tr[name] = dfMergedTraining
store_tr.close()

print('** INFO: saving file ' + outFileValidation_dict[name])
store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
store_val[name] = dfMergedValidation
store_val.close()









