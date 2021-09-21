# THIS SCRIPT CALCULATES AND STORES 3 CLUSTER-BASED ISOLATION FEATURES AND 11 TOWER-BASED ISOLATION FEATURES
# FOR THE ISOLATION CALCULATION ALL CLUSTERS ARE RETAINED, ALSO THOSE NOT PASSING THE PUBDTWP

import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import argparse


def deltar2cluster ( df ):
    delta_eta = np.abs(df['cl3d_eta']-df['cl3d_eta_ass'])
    delta_phi = np.abs(df['cl3d_phi']-df['cl3d_phi_ass'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def L1Cl3dEtIso ( dfL1Candidates, dfL1associated2Candidates, dR ):
    df_joined  = dfL1Candidates.join(dfL1associated2Candidates, on='event', how='left', rsuffix='_ass', sort=False)

    df_joined['deltar2cluster'] = deltar2cluster(df_joined)
    sel = (df_joined['deltar2cluster'] <= dR) & (df_joined['deltar2cluster'] > 0.0001)
    df_joined = df_joined[sel].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt_c3'], inplace=True)
    
    dfL1Candidates['cl3d_etIso_dR{0}'.format(int(dR*10))] = df_joined.groupby(['event', 'cl3d_pt_c3'])['cl3d_pt_c3_ass'].sum()
    dfL1Candidates['cl3d_NclIso_dR{0}'.format(int(dR*10))] = df_joined.groupby(['event', 'cl3d_pt_c3'])['cl3d_pt_c3_ass'].size()

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

def deltar2tower ( df ):
    delta_eta = np.abs(df['cl3d_eta'] - df['tower_eta'])
    delta_phi = np.abs(df['cl3d_phi'] - df['tower_phi'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def L1TowerEtIso ( dfL1Candidates, dfL1Towers, dRsgn, dRiso, dRisoEm, dRisoHad ):
    df_joined  = dfL1Candidates.join(dfL1Towers, on='event', how='inner', rsuffix='_tow', sort=False) # use 'inner' so that only the events present in the candidates dataframe are actually joined

    df_joined['deltar2tower'] = deltar2tower(df_joined)
    sel_sgn = (df_joined['deltar2tower'] <= dRsgn)
    sel_iso = (df_joined['deltar2tower'] <= dRiso) & (df_joined['deltar2tower'] > dRsgn)
    df_joined_sgn = df_joined[sel_sgn].copy(deep=True)
    df_joined_iso = df_joined[sel_iso].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt'], inplace=True)

    dfL1Candidates['tower_etSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined_sgn.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
    dfL1Candidates['tower_eSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined_sgn.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()

    dfL1Candidates['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
    dfL1Candidates['tower_eIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()
    dfL1Candidates['tower_etEmIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoEm*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_etEm'].sum()
    dfL1Candidates['tower_etHadIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoHad*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_etHad'].sum() 

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

def IsoCalculation(dfTr, dfVal, dfTowers, mode='candidate'):
    if mode == 'Nu':
        print('       doing nu special L1 candidate selection')
        dfTr.reset_index(inplace=True)
        dfL1CandidatesTr = dfTr.query('cl3d_pubdt_passWP99==True').copy(deep=True) # selecting WP 99 we select also 95 and 90
        dfL1CandidatesTr.sort_values('cl3d_pt_c3', inplace=True)
        dfL1CandidatesTr.drop_duplicates('event', keep='last', inplace=True) # keep only highest pt cluster
        sel = dfTr['cl3d_pt_c3'].isin(dfL1CandidatesTr['cl3d_pt_c3'])
        dfL1ass2candTr = dfTr.drop(dfTr[sel].index)
        dfL1CandidatesTr.set_index('event', inplace=True)
        dfL1CandidatesTr.sort_values('event', inplace=True)
        dfL1ass2candTr.set_index('event', inplace=True)
        dfL1ass2candTr.sort_values('event', inplace=True)

        dfVal.reset_index(inplace=True)
        dfL1CandidatesVal = dfVal.query('cl3d_pubdt_passWP99==True').copy(deep=True) # selecting WP 99 we select also 95 and 90
        dfL1CandidatesVal.sort_values('cl3d_pt_c3', inplace=True)
        dfL1CandidatesVal.drop_duplicates('event', keep='last', inplace=True) # keep only highest pt cluster
        sel = dfVal['cl3d_pt_c3'].isin(dfL1CandidatesVal['cl3d_pt_c3'])
        dfL1ass2candVal = dfVal.drop(dfVal[sel].index)
        dfL1CandidatesVal.set_index('event', inplace=True)
        dfL1CandidatesVal.sort_values('event', inplace=True)
        dfL1ass2candVal.set_index('event', inplace=True)
        dfL1ass2candVal.sort_values('event', inplace=True)
    else:
        dfL1CandidatesTr = dfTr.query('cl3d_isbestmatch==True').copy(deep=True)
        dfL1ass2candTr = dfTr.query('cl3d_isbestmatch==False').copy(deep=True)
        dfL1CandidatesVal = dfVal.query('cl3d_isbestmatch==True').copy(deep=True)
        dfL1ass2candVal = dfVal.query('cl3d_isbestmatch==False').copy(deep=True)

    # split the two endcaps to make the loops over the rows faster
    dfL1CandidatesTr_p = dfL1CandidatesTr.query('cl3d_eta>=0').copy(deep=True)
    dfL1CandidatesTr_m = dfL1CandidatesTr.query('cl3d_eta<0').copy(deep=True)
    dfL1ass2candTr_p = dfL1ass2candTr.query('cl3d_eta>=0').copy(deep=True)
    dfL1ass2candTr_m = dfL1ass2candTr.query('cl3d_eta<0').copy(deep=True)
    dfL1CandidatesVal_p = dfL1CandidatesVal.query('cl3d_eta>=0').copy(deep=True)
    dfL1CandidatesVal_m = dfL1CandidatesVal.query('cl3d_eta<0').copy(deep=True)
    dfL1ass2candVal_p = dfL1ass2candVal.query('cl3d_eta>=0').copy(deep=True)
    dfL1ass2candVal_m = dfL1ass2candVal.query('cl3d_eta<0').copy(deep=True)
    
    dfL1Towers_p = dfTowers.query('tower_eta>=0').copy(deep=True)
    dfL1Towers_m = dfTowers.query('tower_eta<0').copy(deep=True)

    print('       Training dataset')
    L1Cl3dEtIso(dfL1CandidatesTr_p, dfL1ass2candTr_p, 0.4)
    L1Cl3dEtIso(dfL1CandidatesTr_m, dfL1ass2candTr_m, 0.4)
    L1TowerEtIso(dfL1CandidatesTr_p, dfL1Towers_p, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
    L1TowerEtIso(dfL1CandidatesTr_m, dfL1Towers_m, 0.1, 0.3, 0.3, 0.7) # "
    L1TowerEtIso(dfL1CandidatesTr_p, dfL1Towers_p, 0.2, 0.4, 0.4, 0.7) # "
    L1TowerEtIso(dfL1CandidatesTr_m, dfL1Towers_m, 0.2, 0.4, 0.4, 0.7) # "
    print('       Validation dataset')
    L1Cl3dEtIso(dfL1CandidatesVal_p, dfL1ass2candVal_p, 0.4)
    L1Cl3dEtIso(dfL1CandidatesVal_m, dfL1ass2candVal_m, 0.4)
    L1TowerEtIso(dfL1CandidatesVal_p, dfL1Towers_p, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
    L1TowerEtIso(dfL1CandidatesVal_m, dfL1Towers_m, 0.1, 0.3, 0.3, 0.7) # "
    L1TowerEtIso(dfL1CandidatesVal_p, dfL1Towers_p, 0.2, 0.4, 0.4, 0.7) # "
    L1TowerEtIso(dfL1CandidatesVal_m, dfL1Towers_m, 0.2, 0.4, 0.4, 0.7) # "

    dfOutTr = pd.concat([dfL1CandidatesTr_p, dfL1CandidatesTr_m], sort=False)
    dfOutVal = pd.concat([dfL1CandidatesVal_p, dfL1CandidatesVal_m], sort=False)

    return dfOutTr, dfOutVal

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/PUrejected_C1fullC2C3_fullPUnoPt'
    matchdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_C1fullC2C3_fullPUnoPt'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileHHTowers_dict = {
        'threshold'    : matchdir+'/GluGluHHTo2b2Tau_PU200_th_towers.hdf5',
        'supertrigger' : matchdir+'/',
        'bestchoice'   : matchdir+'/',
        'bestcoarse'   : matchdir+'/',
        'mixed'        : matchdir+'/'
    }

    inFileTenTauTowers_dict = {
        'threshold'    : matchdir+'/RelValTenTau_PU200_th_towers.hdf5',
        'supertrigger' : matchdir+'/',
        'bestchoice'   : matchdir+'/',
        'bestcoarse'   : matchdir+'/',
        'mixed'        : matchdir+'/'
    }

    inFileSingleTauTowers_dict = {
        'threshold'    : matchdir+'/RelValSingleTau_PU200_th_towers.hdf5',
        'supertrigger' : matchdir+'/',
        'bestchoice'   : matchdir+'/',
        'bestcoarse'   : matchdir+'/',
        'mixed'        : matchdir+'/'
    }

    inFileQCDTowers_dict = {
        'threshold'    : matchdir+'/QCD_PU200_th_towers.hdf5',
        'supertrigger' : matchdir+'/',
        'bestchoice'   : matchdir+'/',
        'bestcoarse'   : matchdir+'/',
        'mixed'        : matchdir+'/'
    }

    inFileNuTowers_dict = {
        'threshold'    : matchdir+'/RelValNu_PU200_th_towers.hdf5',
        'supertrigger' : matchdir+'/',
        'bestchoice'   : matchdir+'/',
        'bestcoarse'   : matchdir+'/',
        'mixed'        : matchdir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_isoCalculated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_isoCalculated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    dfTraining_dict = {}   # dictionary of the merged training dataframes
    dfValidation_dict = {} # dictionary of the merged test dataframes


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
            
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting DM sorting for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()

        store = pd.HDFStore(inFileHHTowers_dict[name], mode='r')
        dfHHTowers = store[name]
        store.close()

        store = pd.HDFStore(inFileTenTauTowers_dict[name], mode='r')
        dfTenTauTowers = store[name]
        store.close()

        store = pd.HDFStore(inFileSingleTauTowers_dict[name], mode='r')
        dfSingleTauTowers = store[name]
        store.close()

        store = pd.HDFStore(inFileQCDTowers_dict[name], mode='r')
        dfQCDTowers = store[name]
        store.close()

        store = pd.HDFStore(inFileNuTowers_dict[name], mode='r')
        dfNuTowers = store[name]
        store.close()


        ######################### SELECT EVENTS FOR TRAINING #########################  

        print('\n** INFO: calculating eT Iso for L1Tau candidates - HH dataset ')
        dfHHTr = dfTraining_dict[name].query('dataset==0').copy(deep=True)
        dfHHVal = dfValidation_dict[name].query('dataset==0').copy(deep=True)
        dfHHOutTr, dfHHOutVal = IsoCalculation(dfHHTr, dfHHVal, dfHHTowers)

        print('\n** INFO: calculating eT Iso for L1Tau candidates - TenTau dataset ')
        dfTenTauTr = dfTraining_dict[name].query('dataset==1').copy(deep=True)
        dfTenTauVal = dfValidation_dict[name].query('dataset==1').copy(deep=True)
        dfTenTauOutTr, dfTenTauOutVal = IsoCalculation(dfTenTauTr, dfTenTauVal, dfTenTauTowers)

        print('\n** INFO: calculating eT Iso for L1Tau candidates - SingleTau dataset ')
        dfSingleTauTr = dfTraining_dict[name].query('dataset==2').copy(deep=True)
        dfSingleTauVal = dfValidation_dict[name].query('dataset==2').copy(deep=True)
        dfSingleTauOutTr, dfSingleTauOutVal = IsoCalculation(dfSingleTauTr, dfSingleTauVal, dfSingleTauTowers)

        print('\n** INFO: calculating eT Iso for L1Tau candidates - QCD dataset ')
        dfQCDTr = dfTraining_dict[name].query('dataset==3').copy(deep=True)
        dfQCDVal = dfValidation_dict[name].query('dataset==3').copy(deep=True)
        dfQCDOutTr, dfQCDOutVal = IsoCalculation(dfQCDTr, dfQCDVal, dfQCDTowers)

        print('\n** INFO: calculating eT Iso for L1Tau candidates - Nu dataset ')
        dfNuTr = dfTraining_dict[name].query('dataset==4').copy(deep=True)
        dfNuVal = dfValidation_dict[name].query('dataset==4').copy(deep=True)
        dfNuOutTr, dfNuOutVal = IsoCalculation(dfNuTr, dfNuVal, dfNuTowers, mode='Nu')

        
        ######################### SAVE FILES #########################

        dfTraining_dict[name] = pd.concat([dfHHOutTr, dfTenTauOutTr, dfSingleTauOutTr, dfQCDOutTr, dfNuOutTr], sort=False)
        dfValidation_dict[name] = pd.concat([dfHHOutVal, dfTenTauOutVal, dfSingleTauOutVal, dfQCDOutVal, dfNuOutVal], sort=False)

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()







