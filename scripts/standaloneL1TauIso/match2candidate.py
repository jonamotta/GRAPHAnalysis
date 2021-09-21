import os
import numpy as np
import pandas as pd
import dask.dataframe as ddf
import root_pandas
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import argparse
import time

def deltar2cluster ( df ):
    delta_eta = np.abs(df['cl3d_eta']-df['cl3d_eta_ass'])
    delta_phi = np.abs(df['cl3d_phi']-df['cl3d_phi_ass'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def L1Cl3dEtIso ( dfL1Candidates, dfL1associated2Candidates, dR ):
    df_joined  = dfL1Candidates.join(dfL1associated2Candidates, on='event', how='left', rsuffix='_ass', sort=False)

    df_joined['deltar2cluster'] = deltar2cluster(df_joined)
    df_joined.query('deltar2cluster<={0} and deltar2cluster>0.0001'.format(dR), inplace=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt_c3'], inplace=True)
    
    dfL1Candidates['cl3d_etIso_dR{0}'.format(int(dR*10))] = df_joined.groupby(['event', 'cl3d_pt_c3'])['cl3d_pt_c3_ass'].sum()
    dfL1Candidates['cl3d_NclIso_dR{0}'.format(int(dR*10))] = df_joined.groupby(['event', 'cl3d_pt_c3'])['cl3d_pt_c3_ass'].size()

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined # free memory

def deltar2tower ( df ):
    delta_eta = np.abs(df['cl3d_eta'] - df['tower_eta'])
    delta_phi = np.abs(df['cl3d_phi'] - df['tower_phi'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

#def L1TowerEtIso ( dfL1Candidates, dfL1Towers, dRsgn, dRiso, dRisoEm, dRisoHad ):
#    df_joined  = ddf.from_pandas(dfL1Candidates.join(dfL1Towers, on='event', how='inner', rsuffix='_tow', sort=False), npartitions=130) # use 'inner' so that only the events present in the candidates dataframe are actually joined
#
#    df_joined['deltar2tower'] = deltar2tower(df_joined)
#    df_joined.query('deltar2tower <= {0} and deltar2tower > {1}'.format(dRiso,dRsgn), inplace=True)
#
#    dfL1Candidates.reset_index(inplace=True)
#    dfL1Candidates.set_index(['event', 'cl3d_pt'], inplace=True)
#
#    dfL1Candidates['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
#    dfL1Candidates['tower_eIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()
#    dfL1Candidates['tower_etEmIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoEm*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_etEm'].sum()
#    dfL1Candidates['tower_etHadIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoHad*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_etHad'].sum() 
#
#    dfL1Candidates.reset_index(inplace=True)
#    dfL1Candidates.set_index('event',inplace=True)
#    dfL1Candidates.fillna(0.0,inplace=True)
#
#    del df_joined # free memory
#
#def L1TowerEtSgn ( dfL1Candidates, dfL1Towers, dRsgn):
#    df_joined  = dfL1Candidates.join(dfL1Towers, on='event', how='inner', rsuffix='_tow', sort=False) # use 'inner' so that only the events present in the candidates dataframe are actually joined
#
#    df_joined['deltar2tower'] = deltar2tower(df_joined)
#    df_joined.query('deltar2tower <= {0}'.format(dRsgn), inplace=True)
#
#    dfL1Candidates.reset_index(inplace=True)
#    dfL1Candidates.set_index(['event', 'cl3d_pt'], inplace=True)
#
#    dfL1Candidates['tower_etSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
#    dfL1Candidates['tower_eSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()
#
#    dfL1Candidates.reset_index(inplace=True)
#    dfL1Candidates.set_index('event',inplace=True)
#    dfL1Candidates.fillna(0.0,inplace=True)
#
#    del df_joined # free memory

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

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'r') as f:
        return pickle.load(f)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--doHH', dest='doHH', help='match the HH samples?',  action='store_true', default=False)
    parser.add_argument('--doTenTau', dest='doTenTau', help='match the TenTau samples?',  action='store_true', default=False)
    parser.add_argument('--doSingleTau', dest='doSingleTau', help='match the SingleTau samples?',  action='store_true', default=False)
    parser.add_argument('--doAllTau', dest='doAllTau', help='match all the Tau samples',  action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='match the QCD samples?',  action='store_true', default=False)
    parser.add_argument('--doMinbias', dest='doMinbias', help='match the Minbias samples?',  action='store_true', default=False)
    parser.add_argument('--doNu', dest='doNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--testRun', dest='testRun', help='do test run with reduced number of events?',  action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    start = time.time()

    if not args.doHH and not args.doTenTau and not args.doSingleTau and not args.doAllTau and not args.doQCD and not args.doMinbias and not args.doNu:
        print('** WARNING: no matching dataset specified. What do you want to do (doHH, doTenTau, doSingleTau doAllTau, doQCD, doMinbias, doNu)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    if args.doAllTau:
        args.doSingleTau = True
        args.doTenTau = True
        args.doHH = True

    print('** INFO: using front-end option: '+args.FE)

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/rateEvaluation'

    # DICTIONARIES FOR THE MATCHING
    if args.doHH:
        inFileHH_match = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileHH_towers = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        inFileTenTau_match = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileTenTau_towers = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        inFileSingleTau_match = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileSingleTau_towers = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        inFileQCD_match = {
            'threshold'    : indir+'/QCD_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileQCD_towers = {
            'threshold'    : indir+'/QCD_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        inFileMinbias_match = {
            'threshold'    : indir+'/Minbias_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileMinbias_towers = {
            'threshold'    : indir+'/Minbias_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        inFileNu_match = {
            'threshold'    : indir+'/RelValNu_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileNu_towers = {
            'threshold'    : indir+'/RelValNu_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE CALIBRATION
    calib_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration'

    modelC1_calib = {
        'threshold'    : calib_model_indir+'/model_c1_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    modelC2_calib = {
        'threshold'    : calib_model_indir+'/model_c2_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    modelC3_calib = {
        'threshold'    : calib_model_indir+'/model_c3_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    if args.doHH:
        outFileHH_calib = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_calib = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_calib = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_calib = {
            'threshold'    : indir+'/QCD_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_calib = {
            'threshold'    : indir+'/Minbias_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_calib = {
            'threshold'    : indir+'/RelValNu_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE PILEUP REJECTION
    PUrej_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/PUrejection'

    model_PUrej = {
        'threshold'    : PUrej_model_indir+'/model_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP90_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP90_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP95_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP95_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP99_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP99_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    if args.doHH:
        outFileHH_PUrej = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_PUrej = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_PUrej = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_PUrej = {
            'threshold'    : indir+'/QCD_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_PUrej = {
            'threshold'    : indir+'/Minbias_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_PUrej = {
            'threshold'    : indir+'/RelValNu_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE ISO QCD REJECTION
    isoQCDrej_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/isolation'

    model_isoQCDrej = {
        'threshold'    : isoQCDrej_model_indir+'_PUWP{0}/model_isolation_PUWP{0}_th_PU200.pkl',
        'supertrigger' : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestchoice'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestcoarse'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'mixed'        : isoQCDrej_model_indir+'_PUWP{0}/'
    }

    WP10_isoQCDrej = {
        'threshold'    : isoQCDrej_model_indir+'_PUWP{0}/WP10_isolation_PUWP{0}_th_PU200.pkl',
        'supertrigger' : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestchoice'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestcoarse'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'mixed'        : isoQCDrej_model_indir+'_PUWP{0}/'
    }

    WP05_isoQCDrej = {
        'threshold'    : isoQCDrej_model_indir+'_PUWP{0}/WP05_isolation_PUWP{0}_th_PU200.pkl',
        'supertrigger' : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestchoice'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestcoarse'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'mixed'        : isoQCDrej_model_indir+'_PUWP{0}/'
    }

    WP01_isoQCDrej = {
        'threshold'    : isoQCDrej_model_indir+'_PUWP{0}/WP01_isolation_PUWP{0}_th_PU200.pkl',
        'supertrigger' : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestchoice'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'bestcoarse'   : isoQCDrej_model_indir+'_PUWP{0}/',
        'mixed'        : isoQCDrej_model_indir+'_PUWP{0}/'
    }

    if args.doHH:
        outFileHH_isoQCDrej = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_isoQCDrej = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_isoQCDrej = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_isoQCDrej = {
            'threshold'    : indir+'/QCD_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_isoQCDrej = {
            'threshold'    : indir+'/Minbias_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_isoQCDrej = {
            'threshold'    : indir+'/RelValNu_PU200_th_isoQCDrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }


    # DICTIONARIES FOR THE DM sorting
    DMsort_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/DMsorting'

    model_DMsort = {
        'threshold'    : DMsort_model_indir+'/model_DMsorting_th_PU200_PUWP{0}_ISOWP{1}.pkl',
        'supertrigger' : DMsort_model_indir+'/',
        'bestchoice'   : DMsort_model_indir+'/',
        'bestcoarse'   : DMsort_model_indir+'/',
        'mixed'        : DMsort_model_indir+'/'
    }

    if args.doHH:
        outFileHH_DMsort = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_DMsort = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_DMsort = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_DMsort = {
            'threshold'    : indir+'/QCD_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_DMsort = {
            'threshold'    : indir+'/Minbias_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_DMsort = {
            'threshold'    : indir+'/RelValNu_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    ###########################################
    # CREATE LISTS OF FILES TO CREATE AND SAVE

    dfs4Iso = []
    dfsTowers = []
    outFiles4Iso_calib = []
    outFiles4Iso_PUrej = []
    outFiles4Iso_isoQCDrej = []
    outFiles4Iso_DMsort = []

    if args.doHH:
        store = pd.HDFStore(inFileHH_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) or (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileHH_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileHH_calib)
        outFiles4Iso_PUrej.append(outFileHH_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileHH_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileHH_DMsort)

        del tmp, store # free memory

    if args.doTenTau:
        store = pd.HDFStore(inFileTenTau_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) or (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileTenTau_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileTenTau_calib)
        outFiles4Iso_PUrej.append(outFileTenTau_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileTenTau_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileTenTau_DMsort)

        del tmp, store # free memory

    if args.doSingleTau:
        store = pd.HDFStore(inFileSingleTau_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) or (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileSingleTau_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileSingleTau_calib)
        outFiles4Iso_PUrej.append(outFileSingleTau_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileSingleTau_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileSingleTau_DMsort)

        del tmp, store # free memory

    if args.doQCD:
        store = pd.HDFStore(inFileQCD_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) or (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9)))', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileQCD_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileQCD_calib)
        outFiles4Iso_PUrej.append(outFileQCD_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileQCD_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileQCD_DMsort)

        del tmp, store # free memory

    if args.doMinbias:
        store = pd.HDFStore(inFileMinbias_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('cl3d_pt>4. and cl3d_abseta>1.6 and cl3d_abseta<2.9', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileMinbias_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileMinbias_calib)
        outFiles4Iso_PUrej.append(outFileMinbias_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileMinbias_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileMinbias_DMsort)

        del tmp, store # free memory

    if args.doNu:
        store = pd.HDFStore(inFileNu_match[args.FE], mode='r')
        tmp = store[args.FE].copy(deep=True)
        tmp['cl3d_abseta'] = np.abs(tmp['cl3d_eta'])
        tmp.query('cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9', inplace=True)
        dfs4Iso.append(tmp)
        store.close()

        store = pd.HDFStore(inFileNu_towers[args.FE], mode='r')
        dfsTowers.append(store[args.FE])
        store.close()

        outFiles4Iso_calib.append(outFileNu_calib)
        outFiles4Iso_PUrej.append(outFileNu_PUrej)
        outFiles4Iso_isoQCDrej.append(outFileNu_isoQCDrej)
        outFiles4Iso_DMsort.append(outFileNu_DMsort)

        del tmp, store # free memory

    ####################################################################
    # LOAD MODELS FOR CALIBRATION, PU REJECTION, AND DECAY MODE SORTING

    # features used for the C2 calibration step
    features_calib = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean','cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    modelC1 = load_obj(modelC1_calib[args.FE])
    modelC2 = load_obj(modelC2_calib[args.FE])
    modelC3 = load_obj(modelC3_calib[args.FE])

    # features for PU rejection BDT
    features_PUrej = ['cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    model_PU = load_obj(model_PUrej[args.FE])
    PUbdtWP99 = load_obj(WP99_PUrej[args.FE])
    PUbdtWP95 = load_obj(WP95_PUrej[args.FE])
    PUbdtWP90 = load_obj(WP90_PUrej[args.FE])

    # features for ISO QCD rejection BDT
    features_isoQCDrej = ['cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_eSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_eSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_eIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_eIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']

    # features for DM sorting BDT
    features = ['cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']

    ##################################################
    # LOOP OVER THE LISTS OF FILES TO CREATE AND SAVE

    for k in range(len(dfs4Iso)):

        toKeep = ['cl3d_isbestmatch', 'cl3d_pt', 'cl3d_eta', 'cl3d_phi', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
        dfs4Iso[k] = dfs4Iso[k].loc[:,toKeep]
        del toKeep # free memory

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting cluster calibration, PU rejection, and DM sorting to produce '+outFiles4Iso_DMsort[k][args.FE])

        #######################################################################################################################################
        # CLUSTERS CALIBRATION
        # application of calibration 1
        print('\n** INFO: applying calibration C1')    
        dfs4Iso[k]['cl3d_pt_c1'] = dfs4Iso[k]['cl3d_pt'] + modelC1.predict(dfs4Iso[k][['cl3d_abseta']])
        # application of calibration 2
        print('** INFO: applying calibration C2')
        dfs4Iso[k]['cl3d_pt_c2'] = dfs4Iso[k]['cl3d_pt_c1'] * modelC2.predict(dfs4Iso[k][features_calib])
        # application of calibration 3
        print('** INFO: applying calibration C3')
        logpt1 = np.log(abs(dfs4Iso[k]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfs4Iso[k]['cl3d_pt_c3'] = dfs4Iso[k]['cl3d_pt_c2'] / modelC3.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        # save file
        print('** INFO: saving file ' + outFiles4Iso_calib[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_calib[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        del logpt1, logpt2, logpt3, logpt4, modelC1, modelC2, modelC3, store # free memory

        #######################################################################################################################################
        # PILE UP REJECTION
        print('\n** INFO: applying PU rejection BDT')
        full = xgb.DMatrix(data=dfs4Iso[k][features_PUrej], feature_names=features_PUrej)
        dfs4Iso[k]['cl3d_pubdt_score'] = model_PU.predict(full)
        dfs4Iso[k]['cl3d_pubdt_passWP99'] = dfs4Iso[k]['cl3d_pubdt_score'] > PUbdtWP99
        dfs4Iso[k]['cl3d_pubdt_passWP95'] = dfs4Iso[k]['cl3d_pubdt_score'] > PUbdtWP95
        dfs4Iso[k]['cl3d_pubdt_passWP90'] = dfs4Iso[k]['cl3d_pubdt_score'] > PUbdtWP90
        # save file
        print('** INFO: saving file ' + outFiles4Iso_PUrej[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_PUrej[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        del full, model_PU, store # free memory

        #######################################################################################################################################
        # CALCULATION OF ISOLATION FEATURES
        print('\n** INFO: calculationg isolation features')
        if ('Minbias' in outFiles4Iso_isoQCDrej[k][args.FE]) or ('Nu' in outFiles4Iso_isoQCDrej[k][args.FE]):
            print('       doing nu/minbias special L1 candidate selection')
            dfs4Iso[k].reset_index(inplace=True)
            dfL1Candidates = dfs4Iso[k].query('cl3d_pubdt_passWP99==True').copy(deep=True) # selecting WP 99 we select also 95 and 90 --> the selection on the specific WP is applied when calculating turnONs
            dfL1Candidates.sort_values('cl3d_pt_c3', inplace=True)
            dfL1Candidates = dfL1Candidates.groupby('event').tail(5).copy(deep=True) # keep only the 5 highest pt cluster
            #dfL1Candidates.drop_duplicates('event', keep='last', inplace=True) # keep only highest pt cluster
            sel = dfs4Iso[k]['cl3d_pt_c3'].isin(dfL1Candidates['cl3d_pt_c3'])
            dfL1ass2cand = dfs4Iso[k].drop(dfs4Iso[k][sel].index)
            dfL1ass2cand = dfs4Iso[k]
            dfL1Candidates.set_index('event', inplace=True)
            dfL1Candidates.sort_values('event', inplace=True)
            dfL1ass2cand.set_index('event', inplace=True)
            dfL1ass2cand.sort_values('event', inplace=True)
            del sel # free memory
        else:
            dfL1Candidates = dfs4Iso[k].query('cl3d_isbestmatch==True').copy(deep=True)
            dfL1ass2cand = dfs4Iso[k].query('cl3d_isbestmatch==False').copy(deep=True)

        # split the two endcaps to make the loops over the rows faster
        dfL1Candidates_p = dfL1Candidates.query('cl3d_eta>=0').copy(deep=True)
        dfL1Candidates_m = dfL1Candidates.query('cl3d_eta<0').copy(deep=True)
        dfL1ass2cand_p = dfL1ass2cand.query('cl3d_eta>=0').copy(deep=True)
        dfL1ass2cand_m = dfL1ass2cand.query('cl3d_eta<0').copy(deep=True)
        dfL1Towers_p = dfsTowers[k].query('tower_eta>=0').copy(deep=True)
        dfL1Towers_m = dfsTowers[k].query('tower_eta<0').copy(deep=True)

        del dfL1Candidates, dfL1ass2cand # free memory --> these dataframes are not needed anymore, we already split it in plus/minus endcaps
        dfsTowers[k] = None # free memory --> this dataframe is not needed anymore so we can just empty the entry of the array
        dfs4Iso[k] = None # free memory --> this dataframe is not needed anymore and it is ovewritten after the iso features calculation

        if 'Minbias' in outFiles4Iso_isoQCDrej[k][args.FE]:
            dfL1Candidates_p_d = dfL1Candidates_p.query('event<300000').copy(deep=True)
            dfL1Candidates_p_m = dfL1Candidates_p.query('event>=300000 and event<600000').copy(deep=True)
            dfL1Candidates_p_u = dfL1Candidates_p.query('event>=600000').copy(deep=True)
            del dfL1Candidates_p # free memory --> this dataframe are not needed anymore
            dfL1Candidates_m_d = dfL1Candidates_m.query('event<300000').copy(deep=True)
            dfL1Candidates_m_m = dfL1Candidates_m.query('event>=300000 and event<600000').copy(deep=True)
            dfL1Candidates_m_u = dfL1Candidates_m.query('event>=600000').copy(deep=True)
            del dfL1Candidates_m # free memory --> this dataframe are not needed anymore
            dfL1ass2cand_p_d = dfL1ass2cand_p.query('event<300000').copy(deep=True)
            dfL1ass2cand_p_m = dfL1ass2cand_p.query('event>=300000 and event<600000').copy(deep=True)
            dfL1ass2cand_p_u = dfL1ass2cand_p.query('event>=600000').copy(deep=True)
            del dfL1ass2cand_p # free memory --> this dataframe are not needed anymore
            dfL1ass2cand_m_d = dfL1ass2cand_m.query('event<300000').copy(deep=True)
            dfL1ass2cand_m_m = dfL1ass2cand_m.query('event>=300000 and event<600000').copy(deep=True)
            dfL1ass2cand_m_u = dfL1ass2cand_m.query('event>=600000').copy(deep=True)
            del dfL1ass2cand_m # free memory --> this dataframe are not needed anymore
            dfL1Towers_p_d = dfL1Towers_p.query('event<300000').copy(deep=True)
            dfL1Towers_p_m = dfL1Towers_p.query('event>=300000 and event<600000').copy(deep=True)
            dfL1Towers_p_u = dfL1Towers_p.query('event>=600000').copy(deep=True)
            del dfL1Towers_p # free memory --> this dataframe are not needed anymore
            dfL1Towers_m_d = dfL1Towers_m.query('event<300000').copy(deep=True)
            dfL1Towers_m_m = dfL1Towers_m.query('event>=300000 and event<600000').copy(deep=True)
            dfL1Towers_m_u = dfL1Towers_m.query('event>=600000').copy(deep=True)
            del dfL1Towers_m # free memory --> this dataframe are not needed anymore

            print('       positive z encap - down chunk')
            #L1TowerEtSgn(dfL1Candidates_p_d, dfL1Towers_p_d, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_p_d, dfL1Towers_p_d, 0.2) # "
            L1TowerEtIso(dfL1Candidates_p_d, dfL1Towers_p_d, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_p_d, dfL1Towers_p_d, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_p_d, dfL1ass2cand_p_d, 0.4)
            print('       positive z encap - mid chunk')
            #L1TowerEtSgn(dfL1Candidates_p_m, dfL1Towers_p_m, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_p_m, dfL1Towers_p_m, 0.2) # "
            L1TowerEtIso(dfL1Candidates_p_m, dfL1Towers_p_m, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_p_m, dfL1Towers_p_m, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_p_m, dfL1ass2cand_p_m, 0.4)
            print('       positive z encap - up chunk')
            #L1TowerEtSgn(dfL1Candidates_p_u, dfL1Towers_p_u, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_p_u, dfL1Towers_p_u, 0.2) # "
            L1TowerEtIso(dfL1Candidates_p_u, dfL1Towers_p_u, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_p_u, dfL1Towers_p_u, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_p_u, dfL1ass2cand_p_u, 0.4)
            print('       negative z encap - down chunk')
            #L1TowerEtSgn(dfL1Candidates_m_d, dfL1Towers_m_d, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_m_d, dfL1Towers_m_d, 0.2) # "
            L1TowerEtIso(dfL1Candidates_m_d, dfL1Towers_m_d, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_m_d, dfL1Towers_m_d, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_m_d, dfL1ass2cand_m_d, 0.4)
            print('       negative z encap - mid chunk')
            #L1TowerEtSgn(dfL1Candidates_m_m, dfL1Towers_m_m, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_m_m, dfL1Towers_m_m, 0.2) # "
            L1TowerEtIso(dfL1Candidates_m_m, dfL1Towers_m_m, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_m_m, dfL1Towers_m_m, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_m_m, dfL1ass2cand_m_m, 0.4)
            print('       negative z encap - up chunk')
            #L1TowerEtSgn(dfL1Candidates_m_u, dfL1Towers_m_u, 0.1) # dRsgn
            #L1TowerEtSgn(dfL1Candidates_m_u, dfL1Towers_m_u, 0.2) # "
            L1TowerEtIso(dfL1Candidates_m_u, dfL1Towers_m_u, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_m_u, dfL1Towers_m_u, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_m_u, dfL1ass2cand_m_u, 0.4)

            dfs4Iso[k] = pd.concat([dfL1Candidates_p_d, dfL1Candidates_p_m, dfL1Candidates_p_u, dfL1Candidates_m_d, dfL1Candidates_m_m, dfL1Candidates_m_u], sort=False)

        else:
            print('       positive z encap')
            L1TowerEtSgn(dfL1Candidates_p, dfL1Towers_p, 0.1) # dRsgn
            L1TowerEtSgn(dfL1Candidates_p, dfL1Towers_p, 0.2) # "
            L1TowerEtIso(dfL1Candidates_p, dfL1Towers_p, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_p, dfL1Towers_p, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_p, dfL1ass2cand_p, 0.4)
            print('       negative z encap')
            L1TowerEtSgn(dfL1Candidates_m, dfL1Towers_m, 0.1) # dRsgn
            L1TowerEtSgn(dfL1Candidates_m, dfL1Towers_m, 0.2) # "
            L1TowerEtIso(dfL1Candidates_m, dfL1Towers_m, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
            L1TowerEtIso(dfL1Candidates_m, dfL1Towers_m, 0.2, 0.4, 0.4, 0.7) # "
            L1Cl3dEtIso(dfL1Candidates_m, dfL1ass2cand_m, 0.4)

            dfs4Iso[k] = pd.concat([dfL1Candidates_p, dfL1Candidates_m], sort=False)

        #######################################################################################################################################
        # ISO QCD REJECTION
        print('\n** INFO: applying ISO QCD rejection BDT')
        for PUWP in ['99', '95', '90']:
            print('    - PUWP = {0}'.format(PUWP))
            model_iso = load_obj(model_isoQCDrej[args.FE].format(PUWP))
            ISObdtWP01 = load_obj(WP01_isoQCDrej[args.FE].format(PUWP))
            ISObdtWP05 = load_obj(WP05_isoQCDrej[args.FE].format(PUWP))
            ISObdtWP10 = load_obj(WP10_isoQCDrej[args.FE].format(PUWP))
            full = xgb.DMatrix(data=dfs4Iso[k][features_isoQCDrej], feature_names=features_isoQCDrej)
            dfs4Iso[k]['cl3d_isobdt_score'] = model_iso.predict(full)
            dfs4Iso[k]['cl3d_isobdt_passWP01_PUWP{0}'.format(PUWP)] = dfs4Iso[k]['cl3d_isobdt_score'] > ISObdtWP01
            dfs4Iso[k]['cl3d_isobdt_passWP05_PUWP{0}'.format(PUWP)] = dfs4Iso[k]['cl3d_isobdt_score'] > ISObdtWP05
            dfs4Iso[k]['cl3d_isobdt_passWP10_PUWP{0}'.format(PUWP)] = dfs4Iso[k]['cl3d_isobdt_score'] > ISObdtWP10
            del full, model_iso, ISObdtWP01, ISObdtWP05, ISObdtWP10
        
        # save file
        print('** INFO: saving file ' + outFiles4Iso_isoQCDrej[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_isoQCDrej[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        #######################################################################################################################################
        # DM SORTING
        print('\n** INFO: starting DM sorting')
        for PUWP in ['99', '95', '90']:
            print('    - PUWP = {0}'.format(PUWP))
            for ISOWP in ['01', '05', '10']:
                print('        - ISOWP = {0}'.format(ISOWP))
                model_DM = load_obj(model_DMsort[args.FE].format(PUWP,ISOWP))
                dfs4Iso[k]['cl3d_predDM_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = model_DM.predict(dfs4Iso[k][features])
                probas_DM = model_DM.predict_proba(dfs4Iso[k][features])
                dfs4Iso[k]['cl3d_probDM0_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,0]
                dfs4Iso[k]['cl3d_probDM1_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,1]
                dfs4Iso[k]['cl3d_probDM2_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,2]
                del probas_DM, model_DM
        
        # save file
        print('\n** INFO: saving file ' + outFiles4Iso_DMsort[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_DMsort[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        print('\n** INFO: finished producing and saving '+outFiles4Iso_DMsort[k][args.FE]+' and all intermediate files')
        print('---------------------------------------------------------------------------------------')

    end = time.time()
    print '\nRunning time = %02dh %02dm %02ds'%((end-start)/3600, ((end-start)%3600)/60, (end-start)% 60)
