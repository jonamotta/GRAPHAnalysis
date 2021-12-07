import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import xgboost as xgb
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
from decimal import *
getcontext().prec = 30
from sklearn.preprocessing import MinMaxScaler


class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

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

def L1TowerEtIso ( dfL1Candidates, dfL1Towers, dRsgn, dRiso, dRisoEm, dRisoHad ):
    df_joined  = dfL1Candidates.join(dfL1Towers, on='event', how='inner', rsuffix='_tow', sort=False) # use 'inner' so that only the events present in the candidates dataframe are actually joined

    df_joined['deltar2tower'] = deltar2tower(df_joined)
    sel_sgn = (df_joined['deltar2tower'] <= dRsgn)
    sel_iso = (df_joined['deltar2tower'] <= dRiso) & (df_joined['deltar2tower'] > dRsgn)
    df_joined_sgn = df_joined[sel_sgn].copy(deep=True)
    df_joined_iso = df_joined[sel_iso].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt_c3'], inplace=True)

    dfL1Candidates['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt_c3'])['tower_pt'].sum()
    dfL1Candidates['tower_etEmIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoEm*10))] = df_joined_iso.groupby(['event', 'cl3d_pt_c3'])['tower_etEm'].sum()
    dfL1Candidates['tower_etHadIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRisoHad*10))] = df_joined_iso.groupby(['event', 'cl3d_pt_c3'])['tower_etHad'].sum() 

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

def L1TowerEtSgn ( dfL1Candidates, dfL1Towers, dRsgn ):
    df_joined  = dfL1Candidates.join(dfL1Towers, on='event', how='inner', rsuffix='_tow', sort=False) # use 'inner' so that only the events present in the candidates dataframe are actually joined

    df_joined['deltar2tower'] = deltar2tower(df_joined)
    sel_sgn = (df_joined['deltar2tower'] <= dRsgn)
    df_joined_sgn = df_joined[sel_sgn].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt_c3'], inplace=True)

    dfL1Candidates['tower_etSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined_sgn.groupby(['event', 'cl3d_pt_c3'])['tower_pt'].sum()

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

def applyDRcut ( df, dR ):
    df = df.join(df, on='event', how='left', rsuffix='_ass', sort=False)
    df['deltar2cluster'] = deltar2cluster(df)
    df.query('deltar2cluster>{0}'.format(dR), inplace=True)

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--ptcut', dest='ptcut', help='baseline 3D cluster pT cut to use', default='0')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_argument('--hardPUrej', dest='hardPUrej', help='apply hard PU rejection and do not consider PU categorized clusters for Iso variables? (99, 95, 90)', default='NO')
    parser.add_argument('--testRun', dest='testRun', help='do test run with reduced number of events?',  action='store_true', default=False)
    parser.add_argument('--doRead', dest='doRead', help='only read the root files and create base .hdf5 files', action='store_true', default=False)
    parser.add_argument('--doApply', dest='doApply', help='apply L1 algorithm', action='store_true', default=False)
    parser.add_argument('--doRate', dest='doRate', help='L1 algorithm already applied, jump to rate evluation directly', action='store_true', default=False)
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to only produce the plots?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    tag1 = "{0}hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""
    tag2 = "_{0}hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""
    tag3 = "hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    print('** INFO: using front-end option: '+args.FE)

    ##################### READ TTREES AND MATCH EVENTS ####################

    if args.doRead:
        print('---------------------------------------------------------------------------------------')
        print('** INFO: reading root file to create hdf5')

        import root_pandas

        # create needed folders
        indir   = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_2021_05_12'
        outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/rateEvluation'
        os.system('mkdir -p '+outdir)

        # DICTIONARIES FOR THE MATCHING
        inFileMinbias = {
            'threshold'    : indir+'/SKIM_Minbias_PU200/mergedOutput{0}.root',
            'mixed'        : indir+'/'
        }

        outFileMinbias = {
            'threshold'    : outdir+'/Minbias_PU200_th_matched_pt{0}.hdf5'.format(args.ptcut),
            'mixed'        : outdir+'/'
        }

        outFileMinbias_towers = {
            'threshold'    : outdir+'/Minbias_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

        # TTree to be read
        treename = 'HGCALskimmedTree'
        #TBranches to be stored containing the 3D clusters' info
        branches_event_cl3d = ['event','cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz', 'cl3d_srrtot','cl3d_srrmax','cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
        branches_cl3d       = ['cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz', 'cl3d_srrtot','cl3d_srrmax','cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
        #TBranches to be stored containing the towers' info
        branches_event_towers = ['event','tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']
        branches_towers       = ['tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']

        df4Rate = pd.DataFrame()
        df4Rate_towers = pd.DataFrame()

        if args.testRun:
            inFileNu = {
                'threshold'    : indir+'/SKIM_RelValNu_PU200/mergedOutput.root',
                'mixed'        : indir+'/'
            }
            # fill the dataframes with the needed info from the branches defined above
            print('\n** INFO: creating dataframes for ' + inFileNu[args.FE].format('')) 
            df_cl3d = root_pandas.read_root(inFileNu[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)#.query('event<=1000')
            towers = root_pandas.read_root(inFileNu[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)#.query('event<=1000')
            df4Rate = df_cl3d; df4Rate_towers = towers
        else:
            # fill the dataframes with the needed info from the branches defined above
            print('\n** INFO: creating dataframes for ' + inFileMinbias[args.FE].format(''))
            for i in ['', '_1', '_2']:     
                df_cl3d = root_pandas.read_root(inFileMinbias[args.FE].format(i), key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
                towers = root_pandas.read_root(inFileMinbias[args.FE].format(i), key=treename, columns=branches_event_towers, flatten=branches_towers)

                df4Rate = pd.concat([df4Rate,df_cl3d], sort=False)
                df4Rate_towers = pd.concat([df4Rate_towers,towers], sort=False)

        df4Rate.drop('__array_index', inplace=True, axis=1)
        df4Rate['geom_match'] = False
        df4Rate['cl3d_isbestmatch'] = False
        df4Rate['cl3d_abseta'] = np.abs(df4Rate['cl3d_eta'])
        df4Rate.query('cl3d_pt>{0} and cl3d_abseta>1.6 and cl3d_abseta<2.9'.format(args.ptcut), inplace=True)
        df4Rate.set_index('event', inplace=True)
        df4Rate.sort_values('event', inplace=True)

        df4Rate_towers.drop('__array_index', inplace=True, axis=1)
        df4Rate_towers.set_index('event', inplace=True)
        df4Rate_towers.sort_values('event', inplace=True)

        print('** INFO: saving file ' + outFileMinbias_towers[args.FE])
        store_towers = pd.HDFStore(outFileMinbias_towers[args.FE], mode='w')
        store_towers[args.FE] = df4Rate_towers
        store_towers.close()

        print('** INFO: saving file ' + outFileMinbias[args.FE])
        store_towers = pd.HDFStore(outFileMinbias[args.FE], mode='w')
        store_towers[args.FE] = df4Rate
        store_towers.close()

        del df_cl3d, towers, store_towers, treename, branches_event_cl3d, branches_cl3d, branches_event_towers, branches_towers # free memory
        
        print('\n** INFO: done creating hdf5')
        print('---------------------------------------------------------------------------------------')
        exit()
    else:
        print('---------------------------------------------------------------------------------------')
        print('** INFO: reading hdf5 file')

        # create needed folders
        indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/rateEvluation'
        outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/rateEvluation'
        os.system('mkdir -p '+outdir)

        inFileMinbias = {
            'threshold'    : indir+'/Minbias_PU200_th_matched_pt{0}.hdf5'.format(args.ptcut),
            'mixed'        : indir+'/'
        }

        outFileMinbias = {
            'threshold'    : outdir+'/Minbias_PU200_th_rateEvaluated_pt{0}{1}.hdf5'.format(args.ptcut, tag2),
            'mixed'        : outdir+'/'
        }

        store_tr = pd.HDFStore(inFileMinbias[args.FE], mode='r')
        df4Rate = store_tr[args.FE]
        store_tr.close()

        save_obj(np.unique(df4Rate.reset_index()['event']).shape[0], outdir+'/events_total.pkl')

    if args.doApply:

        # DICTIONARIES FOR THE CALIBRATION
        calib_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/calibration_C1skimC2C3'

        modelC1_calib = {
            'threshold'    : calib_model_indir+'/model_c1_th_PU200.pkl',
            'mixed'        : calib_model_indir+'/'
        }

        modelC2_calib = {
            'threshold'    : calib_model_indir+'/model_c2_th_PU200.pkl',
            'mixed'        : calib_model_indir+'/'
        }

        modelC3_calib = {
            'threshold'    : calib_model_indir+'/model_c3_th_PU200.pkl',
            'mixed'        : calib_model_indir+'/'
        }

        # DICTIONARIES FOR THE PILEUP REJECTION
        PUrej_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/PUrejection_skimPUnoPtRscld'

        model_PUrej = {
            'threshold'    : PUrej_model_indir+'/model_PUrejection_th_PU200.pkl',
            'mixed'        : PUrej_model_indir+'/'
        }

        WP90_PUrej = {
            'threshold'    : PUrej_model_indir+'/WP90_PUrejection_th_PU200.pkl',
            'mixed'        : PUrej_model_indir+'/'
        }

        WP95_PUrej = {
            'threshold'    : PUrej_model_indir+'/WP95_PUrejection_th_PU200.pkl',
            'mixed'        : PUrej_model_indir+'/'
        }

        WP99_PUrej = {
            'threshold'    : PUrej_model_indir+'/WP99_PUrejection_th_PU200.pkl',
            'mixed'        : PUrej_model_indir+'/'
        }

        # DICTIONARIES FOR THE ISO QCD REJECTION
        isoQCDrej_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/isolation_skimPUnoPtRscld_fullISORscld{0}{1}_againstPU'

        lowEta_model_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/model_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP10_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP10_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP15_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP15_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP20_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP20_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP90_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP90_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP95_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP95_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        lowEta_WP99_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP99_isolation_lowEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_model_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/model_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP10_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP10_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP15_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP15_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP20_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP20_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP90_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP90_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP95_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP95_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        midEta_WP99_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP99_isolation_midEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_model_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/model_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP10_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP10_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP15_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP15_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP20_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP20_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP90_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP90_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP95_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP95_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        highEta_WP99_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP99_isolation_highEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_model_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/model_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP10_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP10_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP15_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP15_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP20_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP20_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP90_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP90_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP95_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP95_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        vhighEta_WP99_isoQCDrej = {
            'threshold'    : isoQCDrej_model_indir+'/WP99_isolation_vhighEta_PUWP{0}_th_PU200.pkl',
            'mixed'        : isoQCDrej_model_indir+'/'
        }

        # DICTIONARIES FOR THE DM SORTING
        # DMsort_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/DMsorting_skimPUnoPtRscld_skimISORscld{0}'.format(tag1)

        # model_DMsort = {
        #     'threshold'    : DMsort_model_indir+'/model_DMsorting_th_PU200_PUWP{0}_ISOWP{1}.pkl',
        #     'mixed'        : DMsort_model_indir+'/'
        # }

        #######################################################################################################################################
        # CLUSTERS CALIBRATION
        features_calib = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
        modelC1 = load_obj(modelC1_calib[args.FE])
        modelC2 = load_obj(modelC2_calib[args.FE])
        modelC3 = load_obj(modelC3_calib[args.FE])

        # application of calibration 1
        print('\n** INFO: applying calibration C1') 
        df4Rate['cl3d_c1'] = modelC1.predict(df4Rate[['cl3d_abseta']])
        df4Rate['cl3d_pt_c1'] = df4Rate['cl3d_pt'] + df4Rate['cl3d_c1']
        # application of calibration 2
        print('** INFO: applying calibration C2')
        df4Rate['cl3d_c2'] = modelC2.predict(df4Rate[features_calib])
        df4Rate['cl3d_pt_c2'] = df4Rate['cl3d_pt_c1'] * df4Rate['cl3d_c2']
        # application of calibration 3
        print('** INFO: applying calibration C3')
        logpt1 = np.log(abs(df4Rate['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        df4Rate['cl3d_c3'] = modelC3.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        df4Rate['cl3d_pt_c3'] = df4Rate['cl3d_pt_c2'] / df4Rate['cl3d_c3']

        del features_calib, logpt1, logpt2, logpt3, logpt4, modelC1, modelC2, modelC3, modelC1_calib, modelC2_calib, modelC3_calib # free memory

        #######################################################################################################################################
        # PILE UP REJECTION
        features_PUrej = ['cl3d_c3', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

        if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            features2shift = ['cl3d_coreshowerlength']
            features2saturate = ['cl3d_c3', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
            saturation_dict = {'cl3d_c3': [0,30],
                               'cl3d_srrtot': [0, 0.02],
                               'cl3d_srrmean': [0, 0.01],
                               'cl3d_hoe': [0, 63],
                               'cl3d_meanz': [305, 535]
                              }

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            # shift features to be shifted
            for feat in features2shift:
                df4Rate[feat] = df4Rate[feat] - 25

            # saturate features
            for feat in features2saturate:
                df4Rate[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                
                # fill the bounds DF
                bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

            scale_range = [-32,32]
            MMS = MinMaxScaler(scale_range)

            for feat in features2saturate:
                MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
                df4Rate[feat] = MMS.transform( np.array(df4Rate[feat]).reshape(-1,1) )

            del features2shift, features2saturate, saturation_dict

        model_PU = load_obj(model_PUrej[args.FE])
        PUbdtWP99 = load_obj(WP99_PUrej[args.FE])
        PUbdtWP95 = load_obj(WP95_PUrej[args.FE])
        PUbdtWP90 = load_obj(WP90_PUrej[args.FE])

        print('\n** INFO: applying PU rejection BDT')
        full = xgb.DMatrix(data=df4Rate[features_PUrej], feature_names=features_PUrej)
        df4Rate['cl3d_pubdt_score'] = model_PU.predict(full)
        df4Rate['cl3d_pubdt_passWP99'] = df4Rate['cl3d_pubdt_score'] > PUbdtWP99
        df4Rate['cl3d_pubdt_passWP95'] = df4Rate['cl3d_pubdt_score'] > PUbdtWP95
        df4Rate['cl3d_pubdt_passWP90'] = df4Rate['cl3d_pubdt_score'] > PUbdtWP90

        if args.hardPUrej != 'NO':
            print("\n** INFO: applying hard PU rejection")
            df4Rate.query('cl3d_pubdt_passWP{0}==True'.format(args.hardPUrej), inplace=True)

        del features_PUrej, full, model_PU, PUbdtWP99, PUbdtWP95, PUbdtWP90, WP99_PUrej, WP95_PUrej, WP90_PUrej # free memory

        #######################################################################################################################################
        # CALCULATION OF ISOLATION FEATURES
        inFileMinbias_towers = {
            'threshold'    : outdir+'/Minbias_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }
       
        store_towers = pd.HDFStore(inFileMinbias_towers[args.FE], mode='r')
        df4Rate_towers = store_towers[args.FE]
        store_towers.close()

        df4Rate.reset_index(inplace=True)
        dfL1Candidates = df4Rate.query('cl3d_pubdt_passWP99==True').copy(deep=True) # selecting WP 99 we select also 95 and 90
        dfL1Candidates.sort_values('cl3d_pt_c3', inplace=True)
        dfL1Candidates = dfL1Candidates.groupby('event').tail(5).copy(deep=True) # keep only the 5 highest pt cluster
        #dfL1Candidates.drop_duplicates('event', keep='last', inplace=True) # keep only highest pt cluster
        sel = df4Rate['cl3d_pt_c3'].isin(dfL1Candidates['cl3d_pt_c3'])
        dfL1ass2cand = df4Rate.drop(df4Rate[sel].index)
        dfL1ass2cand = df4Rate
        dfL1Candidates.set_index('event', inplace=True)
        dfL1Candidates.sort_values('event', inplace=True)
        dfL1ass2cand.set_index('event', inplace=True)
        dfL1ass2cand.sort_values('event', inplace=True)
        del sel # free memory

        # split the two endcaps to make the loops over the rows faster
        dfL1Candidates_p = dfL1Candidates.query('cl3d_eta>=0').copy(deep=True)
        dfL1Candidates_m = dfL1Candidates.query('cl3d_eta<0').copy(deep=True)
        dfL1ass2cand_p = dfL1ass2cand.query('cl3d_eta>=0').copy(deep=True)
        dfL1ass2cand_m = dfL1ass2cand.query('cl3d_eta<0').copy(deep=True)
        dfL1Towers_p = df4Rate_towers.query('tower_eta>=0').copy(deep=True)
        dfL1Towers_m = df4Rate_towers.query('tower_eta<0').copy(deep=True)

        del dfL1Candidates, dfL1ass2cand, df4Rate_towers # free memory

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

        print('\n** INFO: calculating variables for ISO QCD rejection BDT')
        print('       positive z encap - down chunk')
        L1TowerEtSgn(dfL1Candidates_p_d, dfL1Towers_p_d, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_p_d, dfL1Towers_p_d, 0.2) # "
        L1TowerEtIso(dfL1Candidates_p_d, dfL1Towers_p_d, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_p_d, dfL1Towers_p_d, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_p_d, dfL1ass2cand_p_d, 0.4)
        print('       positive z encap - mid chunk')
        L1TowerEtSgn(dfL1Candidates_p_m, dfL1Towers_p_m, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_p_m, dfL1Towers_p_m, 0.2) # "
        L1TowerEtIso(dfL1Candidates_p_m, dfL1Towers_p_m, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_p_m, dfL1Towers_p_m, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_p_m, dfL1ass2cand_p_m, 0.4)
        print('       positive z encap - up chunk')
        L1TowerEtSgn(dfL1Candidates_p_u, dfL1Towers_p_u, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_p_u, dfL1Towers_p_u, 0.2) # "
        L1TowerEtIso(dfL1Candidates_p_u, dfL1Towers_p_u, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_p_u, dfL1Towers_p_u, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_p_u, dfL1ass2cand_p_u, 0.4)
        print('       negative z encap - down chunk')
        L1TowerEtSgn(dfL1Candidates_m_d, dfL1Towers_m_d, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_m_d, dfL1Towers_m_d, 0.2) # "
        L1TowerEtIso(dfL1Candidates_m_d, dfL1Towers_m_d, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_m_d, dfL1Towers_m_d, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_m_d, dfL1ass2cand_m_d, 0.4)
        print('       negative z encap - mid chunk')
        L1TowerEtSgn(dfL1Candidates_m_m, dfL1Towers_m_m, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_m_m, dfL1Towers_m_m, 0.2) # "
        L1TowerEtIso(dfL1Candidates_m_m, dfL1Towers_m_m, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_m_m, dfL1Towers_m_m, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_m_m, dfL1ass2cand_m_m, 0.4)
        print('       negative z encap - up chunk')
        L1TowerEtSgn(dfL1Candidates_m_u, dfL1Towers_m_u, 0.1) # dRsgn
        L1TowerEtSgn(dfL1Candidates_m_u, dfL1Towers_m_u, 0.2) # "
        L1TowerEtIso(dfL1Candidates_m_u, dfL1Towers_m_u, 0.1, 0.3, 0.3, 0.7) # dRsgn, dRiso, dRisoEm, dRisoHad
        L1TowerEtIso(dfL1Candidates_m_u, dfL1Towers_m_u, 0.2, 0.4, 0.4, 0.7) # "
        L1Cl3dEtIso(dfL1Candidates_m_u, dfL1ass2cand_m_u, 0.4)

        df4Rate = pd.concat([dfL1Candidates_p_d, dfL1Candidates_p_m, dfL1Candidates_p_u, dfL1Candidates_m_d, dfL1Candidates_m_m, dfL1Candidates_m_u], sort=False)

        del dfL1Candidates_p_d, dfL1Candidates_p_m, dfL1Candidates_p_u, dfL1Candidates_m_d, dfL1Candidates_m_m, dfL1Candidates_m_u # free memory
        del dfL1Towers_p_d, dfL1Towers_p_m, dfL1Towers_p_u, dfL1Towers_m_d, dfL1Towers_m_m, dfL1Towers_m_u # free memory
        del dfL1ass2cand_p_d, dfL1ass2cand_p_m, dfL1ass2cand_p_u, dfL1ass2cand_m_d, dfL1ass2cand_m_m, dfL1ass2cand_m_u # free memory

        #######################################################################################################################################
        # ISO QCD REJECTION
        print('\n** INFO: applying ISO QCD rejection BDT')
        
        df4Rate['cl3d_pt_tr'] = df4Rate['cl3d_pt_c3'].copy(deep=True)

        df4Rate_lowEta = df4Rate.query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)
        df4Rate_midEta = df4Rate.query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)
        df4Rate_highEta = df4Rate.query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)
        df4Rate_vhighEta = df4Rate.query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)
        del df4Rate

        if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            features2shift = ['cl3d_NclIso_dR4', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',]
            features2saturate = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
            saturation_dict = {'cl3d_pt_tr': [0, 200],
                               'cl3d_abseta': [1.45, 3.2],
                               'cl3d_seetot': [0, 0.17],
                               'cl3d_seemax': [0, 0.6],
                               'cl3d_spptot': [0, 0.17],
                               'cl3d_sppmax': [0, 0.53],
                               'cl3d_szz': [0, 141.09],
                               'cl3d_srrtot': [0, 0.02],
                               'cl3d_srrmax': [0, 0.02],
                               'cl3d_srrmean': [0, 0.01],
                               'cl3d_hoe': [0, 63],
                               'cl3d_meanz': [305, 535],
                               'cl3d_etIso_dR4': [0, 58],
                               'tower_etSgn_dRsgn1': [0, 194],
                               'tower_etSgn_dRsgn2': [0, 228],
                               'tower_etIso_dRsgn1_dRiso3': [0, 105],
                               'tower_etEmIso_dRsgn1_dRiso3': [0, 72],
                               'tower_etHadIso_dRsgn1_dRiso7': [0, 43],
                               'tower_etIso_dRsgn2_dRiso4': [0, 129],
                               'tower_etEmIso_dRsgn2_dRiso4': [0, 95],
                               'tower_etHadIso_dRsgn2_dRiso7': [0, 42]
                            }

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            dfs = [df4Rate_lowEta, df4Rate_midEta, df4Rate_highEta, df4Rate_vhighEta]
            for df in dfs:
                # shift features to be shifted
                for feat in features2shift:
                    df[feat] = df[feat] - 32

                # saturate features
                for feat in features2saturate:
                    df[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                    
                    # fill the bounds DF
                    bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

                scale_range = [-32,32]
                MMS = MinMaxScaler(scale_range)

                for feat in features2saturate:
                    MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
                    df[feat] = MMS.transform( np.array(df[feat]).reshape(-1,1) )

            del features2shift, features2saturate, saturation_dict


        for PUWP in ['99', '95', '90']:
            features_isoQCDrej = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']

            print('    - PUWP = {0}'.format(PUWP))
            lowEta_model_iso = load_obj(lowEta_model_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP10 = load_obj(lowEta_WP10_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP15 = load_obj(lowEta_WP15_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP20 = load_obj(lowEta_WP20_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP90 = load_obj(lowEta_WP90_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP95 = load_obj(lowEta_WP95_isoQCDrej[args.FE].format(PUWP,tag3))
            lowEta_ISObdtWP99 = load_obj(lowEta_WP99_isoQCDrej[args.FE].format(PUWP,tag3))
            full = xgb.DMatrix(data=df4Rate_lowEta[features_isoQCDrej], feature_names=features_isoQCDrej)
            df4Rate_lowEta['cl3d_isobdt_score'] = lowEta_model_iso.predict(full)
            df4Rate_lowEta['cl3d_isobdt_passWP10_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP10
            df4Rate_lowEta['cl3d_isobdt_passWP15_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP15
            df4Rate_lowEta['cl3d_isobdt_passWP20_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP20
            df4Rate_lowEta['cl3d_isobdt_passWP90_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP90
            df4Rate_lowEta['cl3d_isobdt_passWP95_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP95
            df4Rate_lowEta['cl3d_isobdt_passWP99_PUWP{0}'.format(PUWP)] = df4Rate_lowEta['cl3d_isobdt_score'] > lowEta_ISObdtWP99
            del full, lowEta_model_iso, lowEta_ISObdtWP10, lowEta_ISObdtWP15, lowEta_ISObdtWP20, lowEta_ISObdtWP90, lowEta_ISObdtWP95, lowEta_ISObdtWP99

            midEta_model_iso = load_obj(midEta_model_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP10 = load_obj(midEta_WP10_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP15 = load_obj(midEta_WP15_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP20 = load_obj(midEta_WP20_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP90 = load_obj(midEta_WP90_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP95 = load_obj(midEta_WP95_isoQCDrej[args.FE].format(PUWP,tag3))
            midEta_ISObdtWP99 = load_obj(midEta_WP99_isoQCDrej[args.FE].format(PUWP,tag3))
            full = xgb.DMatrix(data=df4Rate_midEta[features_isoQCDrej], feature_names=features_isoQCDrej)
            df4Rate_midEta['cl3d_isobdt_score'] = midEta_model_iso.predict(full)
            df4Rate_midEta['cl3d_isobdt_passWP10_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP10
            df4Rate_midEta['cl3d_isobdt_passWP15_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP15
            df4Rate_midEta['cl3d_isobdt_passWP20_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP20
            df4Rate_midEta['cl3d_isobdt_passWP90_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP90
            df4Rate_midEta['cl3d_isobdt_passWP95_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP95
            df4Rate_midEta['cl3d_isobdt_passWP99_PUWP{0}'.format(PUWP)] = df4Rate_midEta['cl3d_isobdt_score'] > midEta_ISObdtWP99
            del full, midEta_model_iso, midEta_ISObdtWP10, midEta_ISObdtWP15, midEta_ISObdtWP20, midEta_ISObdtWP90, midEta_ISObdtWP95, midEta_ISObdtWP99

            highEta_model_iso = load_obj(highEta_model_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP10 = load_obj(highEta_WP10_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP15 = load_obj(highEta_WP15_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP20 = load_obj(highEta_WP20_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP90 = load_obj(highEta_WP90_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP95 = load_obj(highEta_WP95_isoQCDrej[args.FE].format(PUWP,tag3))
            highEta_ISObdtWP99 = load_obj(highEta_WP99_isoQCDrej[args.FE].format(PUWP,tag3))
            full = xgb.DMatrix(data=df4Rate_highEta[features_isoQCDrej], feature_names=features_isoQCDrej)
            df4Rate_highEta['cl3d_isobdt_score'] = highEta_model_iso.predict(full)
            df4Rate_highEta['cl3d_isobdt_passWP10_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP10
            df4Rate_highEta['cl3d_isobdt_passWP15_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP15
            df4Rate_highEta['cl3d_isobdt_passWP20_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP20
            df4Rate_highEta['cl3d_isobdt_passWP90_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP90
            df4Rate_highEta['cl3d_isobdt_passWP95_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP95
            df4Rate_highEta['cl3d_isobdt_passWP99_PUWP{0}'.format(PUWP)] = df4Rate_highEta['cl3d_isobdt_score'] > highEta_ISObdtWP99
            del full, highEta_model_iso, highEta_ISObdtWP10, highEta_ISObdtWP15, highEta_ISObdtWP20, highEta_ISObdtWP90, highEta_ISObdtWP95, highEta_ISObdtWP99

            vhighEta_model_iso = load_obj(vhighEta_model_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP10 = load_obj(vhighEta_WP10_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP15 = load_obj(vhighEta_WP15_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP20 = load_obj(vhighEta_WP20_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP90 = load_obj(vhighEta_WP90_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP95 = load_obj(vhighEta_WP95_isoQCDrej[args.FE].format(PUWP,tag3))
            vhighEta_ISObdtWP99 = load_obj(vhighEta_WP99_isoQCDrej[args.FE].format(PUWP,tag3))
            full = xgb.DMatrix(data=df4Rate_vhighEta[features_isoQCDrej], feature_names=features_isoQCDrej)
            df4Rate_vhighEta['cl3d_isobdt_score'] = vhighEta_model_iso.predict(full)
            df4Rate_vhighEta['cl3d_isobdt_passWP10_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP10
            df4Rate_vhighEta['cl3d_isobdt_passWP15_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP15
            df4Rate_vhighEta['cl3d_isobdt_passWP20_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP20
            df4Rate_vhighEta['cl3d_isobdt_passWP90_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP90
            df4Rate_vhighEta['cl3d_isobdt_passWP95_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP95
            df4Rate_vhighEta['cl3d_isobdt_passWP99_PUWP{0}'.format(PUWP)] = df4Rate_vhighEta['cl3d_isobdt_score'] > vhighEta_ISObdtWP99
            del full, vhighEta_model_iso, vhighEta_ISObdtWP10, vhighEta_ISObdtWP15, vhighEta_ISObdtWP20, vhighEta_ISObdtWP90, vhighEta_ISObdtWP95, vhighEta_ISObdtWP99

        del features_isoQCDrej, lowEta_model_isoQCDrej, lowEta_WP10_isoQCDrej, lowEta_WP15_isoQCDrej, lowEta_WP20_isoQCDrej, lowEta_WP90_isoQCDrej, lowEta_WP95_isoQCDrej, lowEta_WP99_isoQCDrej # free memory
        del midEta_model_isoQCDrej, midEta_WP10_isoQCDrej, midEta_WP15_isoQCDrej, midEta_WP20_isoQCDrej, midEta_WP90_isoQCDrej, midEta_WP95_isoQCDrej, midEta_WP99_isoQCDrej # free memory
        del highEta_model_isoQCDrej, highEta_WP10_isoQCDrej, highEta_WP15_isoQCDrej, highEta_WP20_isoQCDrej, highEta_WP90_isoQCDrej, highEta_WP95_isoQCDrej, highEta_WP99_isoQCDrej # free memory
        del vhighEta_model_isoQCDrej, vhighEta_WP10_isoQCDrej, vhighEta_WP15_isoQCDrej, vhighEta_WP20_isoQCDrej, vhighEta_WP90_isoQCDrej, vhighEta_WP95_isoQCDrej, vhighEta_WP99_isoQCDrej # free memory

        df4Rate = pd.concat([df4Rate_lowEta, df4Rate_midEta, df4Rate_highEta, df4Rate_vhighEta], sort=False).copy(deep=True)

        #######################################################################################################################################
        # # DM SORTING
        # features_DMsort = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

        # if args.doRescale:
        #     print('\n** INFO: rescaling features to bound their values')

        #     features2shift = ['cl3d_showerlength', 'cl3d_firstlayer',]
        #     features2saturate = [ 'cl3d_seetot', 'cl3d_seemax', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrmax']
        #     saturation_dict = {'cl3d_seetot': [0, 0.17],
        #                        'cl3d_seemax': [0, 0.6],
        #                        'cl3d_sppmax': [0, 0.53],
        #                        'cl3d_szz': [0, 141.09],
        #                        'cl3d_srrmax': [0, 0.02]
        #                     }

        #     #define a DF with the bound values of the features to use for the MinMaxScaler fit
        #     bounds4features = pd.DataFrame(columns=features2saturate)

        #     # shift features to be shifted
        #     for feat in features2shift:
        #         df4Rate[feat] = df4Rate[feat] - 25

        #     # saturate features
        #     for feat in features2saturate:
        #         df4Rate[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                
        #         # fill the bounds DF
        #         bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

        #     scale_range = [-32,32]
        #     MMS = MinMaxScaler(scale_range)

        #     for feat in features2saturate:
        #         MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
        #         df4Rate[feat] = MMS.transform( np.array(df4Rate[feat]).reshape(-1,1) )

        #     del features2shift, features2saturate, saturation_dict


        # print('\n** INFO: starting DM sorting')
        # for PUWP in ['99', '95', '90']:
        #     print('    - PUWP = {0}'.format(PUWP))
        #     for ISOWP in ['10', '15', '20', '90', '95', '99']:
        #         print('        - ISOWP = {0}'.format(ISOWP))
        #         model_DM = load_obj(model_DMsort[args.FE].format(PUWP,ISOWP))
        #         df4Rate['cl3d_predDM_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = model_DM.predict(df4Rate[features_DMsort])
        #         probas_DM = model_DM.predict_proba(df4Rate[features_DMsort])
        #         df4Rate['cl3d_probDM0_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,0]
        #         df4Rate['cl3d_probDM1_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,1]
        #         df4Rate['cl3d_probDM2_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = probas_DM[:,2]
        #         del probas_DM, model_DM

        # del features_DMsort, model_DMsort

        #######################################################################################################################################
        # SAVE MINBIAS TO FILE
        print('\n** INFO: saving file ' + outFileMinbias[args.FE])
        store = pd.HDFStore(outFileMinbias[args.FE], mode='w')
        store[args.FE] = df4Rate
        store.close()
        del store

    #######################################################################################################################################
    # RATE EVALUATION
    
    if args.doRate:
        indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/rateEvluation'
        inFileMinbias = {
            'threshold'    : indir+'/Minbias_PU200_th_rateEvaluated_pt{0}{1}.hdf5'.format(args.ptcut, tag2),
            'mixed'        : indir+'/'
        }

        store_tr = pd.HDFStore(inFileMinbias[args.FE], mode='r')
        df4Rate = store_tr[args.FE]
        store_tr.close()

        # DICTIONARIES FOR RATE EVALUATION
        mapping_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/mapping_skimPUnoPtRscld_fullISORscld{0}'.format(tag1)
        ratedir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/rateEvaluation_C1skimC2C3_skimPUnoPtRscld_fullISORscld{0}'.format(tag1)
        os.system('mkdir -p '+ratedir)

        mappings_dict = {
            'threshold' : {
                'PUWP99' : {
                    'ISOWP10' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP10_mapping.pkl',
                    'ISOWP15' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP15_mapping.pkl',
                    'ISOWP20' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP20_mapping.pkl',
                    'ISOWP90' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP90_mapping.pkl',
                    'ISOWP95' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP95_mapping.pkl',
                    'ISOWP99' : mapping_indir+'/Tau_PU200_th_PUWP99_ISOWP99_mapping.pkl'
                },

                'PUWP95' : {
                    'ISOWP10' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP10_mapping.pkl',
                    'ISOWP15' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP15_mapping.pkl',
                    'ISOWP20' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP20_mapping.pkl',
                    'ISOWP90' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP90_mapping.pkl',
                    'ISOWP95' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP95_mapping.pkl',
                    'ISOWP99' : mapping_indir+'/Tau_PU200_th_PUWP95_ISOWP99_mapping.pkl'
                },

                'PUWP90' : {
                    'ISOWP10' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP10_mapping.pkl',
                    'ISOWP15' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP15_mapping.pkl',
                    'ISOWP20' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP20_mapping.pkl',
                    'ISOWP90' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP90_mapping.pkl',
                    'ISOWP95' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP95_mapping.pkl',
                    'ISOWP99' : mapping_indir+'/Tau_PU200_th_PUWP90_ISOWP99_mapping.pkl'
                },
            },

            'mixed' : {
                'PUWP99' : {
                    'ISOWP10' : mapping_indir+'',
                    'ISOWP15' : mapping_indir+'',
                    'ISOWP20' : mapping_indir+'',
                    'ISOWP90' : mapping_indir+'',
                    'ISOWP95' : mapping_indir+'',
                    'ISOWP99' : mapping_indir+''
                },

                'PUWP95' : {
                    'ISOWP10' : mapping_indir+'',
                    'ISOWP15' : mapping_indir+'',
                    'ISOWP20' : mapping_indir+'',
                    'ISOWP90' : mapping_indir+'',
                    'ISOWP95' : mapping_indir+'',
                    'ISOWP99' : mapping_indir+''
                },

                'PUWP90' : {
                    'ISOWP10' : mapping_indir+'',
                    'ISOWP15' : mapping_indir+'',
                    'ISOWP20' : mapping_indir+'',
                    'ISOWP90' : mapping_indir+'',
                    'ISOWP95' : mapping_indir+'',
                    'ISOWP99' : mapping_indir+''
                },
            }

        }

        events_frequency=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf

        df4Rate.reset_index(inplace=True)
        tokeep = ['event', 'cl3d_pt_c3', 'cl3d_eta', 'cl3d_phi', 'cl3d_pubdt_passWP90', 'cl3d_pubdt_passWP95', 'cl3d_pubdt_passWP99', 'cl3d_isobdt_passWP10_PUWP90', 'cl3d_isobdt_passWP15_PUWP90', 'cl3d_isobdt_passWP20_PUWP90', 'cl3d_isobdt_passWP90_PUWP90', 'cl3d_isobdt_passWP95_PUWP90', 'cl3d_isobdt_passWP99_PUWP90', 'cl3d_isobdt_passWP10_PUWP95', 'cl3d_isobdt_passWP15_PUWP95', 'cl3d_isobdt_passWP20_PUWP95', 'cl3d_isobdt_passWP90_PUWP95', 'cl3d_isobdt_passWP95_PUWP95', 'cl3d_isobdt_passWP99_PUWP95', 'cl3d_isobdt_passWP10_PUWP99', 'cl3d_isobdt_passWP15_PUWP99', 'cl3d_isobdt_passWP20_PUWP99', 'cl3d_isobdt_passWP90_PUWP99', 'cl3d_isobdt_passWP95_PUWP99', 'cl3d_isobdt_passWP99_PUWP99']
        df4Rate = df4Rate[tokeep]

        events_total = load_obj(indir+'/events_total.pkl')
        rates_online = {'singleTau' : {}, 'doubleTau' : {}}
        rates_offline = {'singleTau' : {}, 'doubleTau' : {}}

        print('\n** INFO: starting rate evaluation')
        for PUWP in ['99', '95', '90']:
            if PUWP != args.hardPUrej: continue
            print('    - PUWP = {0}'.format(PUWP))
            for ISOWP in ['10', '15', '20', '90', '95', '99']:
                print('        - ISOWP = {0}'.format(ISOWP))

                tmp = df4Rate.query('cl3d_pt_c3>10 and cl3d_pubdt_passWP{0}==True and cl3d_isobdt_passWP{1}_PUWP{0}==True'.format(PUWP,ISOWP)).copy(deep=True)

                ptXXs = load_obj(mappings_dict[args.FE]['PUWP'+PUWP]['ISOWP'+ISOWP])

                # find the gentau_vis_pt with XX% efficiency of selection if we were to apply a threshold on cl3d_pt equal to the specific cl3d_pt we are considering
                # this  essentially returns the offline threshold that when applied corresponds to applying an online threshold equal to cl3d_pt_c3
                tmp['cl3d_pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = tmp['cl3d_pt_c3'].apply(lambda x : np.interp(x, ptXXs.threshold, ptXXs.pt95))
                tmp['cl3d_pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = tmp['cl3d_pt_c3'].apply(lambda x : np.interp(x, ptXXs.threshold, ptXXs.pt90))
                tmp['cl3d_pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = tmp['cl3d_pt_c3'].apply(lambda x : np.interp(x, ptXXs.threshold, ptXXs.pt50))

                #********** single tau rate **********#
                sel_events = np.unique(tmp['event']).shape[0]
                print('             singleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

                tmp.sort_values('cl3d_pt_c3', inplace=True)
                tmp['COTcount_singleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

                rates_online['singleTau']['PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt_c3']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
                rates_offline['singleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
                rates_offline['singleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
                rates_offline['singleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )

                #********** double tau rate **********#
                tmp['doubleTau'] = tmp.duplicated('event', keep=False)
                tmp.query('doubleTau==True', inplace=True)
                applyDRcut(tmp, 0.5)
                sel_events = np.unique(tmp['event']).shape[0]
                print('             doubleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

                tmp.sort_values('cl3d_pt_c3', inplace=True)
                tmp['COTcount_doubleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

                rates_online['doubleTau']['PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt_c3']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
                rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
                rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
                rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)] = ( np.sort(tmp['cl3d_pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)]), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )

                del tmp
        
        save_obj(rates_online, ratedir+'/onlineRate_pt{0}.pkl'.format(args.ptcut))
        save_obj(rates_offline, ratedir+'/offlineRate_pt{0}.pkl'.format(args.ptcut))


    if args.doPlots:
        plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/plots/rateEvaluation_C1skimC2C3_skimPUnoPtRscld_fullISORscld{0}_pt{1}'.format(tag1,args.ptcut)
        ratedir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/rateEvaluation_C1skimC2C3_skimPUnoPtRscld_fullISORscld{0}'.format(tag1)
        os.system('mkdir -p '+plotdir)
        rates_offline = load_obj(ratedir+'/offlineRate_pt{0}.pkl'.format(args.ptcut))

        # set output to go both to terminal and to file
        sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/rateEvaluation_C1skimC2C3_skimPUnoPtRscld_fullISORscld{0}/offlineThresholds_pt{1}.txt".format(tag1, args.ptcut))

        for PUWP in ['99', '95', '90']:
            if PUWP != args.hardPUrej: continue
            print('    - PUWP = {0}'.format(PUWP))
            for ISOWP in ['10', '15', '20', '90', '95', '99']:
                print('        - ISOWP = {0}'.format(ISOWP))

                plt.figure(figsize=(8,8))
                plt.plot(rates_offline['singleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['singleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='blue', label='Offline threshold @ 95%')
                plt.plot(rates_offline['singleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['singleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='red', label='Offline threshold @ 90%')
                plt.plot(rates_offline['singleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['singleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='green', label='Offline threshold @ 50%')
                plt.yscale("log")
                plt.ylim(bottom=1)
                legend = plt.legend(loc = 'upper right')
                plt.grid(linestyle=':')
                plt.xlabel('Offline threshold [GeV]')
                plt.ylabel('Rate [kHz]')
                plt.title('Single Tau Rate - PUWP={0} ISOWP={1}'.format(PUWP,ISOWP), fontsize=15)
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/rate_singleTau_offline_PUWP{0}_ISOWP{1}.pdf'.format(PUWP,ISOWP))
                plt.close()

                print('            - single tau offline threshold for 25kHz rate @ 95% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['singleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))
                print('            - single tau offline threshold for 25kHz rate @ 90% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['singleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))
                print('            - single tau offline threshold for 25kHz rate @ 50% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['singleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))

                plt.figure(figsize=(8,8))
                plt.plot(rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='blue', label='Offline threshold @ 95%')
                plt.plot(rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='red', label='Offline threshold @ 90%')
                plt.plot(rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], linewidth=2, color='green', label='Offline threshold @ 50%')
                plt.yscale("log")
                plt.ylim(bottom=1)
                legend = plt.legend(loc = 'upper right')
                plt.grid(linestyle=':')
                plt.xlabel('Offline threshold [GeV]')
                plt.ylabel('Rate [kHz]')
                plt.title('Double Tau Rate - PUWP={0} ISOWP={1}'.format(PUWP,ISOWP), fontsize=15)
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/rate_doubleTau_offline_PUWP{0}_ISOWP{1}.pdf'.format(PUWP,ISOWP))
                plt.close()

                print('            - double tau offline threshold for 25kHz rate @ 95% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))
                print('            - double tau offline threshold for 25kHz rate @ 90% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))
                print('            - double tau offline threshold for 25kHz rate @ 50% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1]), np.flip(rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0]), left=0.0, right=0.0),0) ))

                if PUWP=='99' and ISOWP=='99':
                    plt.figure(figsize=(8,8))
                    plt.plot(rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], rates_offline['doubleTau']['pt95_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], linewidth=2, color='blue', label='Offline threshold @ 95%')
                    plt.plot(rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], rates_offline['doubleTau']['pt90_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], linewidth=2, color='red', label='Offline threshold @ 90%')
                    plt.plot(rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][1], rates_offline['doubleTau']['pt50_PUWP{0}_ISOWP{1}'.format(PUWP,ISOWP)][0], linewidth=2, color='green', label='Offline threshold @ 50%')
                    plt.yscale("log")
                    plt.ylim(bottom=1)
                    legend = plt.legend(loc = 'upper right')
                    plt.grid(linestyle=':')
                    plt.ylabel('Offline threshold [GeV]')
                    plt.xlabel('Rate [kHz]')
                    plt.title('Double Tau Rate - PUWP={0} ISOWP={1}'.format(PUWP,ISOWP), fontsize=15)
                    plt.subplots_adjust(bottom=0.12)
                    plt.savefig(plotdir+'/test.pdf'.format(PUWP,ISOWP))
                    plt.close()

        # restore normal output
        sys.stdout = sys.__stdout__
