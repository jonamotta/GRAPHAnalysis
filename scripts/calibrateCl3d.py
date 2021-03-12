import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import argparse


# binned responses and resolutions
def effrms(df, c=0.68):
    """ Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out_dict = {}
    for col in df:
        x = df[col]
        x = np.sort(x, kind="mergesort")
        m = int(c * len(x)) + 1
        if len(x) > 3: out_dict[col] = [np.min(x[m:] - x[:-m]) / 2.0] 
        elif len(x) > 0: out_dict[col] = [np.min(x[len(x)-1] - x[0]) / 2.0]
        else: out_dict[col] = [0]
    return pd.DataFrame(out_dict).iloc[0]

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'r') as f:
        pickle.load(f)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--noTraining', dest='doTraining', help='skip training and do only calibration?',  action='store_false', default=True)
    parser.add_argument('--calibrateHH', dest='calibrateHH', help='match the HH samples?',  action='store_true', default=False)
    parser.add_argument('--calibrateNu', dest='calibrateNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--calibrateQCD', dest='calibrateQCD', help='match the QCD samples?',  action='store_true', default=False)
    parser.add_argument('--doPlots', dest='doPlots', help='do plots?',  action='store_true', default=False)
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    # store parsed options
    args = parser.parse_args()

    if not args.doTraining and not args.calibrateQCD and not args.calibrateNu and not args.calibrateHH:
        print('** WARNING: neither training nor calibration work specified. What do you want to do (calibrateHH, calibrateNu, calibrateQCD)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    #################### INITIALIZATION OF ALL USEFUL VARIABLES AND DICTIONARIES ####################

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/calibration'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTau_dict = {
        'threshold'    : indir+'/RelValTenTau_PU200_th_matched.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_dict = {
        'threshold'    : outdir+'/RelValTenTau_PU200_th_calibrated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    inFileHH_dict = {
        'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }


    outFileHH_dict = {
        'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_calibrated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_modelC1_dict = {
        'threshold'    : model_outdir+'/model_c1_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC2_dict = {
        'threshold'    : model_outdir+'/model_c2_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC3_dict = {
        'threshold'    : model_outdir+'/model_c3_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    if args.calibrateQCD:
        inFileQCD_dict = {
            'threshold'    : indir+'/QCD_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }


        outFileQCD_dict = {
            'threshold'    : outdir+'/QCD_PU200_th_calibrated.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        dfQCD_dict = {} # dictionary of the skim level QCD dataframes 

    if args.calibrateNu:
        inFileNu_dict = {
            'threshold'    : indir+'/RelValNu_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }


        outFileNu_dict = {
            'threshold'    : outdir+'/RelValNu_PU200_th_calibrated.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        dfNu_dict = {} # dictionary of the skim level Nu dataframes 

    dfTau_dict = {}         # dictionary of the skim level tau dataframes
    dfHH_dict = {}          # dictionary of the skim level HH dataframes 
    dfTraining_dict = {}    # dictionary of the training tau dataframes
    C1model_dict = {}       # dictionary of models from C1 calibration step
    C2model_dict = {}       # dictionary of models from C2 calibration step
    C3model_dict = {}       # dictionary of models from C3 calibration step    
    meansTrainPt_dict = {}  # dictionary of
    rmssTrainPt_dict = {}   # dictionary of
    etameans_dict = {}      # dictionary of eta means - used for plotting
    etarmss_dict = {}       # dictionary of eta rms(std) - used for plotting
    etaeffrmss_dict = {}    # dictionary of
    ptmeans_dict = {}       # dictionary of pt means - used for plotting
    ptrmss_dict = {}        # dictionary of pt rms(std) - used for plotting
    pteffrmss_dict = {}     # dictionary of

    # minimum pt and eta requirements 
    ptmin = 20
    etamin = 1.6

    # features used for the C2 calibration step
    features = ['n_matched_cl3d', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean','cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']

    # features used for the C3 calibration step
    vars = ['gentau_vis_pt', 'gentau_vis_bin_pt', 'cl3d_pt_c2', 'cl3d_response_c2']

    # variables to plot
    plot_var = ['gentau_vis_pt', 'gentau_vis_abseta', 'gentau_vis_bin_eta', 'gentau_vis_bin_pt', 'cl3d_pt', 'cl3d_response', 'cl3d_abseta', 'cl3d_pt_c1', 'cl3d_response_c1', 'cl3d_pt_c2', 'cl3d_response_c2', 'cl3d_pt_c3', 'cl3d_response_c3']

    # colors to use
    colors_dict = {
        'threshold'    : 'blue',
        'supertrigger' : 'red',
        'bestchoice'   : 'olive',
        'bestcoarse'   : 'orange',
        'mixed'        : 'fuchsia'
    }

    # legend to use
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'supertrigger' : 'STC4+16',
        'bestchoice'   : 'BC Decentral',
        'bestcoarse'   : 'BC Coarse 2x2 TC',
        'mixed'        : 'Mixed BC + STC'
    }

    matplotlib.rcParams.update({'font.size': 16})


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end option that we do not want to do
        
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting energy calibration for the front-end option '+feNames_dict[name])

        if args.doTraining:
            # fill tau dataframes and dictionaries
            store_tau = pd.HDFStore(inFileTau_dict[name], mode='r')
            dfTau_dict[name] = store_tau[name] 
            store_tau.close()
            dfTau_dict[name]['cl3d_abseta'] = np.abs(dfTau_dict[name]['cl3d_eta'])
            # fill hh dataframes and dictionaries
            store_hh = pd.HDFStore(inFileHH_dict[name], mode='r')
            dfHH_dict[name] = store_hh[name]
            store_hh.close()
            dfHH_dict[name]['cl3d_abseta'] = np.abs(dfHH_dict[name]['cl3d_eta'])

        if args.calibrateQCD:
            # fill background dataframes and dictionaries
            store_qcd = pd.HDFStore(inFileQCD_dict[name], mode='r')
            dfQCD_dict[name] = store_qcd[name]
            store_qcd.close()
            dfQCD_dict[name]['cl3d_abseta'] = np.abs(dfQCD_dict[name]['cl3d_eta'])

        if args.calibrateNu:
            # fill background dataframes and dictionaries
            store_nu = pd.HDFStore(inFileNu_dict[name], mode='r')
            dfNu_dict[name] = store_nu[name]
            store_nu.close()
            dfNu_dict[name]['cl3d_abseta'] = np.abs(dfNu_dict[name]['cl3d_eta'])
            dfNu_dict[name]['n_matched_cl3d'] = 1 # we have to set this because the matching is not done as no taus are present

    
        ######################### SELECT EVENTS FOR TRAINING #########################
        
        dfTau_dict[name]['cl3d_response'] = dfTau_dict[name]['cl3d_pt']/dfTau_dict[name]['gentau_vis_pt']
        dfHH_dict[name]['cl3d_response'] = dfHH_dict[name]['cl3d_pt']/dfHH_dict[name]['gentau_vis_pt']

        # initialize the dictioary entry to the correct dataframe 
        dfTraining_dict[name] = pd.concat([dfHH_dict[name],dfTau_dict[name]],sort=False)

        # define selections for the training dataset
        genPt_sel  = dfTraining_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfTraining_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfTraining_dict[name]['gentau_vis_eta']) < 2.9
        cl3dPt_sel = dfTraining_dict[name]['cl3d_pt'] > 4
        cl3dBest_sel = dfTraining_dict[name]['cl3d_isbestmatch'] == True
        # apply slections for the training dataset
        dfTraining_dict[name] = dfTraining_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dPt_sel & cl3dBest_sel]
      

        ######################### C1 CALIBRATION TRAINING (PU eta dependent calibration) #########################

        print('\n** INFO: training calibration C1')

        input_c1 = dfTraining_dict[name][['cl3d_abseta']]
        target_c1 = dfTraining_dict[name].gentau_vis_pt - dfTraining_dict[name].cl3d_pt
        C1model_dict[name] = LinearRegression().fit(input_c1, target_c1)

        save_obj(C1model_dict[name], outFile_modelC1_dict[name])

        dfTraining_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfTraining_dict[name][['cl3d_abseta']])
        dfTraining_dict[name]['cl3d_pt_c1'] = dfTraining_dict[name].cl3d_c1 + dfTraining_dict[name].cl3d_pt
        dfTraining_dict[name]['cl3d_response_c1'] = dfTraining_dict[name].cl3d_pt_c1 / dfTraining_dict[name].gentau_vis_pt


        ######################### C2 CALIBRATION TRAINING (DM dependent calibration) #########################

        print('\n** INFO: training calibration C2')

        input_c2 = dfTraining_dict[name][features]
        target_c2 = dfTraining_dict[name].gentau_vis_pt / dfTraining_dict[name].cl3d_pt_c1
        C2model_dict[name] = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=0, loss='huber').fit(input_c2, target_c2)

        save_obj(C2model_dict[name], outFile_modelC2_dict[name])

        dfTraining_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfTraining_dict[name][features])
        dfTraining_dict[name]['cl3d_pt_c2'] = dfTraining_dict[name].cl3d_c2 * dfTraining_dict[name].cl3d_pt_c1
        dfTraining_dict[name]['cl3d_response_c2'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].gentau_vis_pt

        # print importance of the features used for training
        feature_importance = C2model_dict[name].feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(8, 6))
        plt.gcf().subplots_adjust(left=0.3)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(features)[sorted_idx])
        plt.title('Feature Importance (MDI)')
        plt.savefig(plotdir+'/'+name+'_featureImportance_modelC2.pdf')


        ######################### C3 CALIBRATION TRAINING (DM dependent calibration) #########################

        print('\n** INFO: training calibration C3')

        dfTraining_dict[name]['gentau_vis_abseta'] = np.abs(dfTraining_dict[name]['gentau_vis_eta'])
        dfTraining_dict[name]['gentau_vis_bin_eta'] = ((dfTraining_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTraining_dict[name]['gentau_vis_bin_pt']  = ((dfTraining_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

        meansTrainPt_dict[name] = dfTraining_dict[name][vars].groupby('gentau_vis_bin_pt').mean() 
        rmssTrainPt_dict[name] = dfTraining_dict[name][vars].groupby('gentau_vis_bin_pt').std() 

        meansTrainPt_dict[name]['logpt1'] = np.log(meansTrainPt_dict[name]['cl3d_pt_c2'])
        meansTrainPt_dict[name]['logpt2'] = meansTrainPt_dict[name].logpt1**2
        meansTrainPt_dict[name]['logpt3'] = meansTrainPt_dict[name].logpt1**3
        meansTrainPt_dict[name]['logpt4'] = meansTrainPt_dict[name].logpt1**4

        input_c3 = meansTrainPt_dict[name][['logpt1', 'logpt2', 'logpt3', 'logpt4']]
        target_c3 = meansTrainPt_dict[name]['cl3d_response_c2']
        C3model_dict[name] = LinearRegression().fit(input_c3, target_c3)

        save_obj(C3model_dict[name], outFile_modelC3_dict[name])

        logpt1 = np.log(abs(dfTraining_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4

        dfTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTraining_dict[name]['cl3d_pt_c3'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].cl3d_c3
        dfTraining_dict[name]['cl3d_response_c3'] = dfTraining_dict[name].cl3d_pt_c3 / dfTraining_dict[name].gentau_vis_pt


        ######################### C1+C2+C3 CALIBRATION APPLICATION #########################

        print('\n** INFO: applying calibration C1+C2+C3')            

        # CALIBRATE TENTAU
        eta_sel1   = np.abs(dfTau_dict[name]['cl3d_eta']) > 1.6
        eta_sel2   = np.abs(dfTau_dict[name]['cl3d_eta']) < 2.9
        cl3dPt_sel = dfTau_dict[name]['cl3d_pt'] > 4
        dfTau_dict[name] = dfTau_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
        # application calibration 1
        dfTau_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfTau_dict[name][['cl3d_abseta']])
        dfTau_dict[name]['cl3d_pt_c1'] = dfTau_dict[name].cl3d_c1 + dfTau_dict[name].cl3d_pt
        dfTau_dict[name]['cl3d_response_c1'] = dfTau_dict[name].cl3d_pt_c1 / dfTau_dict[name].gentau_vis_pt
        # application calibration 2
        dfTau_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfTau_dict[name][features])
        dfTau_dict[name]['cl3d_pt_c2'] = dfTau_dict[name].cl3d_c2 * dfTau_dict[name].cl3d_pt_c1
        dfTau_dict[name]['cl3d_response_c2'] = dfTau_dict[name].cl3d_pt_c2 / dfTau_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfTau_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTau_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTau_dict[name]['cl3d_pt_c3'] = dfTau_dict[name].cl3d_pt_c2 / dfTau_dict[name].cl3d_c3
        dfTau_dict[name]['cl3d_response_c3'] = dfTau_dict[name].cl3d_pt_c3 / dfTau_dict[name].gentau_vis_pt

        # CALIBRATE HH
        eta_sel1   = np.abs(dfHH_dict[name]['cl3d_eta']) > 1.6
        eta_sel2   = np.abs(dfHH_dict[name]['cl3d_eta']) < 2.9
        cl3dPt_sel = dfHH_dict[name]['cl3d_pt'] > 4
        dfHH_dict[name] = dfHH_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
        # application calibration 1
        dfHH_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfHH_dict[name][['cl3d_abseta']])
        dfHH_dict[name]['cl3d_pt_c1'] = dfHH_dict[name].cl3d_c1 + dfHH_dict[name].cl3d_pt
        dfHH_dict[name]['cl3d_response_c1'] = dfHH_dict[name].cl3d_pt_c1 / dfHH_dict[name].gentau_vis_pt
        # application calibration 2
        dfHH_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfHH_dict[name][features])
        dfHH_dict[name]['cl3d_pt_c2'] = dfHH_dict[name].cl3d_c2 * dfHH_dict[name].cl3d_pt_c1
        dfHH_dict[name]['cl3d_response_c2'] = dfHH_dict[name].cl3d_pt_c2 / dfHH_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfHH_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfHH_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfHH_dict[name]['cl3d_pt_c3'] = dfHH_dict[name].cl3d_pt_c2 / dfHH_dict[name].cl3d_c3
        dfHH_dict[name]['cl3d_response_c3'] = dfHH_dict[name].cl3d_pt_c3 / dfHH_dict[name].gentau_vis_pt

        if args.calibrateQCD:
            eta_sel1   = np.abs(dfQCD_dict[name]['cl3d_eta']) > 1.6
            eta_sel2   = np.abs(dfQCD_dict[name]['cl3d_eta']) < 2.9
            cl3dPt_sel = dfQCD_dict[name]['cl3d_pt'] > 4
            dfQCD_dict[name] = dfQCD_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
            dfQCD_dict[name]['cl3d_response'] = dfQCD_dict[name]['cl3d_pt']/dfQCD_dict[name]['genjet_pt']
            # application calibration 1
            dfQCD_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfQCD_dict[name][['cl3d_abseta']])
            dfQCD_dict[name]['cl3d_pt_c1'] = dfQCD_dict[name].cl3d_c1 + dfQCD_dict[name].cl3d_pt
            dfQCD_dict[name]['cl3d_response_c1'] = dfQCD_dict[name].cl3d_pt_c1 / dfQCD_dict[name].genjet_pt
            # application calibration 2
            dfQCD_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfQCD_dict[name][features])
            dfQCD_dict[name]['cl3d_pt_c2'] = dfQCD_dict[name].cl3d_c2 * dfQCD_dict[name].cl3d_pt_c1
            dfQCD_dict[name]['cl3d_response_c2'] = dfQCD_dict[name].cl3d_pt_c2 / dfQCD_dict[name].genjet_pt
            # application calibration 3
            logpt1 = np.log(abs(dfQCD_dict[name]['cl3d_pt_c2']))
            logpt2 = logpt1**2
            logpt3 = logpt1**3
            logpt4 = logpt1**4
            dfQCD_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
            dfQCD_dict[name]['cl3d_pt_c3'] = dfQCD_dict[name].cl3d_pt_c2 / dfQCD_dict[name].cl3d_c3
            dfQCD_dict[name]['cl3d_response_c3'] = dfQCD_dict[name].cl3d_pt_c3 / dfQCD_dict[name].genjet_pt

        if args.calibrateNu:
            eta_sel1   = np.abs(dfNu_dict[name]['cl3d_eta']) > 1.6
            eta_sel2   = np.abs(dfNu_dict[name]['cl3d_eta']) < 2.9
            cl3dPt_sel = dfNu_dict[name]['cl3d_pt'] > 4
            dfNu_dict[name] = dfNu_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
            # application calibration 1
            dfNu_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfNu_dict[name][['cl3d_abseta']])
            dfNu_dict[name]['cl3d_pt_c1'] = dfNu_dict[name].cl3d_c1 + dfNu_dict[name].cl3d_pt
            # application calibration 2
            dfNu_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfNu_dict[name][features])
            dfNu_dict[name]['cl3d_pt_c2'] = dfNu_dict[name].cl3d_c2 * dfNu_dict[name].cl3d_pt_c1
            # application calibration 3
            logpt1 = np.log(abs(dfNu_dict[name]['cl3d_pt_c2']))
            logpt2 = logpt1**2
            logpt3 = logpt1**3
            logpt4 = logpt1**4
            dfNu_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
            dfNu_dict[name]['cl3d_pt_c3'] = dfNu_dict[name].cl3d_pt_c2 / dfNu_dict[name].cl3d_c3


        ######################### SAVE FILES #########################

        # save the tau dataframes
        store_tau = pd.HDFStore(outFileTau_dict[name], mode='w')
        store_tau[name] = dfTau_dict[name]
        store_tau.close()
        # save the hh dataframes
        store_hh = pd.HDFStore(outFileHH_dict[name], mode='w')
        store_hh[name] = dfHH_dict[name]
        store_hh.close()

        if args.calibrateQCD:
            # save the QCD dataframes
            store_qcd = pd.HDFStore(outFileQCD_dict[name], mode='w')
            store_qcd[name] = dfQCD_dict[name]
            store_qcd.close()

        if args.calibrateNu:
            # save the nu dataframes
            store_nu = pd.HDFStore(outFileNu_dict[name], mode='w')
            store_nu[name] = dfNu_dict[name]
            store_nu.close()


        ######################### EVALUATE MEAN AND RMS OF RESPONSE #########################

        print('\n** INFO: calculation of response MEAN and RMS on training sample')
        
        genPt_sel  = dfTraining_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfTraining_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfTraining_dict[name]['gentau_vis_eta']) < 2.9
        cl3dBest_sel = dfTraining_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfTraining_dict[name]['cl3d_pt'] > 4
        dfTraining_dict[name] = dfTraining_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel]

        dfTraining_dict[name]['gentau_vis_abseta'] = np.abs(dfTraining_dict[name]['gentau_vis_eta'])
        dfTraining_dict[name]['gentau_vis_bin_eta'] = ((dfTraining_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTraining_dict[name]['gentau_vis_bin_pt']  = ((dfTraining_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

        etameans_dict[name]   = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').mean()
        etarmss_dict[name]    = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').std()
        etaeffrmss_dict[name] = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').apply(effrms)
        ptmeans_dict[name]    = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').mean()
        ptrmss_dict[name]     = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').std()
        pteffrmss_dict[name]  = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').apply(effrms)

        print(' -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response'].mean(), dfTraining_dict[name]['cl3d_response'].std(), dfTraining_dict[name]['cl3d_response'].std()/dfTraining_dict[name]['cl3d_response'].mean()))
        print(' -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c1'].mean(), dfTraining_dict[name]['cl3d_response_c1'].std(), dfTraining_dict[name]['cl3d_response_c1'].std()/dfTraining_dict[name]['cl3d_response_c1'].mean()))
        print(' -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c2'].mean(), dfTraining_dict[name]['cl3d_response_c2'].std(), dfTraining_dict[name]['cl3d_response_c2'].std()/dfTraining_dict[name]['cl3d_response_c2'].mean()))
        print(' -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c3'].mean(), dfTraining_dict[name]['cl3d_response_c3'].std(), dfTraining_dict[name]['cl3d_response_c3'].std()/dfTraining_dict[name]['cl3d_response_c3'].mean()))


        ######################### PLOT REULTS #########################
        
        if args.doPlots:    
            print('\n** INFO: plotting of response curves')
            plt.figure(figsize=(8,8))
            plt.hist(dfTau_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response'].mean(),3), round(dfTau_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfTau_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c1'].mean(),3), round(dfTau_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfTau_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c2'].mean(),3), round(dfTau_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfTau_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c3'].mean(),3), round(dfTau_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response'].mean(),3), round(dfTau_dict[name]['cl3d_response'].std(),3)),lw=2)
            cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c1'].mean(),3), round(dfTau_dict[name]['cl3d_response_c1'].std(),3)),lw=2)
            cal2_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c2'].mean(),3), round(dfTau_dict[name]['cl3d_response_c2'].std(),3)),lw=2)
            cal3_line = mlines.Line2D([], [], color='black',markersize=15, label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTau_dict[name]['cl3d_response_c3'].mean(),3), round(dfTau_dict[name]['cl3d_response_c3'].std(),3)),lw=2)
            plt.legend(loc = 'upper right',handles=[uncal_line,cal1_line,cal2_line,cal3_line], fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('TenTau dataset calibration response')
            plt.ylim(0, 2200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_tau_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfHH_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response'].mean(),3), round(dfHH_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfHH_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c1'].mean(),3), round(dfHH_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfHH_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c2'].mean(),3), round(dfHH_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfHH_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c3'].mean(),3), round(dfHH_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response'].mean(),3), round(dfHH_dict[name]['cl3d_response'].std(),3)),lw=2)
            cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c1'].mean(),3), round(dfHH_dict[name]['cl3d_response_c1'].std(),3)),lw=2)
            cal2_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c2'].mean(),3), round(dfHH_dict[name]['cl3d_response_c2'].std(),3)),lw=2)
            cal3_line = mlines.Line2D([], [], color='black',markersize=15, label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfHH_dict[name]['cl3d_response_c3'].mean(),3), round(dfHH_dict[name]['cl3d_response_c3'].std(),3)),lw=2)
            plt.legend(loc = 'upper right',handles=[uncal_line,cal1_line,cal2_line,cal3_line], fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('GluGluHHTo2b2Tau dataset \n calibration response')
            plt.ylim(0, 700)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_hh_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfTraining_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response'].mean(),3), round(dfTraining_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c1'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c2'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c3'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response'].mean(),3), round(dfTraining_dict[name]['cl3d_response'].std(),3)),lw=2)
            cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c1'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c1'].std(),3)),lw=2)
            cal2_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c2'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c2'].std(),3)),lw=2)
            cal3_line = mlines.Line2D([], [], color='black',markersize=15, label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c3'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c3'].std(),3)),lw=2)
            plt.legend(loc = 'upper right',handles=[uncal_line,cal1_line,cal2_line,cal3_line], fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Training (TenTau+GluGluHHTo2b2Tau) dataset \n calibration response')
            plt.ylim(0, 2000)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_'+name+'_PU200.pdf')
            plt.close()

            if args.calibrateQCD:
                plt.figure(figsize=(8,8))
                plt.hist(dfQCD_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response'].mean(),3), round(dfQCD_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
                plt.hist(dfQCD_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c1'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
                plt.hist(dfQCD_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c2'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
                plt.hist(dfQCD_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c3'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
                uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response'].mean(),3), round(dfQCD_dict[name]['cl3d_response'].std(),3)),lw=2)
                cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c1'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c1'].std(),3)),lw=2)
                cal2_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c2'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c2'].std(),3)),lw=2)
                cal3_line = mlines.Line2D([], [], color='black',markersize=15, label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfQCD_dict[name]['cl3d_response_c3'].mean(),3), round(dfQCD_dict[name]['cl3d_response_c3'].std(),3)),lw=2)
                plt.legend(loc = 'upper right',handles=[uncal_line,cal1_line,cal2_line,cal3_line], fontsize=15)
                plt.grid(linestyle=':')
                plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
                plt.ylabel(r'a. u.')
                plt.title('QCD dataset calibration response')
                plt.ylim(0, 6000)
                plt.gcf().subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/calibResponse_qcd_'+name+'_PU200.pdf')
                plt.close()
            
        print('\n** INFO: finished energy calibration for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')



'''
myvariable = 'cl3d_spptot'
myxmin = 0
myxmax = 0.08
mybinsize = 0.005

df_tau_DM0 = {}
df_tau_DM1 = {}
df_tau_DM4 = {}
df_tau_DM5 = {}

for name in dfTau_dict:

  seldm0 = (dfTau_dict[name]['gentau_decayMode'] == 0)
  df_tau_DM0[name] = dfTau_dict[name][seldm0]

  seldm1 = (dfTau_dict[name]['gentau_decayMode'] == 1)
  df_tau_DM1[name] = dfTau_dict[name][seldm1]

  seldm4 = (dfTau_dict[name]['gentau_decayMode'] == 4)
  df_tau_DM4[name] = dfTau_dict[name][seldm4]

  seldm5 = (dfTau_dict[name]['gentau_decayMode'] == 5)
  df_tau_DM5[name] = dfTau_dict[name][seldm5]

  plt.figure(figsize=(8,8))
  #plt.hist(dfTau_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2', color='limegreen',  histtype='step', lw=2)
  plt.hist(df_tau_DM0[name][myvariable], normed=True, bins=np.arange(myxmin,myxmax,mybinsize),  label='1-prong', color='limegreen',  histtype='step', lw=2)
  plt.hist(df_tau_DM1[name][myvariable], normed=True, bins=np.arange(myxmin,myxmax,mybinsize),  label='1-prong + $\pi^{0}$\'s', color='blue',  histtype='step', lw=2)
  plt.hist(df_tau_DM4[name][myvariable], normed=True, bins=np.arange(myxmin,myxmax,mybinsize),  label='3-prongs', color='red',  histtype='step', lw=2)
  plt.hist(df_tau_DM5[name][myvariable], normed=True, bins=np.arange(myxmin,myxmax,mybinsize),  label='3-prongs + $\pi^{0}$\'s', color='fuchsia',  histtype='step', lw=2)
  cal2_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='1-prong',lw=2)
  cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='1-prong + $\pi^{0}$\'s',lw=2)
  uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='3-prongs',lw=2)
  fuchsia_line = mlines.Line2D([], [], color='fuchsia',markersize=15, label='3-prongs + $\pi^{0}$\'s', lw=2)
  plt.legend(loc = 'upper right',handles=[cal2_line,cal1_line,uncal_line,fuchsia_line])
  #plt.legend(loc = 'upper right', fontsize=18)
  plt.grid()
  plt.xlabel(r'Total $\sigma_{\phi\phi}$')
  plt.ylabel(r'Normalized events')
  plt.ylim(0, 50)
  plt.cmstext("CMS"," Phase-2 Simulation")
  plt.lumitext("PU=200","HGCAL")
  plt.subplots_adjust(bottom=0.12)
  #txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
  #t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
  #t.set_bbox(dict(facecolor='white', edgecolor='white'))
  plt.savefig(plotdir+'calib2_'+myvariable+'_DM_TDRv1.png')
  plt.savefig(plotdir+'calib2_'+myvariable+'_DM_TDRv1.pdf')
  plt.show()
'''

###########

# Model c3
'''
matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(8,8))
pt = np.arange(25, 100, 1)

logpt1 = np.log(pt)
logpt2 = logpt1**2
logpt3 = logpt1**3
logpt4 = logpt1**4

for name in dfTau_dict:

  ptmeans_dict[name]['logpt1'] = np.log(ptmeans_dict[name]['cl3d_pt_c2'])
  ptmeans_dict[name]['logpt2'] = ptmeans_dict[name].logpt1**2
  ptmeans_dict[name]['logpt3'] = ptmeans_dict[name].logpt1**3
  ptmeans_dict[name]['logpt4'] = ptmeans_dict[name].logpt1**4

for name in dfTau_dict:

  plt.plot(ptmeans_dict[name]['cl3d_pt_c2'], ptmeans_dict[name]['cl3d_response_c2'], marker='s', markersize=9, ls='None', color=colors_dict[name], label='Observed')
  plt.plot(pt, C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T), ls='--', color=colors_dict[name], label='Predicted')

plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\langle E_{T,calib2}^{L1}\rangle\, [GeV]$')
plt.ylabel(r'$\langle E_{T,calib2}^{L1}/p_{T}^{gen}\rangle$')
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
t.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.savefig(plotdir+'calib_modelc3_TDRv2.png')
plt.savefig(plotdir+'calib_modelc3_TDRv2.pdf')
#plt.show()
'''

###########

# Responses and resolutions
'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = etameans_dict[name]
  plt.plot(df['gentau_vis_abseta'], etaeffrmss_dict[name]['cl3d_response']/df['cl3d_response'], label="Raw",color='red',lw=2)
  plt.plot(df['gentau_vis_abseta'], etaeffrmss_dict[name]['cl3d_response_c3']/df['cl3d_response_c3'], label="Calibrated",color='blue',lw=2)
  y_array_raw = (etaeffrmss_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (etaeffrmss_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.errorbar(x_array,y_array_raw,xerr=2.5,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=2.5,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
plt.xlabel(r'$|\eta|$')
plt.ylabel(r'$RMS_{eff}(E_{T}^{L1}/p_{T}^{gen})$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig(plotdir+'calib_mean_vs_eta.png')
plt.savefig(plotdir+'calib_mean_vs_eta.pdf')
'''

'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = etameans_dict[name]
  axs[0].plot(df['gentau_vis_abseta'], etarmss_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = etarmss_dict[name]['cl3d_response_c3']/etarmss_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_abseta'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(1.6, 3.0)
axs[0].set_ylim(0.10,0.35)
axs[0].grid()
#axs[0].set_xlabel(r'$|\eta|$')
axs[0].set_ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(1.6, 3.0)
axs[1].set_ylim(0.95,1.20)
axs[1].set_xlabel(r'$|\eta|$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_rms_vs_eta_ratio.png')
plt.savefig(plotdir+'calib_rms_vs_eta_ratio.pdf')
'''

'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = etameans_dict[name]
  axs[0].plot(df['gentau_vis_abseta'], etaeffrmss_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = etaeffrmss_dict[name]['cl3d_response_c3']/etaeffrmss_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_abseta'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(1.6, 3.0)
axs[0].set_ylim(0.10,0.30)
axs[0].grid()
#axs[0].set_xlabel(r'$|\eta|$')
axs[0].set_ylabel(r'$RMS_{eff}(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(1.6, 3.0)
axs[1].set_ylim(0.95,1.25)
axs[1].set_xlabel(r'$|\eta|$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_effrms_vs_eta_ratio.png')
plt.savefig(plotdir+'calib_effrms_vs_eta_ratio.pdf')
'''

'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = ptmeans_dict[name]
  plt.plot(df['gentau_vis_pt'], df['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)

plt.ylim(0.9, 1.1)
plt.xlabel(r'$p_{T}\, [GeV]$', fontsize=18)
plt.ylabel(r'$\langle p_{T}^{L1}/p_{T}^{gen}\rangle$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.ylim(0.95, 1.05)
plt.savefig(plotdir+'calib_mean_vs_pt.png')
plt.savefig(plotdir+'calib_mean_vs_pt.pdf')
'''

'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = ptmeans_dict[name]
  plt.plot(df['gentau_vis_pt'], ptrmss_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)

plt.ylim(0., 0.3)
plt.xlabel(r'$p_{T}\, [GeV]$')
plt.ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.ylim(0.0, 0.4)
plt.savefig(plotdir+'calib_rms_vs_pt.png')
plt.savefig(plotdir+'calib_rms_vs_pt.pdf')
'''
'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = ptmeans_dict[name]
  axs[0].plot(df['gentau_vis_pt'], ptrmss_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = ptrmss_dict[name]['cl3d_response_c3']/ptrmss_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_pt'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(20., 100.)
axs[0].set_ylim(0.0, 0.4)
axs[0].grid()
#axs[0].set_xlabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[0].set_ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(20., 100.)
axs[1].set_ylim(0.95,1.40)
axs[1].set_xlabel(r'$p_{T}\, [GeV]$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_rms_vs_pt_ratio.png')
plt.savefig(plotdir+'calib_rms_vs_pt_ratio.pdf')
'''
'''
plt.figure(figsize=(8,9))

for name in dfTau_dict:

  df = ptmeans_dict[name]
  plt.plot(df['gentau_vis_pt'], pteffrmss_dict[name]['cl3d_response']/df['cl3d_response'], label="Raw",color='red',lw=2)
  plt.plot(df['gentau_vis_pt'], pteffrmss_dict[name]['cl3d_response_c3']/df['cl3d_response_c3'], label="Calibrated",color='blue',lw=2)

plt.ylim(0.01, 0.45)
plt.legend(loc = 'lower left')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$RMS_{eff}\ /\ <p_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'calib_effrms_vs_pt_TDR.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_TDR.pdf')
plt.show()
'''

'''
x_array = [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5]

for name in dfTau_dict:
  df = ptmeans_dict[name]
  y_array_raw = (pteffrmss_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (pteffrmss_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.figure(figsize=(8,8))
plt.errorbar(x_array,y_array_raw,xerr=2.5,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=2.5,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
plt.ylim(0.01, 0.45)
uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Raw',lw=2)
cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calibrated',lw=2)
plt.legend(loc = 'lower left',handles=[uncal_line,cal1_line])
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$RMS_{eff}\ /\ <E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'calib_effrms_vs_pt_TDRv1.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_TDRv1.pdf')
plt.show()
'''
'''
x_array = [1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85]

for name in dfTau_dict:
  df = etameans_dict[name]
  y_array_raw = (etaeffrmss_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (etaeffrmss_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.figure(figsize=(8,8))
plt.errorbar(x_array,y_array_raw,xerr=0.05,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=0.05,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Raw',lw=2)
cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calibrated',lw=2)
plt.legend(loc = 'lower left',handles=[uncal_line,cal1_line])
plt.xlabel(r'$|\eta^{gen,\tau}|$')
plt.ylabel(r'$RMS_{eff}\ /\ <E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.savefig(plotdir+'calib_effrms_vs_eta_TDRv1.png')
plt.savefig(plotdir+'calib_effrms_vs_eta_TDRv1.pdf')
plt.show()
'''
'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = ptmeans_dict[name]
  axs[0].plot(df['gentau_vis_pt'], pteffrmss_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = pteffrmss_dict[name]['cl3d_response_c3']/pteffrmss_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_pt'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(20., 100.)
axs[0].set_ylim(0., 0.35)
axs[0].grid()
#axs[0].set_xlabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[0].set_ylabel(r'$RMS_{eff}(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(20., 100.)
axs[1].set_ylim(0.95,1.40)
axs[1].set_xlabel(r'$p_{T}\, [GeV]$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_effrms_vs_pt_ratio.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_ratio.pdf')
'''
'''
plt.figure(figsize=(8,7))

for name in dfTau_dict:
  plt.hist(dfTau_dict[name]['gentau_vis_eta']-dfTau_dict[name]['cl3d_eta'], bins=np.arange(-0.15, 0.15, 0.005), color='blue',lw=2,histtype='step')

#plt.ylim(0.01, 0.4)
plt.xlabel(r'$\eta_{gen,\tau}$ - $\eta_{L1,\tau}$')
plt.ylabel(r'a. u. ')
#plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
plt.grid()
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(-0.13,1400, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#txt2 = ('Mean: 0.00, RMS: 0.02')
#t2 = plt.text(0.025,1700, txt2, ha='left', wrap=True, fontsize=16)
#t2.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'res_eta_TDRv1.png')
plt.savefig(plotdir+'res_eta_TDRv1.pdf')

plt.figure(figsize=(8,7))

for name in dfTau_dict:
  plt.hist(dfTau_dict[name]['gentau_vis_phi']-dfTau_dict[name]['cl3d_phi'], bins=np.arange(-0.15, 0.15, 0.005), color='blue',lw=2,histtype='step')

#plt.ylim(0.01, 0.4)
plt.xlabel(r'$\phi_{gen,\tau}$ - $\phi_{L1,\tau}$')
plt.ylabel(r'a. u. ')
#plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
plt.grid()
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(-0.13,1400, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#txt2 = ('Mean: 0.00, RMS: 0.02')
#t2 = plt.text(0.025,1700, txt2, ha='left', wrap=True, fontsize=16)
#t2.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'res_phi_TDRv1.png')
plt.savefig(plotdir+'res_phi_TDRv1.pdf')
'''
