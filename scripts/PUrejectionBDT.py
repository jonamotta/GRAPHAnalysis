import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
import matplotlib
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import root_pandas
import matplotlib.lines as mlines
import argparse


def train_xgb(dfTrain, features, output, hyperparams, test_fraction=0.3):    
    X_train, X_test, y_train, y_test = train_test_split(dfTrain[features], dfTrain[output], test_size=test_fraction)

    train = xgb.DMatrix(data=X_train,label=y_train, feature_names=features)
    test = xgb.DMatrix(data=X_test,label=y_test,feature_names=features)
    booster = xgb.train(hyperparams, train, num_boost_round=hyperparams['num_trees'])
    X_train['bdt_output'] = booster.predict(train)
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, X_train['bdt_output'])
    X_test['bdt_output'] = booster.predict(test)
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, X_test['bdt_output'])

    auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
    auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])

    return booster, fpr_train, tpr_train, threshold_train, fpr_test, tpr_test, threshold_test, auroc_test, auroc_train

def efficiency(group, cut):
    tot = group.shape[0]
    sel = group[group.cl3d_pubdt_score > cut].shape[0]
    return float(sel)/float(tot)

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
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='do you want to apply PU rejection to QCD?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

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
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/PUrejected'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/PUrejection'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/PUrejection'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTau_dict = {
        'threshold'    : indir+'/RelValTenTau_PU200_th_calibrated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileNu_dict = {
        'threshold'    : indir+'/RelValNu_PU200_th_calibrated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileHH_dict = {
        'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_calibrated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_dict = {
        'threshold'    : outdir+'/RelValTenTau_PU200_th_PUrejected.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileNu_dict = {
        'threshold'    : outdir+'/RelValNu_PU200_th_PUrejected.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileHH_dict = {
        'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_PUrejected.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    if args.doQCD:
        inFileQCD_dict = {
            'threshold'    : indir+'/QCD_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileQCD_dict = {
            'threshold'    : outdir+'/QCD_PU200_th_PUrejected.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    outFile_model_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_FPRnClusters_dict = {
        'threshold'    : model_outdir+'/FPRnClusters_PUrejection_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    dfTau_dict = {}              # dictionary of the skim level tau dataframes
    dfNu_dict = {}               # dictionary of the skim level nu dataframes
    dfHH_dict = {}               # dictionary of the skim level HH dataframes
    dfHHValidation_dict = {}     # dictionary of the validation HH dataframes
    dfTauTraining_dict = {}      # dictionary of the training tau dataframes
    dfNuTraining_dict = {}       # dictionary of the training nu dataframes
    dfNuValidation_dict = {}     # dictionary of the test nu dataframes
    dfMergedTraining_dict = {}   # dictionary of the merged training dataframes
    dfMergedValidation_dict = {} # dictionary of the merged test dataframes
    totalTauEvents_dict = {}     # dictionary of the total number of tau events
    storedTauEvents_dict = {}    # dictionary of the training number of tau events
    totalNuEvents_dict = {}      # dictionary of the total number of nu events
    storedNuEvents_dict = {}     # dictionary of the training number of nu events
    totalHHEvents_dict = {}      # dictionary of the total number of nu events
    storedHHEvents_dict = {}     # dictionary of the training number of nu events
    clustersNperEvent_dict = {}

    if args.doQCD: dfQCD_dict = {} # dictionary of the skim level QCD dataframes

    # features for BDT training
    features = ['cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    output = 'gentau_pid'

    params_dict = {}
    params_dict['nthread']            = 10  # limit number of threads
    params_dict['eta']                = 0.2 # learning rate
    params_dict['max_depth']          = 4   # maximum depth of a tree
    params_dict['subsample']          = 0.8 # fraction of events to train tree on
    params_dict['colsample_bytree']   = 0.8 # fraction of features to train tree on
    params_dict['silent']             = True
    params_dict['objective']          = 'binary:logistic' # objective function
    params_dict['num_trees']          = 81  # number of trees to make

    # dictionaries for BDT training
    model_dict= {}
    fpr_train_dict = {}
    tpr_train_dict = {}
    fpr_test_dict = {}
    tpr_test_dict = {}
    fprNcluster_dict = {}
    threshold_train_dict = {}
    threshold_test_dict = {}
    threshold_validation_dict = {}
    testAuroc_dict = {}
    trainAuroc_dict = {}
    fpr_validation_dict = {}
    tpr_validation_dict = {}

    # working points dictionaries
    bdtWP99_dict = {}
    bdtWP95_dict = {}
    bdtWP90_dict = {}

    # colors to use for plotting
    colors_dict = {
        'threshold'    : 'blue',
        'supertrigger' : 'red',
        'bestchoice'   : 'olive',
        'bestcoarse'   : 'orange',
        'mixed'        : 'fuchsia'
    }

    # legend to use for plotting
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'supertrigger' : 'STC4+16',
        'bestchoice'   : 'BC Decentral',
        'bestcoarse'   : 'BC Coarse 2x2 TC',
        'mixed'        : 'Mixed BC + STC'
    }

    matplotlib.rcParams.update({'font.size': 22})


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting PU rejection for the front-end option '+feNames_dict[name])

        # fill tau dataframes and dictionaries -> training 
        store_tau = pd.HDFStore(inFileTau_dict[name], mode='r')
        dfTau_dict[name] = store_tau[name] 
        store_tau.close()
        dfTau_dict[name]['gentau_pid'] = 1 # tag as signal

        # fill nu pileup dataframes and dictionaries -> 1/2 training + 1/2 validation 
        store_nu = pd.HDFStore(inFileNu_dict[name], mode='r')
        dfNu_dict[name] = store_nu[name]
        store_nu.close()
        dfNu_dict[name]['gentau_pid'] = 0 # tag as PU

        # fill HH dataframes and dictionaries -> validation
        store_hh = pd.HDFStore(inFileHH_dict[name], mode='r')
        dfHH_dict[name] = store_hh[name]
        store_hh.close()
        dfHH_dict[name]['gentau_pid'] = 1 # tag as signal

        if args.doQCD:
            # fill QCD dataframes and dictionaries
            store_qcd = pd.HDFStore(inFileQCD_dict[name], mode='r')
            dfQCD_dict[name] = store_qcd[name]
            store_qcd.close()
            dfQCD_dict[name]['gentau_pid'] = 1 # tag as background


        ######################### SELECT EVENTS FOR TRAINING #########################
  
        # SIGNAL
        totalTauEvents_dict[name] = np.unique(dfTau_dict[name].reset_index()['event']).shape[0]
        dfTauTraining_dict[name] = dfTau_dict[name]
        # define selections for the training dataset
        genPt_sel  = dfTauTraining_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfTauTraining_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfTauTraining_dict[name]['gentau_vis_eta']) < 2.9
        cl3dBest_sel = dfTauTraining_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfTauTraining_dict[name]['cl3d_pt_c3'] > 4
        # apply slections for the training dataset
        dfTauTraining_dict[name] = dfTauTraining_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel]
        # store the absolute number of events of sgn used for training
        storedTauEvents_dict[name] = np.unique(dfTauTraining_dict[name].reset_index()['event']).shape[0]

        # PILEUP
        totalNuEvents_dict[name] = np.unique(dfNu_dict[name].reset_index()['event']).shape[0] 
        dfNuTraining_dict[name] = dfNu_dict[name].sample(frac=0.5,random_state=10)
        dfNuValidation_dict[name] = dfNu_dict[name].drop(dfNuTraining_dict[name].index)
        # define selections for the nu dataset
        cl3dPt_selTrain = dfNuTraining_dict[name]['cl3d_pt_c3'] > 4
        cl3dPt_selTest = dfNuValidation_dict[name]['cl3d_pt_c3'] > 4
        # apply slections for the nu dataset
        dfNuTraining_dict[name] = dfNuTraining_dict[name][cl3dPt_selTrain]
        dfNuValidation_dict[name] = dfNuValidation_dict[name][cl3dPt_selTest]
        # store the absolute number of events of sgn used for training
        storedNuEvents_dict[name] = np.unique(dfNuTraining_dict[name].reset_index()['event']).shape[0]

        # VALIDATION
        dfHHValidation_dict[name] = dfHH_dict[name]
        eta_sel1   = np.abs(dfHHValidation_dict[name]['cl3d_eta']) > 1.6
        eta_sel2   = np.abs(dfHHValidation_dict[name]['cl3d_eta']) < 2.9
        cl3dPt_sel = dfHHValidation_dict[name]['cl3d_pt_c3'] > 4
        # apply slections for the validation dataset
        dfHHValidation_dict[name] = dfHHValidation_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
        
        # MERGE
        dfMergedTraining_dict[name] = pd.concat([dfTauTraining_dict[name],dfNuTraining_dict[name]],sort=False)
        dfMergedValidation_dict[name] = pd.concat([dfHHValidation_dict[name],dfNuValidation_dict[name]],sort=False)

        print('\n**INFO: with front-end option {0} - #cl3d PU: {1} - #cl3s SGN: {2}'.format(feNames_dict[name], dfNuTraining_dict[name].shape[0], dfTauTraining_dict[name].shape[0]))


        ######################### PU CLUSTERS PER EVENT #########################

        sel = dfMergedTraining_dict[name]['gentau_pid']==0
        dfPU = dfMergedTraining_dict[name][sel]
        clusters_per_event = dfPU.groupby('event').count()
        clustersNperEvent_dict[name] = np.mean(clusters_per_event.gentau_pid) * storedNuEvents_dict[name]/totalNuEvents_dict[name]
        print('\n**INFO: with front-end option {0} - #events stored: {1} - #events total: {2} - clusters/event: {3}'.format(feNames_dict[name], storedNuEvents_dict[name], totalNuEvents_dict[name], clustersNperEvent_dict[name]))


        ######################### TRAINING OF BDT #########################

        dfMergedTraining_dict[name]['cl3d_abseta'] = np.abs(dfMergedTraining_dict[name].cl3d_eta)

        model_dict[name], fpr_train_dict[name], tpr_train_dict[name], threshold_train_dict[name], fpr_test_dict[name], tpr_test_dict[name], threshold_test_dict[name], testAuroc_dict[name], trainAuroc_dict[name] = train_xgb(dfMergedTraining_dict[name], features, output, params_dict, test_fraction=0.3)
        fprNcluster_dict[name] = fpr_train_dict[name]*clustersNperEvent_dict[name]

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_dict[name]))

        save_obj(model_dict[name], outFile_model_dict[name])
        save_obj(fpr_train_dict[name], outFile_fpr_train_dict[name])
        save_obj(tpr_train_dict[name], outFile_tpr_train_dict[name])
        save_obj(fprNcluster_dict[name], outFile_FPRnClusters_dict[name])

        bdtWP99_dict[name] = np.interp(0.99, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP95_dict[name] = np.interp(0.95, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP90_dict[name] = np.interp(0.90, tpr_train_dict[name], threshold_train_dict[name])
        
        save_obj(bdtWP99_dict[name], outFile_WP99_dict[name])
        save_obj(bdtWP95_dict[name], outFile_WP95_dict[name])
        save_obj(bdtWP90_dict[name], outFile_WP90_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_dict[name], bdtWP95_dict[name], bdtWP90_dict[name]))
        print('\n**INFO: bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.95, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.90, tpr_train_dict[name], fpr_train_dict[name])))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfMergedValidation_dict[name][features], label=dfMergedValidation_dict[name][output], feature_names=features)
        dfMergedValidation_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

        fpr_validation_dict[name], tpr_validation_dict[name], threshold_validation_dict[name] = metrics.roc_curve(dfMergedValidation_dict[name][output], dfMergedValidation_dict[name]['cl3d_pubdt_score'])
        auroc_validation = metrics.roc_auc_score(dfMergedValidation_dict[name]['gentau_pid'],dfMergedValidation_dict[name]['cl3d_pubdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(auroc_validation))

        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTau_dict[name][features], label=dfTau_dict[name][output], feature_names=features)
        dfTau_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

        dfTau_dict[name]['cl3d_pubdt_passWP99'] = dfTau_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfTau_dict[name]['cl3d_pubdt_passWP95'] = dfTau_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfTau_dict[name]['cl3d_pubdt_passWP90'] = dfTau_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfNu_dict[name][features], label=dfNu_dict[name][output], feature_names=features)
        dfNu_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

        dfNu_dict[name]['cl3d_pubdt_passWP99'] = dfNu_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfNu_dict[name]['cl3d_pubdt_passWP95'] = dfNu_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfNu_dict[name]['cl3d_pubdt_passWP90'] = dfNu_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfHH_dict[name][features], label=dfHH_dict[name][output], feature_names=features)
        dfHH_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

        dfHH_dict[name]['cl3d_pubdt_passWP99'] = dfHH_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfHH_dict[name]['cl3d_pubdt_passWP95'] = dfHH_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfHH_dict[name]['cl3d_pubdt_passWP90'] = dfHH_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        if args.doQCD:
            full = xgb.DMatrix(data=dfQCD_dict[name][features], label=dfQCD_dict[name][output], feature_names=features)
            dfQCD_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

            dfQCD_dict[name]['cl3d_pubdt_passWP99'] = dfQCD_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
            dfQCD_dict[name]['cl3d_pubdt_passWP95'] = dfQCD_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
            dfQCD_dict[name]['cl3d_pubdt_passWP90'] = dfQCD_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]
            
            # print('\n**INFO: rejected PU for QCD dataset and obtained:')
            # print('  -- total number of QCD events: {0}'.format(dfQCD_dict[name]['cl3d_pubdt_score'].count()))
            # print(r'  -- number of QCD events passing WP99: {0} ({1}\%)'.format(QCDpassing99, QCDpassing99/QCDtot*100))
            # print(r'  -- number of QCD events passing WP95: {0} ({1}\%)'.format(QCDpassing95, QCDpassing95/QCDtot*100))
            # print(r'  -- number of QCD events passing WP90: {0} ({1}\%)'.format(QCDpassing90, QCDpassing90/QCDtot*100))


        ######################### SAVE FILES #########################

        store_tau = pd.HDFStore(outFileTau_dict[name], mode='w')
        store_tau[name] = dfTau_dict[name]
        store_tau.close()

        store_nu = pd.HDFStore(outFileNu_dict[name], mode='w')
        store_nu[name] = dfNu_dict[name]
        store_nu.close()

        store_hh = pd.HDFStore(outFileHH_dict[name], mode='w')
        store_hh[name] = dfHH_dict[name]
        store_hh.close()

        if args.doQCD:
            store_qcd = pd.HDFStore(outFileQCD_dict[name], mode='w')
            store_qcd[name] = dfQCD_dict[name]
            store_qcd.close()


        ######################### PRINT SOME INFOS #########################

        #print np.unique(dfNu_dict[name].reset_index()['event']).shape[0]
        #sel = dfNu_dict[name]['cl3d_pt_c3'] > 20
        #dfNu_dict[name] = dfNu_dict[name][sel]
        #print np.unique(dfNu_dict[name].reset_index()['event']).shape[0]
        #sel = dfNu_dict[name]['cl3d_pubdt_passWP99'] == True
        #dfNu_dict[name] = dfNu_dict[name][sel]
        #print np.unique(dfNu_dict[name].reset_index()['event']).shape[0]

        print('\n** INFO: finished PU rejection for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')


        
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    ######################### PLOT VARIABLES #########################        
    print('\n**INFO: plotting features')
    
    # name : [title, [min, max, step]
    features_dict = {'cl3d_abseta'           : [r'TCs |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'     : [r'TCs shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'       : [r'TCs first layer',[0.,20.,20]], 
                     'cl3d_maxlayer'         : [r'TCs maximum layer',[0.,50.,50]], 
                     'cl3d_szz'              : [r'TCs szz',[0.,60.,20]], 
                     'cl3d_seetot'           : [r'TCs seetot',[0.,0.15,10]], 
                     'cl3d_spptot'           : [r'TCs spptot',[0.,0.1,10]], 
                     'cl3d_srrtot'           : [r'TCs srrtot',[0.,0.01,10]], 
                     'cl3d_srrmean'          : [r'TCs srrmean',[0.,0.01,10]], 
                     'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,25]], 
                     'cl3d_meanz'            : [r'TCs meanz',[325.,375.,30]], 
                     'cl3d_layer10'          : [r'TCs 10th layer',[0.,15.,30]], 
                     'cl3d_layer50'          : [r'TCs 50th layer',[0.,30.,60]], 
                     'cl3d_layer90'          : [r'TCs 90th layer',[0.,40.,40]], 
                     'cl3d_ntc67'            : [r'Number of TCs with 67% of energy',[0.,50.,10]], 
                     'cl3d_ntc90'            : [r'Number of TCs with 90% of energy',[0.,100.,20]]
    }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        for var in features_dict:
            plt.figure(figsize=(8,8))
            plt.hist(dfNu_dict[name][var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='PU events',      color='red',    histtype='step', lw=2, density=True)
            plt.hist(dfTau_dict[name][var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='TenTau Signal',   color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfHH_dict[name][var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='HH Signal',   color='green',    histtype='step', lw=2, density=True)            
            PUline = mlines.Line2D([], [], color='red',markersize=15, label='PU events',lw=2)
            SGNline = mlines.Line2D([], [], color='blue',markersize=15, label='TenTau Signal',lw=2)
            HHline = mlines.Line2D([], [], color='green',markersize=15, label='HH Signal',lw=2)
            plt.legend(loc = 'upper right',handles=[PUline,SGNline,HHline])
            plt.grid(linestyle=':')
            plt.xlabel(features_dict[var][0])
            plt.ylabel(r'Normalized entries')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/'+var+'.pdf')
            plt.close()


    ######################### PLOT ROCS #########################
    
    print('\n** INFO: plotting test ROC curves')
    plt.figure(figsize=(10,10))
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        roc_auc = metrics.auc(fpr_test_dict[name], tpr_test_dict[name])
        print('  -- for front-end option {0} - TEST AUC (bkgRej. vs. sgnEff.): {1}'.format(feNames_dict[name],roc_auc))
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label=legends_dict[name], color=colors_dict[name],lw=2)

    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=22)
    plt.xlim(0.9,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency') 
    plt.title('Test ROC curves')
    plt.savefig(plotdir+'/PUbdt_test_rocs.pdf')
    plt.close()

    print('\n** INFO: plotting validation ROC curves')
    plt.figure(figsize=(10,10))
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        roc_auc = metrics.auc(fpr_validation_dict[name], tpr_validation_dict[name])
        print('  -- for front-end option {0} - VALIDATION AUC (bkgRej. vs. sgnEff.): {1}'.format(feNames_dict[name],roc_auc))
        plt.plot(tpr_validation_dict[name],fpr_validation_dict[name],label=legends_dict[name], color=colors_dict[name],lw=2)

    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=22)
    plt.xlim(0.9,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.title('Validation ROC curves')
    plt.savefig(plotdir+'/PUbdt_validation_rocs.pdf')
    plt.close()

    print('\n** INFO: plotting train-test-validation ROC curves')
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_dict[name],fpr_train_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_dict[name],fpr_validation_dict[name],label='Validation ROC', color='blue',lw=2)
    	plt.grid(linestyle=':')
    	plt.legend(loc = 'upper left', fontsize=22)
    	plt.xlim(0.9,1.001)
    	#plt.yscale('log')
    	#plt.ylim(0.01,1)
    	plt.xlabel('Signal efficiency')
    	plt.ylabel('Background efficiency')
    	plt.title('ROC curves - {0} FE'.format(feNames_dict[name]))
    	plt.savefig(plotdir+'/PUbdt_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
    	plt.close()

    print('\n** INFO: plotting PUcl3d/event. vs. sgnEff. curves')
    plt.figure(figsize=(10,10))
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        plt.plot(tpr_train_dict[name],fprNcluster_dict[name],label=legends_dict[name], color=colors_dict[name],lw=2)

    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=22)
    plt.xlim(0.9,1.001)
    #plt.yscale('log')
    #plt.ylim(0.1,2)
    plt.xlabel('Signal efficiency')
    plt.ylabel('PU clusters / event') 
    plt.title('Cluster/Event ROC curves')
    plt.savefig(plotdir+'/PUbdt_roc_nclusters.pdf')
    plt.close()

    

    ######################### PLOT FEATURE IMPORTANCES #########################
    
    matplotlib.rcParams.update({'font.size': 12})
    print('\n** INFO: plotting features importance and score')
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        plt.figure(figsize=(10,10))
        importance = model_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,700)
        plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_importances_'+name+'.pdf')
        plt.close()


    ######################### PLOT BDT SCORE #########################

    matplotlib.rcParams.update({'font.size': 22})
    for name in dfTau_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        plt.figure(figsize=(10,10))
        plt.hist(dfHH_dict[name]['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), density=True, color='red', histtype='step', lw=2, label='Tau PU=200')
        plt.hist(dfNu_dict[name]['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='blue', histtype='step', lw=2, label='Nu PU=200')
        if args.doQCD: plt.hist(dfQCD_dict[name]['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='green', histtype='step', lw=2, label='QCD PU=200')
        plt.legend(loc = 'upper right', fontsize=22)
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('pu rejection BDT score')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_score_'+name+'.pdf')
        plt.close()

    print('\n** INFO: finished plotting')
    print('---------------------------------------------------------------------------------------')



'''
###########

# EFFICIENCIES

matplotlib.rcParams.update({'font.size': 22})

# Cuts for plotting

for name in dfTau_dict:

  dfTau_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_dict[name]['gentau_vis_eta'])

  sel = dfTau_dict[name]['gentau_vis_pt'] > 20
  dfTau_dict[name] = dfTau_dict[name][sel]
  
  sel = np.abs(dfTau_dict[name]['gentau_vis_eta']) > 1.6
  dfTau_dict[name] = dfTau_dict[name][sel]
  
  sel = np.abs(dfTau_dict[name]['gentau_vis_eta']) < 2.9
  dfTau_dict[name] = dfTau_dict[name][sel]
  
  sel = dfTau_dict[name]['cl3d_isbestmatch'] == True
  dfTau_dict[name] = dfTau_dict[name][sel]

  sel = dfTau_dict[name]['cl3d_pt_c3'] > 4
  dfTau_dict[name] = dfTau_dict[name][sel]

ptmin = 20
etamin = 1.6

for name in dfTau_dict:

  dfTau_dict[name]['gentau_bin_eta'] = ((dfTau_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
  dfTau_dict[name]['gentau_bin_pt']  = ((dfTau_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

# EFFICIENCY VS ETA

efficiencies_vs_eta = {}

for name in dfTau_dict:

  efficiencies_vs_eta[name] = dfTau_dict[name].groupby('gentau_bin_eta').mean()

for name in dfTau_dict:

  efficiencies_vs_eta[name]['efficiency99'] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, bdtWP99_dict[name]))
  efficiencies_vs_eta[name]['efficiency95'] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, bdtWP95_dict[name]))
  efficiencies_vs_eta[name]['efficiency90'] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, bdtWP90_dict[name]))

plt.figure(figsize=(8,8))

for name in dfTau_dict:

  df = efficiencies_vs_eta[name]
  plt.plot(df.gentau_vis_abseta, df.efficiency99, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.9, 1.01)
plt.xlabel(r'$|\eta^{gen}|$')
plt.ylabel('Efficiency')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.savefig(plotdir+'pubdt_eff99_eta_TDR.png')
plt.savefig(plotdir+'pubdt_eff99_eta_TDR.pdf')


plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = efficiencies_vs_eta[name]
  plt.plot(df.gentau_vis_abseta, df.efficiency95, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.8, 1.01)
plt.legend(loc = 'lower left', fontsize=16)
plt.xlabel(r'$|\eta|$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'pubdt_eff95_eta.png')
plt.savefig(plotdir+'pubdt_eff95_eta.pdf')

plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = efficiencies_vs_eta[name]
  plt.plot(df.gentau_vis_abseta, df.efficiency90, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.7, 1.01)
plt.legend(loc = 'lower left', fontsize=16)
plt.xlabel(r'$|\eta|$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'pubdt_eff90_eta.png')
plt.savefig(plotdir+'pubdt_eff90_eta.pdf')


# EFFICIENCY VS PT

efficiencies_vs_pt = {}

for name in dfTau_dict:

  efficiencies_vs_pt[name] = dfTau_dict[name].groupby('gentau_bin_pt').mean()

for name in dfTau_dict:

  efficiencies_vs_pt[name]['efficiency99'] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, bdtWP99_dict[name]))
  efficiencies_vs_pt[name]['efficiency95'] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, bdtWP95_dict[name]))
  efficiencies_vs_pt[name]['efficiency90'] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, bdtWP90_dict[name]))

plt.figure(figsize=(8,8))

for name in dfTau_dict:

  df = efficiencies_vs_pt[name]
  plt.plot(df.gentau_vis_pt, df.efficiency99, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.9, 1.01)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'$p_{T}^{gen}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.savefig(plotdir+'pubdt_eff99_pt_TDR.png')
plt.savefig(plotdir+'pubdt_eff99_pt_TDR.pdf')


plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = efficiencies_vs_pt[name]
  plt.plot(df.gentau_vis_pt, df.efficiency95, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.75, 1.01)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'$p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'pubdt_eff95_pt.png')
plt.savefig(plotdir+'pubdt_eff95_pt.pdf')

plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = efficiencies_vs_pt[name]
  plt.plot(df.gentau_vis_pt, df.efficiency90, label=legends_dict[name], color=colors_dict[name],lw=2)

plt.ylim(0.7, 1.01)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'$p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'pubdt_eff90_pt.png')
plt.savefig(plotdir+'pubdt_eff90_pt.pdf')
'''
