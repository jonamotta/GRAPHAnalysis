import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
import matplotlib
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
import argparse
import sys

class Logger(object):
    def __init__(self, file):
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


def train_xgb(dfTr, features, output, hyperparams, num_trees):
    X_train, X_test, y_train, y_test = train_test_split(dfTr[features], dfTr[output], stratify=dfTr[output], test_size=0.3)

    train = xgb.DMatrix(data=X_train,label=y_train, feature_names=features)
    test = xgb.DMatrix(data=X_test,label=y_test,feature_names=features)
    booster = xgb.train(hyperparams, train, num_boost_round=num_trees)
    X_train['bdt_output'] = booster.predict(train)
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, X_train['bdt_output'])
    X_test['bdt_output'] = booster.predict(test)
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, X_test['bdt_output'])

    auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
    auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])

    return booster, fpr_train, tpr_train, threshold_train, fpr_test, tpr_test, threshold_test, auroc_test, auroc_train

def efficiency(group, threshold, PUWP):
    tot = group.shape[0]
    if   PUWP == 99: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
    elif PUWP == 95: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
    else:            sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
    return float(sel)/float(tot)

def efficiency_err(group, threshold, PUWP, upper=False):
    tot = group.shape[0]
    if   PUWP == 99: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
    elif PUWP == 95: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
    else:            sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
    
    # clopper pearson errors --> ppf gives the boundary of the cinfidence interval, therefore for plotting we have to subtract the value of the central value float(sel)/float(tot)!!
    alpha = (1 - 0.9) / 2
    if upper:
        if sel == tot:
             return 0.
        else:
            return abs(btdtri(sel+1, tot-sel, 1-alpha) - float(sel)/float(tot))
    else:
        if sel == 0:
            return 0.
        else:
            return abs(float(sel)/float(tot) - btdtri(sel, tot-sel+1, alpha))

    # naive errors calculated as propagation of the statistical error on sel and tot
    # eff = sel / (sel + not_sel) --> propagate stat error of sel and not_sel to efficiency
    #return np.sqrt( (np.sqrt(tot-sel)+np.sqrt(sel))**2 * sel**2/tot**4 + np.sqrt(sel)**2 * 1./tot**4 )

def sigmoid(x , x0, k):
    return 1 / ( 1 + np.exp(-k*(x-x0)) )

def poly(x, k, a, b, c, d):
    return -a * (x+k)**4 -b * (x+k)**3 -c * (x+k)**2 -d * (x+k)

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
    parser.add_argument('--doEfficiency', dest='doEfficiency', help='do you want calculate the efficiencies?', action='store_true', default=False)
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
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_sectioned/hdf5dataframes/calibrated_C1fullC2C3'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_sectioned/hdf5dataframes/PUrejected_C1fullC2C3_fullPUnoPt'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_sectioned/plots/PUrejection_C1fullC2C3_fullPUnoPt'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_sectioned/pklModels/PUrejection_C1fullC2C3_fullPUnoPt'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_sectioned/pklModels/PUrejection_C1fullC2C3_fullPUnoPt/performance.log")

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_calibrated.hdf5',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_calibrated.hdf5',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_PUrejected.hdf5',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUrejected.hdf5',
        'mixed'        : outdir+'/'
    }

    outFile_model_lowEta_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_lowEta_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_lowEta_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_lowEta_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_lowEta_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_lowEta_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_lowEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_model_midEta_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_midEta_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_midEta_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_midEta_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_midEta_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_midEta_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_midEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_model_highEta_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_highEta_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_highEta_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_highEta_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_highEta_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_highEta_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_highEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_model_vhighEta_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_vhighEta_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_vhighEta_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_vhighEta_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}

    # features for BDT training - FULL AVAILABLE
    features = ['cl3d_c1', 'cl3d_c2', 'cl3d_c3', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

    features = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_hoe', 'cl3d_meanz']


    output = 'gentau_pid'

    params_dict = {}
    params_dict['eval_metric']        = 'logloss'
    params_dict['nthread']            = 10   # limit number of threads
    params_dict['eta']                = 0.2 # learning rate
    params_dict['max_depth']          = 5    # maximum depth of a tree
    params_dict['subsample']          = 0.6 # fraction of events to train tree on
    params_dict['colsample_bytree']   = 0.7 # fraction of features to train tree on
    params_dict['objective']          = 'binary:logistic' # objective function
    params_dict['alpha']              = 10
    params_dict['lambda']             = 0.3
    
    num_trees = 60  # number of trees to make

    # dictionaries for BDT training
    model_lowEta_dict= {}
    fpr_train_lowEta_dict = {}
    tpr_train_lowEta_dict = {}
    fpr_test_lowEta_dict = {}
    tpr_test_lowEta_dict = {}
    threshold_train_lowEta_dict = {}
    threshold_test_lowEta_dict = {}
    threshold_validation_lowEta_dict = {}
    testAuroc_lowEta_dict = {}
    trainAuroc_lowEta_dict = {}
    fpr_validation_lowEta_dict = {}
    tpr_validation_lowEta_dict = {}

    model_midEta_dict= {}
    fpr_train_midEta_dict = {}
    tpr_train_midEta_dict = {}
    fpr_test_midEta_dict = {}
    tpr_test_midEta_dict = {}
    threshold_train_midEta_dict = {}
    threshold_test_midEta_dict = {}
    threshold_validation_midEta_dict = {}
    testAuroc_midEta_dict = {}
    trainAuroc_midEta_dict = {}
    fpr_validation_midEta_dict = {}
    tpr_validation_midEta_dict = {}

    model_highEta_dict= {}
    fpr_train_highEta_dict = {}
    tpr_train_highEta_dict = {}
    fpr_test_highEta_dict = {}
    tpr_test_highEta_dict = {}
    threshold_train_highEta_dict = {}
    threshold_test_highEta_dict = {}
    threshold_validation_highEta_dict = {}
    testAuroc_highEta_dict = {}
    trainAuroc_highEta_dict = {}
    fpr_validation_highEta_dict = {}
    tpr_validation_highEta_dict = {}

    model_vhighEta_dict= {}
    fpr_train_vhighEta_dict = {}
    tpr_train_vhighEta_dict = {}
    fpr_test_vhighEta_dict = {}
    tpr_test_vhighEta_dict = {}
    threshold_train_vhighEta_dict = {}
    threshold_test_vhighEta_dict = {}
    threshold_validation_vhighEta_dict = {}
    testAuroc_vhighEta_dict = {}
    trainAuroc_vhighEta_dict = {}
    fpr_validation_vhighEta_dict = {}
    tpr_validation_vhighEta_dict = {}

    # working points dictionaries
    bdtWP99_lowEta_dict = {}
    bdtWP95_lowEta_dict = {}
    bdtWP90_lowEta_dict = {}

    bdtWP99_midEta_dict = {}
    bdtWP95_midEta_dict = {}
    bdtWP90_midEta_dict = {}

    bdtWP99_highEta_dict = {}
    bdtWP95_highEta_dict = {}
    bdtWP90_highEta_dict = {}

    bdtWP99_vhighEta_dict = {}
    bdtWP95_vhighEta_dict = {}
    bdtWP90_vhighEta_dict = {}

    # colors to use for plotting
    colors_dict = {
        'threshold'    : 'blue',
        'mixed'        : 'fuchsia'
    }

    # legend to use for plotting
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'mixed'        : 'Mixed BC + STC'
    }


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting PU rejection for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()

        ######################### SECTION DEPENDING ON ETA #########################
        
        dfTraining_lowEta = dfTraining_dict[name].query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)
        dfValidation_lowEta = dfValidation_dict[name].query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)

        dfTraining_midEta = dfTraining_dict[name].query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)
        dfValidation_midEta = dfValidation_dict[name].query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)

        dfTraining_highEta = dfTraining_dict[name].query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)
        dfValidation_highEta = dfValidation_dict[name].query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)

        dfTraining_vhighEta = dfTraining_dict[name].query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)
        dfValidation_vhighEta = dfValidation_dict[name].query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)

        ######################### SELECT EVENTS FOR TRAINING #########################

        dfTr_lowEta = dfTraining_lowEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True)    # take all the taus and all the PU
        dfVal_lowEta = dfValidation_lowEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True) # "

        dfTr_midEta = dfTraining_midEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True)    # take all the taus and all the PU
        dfVal_midEta = dfValidation_midEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True) # "

        dfTr_highEta = dfTraining_highEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True)    # take all the taus and all the PU
        dfVal_highEta = dfValidation_highEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True) # "

        dfTr_vhighEta = dfTraining_vhighEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True)    # take all the taus and all the PU
        dfVal_vhighEta = dfValidation_vhighEta.query('gentau_pid==1 or cl3d_isbestmatch==False').copy(deep=True) # "
        
        dfQCDTr_lowEta = dfTraining_lowEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)
        dfQCDVal_lowEta = dfValidation_lowEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)

        dfQCDTr_midEta = dfTraining_midEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)
        dfQCDVal_midEta = dfValidation_midEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)

        dfQCDTr_highEta = dfTraining_highEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)
        dfQCDVal_highEta = dfValidation_highEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)

        dfQCDTr_vhighEta = dfTraining_vhighEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)
        dfQCDVal_vhighEta = dfValidation_vhighEta.query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)

        ######################### TRAINING OF BDT #########################

        #------------- lowEta region -------------#
        model_lowEta_dict[name], fpr_train_lowEta_dict[name], tpr_train_lowEta_dict[name], threshold_train_lowEta_dict[name], fpr_test_lowEta_dict[name], tpr_test_lowEta_dict[name], threshold_test_lowEta_dict[name], testAuroc_lowEta_dict[name], trainAuroc_lowEta_dict[name] = train_xgb(dfTr_lowEta, features, output, params_dict, num_trees)
        
        print('\n** INFO: training and test AUROC low eta region:')
        print('  -- training AUROC: {0}'.format(trainAuroc_lowEta_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_lowEta_dict[name]))

        save_obj(model_lowEta_dict[name], outFile_model_lowEta_dict[name])
        save_obj(fpr_train_lowEta_dict[name], outFile_fpr_train_lowEta_dict[name])
        save_obj(tpr_train_lowEta_dict[name], outFile_tpr_train_lowEta_dict[name])

        bdtWP99_lowEta_dict[name] = np.interp(0.99, tpr_train_lowEta_dict[name], threshold_train_lowEta_dict[name])
        bdtWP95_lowEta_dict[name] = np.interp(0.95, tpr_train_lowEta_dict[name], threshold_train_lowEta_dict[name])
        bdtWP90_lowEta_dict[name] = np.interp(0.90, tpr_train_lowEta_dict[name], threshold_train_lowEta_dict[name])

        save_obj(bdtWP99_lowEta_dict[name], outFile_WP99_lowEta_dict[name])
        save_obj(bdtWP95_lowEta_dict[name], outFile_WP95_lowEta_dict[name])
        save_obj(bdtWP90_lowEta_dict[name], outFile_WP90_lowEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n** INFO: score threshold and associated bkg efficiency for low eta region')
        print('         BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_lowEta_dict[name], bdtWP95_lowEta_dict[name], bdtWP90_lowEta_dict[name]))
        print('         train bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_lowEta_dict[name], fpr_train_lowEta_dict[name]),np.interp(0.95, tpr_train_lowEta_dict[name], fpr_train_lowEta_dict[name]),np.interp(0.90, tpr_train_lowEta_dict[name], fpr_train_lowEta_dict[name])))
        print('         test bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_test_lowEta_dict[name], fpr_test_lowEta_dict[name]),np.interp(0.95, tpr_test_lowEta_dict[name], fpr_test_lowEta_dict[name]),np.interp(0.90, tpr_test_lowEta_dict[name], fpr_test_lowEta_dict[name])))

        #------------- midEta region -------------#
        model_midEta_dict[name], fpr_train_midEta_dict[name], tpr_train_midEta_dict[name], threshold_train_midEta_dict[name], fpr_test_midEta_dict[name], tpr_test_midEta_dict[name], threshold_test_midEta_dict[name], testAuroc_midEta_dict[name], trainAuroc_midEta_dict[name] = train_xgb(dfTr_midEta, features, output, params_dict, num_trees)

        print('\n** INFO: training and test AUROC mid eta region:')
        print('  -- training AUROC: {0}'.format(trainAuroc_midEta_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_midEta_dict[name]))

        save_obj(model_midEta_dict[name], outFile_model_midEta_dict[name])
        save_obj(fpr_train_midEta_dict[name], outFile_fpr_train_midEta_dict[name])
        save_obj(tpr_train_midEta_dict[name], outFile_tpr_train_midEta_dict[name])

        bdtWP99_midEta_dict[name] = np.interp(0.99, tpr_train_midEta_dict[name], threshold_train_midEta_dict[name])
        bdtWP95_midEta_dict[name] = np.interp(0.95, tpr_train_midEta_dict[name], threshold_train_midEta_dict[name])
        bdtWP90_midEta_dict[name] = np.interp(0.90, tpr_train_midEta_dict[name], threshold_train_midEta_dict[name])

        save_obj(bdtWP99_midEta_dict[name], outFile_WP99_midEta_dict[name])
        save_obj(bdtWP95_midEta_dict[name], outFile_WP95_midEta_dict[name])
        save_obj(bdtWP90_midEta_dict[name], outFile_WP90_midEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n** INFO: score threshold and associated bkg efficiency for mid eta region')
        print('         BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_midEta_dict[name], bdtWP95_midEta_dict[name], bdtWP90_midEta_dict[name]))
        print('         train bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_midEta_dict[name], fpr_train_midEta_dict[name]),np.interp(0.95, tpr_train_midEta_dict[name], fpr_train_midEta_dict[name]),np.interp(0.90, tpr_train_midEta_dict[name], fpr_train_midEta_dict[name])))
        print('         test bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_test_midEta_dict[name], fpr_test_midEta_dict[name]),np.interp(0.95, tpr_test_midEta_dict[name], fpr_test_midEta_dict[name]),np.interp(0.90, tpr_test_midEta_dict[name], fpr_test_midEta_dict[name])))

        #------------- highEta region -------------#
        model_highEta_dict[name], fpr_train_highEta_dict[name], tpr_train_highEta_dict[name], threshold_train_highEta_dict[name], fpr_test_highEta_dict[name], tpr_test_highEta_dict[name], threshold_test_highEta_dict[name], testAuroc_highEta_dict[name], trainAuroc_highEta_dict[name] = train_xgb(dfTr_highEta, features, output, params_dict, num_trees)

        print('\n** INFO: training and test AUROC high eta region:')
        print('  -- training AUROC: {0}'.format(trainAuroc_highEta_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_highEta_dict[name]))

        save_obj(model_highEta_dict[name], outFile_model_highEta_dict[name])
        save_obj(fpr_train_highEta_dict[name], outFile_fpr_train_highEta_dict[name])
        save_obj(tpr_train_highEta_dict[name], outFile_tpr_train_highEta_dict[name])

        bdtWP99_highEta_dict[name] = np.interp(0.99, tpr_train_highEta_dict[name], threshold_train_highEta_dict[name])
        bdtWP95_highEta_dict[name] = np.interp(0.95, tpr_train_highEta_dict[name], threshold_train_highEta_dict[name])
        bdtWP90_highEta_dict[name] = np.interp(0.90, tpr_train_highEta_dict[name], threshold_train_highEta_dict[name])

        save_obj(bdtWP99_highEta_dict[name], outFile_WP99_highEta_dict[name])
        save_obj(bdtWP95_highEta_dict[name], outFile_WP95_highEta_dict[name])
        save_obj(bdtWP90_highEta_dict[name], outFile_WP90_highEta_dict[name])
        
        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n** INFO: score threshold and associated bkg efficiency for high eta region')
        print('         BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_highEta_dict[name], bdtWP95_highEta_dict[name], bdtWP90_highEta_dict[name]))
        print('         train bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_highEta_dict[name], fpr_train_highEta_dict[name]),np.interp(0.95, tpr_train_highEta_dict[name], fpr_train_highEta_dict[name]),np.interp(0.90, tpr_train_highEta_dict[name], fpr_train_highEta_dict[name])))
        print('         test bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_test_highEta_dict[name], fpr_test_highEta_dict[name]),np.interp(0.95, tpr_test_highEta_dict[name], fpr_test_highEta_dict[name]),np.interp(0.90, tpr_test_highEta_dict[name], fpr_test_highEta_dict[name])))

        #------------- vhighEta region -------------#
        model_vhighEta_dict[name], fpr_train_vhighEta_dict[name], tpr_train_vhighEta_dict[name], threshold_train_vhighEta_dict[name], fpr_test_vhighEta_dict[name], tpr_test_vhighEta_dict[name], threshold_test_vhighEta_dict[name], testAuroc_vhighEta_dict[name], trainAuroc_vhighEta_dict[name] = train_xgb(dfTr_vhighEta, features, output, params_dict, num_trees)

        print('\n** INFO: training and test AUROC high eta region:')
        print('  -- training AUROC: {0}'.format(trainAuroc_vhighEta_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_vhighEta_dict[name]))

        save_obj(model_vhighEta_dict[name], outFile_model_vhighEta_dict[name])
        save_obj(fpr_train_vhighEta_dict[name], outFile_fpr_train_vhighEta_dict[name])
        save_obj(tpr_train_vhighEta_dict[name], outFile_tpr_train_vhighEta_dict[name])

        bdtWP99_vhighEta_dict[name] = np.interp(0.99, tpr_train_vhighEta_dict[name], threshold_train_vhighEta_dict[name])
        bdtWP95_vhighEta_dict[name] = np.interp(0.95, tpr_train_vhighEta_dict[name], threshold_train_vhighEta_dict[name])
        bdtWP90_vhighEta_dict[name] = np.interp(0.90, tpr_train_vhighEta_dict[name], threshold_train_vhighEta_dict[name])

        save_obj(bdtWP99_vhighEta_dict[name], outFile_WP99_vhighEta_dict[name])
        save_obj(bdtWP95_vhighEta_dict[name], outFile_WP95_vhighEta_dict[name])
        save_obj(bdtWP90_vhighEta_dict[name], outFile_WP90_vhighEta_dict[name])
        
        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n** INFO: score threshold and associated bkg efficiency for high eta region')
        print('         BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_vhighEta_dict[name], bdtWP95_vhighEta_dict[name], bdtWP90_vhighEta_dict[name]))
        print('         train bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_vhighEta_dict[name], fpr_train_vhighEta_dict[name]),np.interp(0.95, tpr_train_vhighEta_dict[name], fpr_train_vhighEta_dict[name]),np.interp(0.90, tpr_train_vhighEta_dict[name], fpr_train_vhighEta_dict[name])))
        print('         test bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_test_vhighEta_dict[name], fpr_test_vhighEta_dict[name]),np.interp(0.95, tpr_test_vhighEta_dict[name], fpr_test_vhighEta_dict[name]),np.interp(0.90, tpr_test_vhighEta_dict[name], fpr_test_vhighEta_dict[name])))


        ######################### VALIDATION OF BDT #########################

        #------------- lowEta region -------------#
        full_lowEta = xgb.DMatrix(data=dfVal_lowEta[features], label=dfVal_lowEta[output], feature_names=features)
        dfVal_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)

        fpr_validation_lowEta_dict[name], tpr_validation_lowEta_dict[name], threshold_validation_lowEta_dict[name] = metrics.roc_curve(dfVal_lowEta[output], dfVal_lowEta['cl3d_pubdt_score'])
        auroc_validation_lowEta = metrics.roc_auc_score(dfVal_lowEta['gentau_pid'],dfVal_lowEta['cl3d_pubdt_score'])

        #------------- midEta region -------------#
        full_midEta = xgb.DMatrix(data=dfVal_midEta[features], label=dfVal_midEta[output], feature_names=features)
        dfVal_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)

        fpr_validation_midEta_dict[name], tpr_validation_midEta_dict[name], threshold_validation_midEta_dict[name] = metrics.roc_curve(dfVal_midEta[output], dfVal_midEta['cl3d_pubdt_score'])
        auroc_validation_midEta = metrics.roc_auc_score(dfVal_midEta['gentau_pid'],dfVal_midEta['cl3d_pubdt_score'])

        #------------- highEta region -------------#
        full_highEta = xgb.DMatrix(data=dfVal_highEta[features], label=dfVal_highEta[output], feature_names=features)
        dfVal_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)

        fpr_validation_highEta_dict[name], tpr_validation_highEta_dict[name], threshold_validation_highEta_dict[name] = metrics.roc_curve(dfVal_highEta[output], dfVal_highEta['cl3d_pubdt_score'])
        auroc_validation_highEta = metrics.roc_auc_score(dfVal_highEta['gentau_pid'],dfVal_highEta['cl3d_pubdt_score'])

        #------------- vhighEta region -------------#
        full_vhighEta = xgb.DMatrix(data=dfVal_vhighEta[features], label=dfVal_vhighEta[output], feature_names=features)
        dfVal_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)

        fpr_validation_vhighEta_dict[name], tpr_validation_vhighEta_dict[name], threshold_validation_vhighEta_dict[name] = metrics.roc_curve(dfVal_vhighEta[output], dfVal_vhighEta['cl3d_pubdt_score'])
        auroc_validation_vhighEta = metrics.roc_auc_score(dfVal_vhighEta['gentau_pid'],dfVal_vhighEta['cl3d_pubdt_score'])


        print('\n** INFO: validation of the BDT')
        print('     -- validation AUC low eta region: {0}'.format(auroc_validation_lowEta))
        print('     -- validation AUC mid eta region: {0}'.format(auroc_validation_midEta))
        print('     -- validation AUC high eta region: {0}'.format(auroc_validation_highEta))
        print('     -- validation AUC high eta region: {0}'.format(auroc_validation_vhighEta))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        #------------- lowEta region -------------#
        full_lowEta = xgb.DMatrix(data=dfTraining_lowEta[features], label=dfTraining_lowEta[output], feature_names=features)
        dfTraining_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)
        dfTraining_lowEta['cl3d_pubdt_passWP99'] = dfTraining_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfTraining_lowEta['cl3d_pubdt_passWP95'] = dfTraining_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfTraining_lowEta['cl3d_pubdt_passWP90'] = dfTraining_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        full_lowEta = xgb.DMatrix(data=dfValidation_lowEta[features], label=dfValidation_lowEta[output], feature_names=features)
        dfValidation_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)
        dfValidation_lowEta['cl3d_pubdt_passWP99'] = dfValidation_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfValidation_lowEta['cl3d_pubdt_passWP95'] = dfValidation_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfValidation_lowEta['cl3d_pubdt_passWP90'] = dfValidation_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        full_lowEta = xgb.DMatrix(data=dfTr_lowEta[features], label=dfTr_lowEta[output], feature_names=features)
        dfTr_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)
        dfTr_lowEta['cl3d_pubdt_passWP99'] = dfTr_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfTr_lowEta['cl3d_pubdt_passWP95'] = dfTr_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfTr_lowEta['cl3d_pubdt_passWP90'] = dfTr_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        dfVal_lowEta['cl3d_pubdt_passWP99'] = dfVal_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfVal_lowEta['cl3d_pubdt_passWP95'] = dfVal_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfVal_lowEta['cl3d_pubdt_passWP90'] = dfVal_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        full_lowEta = xgb.DMatrix(data=dfQCDTr_lowEta[features], label=dfQCDTr_lowEta[output], feature_names=features)
        dfQCDTr_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)
        dfQCDTr_lowEta['cl3d_pubdt_passWP99'] = dfQCDTr_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfQCDTr_lowEta['cl3d_pubdt_passWP95'] = dfQCDTr_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfQCDTr_lowEta['cl3d_pubdt_passWP90'] = dfQCDTr_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        full_lowEta = xgb.DMatrix(data=dfQCDVal_lowEta[features], label=dfQCDVal_lowEta[output], feature_names=features)
        dfQCDVal_lowEta['cl3d_pubdt_score'] = model_lowEta_dict[name].predict(full_lowEta)
        dfQCDVal_lowEta['cl3d_pubdt_passWP99'] = dfQCDVal_lowEta['cl3d_pubdt_score'] > bdtWP99_lowEta_dict[name]
        dfQCDVal_lowEta['cl3d_pubdt_passWP95'] = dfQCDVal_lowEta['cl3d_pubdt_score'] > bdtWP95_lowEta_dict[name]
        dfQCDVal_lowEta['cl3d_pubdt_passWP90'] = dfQCDVal_lowEta['cl3d_pubdt_score'] > bdtWP90_lowEta_dict[name]

        QCDtot_lowEta = pd.concat([dfQCDTr_lowEta,dfQCDVal_lowEta],sort=False)
        QCD99_lowEta = QCDtot_lowEta.query('cl3d_pubdt_passWP99==True')
        QCD95_lowEta = QCDtot_lowEta.query('cl3d_pubdt_passWP95==True')
        QCD90_lowEta = QCDtot_lowEta.query('cl3d_pubdt_passWP90==True')

        print('\n**INFO: QCD cluster passing the PU rejection in the low eta region:')
        print('     -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99_lowEta['cl3d_pubdt_passWP99'].count())/float(QCDtot_lowEta['cl3d_pubdt_passWP99'].count())*100,2)))
        print('     -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95_lowEta['cl3d_pubdt_passWP95'].count())/float(QCDtot_lowEta['cl3d_pubdt_passWP95'].count())*100,2)))
        print('     -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90_lowEta['cl3d_pubdt_passWP90'].count())/float(QCDtot_lowEta['cl3d_pubdt_passWP90'].count())*100,2)))

        #------------- midEta region -------------#
        full_midEta = xgb.DMatrix(data=dfTraining_midEta[features], label=dfTraining_midEta[output], feature_names=features)
        dfTraining_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)
        dfTraining_midEta['cl3d_pubdt_passWP99'] = dfTraining_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfTraining_midEta['cl3d_pubdt_passWP95'] = dfTraining_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfTraining_midEta['cl3d_pubdt_passWP90'] = dfTraining_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        full_midEta = xgb.DMatrix(data=dfValidation_midEta[features], label=dfValidation_midEta[output], feature_names=features)
        dfValidation_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)
        dfValidation_midEta['cl3d_pubdt_passWP99'] = dfValidation_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfValidation_midEta['cl3d_pubdt_passWP95'] = dfValidation_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfValidation_midEta['cl3d_pubdt_passWP90'] = dfValidation_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        full_midEta = xgb.DMatrix(data=dfTr_midEta[features], label=dfTr_midEta[output], feature_names=features)
        dfTr_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)
        dfTr_midEta['cl3d_pubdt_passWP99'] = dfTr_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfTr_midEta['cl3d_pubdt_passWP95'] = dfTr_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfTr_midEta['cl3d_pubdt_passWP90'] = dfTr_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        dfVal_midEta['cl3d_pubdt_passWP99'] = dfVal_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfVal_midEta['cl3d_pubdt_passWP95'] = dfVal_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfVal_midEta['cl3d_pubdt_passWP90'] = dfVal_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        full_midEta = xgb.DMatrix(data=dfQCDTr_midEta[features], label=dfQCDTr_midEta[output], feature_names=features)
        dfQCDTr_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)
        dfQCDTr_midEta['cl3d_pubdt_passWP99'] = dfQCDTr_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfQCDTr_midEta['cl3d_pubdt_passWP95'] = dfQCDTr_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfQCDTr_midEta['cl3d_pubdt_passWP90'] = dfQCDTr_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        full_midEta = xgb.DMatrix(data=dfQCDVal_midEta[features], label=dfQCDVal_midEta[output], feature_names=features)
        dfQCDVal_midEta['cl3d_pubdt_score'] = model_midEta_dict[name].predict(full_midEta)
        dfQCDVal_midEta['cl3d_pubdt_passWP99'] = dfQCDVal_midEta['cl3d_pubdt_score'] > bdtWP99_midEta_dict[name]
        dfQCDVal_midEta['cl3d_pubdt_passWP95'] = dfQCDVal_midEta['cl3d_pubdt_score'] > bdtWP95_midEta_dict[name]
        dfQCDVal_midEta['cl3d_pubdt_passWP90'] = dfQCDVal_midEta['cl3d_pubdt_score'] > bdtWP90_midEta_dict[name]

        QCDtot_midEta = pd.concat([dfQCDTr_midEta,dfQCDVal_midEta],sort=False)
        QCD99_midEta = QCDtot_midEta.query('cl3d_pubdt_passWP99==True')
        QCD95_midEta = QCDtot_midEta.query('cl3d_pubdt_passWP95==True')
        QCD90_midEta = QCDtot_midEta.query('cl3d_pubdt_passWP90==True')

        print('\n**INFO: QCD cluster passing the PU rejection in the mid eta region:')
        print('     -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99_midEta['cl3d_pubdt_passWP99'].count())/float(QCDtot_midEta['cl3d_pubdt_passWP99'].count())*100,2)))
        print('     -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95_midEta['cl3d_pubdt_passWP95'].count())/float(QCDtot_midEta['cl3d_pubdt_passWP95'].count())*100,2)))
        print('     -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90_midEta['cl3d_pubdt_passWP90'].count())/float(QCDtot_midEta['cl3d_pubdt_passWP90'].count())*100,2)))

        #------------- highEta region -------------#
        full_highEta = xgb.DMatrix(data=dfTraining_highEta[features], label=dfTraining_highEta[output], feature_names=features)
        dfTraining_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)
        dfTraining_highEta['cl3d_pubdt_passWP99'] = dfTraining_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfTraining_highEta['cl3d_pubdt_passWP95'] = dfTraining_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfTraining_highEta['cl3d_pubdt_passWP90'] = dfTraining_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        full_highEta = xgb.DMatrix(data=dfValidation_highEta[features], label=dfValidation_highEta[output], feature_names=features)
        dfValidation_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)
        dfValidation_highEta['cl3d_pubdt_passWP99'] = dfValidation_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfValidation_highEta['cl3d_pubdt_passWP95'] = dfValidation_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfValidation_highEta['cl3d_pubdt_passWP90'] = dfValidation_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        full_highEta = xgb.DMatrix(data=dfTr_highEta[features], label=dfTr_highEta[output], feature_names=features)
        dfTr_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)
        dfTr_highEta['cl3d_pubdt_passWP99'] = dfTr_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfTr_highEta['cl3d_pubdt_passWP95'] = dfTr_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfTr_highEta['cl3d_pubdt_passWP90'] = dfTr_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        dfVal_highEta['cl3d_pubdt_passWP99'] = dfVal_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfVal_highEta['cl3d_pubdt_passWP95'] = dfVal_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfVal_highEta['cl3d_pubdt_passWP90'] = dfVal_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        full_highEta = xgb.DMatrix(data=dfQCDTr_highEta[features], label=dfQCDTr_highEta[output], feature_names=features)
        dfQCDTr_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)
        dfQCDTr_highEta['cl3d_pubdt_passWP99'] = dfQCDTr_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfQCDTr_highEta['cl3d_pubdt_passWP95'] = dfQCDTr_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfQCDTr_highEta['cl3d_pubdt_passWP90'] = dfQCDTr_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        full_highEta = xgb.DMatrix(data=dfQCDVal_highEta[features], label=dfQCDVal_highEta[output], feature_names=features)
        dfQCDVal_highEta['cl3d_pubdt_score'] = model_highEta_dict[name].predict(full_highEta)
        dfQCDVal_highEta['cl3d_pubdt_passWP99'] = dfQCDVal_highEta['cl3d_pubdt_score'] > bdtWP99_highEta_dict[name]
        dfQCDVal_highEta['cl3d_pubdt_passWP95'] = dfQCDVal_highEta['cl3d_pubdt_score'] > bdtWP95_highEta_dict[name]
        dfQCDVal_highEta['cl3d_pubdt_passWP90'] = dfQCDVal_highEta['cl3d_pubdt_score'] > bdtWP90_highEta_dict[name]

        QCDtot_highEta = pd.concat([dfQCDTr_highEta,dfQCDVal_highEta],sort=False)
        QCD99_highEta = QCDtot_highEta.query('cl3d_pubdt_passWP99==True').copy(deep=True)
        QCD95_highEta = QCDtot_highEta.query('cl3d_pubdt_passWP95==True').copy(deep=True)
        QCD90_highEta = QCDtot_highEta.query('cl3d_pubdt_passWP90==True').copy(deep=True)

        print('\n**INFO: QCD cluster passing the PU rejection in the high eta region:')
        print('     -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99_highEta.shape[0])/float(QCDtot_highEta.shape[0])*100,2)))
        print('     -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95_highEta.shape[0])/float(QCDtot_highEta.shape[0])*100,2)))
        print('     -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90_highEta.shape[0])/float(QCDtot_highEta.shape[0])*100,2)))

        #------------- vhighEta region -------------#
        full_vhighEta = xgb.DMatrix(data=dfTraining_vhighEta[features], label=dfTraining_vhighEta[output], feature_names=features)
        dfTraining_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)
        dfTraining_vhighEta['cl3d_pubdt_passWP99'] = dfTraining_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfTraining_vhighEta['cl3d_pubdt_passWP95'] = dfTraining_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfTraining_vhighEta['cl3d_pubdt_passWP90'] = dfTraining_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        full_vhighEta = xgb.DMatrix(data=dfValidation_vhighEta[features], label=dfValidation_vhighEta[output], feature_names=features)
        dfValidation_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)
        dfValidation_vhighEta['cl3d_pubdt_passWP99'] = dfValidation_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfValidation_vhighEta['cl3d_pubdt_passWP95'] = dfValidation_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfValidation_vhighEta['cl3d_pubdt_passWP90'] = dfValidation_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        full_vhighEta = xgb.DMatrix(data=dfTr_vhighEta[features], label=dfTr_vhighEta[output], feature_names=features)
        dfTr_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)
        dfTr_vhighEta['cl3d_pubdt_passWP99'] = dfTr_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfTr_vhighEta['cl3d_pubdt_passWP95'] = dfTr_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfTr_vhighEta['cl3d_pubdt_passWP90'] = dfTr_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        dfVal_vhighEta['cl3d_pubdt_passWP99'] = dfVal_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfVal_vhighEta['cl3d_pubdt_passWP95'] = dfVal_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfVal_vhighEta['cl3d_pubdt_passWP90'] = dfVal_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        full_vhighEta = xgb.DMatrix(data=dfQCDTr_vhighEta[features], label=dfQCDTr_vhighEta[output], feature_names=features)
        dfQCDTr_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)
        dfQCDTr_vhighEta['cl3d_pubdt_passWP99'] = dfQCDTr_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfQCDTr_vhighEta['cl3d_pubdt_passWP95'] = dfQCDTr_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfQCDTr_vhighEta['cl3d_pubdt_passWP90'] = dfQCDTr_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        full_vhighEta = xgb.DMatrix(data=dfQCDVal_vhighEta[features], label=dfQCDVal_vhighEta[output], feature_names=features)
        dfQCDVal_vhighEta['cl3d_pubdt_score'] = model_vhighEta_dict[name].predict(full_vhighEta)
        dfQCDVal_vhighEta['cl3d_pubdt_passWP99'] = dfQCDVal_vhighEta['cl3d_pubdt_score'] > bdtWP99_vhighEta_dict[name]
        dfQCDVal_vhighEta['cl3d_pubdt_passWP95'] = dfQCDVal_vhighEta['cl3d_pubdt_score'] > bdtWP95_vhighEta_dict[name]
        dfQCDVal_vhighEta['cl3d_pubdt_passWP90'] = dfQCDVal_vhighEta['cl3d_pubdt_score'] > bdtWP90_vhighEta_dict[name]

        QCDtot_vhighEta = pd.concat([dfQCDTr_vhighEta,dfQCDVal_vhighEta],sort=False)
        QCD99_vhighEta = QCDtot_vhighEta.query('cl3d_pubdt_passWP99==True').copy(deep=True)
        QCD95_vhighEta = QCDtot_vhighEta.query('cl3d_pubdt_passWP95==True').copy(deep=True)
        QCD90_vhighEta = QCDtot_vhighEta.query('cl3d_pubdt_passWP90==True').copy(deep=True)

        print('\n**INFO: QCD cluster passing the PU rejection in the very high eta region:')
        print('     -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99_vhighEta.shape[0])/float(QCDtot_vhighEta.shape[0])*100,2)))
        print('     -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95_vhighEta.shape[0])/float(QCDtot_vhighEta.shape[0])*100,2)))
        print('     -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90_vhighEta.shape[0])/float(QCDtot_vhighEta.shape[0])*100,2)))



        ######################### REMERGE ALL THE ETA REGIONS IN SINGLE FILES #########################

        dfTraining_dict[name] = pd.concat([dfTraining_lowEta, dfTraining_midEta, dfTraining_highEta, dfTraining_vhighEta], sort=False)
        dfValidation_dict[name] = pd.concat([dfValidation_lowEta, dfValidation_midEta, dfValidation_highEta, dfValidation_vhighEta], sort=False)

        TOT = pd.concat([dfTraining_dict[name],dfValidation_dict[name]],sort=False).query('gentau_pid==0 and gentau_decayMode!=-2')
        TOT99 = TOT.query('gentau_pid==0 and gentau_decayMode!=-2 and cl3d_pubdt_passWP99==True').copy(deep=True)
        TOT95 = TOT.query('gentau_pid==0 and gentau_decayMode!=-2 and cl3d_pubdt_passWP95==True').copy(deep=True)
        TOT90 = TOT.query('gentau_pid==0 and gentau_decayMode!=-2 and cl3d_pubdt_passWP90==True').copy(deep=True)

        print('\nOVERALL BKG EFFICIENCIES:')
        print('     at 0.99 sgn efficiency: {0}%'.format(round(float(TOT99.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.95 sgn efficiency: {0}%'.format(round(float(TOT95.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.90 sgn efficiency: {0}%'.format(round(float(TOT90.shape[0])/float(TOT.shape[0])*100,2)))

        ######################### SAVE FILES #########################

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()


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


        ######################### PLOT ROCS, FEATURE IMPORTANCES, AND BDT SCORE #########################

        #------------- lowEta region -------------#
        print('\n** INFO: plotting ROC curves, feature importance and BDT score for low eta region')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_lowEta_dict[name],fpr_train_lowEta_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_lowEta_dict[name],fpr_test_lowEta_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_lowEta_dict[name],fpr_validation_lowEta_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.85,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE - low eta region'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_lowEta_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()

        plt.figure(figsize=(10,10))
        importance = model_lowEta_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,450)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_lowEta_importances_'+name+'.pdf')
        plt.close()

        dfNu = dfVal_lowEta.query('gentau_pid==0')
        dfTau = dfVal_lowEta.query('gentau_pid==1')

        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='Nu PU=200', density=True)
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='blue', histtype='step', lw=2, label='Tau PU=200', density=True)
        plt.hist(dfQCDVal_lowEta['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='QCD PU=200', density=True)
        plt.legend(loc = 'upper right')
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('PU rejection BDT score - low eta region')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_lowEta_score_'+name+'.pdf')
        plt.close()

        del dfNu, dfTau

        #------------- midEta region -------------#
        print('\n** INFO: plotting ROC curves, feature importance and BDT score for mid eta region')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_midEta_dict[name],fpr_train_midEta_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_midEta_dict[name],fpr_test_midEta_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_midEta_dict[name],fpr_validation_midEta_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.85,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE - mid eta region'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_midEta_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()

        plt.figure(figsize=(10,10))
        importance = model_midEta_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,450)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_midEta_importances_'+name+'.pdf')
        plt.close()

        dfNu = dfVal_midEta.query('gentau_pid==0')
        dfTau = dfVal_midEta.query('gentau_pid==1')

        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='Nu PU=200', density=True)
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='blue', histtype='step', lw=2, label='Tau PU=200', density=True)
        plt.hist(dfQCDVal_midEta['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='QCD PU=200', density=True)
        plt.legend(loc = 'upper right')
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('PU rejection BDT score - mid eta region')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_midEta_score_'+name+'.pdf')
        plt.close()

        del dfNu, dfTau

        #------------- highEta region -------------#
        print('\n** INFO: plotting ROC curves, feature importance and BDT score for high eta region')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_highEta_dict[name],fpr_train_highEta_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_highEta_dict[name],fpr_test_highEta_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_highEta_dict[name],fpr_validation_highEta_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.85,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE - high eta region'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_highEta_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()

        plt.figure(figsize=(10,10))
        importance = model_highEta_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,450)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_highEta_importances_'+name+'.pdf')
        plt.close()

        dfNu = dfVal_highEta.query('gentau_pid==0')
        dfTau = dfVal_highEta.query('gentau_pid==1')

        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='Nu PU=200', density=True)
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='blue', histtype='step', lw=2, label='Tau PU=200', density=True)
        plt.hist(dfQCDVal_highEta['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='QCD PU=200', density=True)
        plt.legend(loc = 'upper right')
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('PU rejection BDT score - high eta region')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_highEta_score_'+name+'.pdf')
        plt.close()

        del dfNu, dfTau

        #------------- vhighEta region -------------#
        print('\n** INFO: plotting ROC curves, feature importance and BDT score for very high eta region')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_vhighEta_dict[name],fpr_train_vhighEta_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_vhighEta_dict[name],fpr_test_vhighEta_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_vhighEta_dict[name],fpr_validation_vhighEta_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.85,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE - very high eta region'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_vhighEta_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()

        plt.figure(figsize=(10,10))
        importance = model_vhighEta_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,450)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_vhighEta_importances_'+name+'.pdf')
        plt.close()

        dfNu = dfVal_vhighEta.query('gentau_pid==0')
        dfTau = dfVal_vhighEta.query('gentau_pid==1')

        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='Nu PU=200', density=True)
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='blue', histtype='step', lw=2, label='Tau PU=200', density=True)
        plt.hist(dfQCDVal_vhighEta['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='QCD PU=200', density=True)
        plt.legend(loc = 'upper right')
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('PU rejection BDT score - very high eta region')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_vhighEta_score_'+name+'.pdf')
        plt.close()

        del dfNu, dfTau

       
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    features_dict = {'cl3d_c1'               : [r'C1 factor value',[0.,2.,10]], 
                     'cl3d_c2'               : [r'C2 factor value',[0.75,2.,15]], 
                     'cl3d_c3'               : [r'C3 factor value',[0.75,2.,15]],
                     'cl3d_pt_c3'            : [r'3D cluster $p_{T}$ after C3',[0.,500.,50]],
                     'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'     : [r'3D cluster shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'       : [r'3D cluster first layer',[0.,20.,20]], 
                     'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_seemax'           : [r'3D cluster max $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_sppmax'           : [r'3D cluster max $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[0.,60.,20]], 
                     'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmax'           : [r'3D cluster max $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[0.,0.01,10]], 
                     'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     'cl3d_meanz'            : [r'3D cluster meanz',[325.,375.,30]], 
                     
    }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfNu = dfTr.query('gentau_pid==0')
        dfTau = dfTr.query('gentau_pid==1')

        os.system('mkdir -p '+plotdir+'/features/')

        for var in features_dict:
            plt.figure(figsize=(8,8))
            plt.hist(dfNu[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='PU events',      color='red',    histtype='step', lw=2, density=True)
            plt.hist(dfTau[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Tau signal',   color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfQCDTr[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background',   color='blue',    histtype='step', lw=2, density=True)
            PUline = mlines.Line2D([], [], color='red',markersize=15, label='PU events',lw=2)
            SGNline = mlines.Line2D([], [], color='limegreen',markersize=15, label='Tau signal',lw=2)
            QCDline = mlines.Line2D([], [], color='blue',markersize=15, label='QCD background',lw=2)
            plt.legend(loc = 'upper right',handles=[PUline,SGNline,QCDline])
            plt.grid(linestyle=':')
            plt.xlabel(features_dict[var][0])
            plt.ylabel(r'Normalized entries')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/features/'+var+'.pdf')
            plt.close()

        del dfNu, dfTau

        ######################### PLOT SINGLE FE OPTION ROCS #########################

        print('\n** INFO: plotting train-test-validation ROC curves')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_dict[name],fpr_train_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_dict[name],fpr_validation_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.85,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()


        ######################### PLOT FEATURE IMPORTANCES #########################

        print('\n** INFO: plotting features importance and score')
        plt.figure(figsize=(10,10))
        importance = model_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain',lw=2)
        plt.xlim(0.0,450)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_importances_'+name+'.pdf')
        plt.close()


        ######################### PLOT BDT SCORE #########################

        dfNu = dfVal.query('gentau_pid==0')
        dfTau = dfVal.query('gentau_pid==1')

        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='Nu PU=200', density=True)
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='blue', histtype='step', lw=2, label='Tau PU=200', density=True)
        plt.hist(dfQCDVal['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='QCD PU=200', density=True)
        plt.legend(loc = 'upper right')
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('PU rejection BDT score')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/PUbdt_score_'+name+'.pdf')
        plt.close()

        del dfNu, dfTau

    ######################### PLOT ALL FE OPTIONS ROCS #########################
    
    print('\n** INFO: plotting test ROC curves')
    plt.figure(figsize=(10,10))
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        roc_auc = metrics.auc(fpr_test_dict[name], tpr_test_dict[name])
        print('  -- for front-end option {0} - TEST AUC (bkgRej. vs. sgnEff.): {1}'.format(feNames_dict[name],roc_auc))
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label=legends_dict[name], color=colors_dict[name],lw=2)

    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    plt.xlim(0.85,1.001)
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
    plt.legend(loc = 'upper left')
    plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.title('Validation ROC curves')
    plt.savefig(plotdir+'/PUbdt_validation_rocs.pdf')
    plt.close()

    print('\n** INFO: finished plotting')
    print('---------------------------------------------------------------------------------------')


if args.doEfficiency:
    print('\n** INFO: calculating efficiency')
    
    matplotlib.rcParams.update({'font.size': 20})
    
    os.system('mkdir -p '+plotdir+'/efficiencies/')

    # efficiencies related dictionaries
    dfTau_dict = {}
    dfTauDM0_dict = {}
    dfTauDM1_dict = {}
    dfTauDM2_dict = {}
    
    effVSpt_Tau_dict = {}
    effVSpt_TauDM0_dict = {}
    effVSpt_TauDM1_dict = {}
    effVSpt_TauDM2_dict = {}

    effVSeta_Tau_dict = {}
    effVSeta_TauDM0_dict = {}
    effVSeta_TauDM1_dict = {}
    effVSeta_TauDM2_dict = {}

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do

        dfTau_dict[name] = pd.concat([dfTraining_dict[name],dfValidation_dict[name]], sort=False)
        dfTau_dict[name].query('gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11', inplace=True)
        # fill all the DM dataframes
        dfTauDM0_dict[name] = dfTau_dict[name].query('gentau_decayMode==0').copy(deep=True)
        dfTauDM1_dict[name] = dfTau_dict[name].query('gentau_decayMode==1').copy(deep=True)
        dfTauDM2_dict[name] = dfTau_dict[name].query('gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)

        effVSpt_Tau_dict[name] = {}
        effVSpt_TauDM0_dict[name] = {}
        effVSpt_TauDM1_dict[name] = {}
        effVSpt_TauDM2_dict[name] = {}

        effVSpt_Tau_dict[name] = dfTau_dict[name].groupby('gentau_bin_pt').mean() # this means are the x bins for the plotting
        effVSpt_TauDM0_dict[name] = dfTauDM0_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_TauDM1_dict[name] = dfTauDM1_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_TauDM2_dict[name] = dfTauDM2_dict[name].groupby('gentau_bin_pt').mean()

        effVSeta_Tau_dict[name] = {}
        effVSeta_TauDM0_dict[name] = {}
        effVSeta_TauDM1_dict[name] = {}
        effVSeta_TauDM2_dict[name] = {}

        effVSeta_Tau_dict[name] = dfTau_dict[name].groupby('gentau_bin_eta').mean() # this means are the x bins for the plotting
        effVSeta_TauDM0_dict[name] = dfTauDM0_dict[name].groupby('gentau_bin_eta').mean()
        effVSeta_TauDM1_dict[name] = dfTauDM1_dict[name].groupby('gentau_bin_eta').mean()
        effVSeta_TauDM2_dict[name] = dfTauDM2_dict[name].groupby('gentau_bin_eta').mean()

        for PUWP in [99,95,90]:
            for threshold in [0,10,20,30]: # consider the no threshold and the 30GeV offline threshold cases
                # calculate efficiency for the TAU datasets --> calculated per bin that will be plotted
                #                                           --> every efficiency_at{threshold} contains the value of the efficiency when applying the specific threshold 
                effVSpt_Tau_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, PUWP)) # --> the output of this will be a series with idx=bin and entry=efficiency
                effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, PUWP))
                effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, PUWP))
                effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, PUWP))

                effVSpt_Tau_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))

                effVSpt_Tau_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))


                effVSeta_Tau_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, PUWP)) # --> the output of this will be a series with idx=bin and entry=efficiency
                effVSeta_TauDM0_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, PUWP))
                effVSeta_TauDM1_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, PUWP))
                effVSeta_TauDM2_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, PUWP))

                effVSeta_Tau_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSeta_TauDM0_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSeta_TauDM1_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))
                effVSeta_TauDM2_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=False))

                effVSeta_Tau_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSeta_TauDM0_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSeta_TauDM1_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))
                effVSeta_TauDM2_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, PUWP, upper=True))


        # colors to use for plotting
        col = {
            99 : 'green',
            95 : 'blue',
            90 : 'red'
        }
        
        x_Tau = effVSpt_Tau_dict[name]['gentau_vis_pt'] # is binned and the value is the mean of the entries per bin
        for threshold in [0,10,20,30]:
            plt.figure(figsize=(10,10))
            for PUWP in [99,95,90]:
                # all values for turnON curves
                eff_30_Tau = effVSpt_Tau_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)]
                eff_err_low_30_Tau = effVSpt_Tau_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)]
                eff_err_up_30_Tau = effVSpt_Tau_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)]

                plt.errorbar(x_Tau,eff_30_Tau,xerr=1,yerr=[eff_err_low_30_Tau,eff_err_up_30_Tau],ls='None',label=r'PUWP = {0}'.format(PUWP),color=col[PUWP],lw=2,marker='o',mec=col[PUWP], alpha=0.5)

                p0 = [np.median(x_Tau), 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_Tau, eff_30_Tau, p0)
                plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color=col[PUWP], lw=1.5, alpha=0.5)
            
            plt.legend(loc = 'lower right')
            txt2 = (r'$E_{T}^{L1,\tau}$ > %i GeV' % (threshold))
            t2 = plt.text(50,0.25, txt2, ha='left')
            t2.set_bbox(dict(facecolor='white', edgecolor='white'))
            plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
            plt.ylabel(r'$\epsilon$')
            plt.title('Efficiency vs pT')
            plt.grid()
            plt.xlim(0, 100)
            plt.ylim(0., 1.10)
            plt.subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/efficiencies/eff_vs_pt_allPUWP_at{0}GeV.pdf'.format(threshold))
            plt.close()

        x_Tau = effVSeta_Tau_dict[name]['gentau_vis_abseta'] # is binned and the value is the mean of the entries per bin
        for threshold in [0,10,20,30]:
            plt.figure(figsize=(10,10))
            for PUWP in [99,95,90]:
                # all values for turnON curves
                eff_30_Tau = effVSeta_Tau_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)]
                eff_err_low_30_Tau = effVSeta_Tau_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)]
                eff_err_up_30_Tau = effVSeta_Tau_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)]

                plt.errorbar(x_Tau,eff_30_Tau,xerr=0.05,yerr=[eff_err_low_30_Tau,eff_err_up_30_Tau],ls='None',label=r'PUWP = {0}'.format(PUWP),color=col[PUWP],lw=2,marker='o',mec=col[PUWP], alpha=0.5)

                #p0 = [-2, -1, -1, -1, -1] # this is an mandatory initial guess for the fit
                #popt, pcov = curve_fit(poly, x_Tau, eff_30_Tau, p0)
                #plt.plot(x_Tau, poly(x_Tau, *popt), '-', label='_', color=col[PUWP], lw=1.5, alpha=0.5)
            
            plt.legend(loc = 'lower left')
            txt2 = (r'$E_{T}^{L1,\tau}$ > %i GeV' % (threshold))
            t2 = plt.text(1.6,0.25, txt2, ha='left')
            t2.set_bbox(dict(facecolor='white', edgecolor='white'))
            plt.xlabel(r'$\eta^{gen,\tau}\ [GeV]$')
            plt.ylabel(r'$\epsilon$')
            plt.title('Efficiency vs pT')
            plt.grid()
            plt.xlim(1.5, 3.0)
            plt.ylim(0., 1.10)
            plt.subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/efficiencies/eff_vs_eta_allPUWP_at{0}GeV.pdf'.format(threshold))
            plt.close()

        x_DM0_Tau = effVSpt_TauDM0_dict[name]['gentau_vis_pt']
        x_DM1_Tau = effVSpt_TauDM1_dict[name]['gentau_vis_pt']
        x_DM2_Tau = effVSpt_TauDM2_dict[name]['gentau_vis_pt']
        for threshold in [0,10,20,30]:
            for PUWP in [99,95,90]:
                # all values for turnON curves
                effTauDM0 = effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)]
                effTauDM1 = effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)]
                effTauDM2 = effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_at{1}GeV'.format(PUWP,threshold)]
                eff_err_low_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)]
                eff_err_low_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)]
                eff_err_low_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_err_low_at{1}GeV'.format(PUWP,threshold)]
                eff_err_up_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)]
                eff_err_up_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)]
                eff_err_up_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_PUWP{0}_err_up_at{1}GeV'.format(PUWP,threshold)]

                plt.figure(figsize=(10,10))
                plt.errorbar(x_DM0_Tau,effTauDM0,xerr=1,yerr=[eff_err_low_TauDM0,eff_err_up_TauDM0],ls='None',label=r'1-prong',color='limegreen',lw=2,marker='o',mec='limegreen')
                plt.errorbar(x_DM1_Tau,effTauDM1,xerr=1,yerr=[eff_err_low_TauDM1,eff_err_up_TauDM1],ls='None',label=r'1-prong + $\pi^{0}$',color='darkorange',lw=2,marker='o',mec='darkorange')
                plt.errorbar(x_DM2_Tau,effTauDM2,xerr=1,yerr=[eff_err_low_TauDM2,eff_err_up_TauDM2],ls='None',label=r'3-prong (+ $\pi^{0}$)',color='fuchsia',lw=2,marker='o',mec='fuchsia')

                p0 = [np.median(x_DM0_Tau), 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_DM0_Tau, effTauDM0, p0)
                plt.plot(x_DM0_Tau, sigmoid(x_DM0_Tau, *popt), '-', label='_', color='limegreen', lw=1.5)

                p0 = [np.median(x_DM1_Tau), 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_DM1_Tau, effTauDM1, p0)
                plt.plot(x_DM1_Tau, sigmoid(x_DM1_Tau, *popt), '-', label='_', color='darkorange', lw=1.5)

                p0 = [np.median(x_DM2_Tau), 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_DM2_Tau, effTauDM2, p0)
                plt.plot(x_DM2_Tau, sigmoid(x_DM2_Tau, *popt), '-', label='_', color='fuchsia', lw=1.5)

                plt.legend(loc = 'lower right')
                # txt = (r'Gen. $\tau$ decay mode:')
                # t = plt.text(63,0.20, txt, ha='left', wrap=True)
                # t.set_bbox(dict(facecolor='white', edgecolor='white'))
                txt2 = (r'$E_{T}^{L1,\tau}$ > %i GeV' % (threshold))
                t2 = plt.text(55,0.25, txt2, ha='left')
                t2.set_bbox(dict(facecolor='white', edgecolor='white'))
                plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
                plt.ylabel(r'$\epsilon$')
                plt.title('Efficiency vs pT - PUWP={0}'.format(PUWP))
                plt.grid()
                plt.xlim(0, 100)
                plt.ylim(0., 1.10)
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/efficiencies/eff_vs_pt_PUWP{0}_at{1}GeV.pdf'.format(PUWP,threshold))
                plt.close()

# restore normal output
sys.stdout = sys.__stdout__
