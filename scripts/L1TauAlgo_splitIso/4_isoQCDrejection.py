import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
import matplotlib
import pickle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
import argparse
import sys
import shap

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


def train_xgb(dfTrain, features, output, hyperparams, num_trees, test_fraction=0.3):    
    X_train, X_test, y_train, y_test = train_test_split(dfTrain[features], dfTrain[output], test_size=test_fraction)

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

def efficiency(group, threshold, ISOWP):
    tot = group.shape[0]
    if ISOWP ==   '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True)].shape[0]
    elif ISOWP == '15': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True)].shape[0]
    elif ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True)].shape[0]
    elif ISOWP == '90': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP90 == True)].shape[0]
    elif ISOWP == '95': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP95 == True)].shape[0]
    else:               sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP99 == True)].shape[0]
    return float(sel)/float(tot)

def efficiency_err(group, threshold, ISOWP, upper=False):
    tot = group.shape[0]
    if ISOWP ==   '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True)].shape[0]
    elif ISOWP == '15': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True)].shape[0]
    elif ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True)].shape[0]
    elif ISOWP == '90': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP90 == True)].shape[0]
    elif ISOWP == '95': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP95 == True)].shape[0]
    else:               sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP99 == True)].shape[0]
    
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

def sigmoid(x , a, x0, k):
    return a / ( 1 + np.exp(-k*(x-x0)) )

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)

def global_shap_importance(shap_values):
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--PUWP', dest='PUWP', help='which PU working point do you want to use (90, 95, 99)?', default='90')
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--doEfficiency', dest='doEfficiency', help='do you want calculate the efficiencies?', action='store_true', default=False)
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_argument('--hardPUrej', dest='hardPUrej', help='apply hard PU rejection and do not consider PU categorized clusters for Iso variables? (99, 95, 90)', default='NO')
    parser.add_argument('--effFitLimit', dest='effFitLimit', help='how many gentau_pt bins you want to consider for the fit of the turnON? (default: 49 bins = <150GeV)', default=49)
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
    tag1 = "Rscld" if args.doRescale else ""
    tag2 = "{0}hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/isolated_skimPUnoPt{0}{1}'.format(tag1, '_'+tag2)
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/hdf5dataframes/isolated_skimPUnoPt{0}_fullISO{0}{1}_againstPU'.format(tag1, tag2)
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/plots/isolation_skimPUnoPt_fullISO{0}{1}_PUWP{2}_againstPU'.format(tag1, tag2, args.PUWP)
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/isolation_skimPUnoPt{0}_fullISO{0}{1}_againstPU'.format(tag1, tag2)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_splitIso/pklModels/isolation_skimPUnoPt{0}_fullISO{0}{1}_againstPU/performance_PUWP{2}.log".format(tag1, tag2, args.PUWP))

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_isoCalculated.hdf5',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_isoCalculated.hdf5',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'mixed'        : outdir+'/'
    }

    outFile_model_lowEta_dict = {
        'threshold'    : model_outdir+'/model_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_model_midEta_dict = {
        'threshold'    : model_outdir+'/model_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_model_highEta_dict = {
        'threshold'    : model_outdir+'/model_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_model_vhighEta_dict = {
        'threshold'    : model_outdir+'/model_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP10_lowEta_dict = {
        'threshold'    : model_outdir+'/WP10_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP15_lowEta_dict = {
        'threshold'    : model_outdir+'/WP15_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP20_lowEta_dict = {
        'threshold'    : model_outdir+'/WP20_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_lowEta_dict = {
        'threshold'    : model_outdir+'/WP90_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_lowEta_dict = {
        'threshold'    : model_outdir+'/WP95_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_lowEta_dict = {
        'threshold'    : model_outdir+'/WP99_isolation_lowEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP10_midEta_dict = {
        'threshold'    : model_outdir+'/WP10_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP15_midEta_dict = {
        'threshold'    : model_outdir+'/WP15_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP20_midEta_dict = {
        'threshold'    : model_outdir+'/WP20_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_midEta_dict = {
        'threshold'    : model_outdir+'/WP90_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_midEta_dict = {
        'threshold'    : model_outdir+'/WP95_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_midEta_dict = {
        'threshold'    : model_outdir+'/WP99_isolation_midEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP10_highEta_dict = {
        'threshold'    : model_outdir+'/WP10_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP15_highEta_dict = {
        'threshold'    : model_outdir+'/WP15_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP20_highEta_dict = {
        'threshold'    : model_outdir+'/WP20_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_highEta_dict = {
        'threshold'    : model_outdir+'/WP90_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_highEta_dict = {
        'threshold'    : model_outdir+'/WP95_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_highEta_dict = {
        'threshold'    : model_outdir+'/WP99_isolation_highEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP10_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP10_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP15_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP15_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP20_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP20_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP90_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP95_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_vhighEta_dict = {
        'threshold'    : model_outdir+'/WP99_isolation_vhighEta_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}

    # efficiencies
    effVSpt_Tau_dict = {}
    effVSpt_TauDM0_dict = {}
    effVSpt_TauDM1_dict = {}
    effVSpt_TauDM2_dict = {}

    # target of the training
    output = 'sgnId'

    # features used for the ISO QCD rejection - FULL AVAILABLE
    features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
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


    # BDT hyperparameters
    # params_dict = {}
    # params_dict['eval_metric']        = 'logloss'
    # params_dict['nthread']            = 10  # limit number of threads
    # params_dict['eta']                = 0.2 # learning rate
    # params_dict['max_depth']          = 4   # maximum depth of a tree
    # params_dict['subsample']          = 0.8 # fraction of events to train tree on
    # params_dict['colsample_bytree']   = 0.8 # fraction of features to train tree on
    # params_dict['objective']          = 'binary:logistic' # objective function
    # num_trees = 60  # number of trees to make


    # selected features from FS
    # features = []
    # if args.PUWP == '99':
    #     features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    # elif args.PUWP == '95':
    #     features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    # else:
    #     features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
    
    # features2shift = ['cl3d_NclIso_dR4', 'cl3d_coreshowerlength']
    # features2saturate = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
    # saturation_dict = {'cl3d_pt_tr': [0, 200],
    #                    'cl3d_abseta': [1.45, 3.2],
    #                    'cl3d_spptot': [0, 0.17],
    #                    'cl3d_srrtot': [0, 0.02],
    #                    'cl3d_srrmean': [0, 0.01],
    #                    'cl3d_hoe': [0, 63],
    #                    'cl3d_meanz': [305, 535],
    #                    'tower_etSgn_dRsgn1': [0, 194],
    #                    'tower_etSgn_dRsgn2': [0, 228],
    #                    'tower_etIso_dRsgn1_dRiso3': [0, 105]
    #                   }
    # # BDT hyperparameters
    params_dict = {}
    params_dict['objective']          = 'binary:logistic'
    params_dict['eval_metric']        = 'logloss'
    params_dict['nthread']            = 10
    params_dict['alpha']              = 9
    params_dict['lambda']             = 5
    params_dict['max_depth']          = 4 # from HPO
    params_dict['eta']                = 0.37 # from HPO
    params_dict['subsample']          = 0.12 # from HPO
    params_dict['colsample_bytree']   = 0.9 # from HPO
    num_trees = 30 # from HPO


    # dictionaries for BDT training
    lowEta_model_dict= {}
    lowEta_fpr_train_dict = {}
    lowEta_tpr_train_dict = {}
    lowEta_fpr_test_dict = {}
    lowEta_tpr_test_dict = {}
    lowEta_fprNcluster_dict = {}
    lowEta_threshold_train_dict = {}
    lowEta_threshold_test_dict = {}
    lowEta_threshold_validation_dict = {}
    lowEta_testAuroc_dict = {}
    lowEta_trainAuroc_dict = {}
    lowEta_fpr_validation_dict = {}
    lowEta_tpr_validation_dict = {}

    midEta_model_dict= {}
    midEta_fpr_train_dict = {}
    midEta_tpr_train_dict = {}
    midEta_fpr_test_dict = {}
    midEta_tpr_test_dict = {}
    midEta_fprNcluster_dict = {}
    midEta_threshold_train_dict = {}
    midEta_threshold_test_dict = {}
    midEta_threshold_validation_dict = {}
    midEta_testAuroc_dict = {}
    midEta_trainAuroc_dict = {}
    midEta_fpr_validation_dict = {}
    midEta_tpr_validation_dict = {}

    highEta_model_dict= {}
    highEta_fpr_train_dict = {}
    highEta_tpr_train_dict = {}
    highEta_fpr_test_dict = {}
    highEta_tpr_test_dict = {}
    highEta_fprNcluster_dict = {}
    highEta_threshold_train_dict = {}
    highEta_threshold_test_dict = {}
    highEta_threshold_validation_dict = {}
    highEta_testAuroc_dict = {}
    highEta_trainAuroc_dict = {}
    highEta_fpr_validation_dict = {}
    highEta_tpr_validation_dict = {}

    vhighEta_model_dict= {}
    vhighEta_fpr_train_dict = {}
    vhighEta_tpr_train_dict = {}
    vhighEta_fpr_test_dict = {}
    vhighEta_tpr_test_dict = {}
    vhighEta_fprNcluster_dict = {}
    vhighEta_threshold_train_dict = {}
    vhighEta_threshold_test_dict = {}
    vhighEta_threshold_validation_dict = {}
    vhighEta_testAuroc_dict = {}
    vhighEta_trainAuroc_dict = {}
    vhighEta_fpr_validation_dict = {}
    vhighEta_tpr_validation_dict = {}

    # working points dictionaries
    lowEta_bdtWP10_dict = {}
    lowEta_bdtWP15_dict = {}
    lowEta_bdtWP20_dict = {}
    lowEta_bdtWP90_dict = {}
    lowEta_bdtWP95_dict = {}
    lowEta_bdtWP99_dict = {}

    midEta_bdtWP10_dict = {}
    midEta_bdtWP15_dict = {}
    midEta_bdtWP20_dict = {}
    midEta_bdtWP90_dict = {}
    midEta_bdtWP95_dict = {}
    midEta_bdtWP99_dict = {}

    highEta_bdtWP10_dict = {}
    highEta_bdtWP15_dict = {}
    highEta_bdtWP20_dict = {}
    highEta_bdtWP90_dict = {}
    highEta_bdtWP95_dict = {}
    highEta_bdtWP99_dict = {}

    vhighEta_bdtWP10_dict = {}
    vhighEta_bdtWP15_dict = {}
    vhighEta_bdtWP20_dict = {}
    vhighEta_bdtWP90_dict = {}
    vhighEta_bdtWP95_dict = {}
    vhighEta_bdtWP99_dict = {}

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


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        print('** INFO: using PU rejection BDT WP: '+args.PUWP)
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting ISO QCD rejection for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()

        # create a copy to cl3d_pt that will be used only for the training of the BDTs
        # this because if we apply the rescaling of variables we still want the original pt untouched so that we can calculate efficiencies
        dfTraining_dict[name]['cl3d_pt_tr'] = dfTraining_dict[name]['cl3d_pt_c3'].copy(deep=True)
        dfValidation_dict[name]['cl3d_pt_tr'] = dfValidation_dict[name]['cl3d_pt_c3'].copy(deep=True)

        ######################### SELECT EVENTS FOR TRAINING #########################

        # here we select only the genuine QCD and Tau events
        #dfTr = dfTraining_dict[name].query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
        #dfVal = dfValidation_dict[name].query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
        dfTr = dfTraining_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)
        dfVal = dfValidation_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)

        ######################### SECTION DEPENDING ON ETA #########################
        
        dfTr_lowEta = dfTr.query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)
        dfVal_lowEta = dfVal.query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)

        dfTr_midEta = dfTr.query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)
        dfVal_midEta = dfVal.query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)

        dfTr_highEta = dfTr.query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)
        dfVal_highEta = dfVal.query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)

        dfTr_vhighEta = dfTr.query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)
        dfVal_vhighEta = dfVal.query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)

        dfTraining_lowEta = dfTraining_dict[name].query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)
        dfValidation_lowEta = dfValidation_dict[name].query('cl3d_abseta >= 1.5 and cl3d_abseta <= 2.1').copy(deep=True)

        dfTraining_midEta = dfTraining_dict[name].query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)
        dfValidation_midEta = dfValidation_dict[name].query('cl3d_abseta > 2.1 and cl3d_abseta <= 2.5').copy(deep=True)

        dfTraining_highEta = dfTraining_dict[name].query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)
        dfValidation_highEta = dfValidation_dict[name].query('cl3d_abseta > 2.5 and cl3d_abseta <= 2.8').copy(deep=True)

        dfTraining_vhighEta = dfTraining_dict[name].query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)
        dfValidation_vhighEta = dfValidation_dict[name].query('cl3d_abseta > 2.8 and cl3d_abseta <= 3.0').copy(deep=True)


        ######################### DO RESCALING OF THE FEATURES #########################
        if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            dfs = [dfTr_lowEta, dfVal_lowEta, dfTr_midEta, dfVal_midEta, dfTr_highEta, dfVal_highEta, dfTr_vhighEta, dfVal_vhighEta, dfTraining_lowEta, dfValidation_lowEta, dfTraining_midEta, dfValidation_midEta, dfTraining_highEta, dfValidation_highEta, dfTraining_vhighEta, dfValidation_vhighEta]
            
            for df in dfs:
                #define a DF with the bound values of the features to use for the MinMaxScaler fit
                bounds4features = pd.DataFrame(columns=features2saturate)

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


        #***********************************************************************************************************#
                                                        # LOW ETA #
        #***********************************************************************************************************#

        ######################### TRAINING OF BDT #########################

        print('\n##########################################')
        print('\n** INFO: training lowEta BDT')
        lowEta_model_dict[name], lowEta_fpr_train_dict[name], lowEta_tpr_train_dict[name], lowEta_threshold_train_dict[name], lowEta_fpr_test_dict[name], lowEta_tpr_test_dict[name], lowEta_threshold_test_dict[name], lowEta_testAuroc_dict[name], lowEta_trainAuroc_dict[name] = train_xgb(dfTr_lowEta, features, output, params_dict, num_trees, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(lowEta_trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(lowEta_testAuroc_dict[name]))

        lowEta_bdtWP10_dict[name] = np.interp(0.10, lowEta_fpr_train_dict[name], lowEta_threshold_train_dict[name])
        lowEta_bdtWP15_dict[name] = np.interp(0.15, lowEta_fpr_train_dict[name], lowEta_threshold_train_dict[name])
        lowEta_bdtWP20_dict[name] = np.interp(0.20, lowEta_fpr_train_dict[name], lowEta_threshold_train_dict[name])
        lowEta_bdtWP90_dict[name] = np.interp(0.90, lowEta_tpr_train_dict[name], lowEta_threshold_train_dict[name])
        lowEta_bdtWP95_dict[name] = np.interp(0.95, lowEta_tpr_train_dict[name], lowEta_threshold_train_dict[name])
        lowEta_bdtWP99_dict[name] = np.interp(0.99, lowEta_tpr_train_dict[name], lowEta_threshold_train_dict[name])

        save_obj(lowEta_model_dict[name], outFile_model_lowEta_dict[name])
        save_obj(lowEta_bdtWP10_dict[name], outFile_WP10_lowEta_dict[name])
        save_obj(lowEta_bdtWP15_dict[name], outFile_WP15_lowEta_dict[name])
        save_obj(lowEta_bdtWP20_dict[name], outFile_WP20_lowEta_dict[name])
        save_obj(lowEta_bdtWP90_dict[name], outFile_WP90_lowEta_dict[name])
        save_obj(lowEta_bdtWP95_dict[name], outFile_WP95_lowEta_dict[name])
        save_obj(lowEta_bdtWP99_dict[name], outFile_WP99_lowEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.10FPR {0} - 0.15FPR {1} - 0.20FPR {2}'.format(lowEta_bdtWP10_dict[name], lowEta_bdtWP15_dict[name], lowEta_bdtWP20_dict[name]))
        print('\n**INFO: BDT WP for: 0.90TPR {0} - 0.95TPR {1} - 0.99TPR {2}'.format(lowEta_bdtWP90_dict[name], lowEta_bdtWP95_dict[name], lowEta_bdtWP99_dict[name]))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal_lowEta[features], label=dfVal_lowEta[output], feature_names=features)
        dfVal_lowEta['cl3d_isobdt_score'] = lowEta_model_dict[name].predict(full)

        lowEta_fpr_validation_dict[name], lowEta_tpr_validation_dict[name], lowEta_threshold_validation_dict[name] = metrics.roc_curve(dfVal_lowEta[output], dfVal_lowEta['cl3d_isobdt_score'])
        lowEta_auroc_validation = metrics.roc_auc_score(dfVal_lowEta['sgnId'],dfVal_lowEta['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(lowEta_auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_lowEta[features], label=dfTraining_lowEta[output], feature_names=features)
        dfTraining_lowEta['cl3d_isobdt_score'] = lowEta_model_dict[name].predict(full)
        dfTraining_lowEta['cl3d_isobdt_passWP10'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP10_dict[name]
        dfTraining_lowEta['cl3d_isobdt_passWP15'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP15_dict[name]
        dfTraining_lowEta['cl3d_isobdt_passWP20'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP20_dict[name]
        dfTraining_lowEta['cl3d_isobdt_passWP90'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP90_dict[name]
        dfTraining_lowEta['cl3d_isobdt_passWP95'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP95_dict[name]
        dfTraining_lowEta['cl3d_isobdt_passWP99'] = dfTraining_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP99_dict[name]

        full = xgb.DMatrix(data=dfValidation_lowEta[features], label=dfValidation_lowEta[output], feature_names=features)
        dfValidation_lowEta['cl3d_isobdt_score'] = lowEta_model_dict[name].predict(full)
        dfValidation_lowEta['cl3d_isobdt_passWP10'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP10_dict[name]
        dfValidation_lowEta['cl3d_isobdt_passWP15'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP15_dict[name]
        dfValidation_lowEta['cl3d_isobdt_passWP20'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP20_dict[name]
        dfValidation_lowEta['cl3d_isobdt_passWP90'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP90_dict[name]
        dfValidation_lowEta['cl3d_isobdt_passWP95'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP95_dict[name]
        dfValidation_lowEta['cl3d_isobdt_passWP99'] = dfValidation_lowEta['cl3d_isobdt_score'] > lowEta_bdtWP99_dict[name]

        ######################### PLOT FEATURE IMPORTANCES #########################

        print('\n** INFO: plotting features importance and score')
        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('sgnId==0').sample(1500)], sort=False)[features]
        explainer = shap.Explainer(lowEta_model_dict[name])
        shap_values = explainer(df)

        plt.figure(figsize=(16,8))
        shap.plots.beeswarm(shap_values, max_display=99, show=False)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/isobdt_lowEta_SHAPimportance_'+name+'.pdf')
        plt.close()


        #***********************************************************************************************************#
                                                        # MID ETA #
        #***********************************************************************************************************#

        ######################### TRAINING OF BDT #########################

        print('\n##########################################')
        print('\n** INFO: training midEta BDT')
        midEta_model_dict[name], midEta_fpr_train_dict[name], midEta_tpr_train_dict[name], midEta_threshold_train_dict[name], midEta_fpr_test_dict[name], midEta_tpr_test_dict[name], midEta_threshold_test_dict[name], midEta_testAuroc_dict[name], midEta_trainAuroc_dict[name] = train_xgb(dfTr_midEta, features, output, params_dict, num_trees, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(midEta_trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(midEta_testAuroc_dict[name]))

        midEta_bdtWP10_dict[name] = np.interp(0.10, midEta_fpr_train_dict[name], midEta_threshold_train_dict[name])
        midEta_bdtWP15_dict[name] = np.interp(0.15, midEta_fpr_train_dict[name], midEta_threshold_train_dict[name])
        midEta_bdtWP20_dict[name] = np.interp(0.20, midEta_fpr_train_dict[name], midEta_threshold_train_dict[name])
        midEta_bdtWP90_dict[name] = np.interp(0.90, midEta_tpr_train_dict[name], midEta_threshold_train_dict[name])
        midEta_bdtWP95_dict[name] = np.interp(0.95, midEta_tpr_train_dict[name], midEta_threshold_train_dict[name])
        midEta_bdtWP99_dict[name] = np.interp(0.99, midEta_tpr_train_dict[name], midEta_threshold_train_dict[name])

        save_obj(midEta_model_dict[name], outFile_model_midEta_dict[name])
        save_obj(midEta_bdtWP10_dict[name], outFile_WP10_midEta_dict[name])
        save_obj(midEta_bdtWP15_dict[name], outFile_WP15_midEta_dict[name])
        save_obj(midEta_bdtWP20_dict[name], outFile_WP20_midEta_dict[name])
        save_obj(midEta_bdtWP90_dict[name], outFile_WP90_midEta_dict[name])
        save_obj(midEta_bdtWP95_dict[name], outFile_WP95_midEta_dict[name])
        save_obj(midEta_bdtWP99_dict[name], outFile_WP99_midEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.10FPR {0} - 0.15FPR {1} - 0.20FPR {2}'.format(midEta_bdtWP10_dict[name], midEta_bdtWP15_dict[name], midEta_bdtWP20_dict[name]))
        print('\n**INFO: BDT WP for: 0.90TPR {0} - 0.95TPR {1} - 0.99TPR {2}'.format(midEta_bdtWP90_dict[name], midEta_bdtWP95_dict[name], midEta_bdtWP99_dict[name]))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal_midEta[features], label=dfVal_midEta[output], feature_names=features)
        dfVal_midEta['cl3d_isobdt_score'] = midEta_model_dict[name].predict(full)

        midEta_fpr_validation_dict[name], midEta_tpr_validation_dict[name], midEta_threshold_validation_dict[name] = metrics.roc_curve(dfVal_midEta[output], dfVal_midEta['cl3d_isobdt_score'])
        midEta_auroc_validation = metrics.roc_auc_score(dfVal_midEta['sgnId'],dfVal_midEta['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(midEta_auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_midEta[features], label=dfTraining_midEta[output], feature_names=features)
        dfTraining_midEta['cl3d_isobdt_score'] = midEta_model_dict[name].predict(full)
        dfTraining_midEta['cl3d_isobdt_passWP10'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP10_dict[name]
        dfTraining_midEta['cl3d_isobdt_passWP15'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP15_dict[name]
        dfTraining_midEta['cl3d_isobdt_passWP20'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP20_dict[name]
        dfTraining_midEta['cl3d_isobdt_passWP90'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP90_dict[name]
        dfTraining_midEta['cl3d_isobdt_passWP95'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP95_dict[name]
        dfTraining_midEta['cl3d_isobdt_passWP99'] = dfTraining_midEta['cl3d_isobdt_score'] > midEta_bdtWP99_dict[name]

        full = xgb.DMatrix(data=dfValidation_midEta[features], label=dfValidation_midEta[output], feature_names=features)
        dfValidation_midEta['cl3d_isobdt_score'] = midEta_model_dict[name].predict(full)
        dfValidation_midEta['cl3d_isobdt_passWP10'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP10_dict[name]
        dfValidation_midEta['cl3d_isobdt_passWP15'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP15_dict[name]
        dfValidation_midEta['cl3d_isobdt_passWP20'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP20_dict[name]
        dfValidation_midEta['cl3d_isobdt_passWP90'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP90_dict[name]
        dfValidation_midEta['cl3d_isobdt_passWP95'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP95_dict[name]
        dfValidation_midEta['cl3d_isobdt_passWP99'] = dfValidation_midEta['cl3d_isobdt_score'] > midEta_bdtWP99_dict[name]

        ######################### PLOT FEATURE IMPORTANCES #########################

        print('\n** INFO: plotting features importance and score')
        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('sgnId==0').sample(1500)], sort=False)[features]
        explainer = shap.Explainer(midEta_model_dict[name])
        shap_values = explainer(df)

        plt.figure(figsize=(16,8))
        shap.plots.beeswarm(shap_values, max_display=99, show=False)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/isobdt_midEta_SHAPimportance_'+name+'.pdf')
        plt.close()


        #***********************************************************************************************************#
                                                        # HIGH ETA #
        #***********************************************************************************************************#

        ######################### TRAINING OF BDT #########################

        print('\n##########################################')
        print('\n** INFO: training highEta BDT')
        highEta_model_dict[name], highEta_fpr_train_dict[name], highEta_tpr_train_dict[name], highEta_threshold_train_dict[name], highEta_fpr_test_dict[name], highEta_tpr_test_dict[name], highEta_threshold_test_dict[name], highEta_testAuroc_dict[name], highEta_trainAuroc_dict[name] = train_xgb(dfTr_highEta, features, output, params_dict, num_trees, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(highEta_trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(highEta_testAuroc_dict[name]))

        highEta_bdtWP10_dict[name] = np.interp(0.10, highEta_fpr_train_dict[name], highEta_threshold_train_dict[name])
        highEta_bdtWP15_dict[name] = np.interp(0.15, highEta_fpr_train_dict[name], highEta_threshold_train_dict[name])
        highEta_bdtWP20_dict[name] = np.interp(0.20, highEta_fpr_train_dict[name], highEta_threshold_train_dict[name])
        highEta_bdtWP90_dict[name] = np.interp(0.90, highEta_tpr_train_dict[name], highEta_threshold_train_dict[name])
        highEta_bdtWP95_dict[name] = np.interp(0.95, highEta_tpr_train_dict[name], highEta_threshold_train_dict[name])
        highEta_bdtWP99_dict[name] = np.interp(0.99, highEta_tpr_train_dict[name], highEta_threshold_train_dict[name])

        save_obj(highEta_model_dict[name], outFile_model_highEta_dict[name])
        save_obj(highEta_bdtWP10_dict[name], outFile_WP10_highEta_dict[name])
        save_obj(highEta_bdtWP15_dict[name], outFile_WP15_highEta_dict[name])
        save_obj(highEta_bdtWP20_dict[name], outFile_WP20_highEta_dict[name])
        save_obj(highEta_bdtWP90_dict[name], outFile_WP90_highEta_dict[name])
        save_obj(highEta_bdtWP95_dict[name], outFile_WP95_highEta_dict[name])
        save_obj(highEta_bdtWP99_dict[name], outFile_WP99_highEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.10FPR {0} - 0.15FPR {1} - 0.20FPR {2}'.format(highEta_bdtWP10_dict[name], highEta_bdtWP15_dict[name], highEta_bdtWP20_dict[name]))
        print('\n**INFO: BDT WP for: 0.90TPR {0} - 0.95TPR {1} - 0.99TPR {2}'.format(highEta_bdtWP90_dict[name], highEta_bdtWP95_dict[name], highEta_bdtWP99_dict[name]))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal_highEta[features], label=dfVal_highEta[output], feature_names=features)
        dfVal_highEta['cl3d_isobdt_score'] = highEta_model_dict[name].predict(full)

        highEta_fpr_validation_dict[name], highEta_tpr_validation_dict[name], highEta_threshold_validation_dict[name] = metrics.roc_curve(dfVal_highEta[output], dfVal_highEta['cl3d_isobdt_score'])
        highEta_auroc_validation = metrics.roc_auc_score(dfVal_highEta['sgnId'],dfVal_highEta['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(highEta_auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_highEta[features], label=dfTraining_highEta[output], feature_names=features)
        dfTraining_highEta['cl3d_isobdt_score'] = highEta_model_dict[name].predict(full)
        dfTraining_highEta['cl3d_isobdt_passWP10'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP10_dict[name]
        dfTraining_highEta['cl3d_isobdt_passWP15'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP15_dict[name]
        dfTraining_highEta['cl3d_isobdt_passWP20'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP20_dict[name]
        dfTraining_highEta['cl3d_isobdt_passWP90'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP90_dict[name]
        dfTraining_highEta['cl3d_isobdt_passWP95'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP95_dict[name]
        dfTraining_highEta['cl3d_isobdt_passWP99'] = dfTraining_highEta['cl3d_isobdt_score'] > highEta_bdtWP99_dict[name]

        full = xgb.DMatrix(data=dfValidation_highEta[features], label=dfValidation_highEta[output], feature_names=features)
        dfValidation_highEta['cl3d_isobdt_score'] = highEta_model_dict[name].predict(full)
        dfValidation_highEta['cl3d_isobdt_passWP10'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP10_dict[name]
        dfValidation_highEta['cl3d_isobdt_passWP15'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP15_dict[name]
        dfValidation_highEta['cl3d_isobdt_passWP20'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP20_dict[name]
        dfValidation_highEta['cl3d_isobdt_passWP90'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP90_dict[name]
        dfValidation_highEta['cl3d_isobdt_passWP95'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP95_dict[name]
        dfValidation_highEta['cl3d_isobdt_passWP99'] = dfValidation_highEta['cl3d_isobdt_score'] > highEta_bdtWP99_dict[name]

        ######################### PLOT FEATURE IMPORTANCES #########################

        print('\n** INFO: plotting features importance and score')
        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('sgnId==0').sample(1500)], sort=False)[features]
        explainer = shap.Explainer(highEta_model_dict[name])
        shap_values = explainer(df)

        plt.figure(figsize=(16,8))
        shap.plots.beeswarm(shap_values, max_display=99, show=False)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/isobdt_highEta_SHAPimportance_'+name+'.pdf')
        plt.close()


        #***********************************************************************************************************#
                                                    # VERY HIGH ETA #
        #***********************************************************************************************************#

        ######################### TRAINING OF BDT #########################

        print('\n##########################################')
        print('\n** INFO: training vhighEta BDT')
        vhighEta_model_dict[name], vhighEta_fpr_train_dict[name], vhighEta_tpr_train_dict[name], vhighEta_threshold_train_dict[name], vhighEta_fpr_test_dict[name], vhighEta_tpr_test_dict[name], vhighEta_threshold_test_dict[name], vhighEta_testAuroc_dict[name], vhighEta_trainAuroc_dict[name] = train_xgb(dfTr_vhighEta, features, output, params_dict, num_trees, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(vhighEta_trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(vhighEta_testAuroc_dict[name]))

        vhighEta_bdtWP10_dict[name] = np.interp(0.10, vhighEta_fpr_train_dict[name], vhighEta_threshold_train_dict[name])
        vhighEta_bdtWP15_dict[name] = np.interp(0.15, vhighEta_fpr_train_dict[name], vhighEta_threshold_train_dict[name])
        vhighEta_bdtWP20_dict[name] = np.interp(0.20, vhighEta_fpr_train_dict[name], vhighEta_threshold_train_dict[name])
        vhighEta_bdtWP90_dict[name] = np.interp(0.90, vhighEta_tpr_train_dict[name], vhighEta_threshold_train_dict[name])
        vhighEta_bdtWP95_dict[name] = np.interp(0.95, vhighEta_tpr_train_dict[name], vhighEta_threshold_train_dict[name])
        vhighEta_bdtWP99_dict[name] = np.interp(0.99, vhighEta_tpr_train_dict[name], vhighEta_threshold_train_dict[name])

        save_obj(vhighEta_model_dict[name], outFile_model_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP10_dict[name], outFile_WP10_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP15_dict[name], outFile_WP15_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP20_dict[name], outFile_WP20_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP90_dict[name], outFile_WP90_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP95_dict[name], outFile_WP95_vhighEta_dict[name])
        save_obj(vhighEta_bdtWP99_dict[name], outFile_WP99_vhighEta_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.10FPR {0} - 0.15FPR {1} - 0.20FPR {2}'.format(vhighEta_bdtWP10_dict[name], vhighEta_bdtWP15_dict[name], vhighEta_bdtWP20_dict[name]))
        print('\n**INFO: BDT WP for: 0.90TPR {0} - 0.95TPR {1} - 0.99TPR {2}'.format(vhighEta_bdtWP90_dict[name], vhighEta_bdtWP95_dict[name], vhighEta_bdtWP99_dict[name]))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal_vhighEta[features], label=dfVal_vhighEta[output], feature_names=features)
        dfVal_vhighEta['cl3d_isobdt_score'] = vhighEta_model_dict[name].predict(full)

        vhighEta_fpr_validation_dict[name], vhighEta_tpr_validation_dict[name], vhighEta_threshold_validation_dict[name] = metrics.roc_curve(dfVal_vhighEta[output], dfVal_vhighEta['cl3d_isobdt_score'])
        vhighEta_auroc_validation = metrics.roc_auc_score(dfVal_vhighEta['sgnId'],dfVal_vhighEta['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(vhighEta_auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_vhighEta[features], label=dfTraining_vhighEta[output], feature_names=features)
        dfTraining_vhighEta['cl3d_isobdt_score'] = vhighEta_model_dict[name].predict(full)
        dfTraining_vhighEta['cl3d_isobdt_passWP10'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP10_dict[name]
        dfTraining_vhighEta['cl3d_isobdt_passWP15'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP15_dict[name]
        dfTraining_vhighEta['cl3d_isobdt_passWP20'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP20_dict[name]
        dfTraining_vhighEta['cl3d_isobdt_passWP90'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP90_dict[name]
        dfTraining_vhighEta['cl3d_isobdt_passWP95'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP95_dict[name]
        dfTraining_vhighEta['cl3d_isobdt_passWP99'] = dfTraining_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP99_dict[name]

        full = xgb.DMatrix(data=dfValidation_vhighEta[features], label=dfValidation_vhighEta[output], feature_names=features)
        dfValidation_vhighEta['cl3d_isobdt_score'] = vhighEta_model_dict[name].predict(full)
        dfValidation_vhighEta['cl3d_isobdt_passWP10'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP10_dict[name]
        dfValidation_vhighEta['cl3d_isobdt_passWP15'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP15_dict[name]
        dfValidation_vhighEta['cl3d_isobdt_passWP20'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP20_dict[name]
        dfValidation_vhighEta['cl3d_isobdt_passWP90'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP90_dict[name]
        dfValidation_vhighEta['cl3d_isobdt_passWP95'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP95_dict[name]
        dfValidation_vhighEta['cl3d_isobdt_passWP99'] = dfValidation_vhighEta['cl3d_isobdt_score'] > vhighEta_bdtWP99_dict[name]

        ######################### PLOT FEATURE IMPORTANCES #########################

        print('\n** INFO: plotting features importance and score')
        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('sgnId==0').sample(1500)], sort=False)[features]
        explainer = shap.Explainer(vhighEta_model_dict[name])
        shap_values = explainer(df)

        plt.figure(figsize=(16,10))
        shap.plots.beeswarm(shap_values, max_display=99, show=False)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/isobdt_vhighEta_SHAPimportance_'+name+'.pdf')
        plt.close()


        #***********************************************************************************************************#
                                                # RE-MERGE ETA REGIONS TOGETHER #
        #***********************************************************************************************************#

        dfTraining_dict[name] = pd.concat([dfTraining_lowEta, dfTraining_midEta, dfTraining_highEta, dfTraining_vhighEta], sort=False).copy(deep=True)
        dfValidation_dict[name] = pd.concat([dfValidation_lowEta, dfValidation_midEta, dfValidation_highEta, dfValidation_vhighEta], sort=False).copy(deep=True)


        ######################### PRINT OUT EFFICIENCIES #########################

        QCDtot = pd.concat([dfTraining_dict[name], dfValidation_dict[name]], sort=False).query('gentau_decayMode==-2 and cl3d_isbestmatch==True')
        QCD10 = QCDtot.query('cl3d_isobdt_passWP10==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        QCD15 = QCDtot.query('cl3d_isobdt_passWP15==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        QCD20 = QCDtot.query('cl3d_isobdt_passWP20==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        QCD90 = QCDtot.query('cl3d_isobdt_passWP90==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        QCD95 = QCDtot.query('cl3d_isobdt_passWP95==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        QCD99 = QCDtot.query('cl3d_isobdt_passWP99==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))

        print('\n**INFO: QCD cluster passing the ISO QCD rejection:')
        print('  -- number of QCD events passing WP10: {0}%'.format(round(float(QCD10['cl3d_isobdt_passWP10'].count())/float(QCDtot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of QCD events passing WP15: {0}%'.format(round(float(QCD15['cl3d_isobdt_passWP15'].count())/float(QCDtot['cl3d_isobdt_passWP15'].count())*100,2)))
        print('  -- number of QCD events passing WP20: {0}%'.format(round(float(QCD20['cl3d_isobdt_passWP20'].count())/float(QCDtot['cl3d_isobdt_passWP20'].count())*100,2)))
        print('  -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90['cl3d_isobdt_passWP90'].count())/float(QCDtot['cl3d_isobdt_passWP90'].count())*100,2)))
        print('  -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95['cl3d_isobdt_passWP95'].count())/float(QCDtot['cl3d_isobdt_passWP95'].count())*100,2)))
        print('  -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99['cl3d_isobdt_passWP99'].count())/float(QCDtot['cl3d_isobdt_passWP99'].count())*100,2)))

        Nutot = pd.concat([dfTraining_dict[name], dfValidation_dict[name]], sort=False).query('cl3d_isbestmatch==False')
        Nu10 = Nutot.query('cl3d_isobdt_passWP10==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Nu15 = Nutot.query('cl3d_isobdt_passWP15==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Nu20 = Nutot.query('cl3d_isobdt_passWP20==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Nu90 = Nutot.query('cl3d_isobdt_passWP90==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Nu95 = Nutot.query('cl3d_isobdt_passWP95==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Nu99 = Nutot.query('cl3d_isobdt_passWP99==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))

        print('\n**INFO: PU cluster passing the ISO QCD rejection:')
        print('  -- number of PU events passing WP10: {0}%'.format(round(float(Nu10['cl3d_isobdt_passWP10'].count())/float(Nutot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of PU events passing WP15: {0}%'.format(round(float(Nu15['cl3d_isobdt_passWP15'].count())/float(Nutot['cl3d_isobdt_passWP15'].count())*100,2)))
        print('  -- number of PU events passing WP20: {0}%'.format(round(float(Nu20['cl3d_isobdt_passWP20'].count())/float(Nutot['cl3d_isobdt_passWP20'].count())*100,2)))
        print('  -- number of PU events passing WP90: {0}%'.format(round(float(Nu90['cl3d_isobdt_passWP90'].count())/float(Nutot['cl3d_isobdt_passWP90'].count())*100,2)))
        print('  -- number of PU events passing WP95: {0}%'.format(round(float(Nu95['cl3d_isobdt_passWP95'].count())/float(Nutot['cl3d_isobdt_passWP95'].count())*100,2)))
        print('  -- number of PU events passing WP99: {0}%'.format(round(float(Nu99['cl3d_isobdt_passWP99'].count())/float(Nutot['cl3d_isobdt_passWP99'].count())*100,2)))

        Tautot = pd.concat([dfTraining_dict[name].query('sgnId==1'), dfValidation_dict[name].query('sgnId==1')], sort=False)
        Tau10 = Tautot.query('cl3d_isobdt_passWP10==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Tau15 = Tautot.query('cl3d_isobdt_passWP15==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Tau20 = Tautot.query('cl3d_isobdt_passWP20==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Tau90 = Tautot.query('cl3d_isobdt_passWP90==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Tau95 = Tautot.query('cl3d_isobdt_passWP95==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))
        Tau99 = Tautot.query('cl3d_isobdt_passWP99==True and cl3d_pubdt_passWP{0}==True'.format(args.PUWP))

        print('\n**INFO: Tau cluster passing the ISO QCD rejection:')
        print('  -- number of Tau events passing WP10: {0}%'.format(round(float(Tau10['cl3d_isobdt_passWP10'].count())/float(Tautot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of Tau events passing WP15: {0}%'.format(round(float(Tau15['cl3d_isobdt_passWP15'].count())/float(Tautot['cl3d_isobdt_passWP15'].count())*100,2)))
        print('  -- number of Tau events passing WP20: {0}%'.format(round(float(Tau20['cl3d_isobdt_passWP20'].count())/float(Tautot['cl3d_isobdt_passWP20'].count())*100,2)))
        print('  -- number of Tau events passing WP90: {0}%'.format(round(float(Tau90['cl3d_isobdt_passWP90'].count())/float(Tautot['cl3d_isobdt_passWP90'].count())*100,2)))
        print('  -- number of Tau events passing WP95: {0}%'.format(round(float(Tau95['cl3d_isobdt_passWP95'].count())/float(Tautot['cl3d_isobdt_passWP95'].count())*100,2)))
        print('  -- number of Tau events passing WP99: {0}%'.format(round(float(Tau99['cl3d_isobdt_passWP99'].count())/float(Tautot['cl3d_isobdt_passWP99'].count())*100,2)))
        print('')
        print('  -- number of Tau pT>30GeV events passing WP10: {0}%'.format(round(float(Tau10.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP10'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP15: {0}%'.format(round(float(Tau15.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP15'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP15'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP20: {0}%'.format(round(float(Tau20.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP20'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP20'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP90: {0}%'.format(round(float(Tau90.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP90'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP90'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP95: {0}%'.format(round(float(Tau95.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP95'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP95'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP99: {0}%'.format(round(float(Tau99.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP99'].count())/float(Tautot.query('cl3d_pt_c3>30')['cl3d_isobdt_passWP99'].count())*100,2)))

        ######################### SAVE FILES #########################

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()


        print('\n** INFO: finished ISO QCD rejection for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')

      
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    os.system('mkdir -p '+plotdir+'/features/')

    # name : [title, [min, max, step]
    features_dict = {'cl3d_pt_tr'                   : [r'3D cluster $p_{T}$',[0.,500.,50]],
                     'cl3d_abseta'                  : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'            : [r'3D cluster shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength'        : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'              : [r'3D cluster first layer',[0.,20.,20]], 
                     'cl3d_seetot'                  : [r'3D cluster total $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_seemax'                  : [r'3D cluster max $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_spptot'                  : [r'3D cluster total $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_sppmax'                  : [r'3D cluster max $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_szz'                     : [r'3D cluster $\sigma_{zz}$',[0.,60.,20]], 
                     'cl3d_srrtot'                  : [r'3D cluster total $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmax'                  : [r'3D cluster max $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmean'                 : [r'3D cluster mean $\sigma_{rr}$',[0.,0.01,10]], 
                     'cl3d_hoe'                     : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     'cl3d_meanz'                   : [r'3D cluster meanz',[325.,375.,30]], 
                     'cl3d_NclIso_dR4'              : [r'Number of clusters inside an isolation cone of dR=0.4',[0.,10.,10]],
                     'cl3d_etIso_dR4'               : [r'Clusters $E_{T}$ inside an isolation cone of dR=0.4',[0.,200.,40]],
                     'tower_etSgn_dRsgn1'           : [r'$E_{T}$ inside a signal cone of dR=0.1',[0.,200.,40]],
                     'tower_etSgn_dRsgn2'           : [r'$E_{T}$ inside a signal cone of dR=0.2',[0.,200.,40]],
                     'tower_etIso_dRsgn1_dRiso3'    : [r'Towers $E_{T}$ between dR=0.1-0.3 around L1 candidate',[0.,200.,40]],
                     'tower_etEmIso_dRsgn1_dRiso3'  : [r'Towers $E_{T}^{em}$ between dR=0.1-0.3 around L1 candidate',[0.,150.,30]],
                     'tower_etHadIso_dRsgn1_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.1-0.7 around L1 candidate',[0.,200.,40]],
                     'tower_etIso_dRsgn2_dRiso4'    : [r'Towers $E_{T}$ between dR=0.2-0.4 around L1 candidate',[0.,200.,40]],
                     'tower_etEmIso_dRsgn2_dRiso4'  : [r'Towers $E_{T}^{em}$ between dR=0.2-0.4 around L1 candidate',[0.,150.,30]],
                     'tower_etHadIso_dRsgn2_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.2-0.7 around L1 candidate',[0.,200.,40]]
    }
    if args.doRescale:
        features_dict = {'cl3d_pt_tr'                   : [r'3D cluster $p_{T}$',[-33.,33.,66]],
                         'cl3d_abseta'                  : [r'3D cluster |$\eta$|',[-33.,33.,66]], 
                         'cl3d_showerlength'            : [r'3D cluster shower length',[-33.,33.,66]], 
                         'cl3d_coreshowerlength'        : [r'Core shower length ',[-33.,33.,66]], 
                         'cl3d_firstlayer'              : [r'3D cluster first layer',[-33.,33.,66]], 
                         'cl3d_seetot'                  : [r'3D cluster total $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_seemax'                  : [r'3D cluster max $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_spptot'                  : [r'3D cluster total $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_sppmax'                  : [r'3D cluster max $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_szz'                     : [r'3D cluster $\sigma_{zz}$',[-33.,33.,66]], 
                         'cl3d_srrtot'                  : [r'3D cluster total $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmax'                  : [r'3D cluster max $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmean'                 : [r'3D cluster mean $\sigma_{rr}$',[-33.,33.,66]], 
                         'cl3d_hoe'                     : [r'Energy in CE-H / Energy in CE-E',[-33.,33.,66]], 
                         'cl3d_meanz'                   : [r'3D cluster meanz',[-33.,33.,66]], 
                         'cl3d_NclIso_dR4'              : [r'Number of clusters inside an isolation cone of dR=0.4',[-33.,33.,66]],
                         'cl3d_etIso_dR4'               : [r'Clusters $E_{T}$ inside an isolation cone of dR=0.4',[-33.,33.,66]],
                         'tower_etSgn_dRsgn1'           : [r'$E_{T}$ inside a signal cone of dR=0.1',[-33.,33.,66]],
                         'tower_etSgn_dRsgn2'           : [r'$E_{T}$ inside a signal cone of dR=0.2',[-33.,33.,66]],
                         'tower_etIso_dRsgn1_dRiso3'    : [r'Towers $E_{T}$ between dR=0.1-0.3 around L1 candidate',[-33.,33.,66]],
                         'tower_etEmIso_dRsgn1_dRiso3'  : [r'Towers $E_{T}^{em}$ between dR=0.1-0.3 around L1 candidate',[-33.,33.,66]],
                         'tower_etHadIso_dRsgn1_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.1-0.7 around L1 candidate',[-33.,33.,66]],
                         'tower_etIso_dRsgn2_dRiso4'    : [r'Towers $E_{T}$ between dR=0.2-0.4 around L1 candidate',[-33.,33.,66]],
                         'tower_etEmIso_dRsgn2_dRiso4'  : [r'Towers $E_{T}^{em}$ between dR=0.2-0.4 around L1 candidate',[-33.,33.,66]],
                         'tower_etHadIso_dRsgn2_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.2-0.7 around L1 candidate',[-33.,33.,66]],
    }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfQCD = dfTr.query('gentau_decayMode==-2 and cl3d_isbestmatch==True')
        dfNu  = dfTr.query('gentau_decayMode==-1')
        dfTau = dfTr.query('sgnId==1')
        dfQCDNu = dfTr.query('sgnId==0')

        for var in features_dict:
            plt.figure(figsize=(10,10))
            plt.hist(dfQCD[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background',      color='red',    histtype='step', lw=2, density=True)
            plt.hist(dfTau[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Tau signal',   color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfNu[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Residual PU',   color='blue',    histtype='step', lw=2, density=True)
            #plt.hist(dfQCDNu[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD + PU',   color='black',    histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right')
            plt.grid(linestyle=':')
            plt.xlabel(features_dict[var][0])
            plt.ylabel(r'Normalized entries')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/features/'+var+'.pdf')
            plt.close()

        del dfQCD, dfTau, dfQCDNu, dfNu

        ######################### PLOT SINGLE FE OPTION ROCS #########################

        print('\n** INFO: plotting train-test-validation ROC curves')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_dict[name],fpr_train_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_dict[name],fpr_validation_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left')
        plt.xlim(0.75,1.001)
        #plt.yscale('log')
        #plt.ylim(0.001,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/isobdt_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()


        ######################### PLOT FEATURE IMPORTANCES #########################

        matplotlib.rcParams.update({'font.size': 8})
        print('\n** INFO: plotting features importance and score')
        plt.figure(figsize=(15,15))
        importance = model_dict[name].get_score(importance_type='gain')
        for key in importance:
            importance[key] = round(importance[key],2)
        xgb.plot_importance(importance, grid=False, importance_type='gain', lw=2)#, fontsize=8)
        plt.xlim(0.0,300)
        plt.subplots_adjust(left=0.4, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/isobdt_importances_'+name+'.pdf')
        plt.close()

        matplotlib.rcParams.update({'font.size': 10})


        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('sgnId==0').sample(1500)], sort=False)[features]
        explainer = shap.Explainer(model_dict[name])
        shap_values = explainer(df)

        plt.figure(figsize=(16,8))
        shap.plots.beeswarm(shap_values, max_display=99, show=False)
        plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.1)
        plt.savefig(plotdir+'/PUbdt_SHAPimportance_'+name+'.pdf')
        plt.close()

        most_importants = list(global_shap_importance(shap_values)['features'])[:3]
        for feat in most_importants:
            plt.figure(figsize=(8,8))
            shap.plots.scatter(shap_values[:,feat], color=shap_values, show=False)
            plt.savefig(plotdir+'/PUbdt_SHAPdependence_'+feat+'_'+name+'.pdf')
            plt.close()

        ######################### PLOT BDT SCORE #########################

        
        dfTot = pd.concat([dfTr,dfVal], sort=False)
        dfQCD = dfTot.query('gentau_decayMode==-2')
        dfNu  = dfTot.query('gentau_decayMode==-1')
        dfTau = dfTot.query('sgnId==1')
        dfQCDNu = dfTot.query('sgnId==0')

        plt.figure(figsize=(10,10))
        plt.hist(dfQCD['cl3d_isobdt_score'], bins=np.arange(-0.0, 1.0, 0.02), density=True, color='red', histtype='step', lw=2, label='QCD background')
        plt.hist(dfTau['cl3d_isobdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='limegreen', histtype='step', lw=2, label='Tau signal')
        plt.hist(dfNu['cl3d_isobdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='blue', histtype='step', lw=2, label='Residual PU')
        #plt.hist(dfQCDNu['cl3d_isobdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='black', histtype='step', lw=2, label='QCD + PU PU=200')
        plt.legend(loc = 'upper right')
        plt.xlabel(r'Iso BDT score')
        plt.ylabel(r'Entries')
        plt.title('Iso BDT score')
        plt.grid(linestyle=':')
        plt.savefig(plotdir+'/isobdt_score_'+name+'.pdf')
        plt.close()

        del dfQCD, dfTau, dfQCDNu, dfNu

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
    plt.xlim(0.75,1.001)
    #plt.yscale('log')
    #plt.ylim(0.001,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency') 
    plt.title('Test ROC curves')
    plt.savefig(plotdir+'/isobdt_test_rocs.pdf')
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
    plt.xlim(0.75,1.001)
    #plt.yscale('log')
    #plt.ylim(0.001,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.title('Validation ROC curves')
    plt.savefig(plotdir+'/isobdt_validation_rocs.pdf')
    plt.close()
        

    print('\n** INFO: finished plotting')
    print('---------------------------------------------------------------------------------------')


if args.doEfficiency:
    print('\n** INFO: calculating efficiency')
    
    os.system('mkdir -p '+plotdir+'/efficiencies/')

    matplotlib.rcParams.update({'font.size': 20})
    
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
        #dfTau_dict[name].query('gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11', inplace=True)
        dfTau_dict[name].query('sgnId==1', inplace=True)
        dfTau_dict[name].query('cl3d_pubdt_passWP{0}==True and gentau_bin_pt<={1}'.format(args.PUWP,args.effFitLimit), inplace=True)
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

        for ISOWP in ['10','15','20','90','95','99']:
            for threshold in [10,20,30]:
                # calculate efficiency for the TAU datasets --> calculated per bin that will be plotted
                #                                           --> every efficiency_at{threshold} contains the value of the efficiency when applying the specific threshold 
                effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, ISOWP)) # --> the output of this will be a series with idx=bin and entry=efficiency
                effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, ISOWP))
                effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, ISOWP))
                effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, ISOWP))

                effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))

                effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))


                effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, ISOWP)) # --> the output of this will be a series with idx=bin and entry=efficiency
                effVSeta_TauDM0_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, ISOWP))
                effVSeta_TauDM1_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, ISOWP))
                effVSeta_TauDM2_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency(x, threshold, ISOWP))

                effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSeta_TauDM0_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSeta_TauDM1_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))
                effVSeta_TauDM2_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=False))

                effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTau_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSeta_TauDM0_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSeta_TauDM1_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))
                effVSeta_TauDM2_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_eta').apply(lambda x : efficiency_err(x, threshold, ISOWP, upper=True))


        # colors to use for plotting
        col = {
            '10' : 'salmon',
            '15' : 'red',
            '20' : 'darkred',
            '90' : 'cyan',
            '95' : 'blue',
            '99' : 'navy'
        }
        
        x_Tau = effVSpt_Tau_dict[name]['gentau_vis_pt'] # is binned and the value is the mean of the entries per bin
        for threshold in [10,20,30]:
            plt.figure(figsize=(10,10))
            for ISOWP in ['10','15','20','90','95','99']:
                # all values for turnON curves
                eff_Tau = effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_low_Tau = effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_up_Tau = effVSpt_Tau_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)]

                plt.errorbar(x_Tau,eff_Tau,xerr=1,yerr=[eff_err_low_Tau,eff_err_up_Tau],ls='None',label=r'ISOWP = {0}'.format(ISOWP),color=col[ISOWP],lw=2,marker='o',mec=col[ISOWP], alpha=0.5)

                p0 = [1, threshold, 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_Tau, eff_Tau, p0)
                plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color=col[ISOWP], lw=1.5, alpha=0.5)
            
                print('\nfitted parameters for PUWP={0} and ISOWP={1} with threshold {2}GeV:'.format(args.PUWP,ISOWP,threshold))
                print('plateau efficiency = {0}'.format(popt[0]))
                print('turning point threshold = {0}GeV'.format(popt[1]))
                print('exponential = {0}'.format(popt[2]))

            plt.legend(loc = 'lower right')
            txt2 = (r'$E_{T}^{L1,\tau}$ > %i GeV' % (threshold))
            t2 = plt.text(50,0.25, txt2, ha='left')
            t2.set_bbox(dict(facecolor='white', edgecolor='white'))
            plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
            plt.ylabel(r'$\epsilon$')
            plt.title('Efficiency vs pT - PUWP={0}'.format(args.PUWP))
            plt.grid()
            plt.xlim(0, args.effFitLimit*3+3)
            plt.ylim(0., 1.10)
            plt.subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/efficiencies/eff_vs_pt_allISOWP_at{0}GeV.pdf'.format(threshold))
            plt.close()

        x_Tau = effVSeta_Tau_dict[name]['gentau_vis_abseta'] # is binned and the value is the mean of the entries per bin
        for threshold in [10,20,30]:
            plt.figure(figsize=(10,10))
            for ISOWP in ['10','15','20','90','95','99']:
                # all values for turnON curves
                eff_Tau = effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_low_Tau = effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_up_Tau = effVSeta_Tau_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)]

                plt.errorbar(x_Tau,eff_Tau,xerr=0.05,yerr=[eff_err_low_Tau,eff_err_up_Tau],ls='None',label=r'ISOWP = {0}'.format(ISOWP),color=col[ISOWP],lw=2,marker='o',mec=col[ISOWP], alpha=0.5)

                #p0 = [-2, -1, -1, -1, -1] # this is an mandatory initial guess for the fit
                #popt, pcov = curve_fit(poly, x_Tau, eff_Tau, p0)
                #plt.plot(x_Tau, poly(x_Tau, *popt), '-', label='_', color=col[ISOWP], lw=1.5, alpha=0.5)
            
            plt.legend(loc = 'lower left')
            txt2 = (r'$E_{T}^{L1,\tau}$ > %i GeV' % (threshold))
            t2 = plt.text(1.6,0.25, txt2, ha='left')
            t2.set_bbox(dict(facecolor='white', edgecolor='white'))
            plt.xlabel(r'$\eta^{gen,\tau}\ [GeV]$')
            plt.ylabel(r'$\epsilon$')
            plt.title('Efficiency vs pT - PUWP={0}'.format(args.PUWP))
            plt.grid()
            plt.xlim(1.5, 3.0)
            plt.ylim(0., 1.10)
            plt.subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/efficiencies/eff_vs_eta_allISOWP_at{0}GeV.pdf'.format(threshold))
            plt.close()

        x_DM0_Tau = effVSpt_TauDM0_dict[name]['gentau_vis_pt']
        x_DM1_Tau = effVSpt_TauDM1_dict[name]['gentau_vis_pt']
        x_DM2_Tau = effVSpt_TauDM2_dict[name]['gentau_vis_pt']
        for threshold in [10,20,30]:
            for ISOWP in ['10','15','20','90','95','99']:
                # all values for turnON curves
                effTauDM0 = effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)]
                effTauDM1 = effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)]
                effTauDM2 = effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_low_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_low_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_low_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_err_low_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_up_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_up_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)]
                eff_err_up_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_ISOWP{0}_err_up_at{1}GeV'.format(ISOWP,threshold)]

                plt.figure(figsize=(10,10))
                plt.errorbar(x_DM0_Tau,effTauDM0,xerr=1,yerr=[eff_err_low_TauDM0,eff_err_up_TauDM0],ls='None',label=r'1-prong',color='limegreen',lw=2,marker='o',mec='limegreen')
                plt.errorbar(x_DM1_Tau,effTauDM1,xerr=1,yerr=[eff_err_low_TauDM1,eff_err_up_TauDM1],ls='None',label=r'1-prong + $\pi^{0}$',color='darkorange',lw=2,marker='o',mec='darkorange')
                plt.errorbar(x_DM2_Tau,effTauDM2,xerr=1,yerr=[eff_err_low_TauDM2,eff_err_up_TauDM2],ls='None',label=r'3-prong (+ $\pi^{0}$)',color='fuchsia',lw=2,marker='o',mec='fuchsia')

                p0 = [1, threshold, 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_DM0_Tau, effTauDM0, p0)
                plt.plot(x_DM0_Tau, sigmoid(x_DM0_Tau, *popt), '-', label='_', color='limegreen', lw=1.5)

                p0 = [1, threshold, 1] # this is an mandatory initial guess for the fit
                popt, pcov = curve_fit(sigmoid, x_DM1_Tau, effTauDM1, p0)
                plt.plot(x_DM1_Tau, sigmoid(x_DM1_Tau, *popt), '-', label='_', color='darkorange', lw=1.5)

                p0 = [1, threshold, 1] # this is an mandatory initial guess for the fit
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
                plt.title('Efficiency vs pT - PUWP={0} - ISOWP={1}'.format(args.PUWP,ISOWP))
                plt.grid()
                plt.xlim(0, args.effFitLimit*3+3)
                plt.ylim(0., 1.10)
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/efficiencies/eff_vs_pt_ISOWP{0}_at{1}GeV.pdf'.format(ISOWP,threshold))
                plt.close()


# restore normal output
sys.stdout = sys.__stdout__
