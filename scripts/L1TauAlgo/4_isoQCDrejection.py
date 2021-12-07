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


def train_xgb(dfTrain, features, output, hyperparams, test_fraction=0.3):    
    X_train, X_test, y_train, y_test = train_test_split(dfTrain[features], dfTrain[output], test_size=test_fraction)

    classifier = xgb.XGBClassifier(objective=hyperparams['objective'], booster=hyperparams['booster'], eval_metric=hyperparams['eval_metric'],
                                   reg_alpha=hyperparams['reg_alpha'], reg_lambda=hyperparams['reg_lambda'], max_depth=hyperparams['max_depth'],
                                   learning_rate=hyperparams['learning_rate'], subsample=hyperparams['subsample'], colsample_bytree=hyperparams['colsample_bytree'], 
                                   n_estimators=hyperparams['num_trees'], use_label_encoder=False)

    classifier.fit(X_train, y_train)
    X_train['bdt_output'] = classifier.predict_proba(X_train)[:,1]
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, X_train['bdt_output'])
    X_test['bdt_output'] = classifier.predict_proba(X_test)[:,1]
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, X_test['bdt_output'])

    auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
    auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])

    return classifier, fpr_train, tpr_train, threshold_train, fpr_test, tpr_test, threshold_test, auroc_test, auroc_train

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
    parser.add_argument('--doONNX', dest='doONNX', help='run ONNX conversione', action='store_true', default=False)
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
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_skimPUnoPt{0}{1}'.format(tag1, '_'+tag2)
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_skimPUnoPt{0}_skimISO{0}{1}_againstPU'.format(tag1, tag2)
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/isolation_skimPUnoPt_skimISO{0}{1}_PUWP{2}_againstPU'.format(tag1, tag2, args.PUWP)
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/isolation_skimPUnoPt{0}_skimISO{0}{1}_againstPU'.format(tag1, tag2)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/isolation_skimPUnoPt{0}_skimISO{0}{1}_againstPU/performance_PUWP{2}.log".format(tag1, tag2, args.PUWP))

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

    outFile_model_dict = {
        'threshold'    : model_outdir+'/model_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP15_dict = {
        'threshold'    : model_outdir+'/WP15_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP10_dict = {
        'threshold'    : model_outdir+'/WP10_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP20_dict = {
        'threshold'    : model_outdir+'/WP20_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_dict = {
        'threshold'    : model_outdir+'/WP90_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_dict = {
        'threshold'    : model_outdir+'/WP95_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_dict = {
        'threshold'    : model_outdir+'/WP99_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
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

    # # features used for the ISO QCD rejection - FULL AVAILABLE
    # features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    # # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
    # features2shift = ['cl3d_NclIso_dR4', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',]
    # features2saturate = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    # saturation_dict = {'cl3d_pt_tr': [0, 200],
    #                    'cl3d_abseta': [1.45, 3.2],
    #                    'cl3d_seetot': [0, 0.17],
    #                    'cl3d_seemax': [0, 0.6],
    #                    'cl3d_spptot': [0, 0.17],
    #                    'cl3d_sppmax': [0, 0.53],
    #                    'cl3d_szz': [0, 141.09],
    #                    'cl3d_srrtot': [0, 0.02],
    #                    'cl3d_srrmax': [0, 0.02],
    #                    'cl3d_srrmean': [0, 0.01],
    #                    'cl3d_hoe': [0, 63],
    #                    'cl3d_meanz': [305, 535],
    #                    'cl3d_etIso_dR4': [0, 58],
    #                    'tower_etSgn_dRsgn1': [0, 194],
    #                    'tower_etSgn_dRsgn2': [0, 228],
    #                    'tower_etIso_dRsgn1_dRiso3': [0, 105],
    #                    'tower_etEmIso_dRsgn1_dRiso3': [0, 72],
    #                    'tower_etHadIso_dRsgn1_dRiso7': [0, 43],
    #                    'tower_etIso_dRsgn2_dRiso4': [0, 129],
    #                    'tower_etEmIso_dRsgn2_dRiso4': [0, 95],
    #                    'tower_etHadIso_dRsgn2_dRiso7': [0, 42]
    #                 }


    # # BDT hyperparameters
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
    features = []
    renaming_dict = {}
    if args.PUWP == '99':
        #features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
        features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']
        renaming_dict = {'cl3d_pt_tr':'f1', 'cl3d_abseta':'f2', 'cl3d_coreshowerlength':'f3', 'cl3d_spptot':'f4', 'cl3d_srrtot':'f5', 'cl3d_srrmean':'f6', 'cl3d_hoe':'f7', 'cl3d_meanz':'f8', 'cl3d_NclIso_dR4':'f9', 'tower_etSgn_dRsgn1':'f10', 'tower_etSgn_dRsgn2':'f11', 'tower_etIso_dRsgn1_dRiso3':'f12'}
        unnaming_dict = {'f1':'cl3d_pt_tr', 'f2':'cl3d_abseta', 'f3':'cl3d_coreshowerlength', 'f4':'cl3d_spptot', 'f5':'cl3d_srrtot', 'f6':'cl3d_srrmean', 'f7':'cl3d_hoe', 'f8':'cl3d_meanz', 'f9':'cl3d_NclIso_dR4', 'f10':'tower_etSgn_dRsgn1', 'f11':'tower_etSgn_dRsgn2', 'f12':'tower_etIso_dRsgn1_dRiso3'}

    elif args.PUWP == '95':
        #features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
        features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
        renaming_dict = {'cl3d_pt_tr':'f1', 'cl3d_abseta':'f2', 'cl3d_coreshowerlength':'f3', 'cl3d_srrtot':'f4', 'cl3d_srrmean':'f5', 'cl3d_hoe':'f6', 'cl3d_meanz':'f7', 'cl3d_NclIso_dR4':'f8', 'tower_etSgn_dRsgn1':'f9', 'tower_etSgn_dRsgn2':'f10', 'tower_etIso_dRsgn1_dRiso3':'f11'}
        unnaming_dict = {'f1':'cl3d_pt_tr', 'f2':'cl3d_abseta', 'f3':'cl3d_coreshowerlength', 'f4':'cl3d_srrtot', 'f5':'cl3d_srrmean', 'f6':'cl3d_hoe', 'f7':'cl3d_meanz', 'f8':'cl3d_NclIso_dR4', 'f9':'tower_etSgn_dRsgn1', 'f10':'tower_etSgn_dRsgn2', 'f11':'tower_etIso_dRsgn1_dRiso3'}

    else:
        #features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
        features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
        renaming_dict = {'cl3d_pt_tr':'f1', 'cl3d_abseta':'f2', 'cl3d_spptot':'f3', 'cl3d_srrtot':'f4', 'cl3d_srrmean':'f5', 'cl3d_hoe':'f6', 'cl3d_meanz':'f7', 'cl3d_NclIso_dR4':'f8', 'tower_etSgn_dRsgn1':'f9', 'tower_etSgn_dRsgn2':'f10', 'tower_etIso_dRsgn1_dRiso3':'f11'}
        unnaming_dict = {'f1':'cl3d_pt_tr', 'f2':'cl3d_abseta', 'f3':'cl3d_spptot', 'f4':'cl3d_srrtot', 'f5':'cl3d_srrmean', 'f6':'cl3d_hoe', 'f7':'cl3d_meanz', 'f8':'cl3d_NclIso_dR4', 'f9':'tower_etSgn_dRsgn1', 'f10':'tower_etSgn_dRsgn2', 'f11':'tower_etIso_dRsgn1_dRiso3'}

    features2shift = ['cl3d_NclIso_dR4', 'cl3d_coreshowerlength']
    features2saturate = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
    saturation_dict = {'cl3d_pt_tr': [0, 200],
                       'cl3d_abseta': [1.45, 3.2],
                       'cl3d_spptot': [0, 0.17],
                       'cl3d_srrtot': [0, 0.02],
                       'cl3d_srrmean': [0, 0.01],
                       'cl3d_hoe': [0, 63],
                       'cl3d_meanz': [305, 535],
                       'tower_etSgn_dRsgn1': [0, 194],
                       'tower_etSgn_dRsgn2': [0, 228],
                       'tower_etIso_dRsgn1_dRiso3': [0, 105]
                      }
    # BDT hyperparameters
    params_dict = {}
    params_dict['objective']          = 'binary:logistic'
    params_dict['booster']            = 'gbtree'
    params_dict['eval_metric']        = 'logloss'
    params_dict['reg_alpha']          = 9
    params_dict['reg_lambda']         = 5
    params_dict['max_depth']          = 4 # from HPO
    params_dict['learning_rate']      = 0.37 # from HPO
    params_dict['subsample']          = 0.12 # from HPO
    params_dict['colsample_bytree']   = 0.9 # from HPO
    params_dict['num_trees']          = 30 # from HPO

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
    bdtWP10_dict = {}
    bdtWP15_dict = {}
    bdtWP20_dict = {}
    bdtWP90_dict = {}
    bdtWP95_dict = {}
    bdtWP99_dict = {}

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


        ######################### DO RESCALING OF THE FEATURES #########################
        if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            # shift features to be shifted
            for feat in features2shift:
                dfTraining_dict[name][feat] = dfTraining_dict[name][feat] - 32
                dfValidation_dict[name][feat] = dfValidation_dict[name][feat] - 32

            # saturate features
            for feat in features2saturate:
                dfTraining_dict[name][feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                dfValidation_dict[name][feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                
                # fill the bounds DF
                bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

            scale_range = [-32,32]
            MMS = MinMaxScaler(scale_range)

            for feat in features2saturate:
                MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
                dfTraining_dict[name][feat] = MMS.transform( np.array(dfTraining_dict[name][feat]).reshape(-1,1) )
                dfValidation_dict[name][feat] = MMS.transform( np.array(dfValidation_dict[name][feat]).reshape(-1,1) )

        ######################### SELECT EVENTS FOR TRAINING #########################

        # here we select only the genuine QCD and Tau events
        #dfTr = dfTraining_dict[name].query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
        #dfVal = dfValidation_dict[name].query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
        dfTr = dfTraining_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)
        dfVal = dfValidation_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)

        ######################### RENAME FEATURES #########################

        dfTr.rename(columns=renaming_dict, inplace=True)
        dfVal.rename(columns=renaming_dict, inplace=True)

        dfTraining_dict[name].rename(columns=renaming_dict, inplace=True)
        dfValidation_dict[name].rename(columns=renaming_dict, inplace=True)


        ######################### TRAINING OF BDT #########################

        print('\n** INFO: training BDT')
        model_dict[name], fpr_train_dict[name], tpr_train_dict[name], threshold_train_dict[name], fpr_test_dict[name], tpr_test_dict[name], threshold_test_dict[name], testAuroc_dict[name], trainAuroc_dict[name] = train_xgb(dfTr, features, output, params_dict, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_dict[name]))

        bdtWP10_dict[name] = np.interp(0.10, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP15_dict[name] = np.interp(0.15, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP20_dict[name] = np.interp(0.20, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP90_dict[name] = np.interp(0.90, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP95_dict[name] = np.interp(0.95, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP99_dict[name] = np.interp(0.99, tpr_train_dict[name], threshold_train_dict[name])

        save_obj(model_dict[name], outFile_model_dict[name])
        save_obj(bdtWP10_dict[name], outFile_WP10_dict[name])
        save_obj(bdtWP15_dict[name], outFile_WP15_dict[name])
        save_obj(bdtWP20_dict[name], outFile_WP20_dict[name])
        save_obj(bdtWP90_dict[name], outFile_WP90_dict[name])
        save_obj(bdtWP95_dict[name], outFile_WP95_dict[name])
        save_obj(bdtWP99_dict[name], outFile_WP99_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.10FPR {0} - 0.15FPR {1} - 0.20FPR {2}'.format(bdtWP10_dict[name], bdtWP15_dict[name], bdtWP20_dict[name]))
        print('\n**INFO: BDT WP for: 0.90TPR {0} - 0.95TPR {1} - 0.99TPR {2}'.format(bdtWP90_dict[name], bdtWP95_dict[name], bdtWP99_dict[name]))

        if args.doONNX:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_type = [('float_input', FloatTensorType([1, 6]))]
            onnx_model = onnxmltools.convert.convert_xgboost(model_dict[name], initial_types=initial_type)
            with open(model_outdir+"/ISOmodel_nonRscld.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())
            with open("/home/llr/cms/motta/HGCAL/CMSSW_11_1_7/src/L1Trigger/L1CaloTrigger/data/ISOmodel_nonRscld.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())


        ######################### VALIDATION OF BDT #########################

        dfVal['cl3d_isobdt_score'] = model_dict[name].predict_proba(dfVal[features])[:,1]

        fpr_validation_dict[name], tpr_validation_dict[name], threshold_validation_dict[name] = metrics.roc_curve(dfVal[output], dfVal['cl3d_isobdt_score'])
        auroc_validation = metrics.roc_auc_score(dfVal['sgnId'],dfVal['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        dfTraining_dict[name]['cl3d_isobdt_score'] = model_dict[name].predict_proba(dfTraining_dict[name][features])[:,1]
        dfTraining_dict[name]['cl3d_isobdt_passWP10'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP15'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP15_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP20'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP20_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP90'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP90_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP95'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP95_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP99'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP99_dict[name]

        dfValidation_dict[name]['cl3d_isobdt_score'] = model_dict[name].predict_proba(dfValidation_dict[name][features])[:,1]
        dfValidation_dict[name]['cl3d_isobdt_passWP10'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP15'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP15_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP20'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP20_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP90'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP90_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP95'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP95_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP99'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP99_dict[name]

        dfTr['cl3d_isobdt_score'] = model_dict[name].predict_proba(dfTr[features])[:,1]
        
        dfTr['cl3d_isobdt_passWP10'] = dfTr['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfTr['cl3d_isobdt_passWP15'] = dfTr['cl3d_isobdt_score'] > bdtWP15_dict[name]
        dfTr['cl3d_isobdt_passWP20'] = dfTr['cl3d_isobdt_score'] > bdtWP20_dict[name]
        dfTr['cl3d_isobdt_passWP90'] = dfTr['cl3d_isobdt_score'] > bdtWP90_dict[name]
        dfTr['cl3d_isobdt_passWP95'] = dfTr['cl3d_isobdt_score'] > bdtWP95_dict[name]
        dfTr['cl3d_isobdt_passWP99'] = dfTr['cl3d_isobdt_score'] > bdtWP99_dict[name]

        dfVal['cl3d_isobdt_passWP10'] = dfVal['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfVal['cl3d_isobdt_passWP15'] = dfVal['cl3d_isobdt_score'] > bdtWP15_dict[name]
        dfVal['cl3d_isobdt_passWP20'] = dfVal['cl3d_isobdt_score'] > bdtWP20_dict[name]
        dfVal['cl3d_isobdt_passWP90'] = dfVal['cl3d_isobdt_score'] > bdtWP90_dict[name]
        dfVal['cl3d_isobdt_passWP95'] = dfVal['cl3d_isobdt_score'] > bdtWP95_dict[name]
        dfVal['cl3d_isobdt_passWP99'] = dfVal['cl3d_isobdt_score'] > bdtWP99_dict[name]


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

        ######################### RENAME FEATURES BACK #########################

        dfTr.rename(columns=unnaming_dict, inplace=True)
        dfVal.rename(columns=unnaming_dict, inplace=True)

        dfTraining_dict[name].rename(columns=unnaming_dict, inplace=True)
        dfValidation_dict[name].rename(columns=unnaming_dict, inplace=True)

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

    if args.PUWP == '99':
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    elif args.PUWP == '95':
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    else:
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']


    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfQCD = dfTr.query('gentau_decayMode==-2 and cl3d_isbestmatch==True')
        dfNu  = dfTr.query('gentau_decayMode==-1')
        dfTau = dfTr.query('sgnId==1')
        dfQCDNu = dfTr.query('sgnId==0')

        for var in features:
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
        importance = model_dict[name].get_booster().get_score(importance_type='gain')
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
