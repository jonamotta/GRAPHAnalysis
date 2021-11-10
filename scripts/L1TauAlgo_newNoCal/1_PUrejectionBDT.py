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
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--doEfficiency', dest='doEfficiency', help='do you want calculate the efficiencies?', action='store_true', default=False)
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
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
    tag = "Rscld" if args.doRescale else ""
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated_C1fullC2C3'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/PUrejected_fullPUnoPt{0}'.format(tag)
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/PUrejection_fullPUnoPt{0}'.format(tag)
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/PUrejection_fullPUnoPt{0}'.format(tag)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/PUrejection_fullPUnoPt{0}/performance.log".format(tag))

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

    outFile_model_dict = {
        'threshold'    : model_outdir+'/model_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP90_dict = {
        'threshold'    : model_outdir+'/WP90_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP95_dict = {
        'threshold'    : model_outdir+'/WP95_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP99_dict = {
        'threshold'    : model_outdir+'/WP99_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_tpr_train_dict = {
        'threshold'    : model_outdir+'/TPR_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_fpr_train_dict = {
        'threshold'    : model_outdir+'/FPR_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_FPRnClusters_dict = {
        'threshold'    : model_outdir+'/FPRnClusters_PUrejection_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}

    # target of the training
    output = 'sgnId'

    # # features for BDT training - FULL AVAILABLE
    # features = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    # # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
    # features2shift = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',]
    # features2saturate = ['cl3d_abseta', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    # saturation_dict = {'cl3d_abseta': [1.45, 3.2],
    #                    'cl3d_seetot': [0, 0.17],
    #                    'cl3d_seemax': [0, 0.6],
    #                    'cl3d_spptot': [0, 0.17],
    #                    'cl3d_sppmax': [0, 0.53],
    #                    'cl3d_szz': [0, 141.09],
    #                    'cl3d_srrtot': [0, 0.02],
    #                    'cl3d_srrmax': [0, 0.02],
    #                    'cl3d_srrmean': [0, 0.01],
    #                    'cl3d_hoe': [0, 63],
    #                    'cl3d_meanz': [305, 535]
    #                 }
    # # BDT hyperparameters
    # params_dict = {}
    # params_dict['eval_metric']        = 'logloss'
    # params_dict['nthread']            = 10   # limit number of threads
    # params_dict['eta']                = 0.2 # learning rate
    # params_dict['max_depth']          = 5    # maximum depth of a tree
    # params_dict['subsample']          = 0.6 # fraction of events to train tree on
    # params_dict['colsample_bytree']   = 0.7 # fraction of features to train tree on
    # params_dict['objective']          = 'binary:logistic' # objective function
    # params_dict['alpha']              = 10
    # params_dict['lambda']             = 0.3
    # num_trees = 60  # number of trees to make

    # selected features from FS
    features = ['cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe']
    features2shift = ['cl3d_coreshowerlength']
    features2saturate = ['cl3d_abseta', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe']
    saturation_dict = {'cl3d_abseta': [1.45, 3.2],
                       'cl3d_srrtot': [0, 0.02],
                       'cl3d_srrmean': [0, 0.01],
                       'cl3d_hoe': [0, 63]
                      }
    # BDT hyperparameters
    params_dict = {}
    params_dict['objective']          = 'binary:logistic'
    params_dict['eval_metric']        = 'logloss'
    params_dict['nthread']            = 10
    params_dict['alpha']              = 9
    params_dict['lambda']             = 5
    params_dict['max_depth']          = 4 # from HPO
    params_dict['eta']                = 0.35 # from HPO
    params_dict['subsample']          = 0.22 # from HPO
    params_dict['colsample_bytree']   = 0.7 # from HPO
    num_trees = 24 # from HPO


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


        ######################### DO RESCALING OF THE FEATURES #########################
        if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            # shift features to be shifted
            for feat in features2shift:
                dfTraining_dict[name][feat] = dfTraining_dict[name][feat] - 25
                dfValidation_dict[name][feat] = dfValidation_dict[name][feat] - 25

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

        dfQCDTr = dfTraining_dict[name].query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)
        dfQCDVal = dfValidation_dict[name].query('gentau_decayMode==-2 and cl3d_isbestmatch==True').copy(deep=True)

        dfTr = dfTraining_dict[name].query('sgnId==1 or cl3d_isbestmatch==False').copy(deep=True)
        dfVal = dfValidation_dict[name].query('sgnId==1 or cl3d_isbestmatch==False').copy(deep=True)


        ######################### TRAINING OF BDT #########################

        print('\n** INFO: training BDT')
        model_dict[name], fpr_train_dict[name], tpr_train_dict[name], threshold_train_dict[name], fpr_test_dict[name], tpr_test_dict[name], threshold_test_dict[name], testAuroc_dict[name], trainAuroc_dict[name] = train_xgb(dfTr, features, output, params_dict, num_trees)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_dict[name]))

        save_obj(model_dict[name], outFile_model_dict[name])
        save_obj(fpr_train_dict[name], outFile_fpr_train_dict[name])
        save_obj(tpr_train_dict[name], outFile_tpr_train_dict[name])

        bdtWP99_dict[name] = np.interp(0.99, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP95_dict[name] = np.interp(0.95, tpr_train_dict[name], threshold_train_dict[name])
        bdtWP90_dict[name] = np.interp(0.90, tpr_train_dict[name], threshold_train_dict[name])
        
        save_obj(bdtWP99_dict[name], outFile_WP99_dict[name])
        save_obj(bdtWP95_dict[name], outFile_WP95_dict[name])
        save_obj(bdtWP90_dict[name], outFile_WP90_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(bdtWP99_dict[name], bdtWP95_dict[name], bdtWP90_dict[name]))
        print('\n**INFO: train bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.95, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.90, tpr_train_dict[name], fpr_train_dict[name])))
        print('**INFO: test bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_test_dict[name], fpr_test_dict[name]),np.interp(0.95, tpr_test_dict[name], fpr_test_dict[name]),np.interp(0.90, tpr_test_dict[name], fpr_test_dict[name])))

        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal[features], label=dfVal[output], feature_names=features)
        dfVal['cl3d_pubdt_score'] = model_dict[name].predict(full)

        fpr_validation_dict[name], tpr_validation_dict[name], threshold_validation_dict[name] = metrics.roc_curve(dfVal[output], dfVal['cl3d_pubdt_score'])
        auroc_validation = metrics.roc_auc_score(dfVal['sgnId'],dfVal['cl3d_pubdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_dict[name][features], label=dfTraining_dict[name][output], feature_names=features)
        dfTraining_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfTraining_dict[name]['cl3d_pubdt_passWP99'] = dfTraining_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfTraining_dict[name]['cl3d_pubdt_passWP95'] = dfTraining_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfTraining_dict[name]['cl3d_pubdt_passWP90'] = dfTraining_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfValidation_dict[name][features], label=dfValidation_dict[name][output], feature_names=features)
        dfValidation_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfValidation_dict[name]['cl3d_pubdt_passWP99'] = dfValidation_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfValidation_dict[name]['cl3d_pubdt_passWP95'] = dfValidation_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfValidation_dict[name]['cl3d_pubdt_passWP90'] = dfValidation_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfTr[features], label=dfTr[output], feature_names=features)
        dfTr['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfTr['cl3d_pubdt_passWP99'] = dfTr['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfTr['cl3d_pubdt_passWP95'] = dfTr['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfTr['cl3d_pubdt_passWP90'] = dfTr['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfVal[features], label=dfVal[output], feature_names=features)
        dfVal['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfVal['cl3d_pubdt_passWP99'] = dfVal['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfVal['cl3d_pubdt_passWP95'] = dfVal['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfVal['cl3d_pubdt_passWP90'] = dfVal['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfQCDTr[features], label=dfQCDTr[output], feature_names=features)
        dfQCDTr['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfQCDTr['cl3d_pubdt_passWP99'] = dfQCDTr['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfQCDTr['cl3d_pubdt_passWP95'] = dfQCDTr['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfQCDTr['cl3d_pubdt_passWP90'] = dfQCDTr['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfQCDVal[features], label=dfQCDVal[output], feature_names=features)
        dfQCDVal['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfQCDVal['cl3d_pubdt_passWP99'] = dfQCDVal['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfQCDVal['cl3d_pubdt_passWP95'] = dfQCDVal['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfQCDVal['cl3d_pubdt_passWP90'] = dfQCDVal['cl3d_pubdt_score'] > bdtWP90_dict[name]

        ######################### OVERALL EFFICIENCIES #########################

        QCDtot = pd.concat([dfQCDTr,dfQCDVal],sort=False)
        QCD99 = QCDtot.query('cl3d_pubdt_passWP99==True')
        QCD95 = QCDtot.query('cl3d_pubdt_passWP95==True')
        QCD90 = QCDtot.query('cl3d_pubdt_passWP90==True')

        print('\n**INFO: QCD cluster passing the PU rejection:')
        print('  -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99['cl3d_pubdt_passWP99'].count())/float(QCDtot['cl3d_pubdt_passWP99'].count())*100,2)))
        print('  -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95['cl3d_pubdt_passWP95'].count())/float(QCDtot['cl3d_pubdt_passWP95'].count())*100,2)))
        print('  -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90['cl3d_pubdt_passWP90'].count())/float(QCDtot['cl3d_pubdt_passWP90'].count())*100,2)))

        del QCDtot, QCD90, QCD95, QCD99

        TOT = pd.concat([dfTraining_dict[name],dfValidation_dict[name]],sort=False).query('cl3d_isbestmatch==False')
        TOT99 = TOT.query('cl3d_isbestmatch==False and cl3d_pubdt_passWP99==True').copy(deep=True)
        TOT95 = TOT.query('cl3d_isbestmatch==False and cl3d_pubdt_passWP95==True').copy(deep=True)
        TOT90 = TOT.query('cl3d_isbestmatch==False and cl3d_pubdt_passWP90==True').copy(deep=True)

        print('\nOVERALL BKG EFFICIENCIES:')
        print('     at 0.99 sgn efficiency: {0}%'.format(round(float(TOT99.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.95 sgn efficiency: {0}%'.format(round(float(TOT95.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.90 sgn efficiency: {0}%'.format(round(float(TOT90.shape[0])/float(TOT.shape[0])*100,2)))

        del TOT, TOT90, TOT95, TOT99

        TOT = pd.concat([dfTraining_dict[name],dfValidation_dict[name]],sort=False).query('sgnId==1 and cl3d_pt_c3>=30')
        TOT99 = TOT.query('cl3d_pubdt_passWP99==True').copy(deep=True)
        TOT95 = TOT.query('cl3d_pubdt_passWP95==True').copy(deep=True)
        TOT90 = TOT.query('cl3d_pubdt_passWP90==True').copy(deep=True)

        print('\nOVERALL SGN EFFICIENCIES FOR cl3d_pt_c3>30GeV:')
        print('     at 0.99 sgn efficiency: {0}%'.format(round(float(TOT99.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.95 sgn efficiency: {0}%'.format(round(float(TOT95.shape[0])/float(TOT.shape[0])*100,2)))
        print('     at 0.90 sgn efficiency: {0}%'.format(round(float(TOT90.shape[0])/float(TOT.shape[0])*100,2)))

        del TOT, TOT90, TOT95, TOT99

        ######################### SAVE FILES #########################

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()


        print('\n** INFO: finished PU rejection for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')

       
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    features_dict = {'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
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
    if args.doRescale:
        features_dict = {'cl3d_abseta'           : [r'3D cluster |$\eta$|',[-33.,33.,66]], 
                         'cl3d_showerlength'     : [r'3D cluster shower length',[-33.,33.,66]], 
                         'cl3d_coreshowerlength' : [r'Core shower length ',[-33.,33.,66]], 
                         'cl3d_firstlayer'       : [r'3D cluster first layer',[-33.,33.,66]], 
                         'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_seemax'           : [r'3D cluster max $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_sppmax'           : [r'3D cluster max $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[-33.,33.,66]], 
                         'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmax'           : [r'3D cluster max $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[-33.,33.,66]], 
                         'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[-33.,33.,66]], 
                         'cl3d_meanz'            : [r'3D cluster meanz',[-33.,33.,66]], 
        }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfNu = dfTr.query('sgnId==0')
        dfTau = dfTr.query('sgnId==1')

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


        df = pd.concat([dfTr.query('sgnId==1').sample(1500), dfTr.query('cl3d_isbestmatch==False').sample(1500)], sort=False)[features]
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

        dfNu = dfVal.query('sgnId==0')
        dfTau = dfVal.query('sgnId==1')

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
