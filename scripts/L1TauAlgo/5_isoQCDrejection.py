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
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
import argparse
import sys

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

def efficiency(group, threshold, ISOWP):
    tot = group.shape[0]
    if   ISOWP == 01: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP01 == True)].shape[0]
    elif ISOWP == 05: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP05 == True)].shape[0]
    elif ISOWP == 10: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True)].shape[0]
    else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True)].shape[0]
    return float(sel)/float(tot)

def efficiency_err(group, threshold, ISOWP, upper=False):
    tot = group.shape[0]
    if   ISOWP == 01: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP01 == True)].shape[0]
    elif ISOWP == 05: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP05 == True)].shape[0]
    elif ISOWP == 10: sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True)].shape[0]
    else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True)].shape[0]
    
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
    with open(source,'r') as f:
        return pickle.load(f)

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
    parser.add_argument('--effFitLimit', dest='effFitLimit', help='how many gentau_pt bins you wnat to consider for the fit of the turnON? (default: 33 bins = <102GeV)', default=33)
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    print('** INFO: using PU rejection BDT WP: '+args.PUWP)

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
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_C1fullC2C3_fullPUnoPt'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_C1fullC2C3_fullPUnoPt'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/isolation_C1fullC2C3_fullPUnoPt_PUWP{0}'.format(args.PUWP)
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/isolation_C1fullC2C3_fullPUnoPt_PUWP{0}'.format(args.PUWP)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/isolation_C1fullC2C3_fullPUnoPt_PUWP{0}/performance.log".format(args.PUWP))

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_isoCalculated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_isoCalculated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_model_dict = {
        'threshold'    : model_outdir+'/model_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
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
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP05_dict = {
        'threshold'    : model_outdir+'/WP05_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_WP01_dict = {
        'threshold'    : model_outdir+'/WP01_isolation_PUWP{0}_th_PU200.pkl'.format(args.PUWP),
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}

    # efficiencies
    effVSpt_Tau_dict = {}
    effVSpt_TauDM0_dict = {}
    effVSpt_TauDM1_dict = {}
    effVSpt_TauDM2_dict = {}

    # features used for the C2 calibration step - FULL AVAILABLE
    features = ['cl3d_c1', 'cl3d_c2', 'cl3d_c3', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_eSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_eSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_eIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_eIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']

    output = 'sgnId'

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
    bdtWP01_dict = {}
    bdtWP05_dict = {}
    bdtWP10_dict = {}
    bdtWP15_dict = {}

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
        
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting ISO QCD rejection for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()


        ######################### SELECT EVENTS FOR TRAINING #########################

        # old naming: iso_pid now substituted by sgnID
        #dfTraining_dict[name]['iso_pid'] = dfTraining_dict[name]['gentau_decayMode'].copy(deep=True)
        #dfValidation_dict[name]['iso_pid'] = dfValidation_dict[name]['gentau_decayMode'].copy(deep=True)
        #dfTraining_dict[name]['iso_pid'].replace([0,1,10,11], 1, inplace=True)
        #dfTraining_dict[name]['iso_pid'].replace([-2,-1], 0, inplace=True)
        #dfValidation_dict[name]['iso_pid'].replace([0,1,10,11], 1, inplace=True)
        #dfValidation_dict[name]['iso_pid'].replace([-2,-1], 0, inplace=True)

        dfTr = dfTraining_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)
        dfVal = dfValidation_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)

        ######################### TRAINING OF BDT #########################

        model_dict[name], fpr_train_dict[name], tpr_train_dict[name], threshold_train_dict[name], fpr_test_dict[name], tpr_test_dict[name], threshold_test_dict[name], testAuroc_dict[name], trainAuroc_dict[name] = train_xgb(dfTr, features, output, params_dict, test_fraction=0.3)

        print('\n** INFO: training and test AUROC:')
        print('  -- training AUROC: {0}'.format(trainAuroc_dict[name]))
        print('  -- test AUROC: {0}'.format(testAuroc_dict[name]))

        bdtWP01_dict[name] = np.interp(0.01, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP05_dict[name] = np.interp(0.05, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP10_dict[name] = np.interp(0.10, fpr_train_dict[name], threshold_train_dict[name])
        bdtWP15_dict[name] = np.interp(0.15, fpr_train_dict[name], threshold_train_dict[name])
        
        save_obj(model_dict[name], outFile_model_dict[name])
        save_obj(bdtWP01_dict[name], outFile_WP01_dict[name])
        save_obj(bdtWP05_dict[name], outFile_WP05_dict[name])
        save_obj(bdtWP10_dict[name], outFile_WP10_dict[name])
        save_obj(bdtWP15_dict[name], outFile_WP15_dict[name])

        # print some info about the WP and FPR at different sgn efficiency levels
        print('\n**INFO: BDT WP for: 0.01FPR {0} -  0.05FPR {1} -  0.10FPR {2} - 0.15FPR {3}'.format(bdtWP01_dict[name], bdtWP05_dict[name], bdtWP10_dict[name], bdtWP15_dict[name]))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfVal[features], label=dfVal[output], feature_names=features)
        dfVal['cl3d_isobdt_score'] = model_dict[name].predict(full)

        fpr_validation_dict[name], tpr_validation_dict[name], threshold_validation_dict[name] = metrics.roc_curve(dfVal[output], dfVal['cl3d_isobdt_score'])
        auroc_validation = metrics.roc_auc_score(dfVal['sgnId'],dfVal['cl3d_isobdt_score'])

        print('\n** INFO: validation of the BDT')
        print('  -- validation AUC: {0}'.format(auroc_validation))


        ######################### APPLICATION OF BDT TO ALL DATASETS #########################

        full = xgb.DMatrix(data=dfTraining_dict[name][features], label=dfTraining_dict[name][output], feature_names=features)
        dfTraining_dict[name]['cl3d_isobdt_score'] = model_dict[name].predict(full)
        dfTraining_dict[name]['cl3d_isobdt_passWP01'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP01_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP05'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP05_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP10'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfTraining_dict[name]['cl3d_isobdt_passWP15'] = dfTraining_dict[name]['cl3d_isobdt_score'] > bdtWP15_dict[name]

        full = xgb.DMatrix(data=dfValidation_dict[name][features], label=dfValidation_dict[name][output], feature_names=features)
        dfValidation_dict[name]['cl3d_isobdt_score'] = model_dict[name].predict(full)
        dfValidation_dict[name]['cl3d_isobdt_passWP01'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP01_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP05'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP05_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP10'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfValidation_dict[name]['cl3d_isobdt_passWP15'] = dfValidation_dict[name]['cl3d_isobdt_score'] > bdtWP15_dict[name]

        full = xgb.DMatrix(data=dfTr[features], label=dfTr[output], feature_names=features)
        dfTr['cl3d_isobdt_score'] = model_dict[name].predict(full)
        dfTr['cl3d_isobdt_passWP01'] = dfTr['cl3d_isobdt_score'] > bdtWP01_dict[name]
        dfTr['cl3d_isobdt_passWP05'] = dfTr['cl3d_isobdt_score'] > bdtWP05_dict[name]
        dfTr['cl3d_isobdt_passWP10'] = dfTr['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfTr['cl3d_isobdt_passWP15'] = dfTr['cl3d_isobdt_score'] > bdtWP15_dict[name]

        dfVal['cl3d_isobdt_passWP01'] = dfVal['cl3d_isobdt_score'] > bdtWP01_dict[name]
        dfVal['cl3d_isobdt_passWP05'] = dfVal['cl3d_isobdt_score'] > bdtWP05_dict[name]
        dfVal['cl3d_isobdt_passWP10'] = dfVal['cl3d_isobdt_score'] > bdtWP10_dict[name]
        dfVal['cl3d_isobdt_passWP15'] = dfVal['cl3d_isobdt_score'] > bdtWP15_dict[name]

        QCDtot = pd.concat([dfTr.query('gentau_decayMode==-2'), dfVal.query('gentau_decayMode==-2')], sort=False)
        QCD01 = QCDtot.query('cl3d_isobdt_passWP01==True')
        QCD05 = QCDtot.query('cl3d_isobdt_passWP05==True')
        QCD10 = QCDtot.query('cl3d_isobdt_passWP10==True')
        QCD15 = QCDtot.query('cl3d_isobdt_passWP15==True')

        print('\n**INFO: QCD cluster passing the ISO QCD rejection:')
        print('  -- number of QCD events passing WP01: {0}%'.format(round(float(QCD01['cl3d_isobdt_passWP01'].count())/float(QCDtot['cl3d_isobdt_passWP01'].count())*100,2)))
        print('  -- number of QCD events passing WP05: {0}%'.format(round(float(QCD05['cl3d_isobdt_passWP05'].count())/float(QCDtot['cl3d_isobdt_passWP05'].count())*100,2)))
        print('  -- number of QCD events passing WP10: {0}%'.format(round(float(QCD10['cl3d_isobdt_passWP10'].count())/float(QCDtot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of QCD events passing WP15: {0}%'.format(round(float(QCD15['cl3d_isobdt_passWP15'].count())/float(QCDtot['cl3d_isobdt_passWP15'].count())*100,2)))

        Nutot = pd.concat([dfTr.query('gentau_decayMode==-1'), dfVal.query('gentau_decayMode==-1')], sort=False)
        Nu01 = Nutot.query('cl3d_isobdt_passWP01==True')
        Nu05 = Nutot.query('cl3d_isobdt_passWP05==True')
        Nu10 = Nutot.query('cl3d_isobdt_passWP10==True')
        Nu15 = Nutot.query('cl3d_isobdt_passWP15==True')

        print('\n**INFO: PU cluster passing the ISO QCD rejection:')
        print('  -- number of PU events passing WP01: {0}%'.format(round(float(Nu01['cl3d_isobdt_passWP01'].count())/float(Nutot['cl3d_isobdt_passWP01'].count())*100,2)))
        print('  -- number of PU events passing WP05: {0}%'.format(round(float(Nu05['cl3d_isobdt_passWP05'].count())/float(Nutot['cl3d_isobdt_passWP05'].count())*100,2)))
        print('  -- number of PU events passing WP10: {0}%'.format(round(float(Nu10['cl3d_isobdt_passWP10'].count())/float(Nutot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of PU events passing WP15: {0}%'.format(round(float(Nu15['cl3d_isobdt_passWP15'].count())/float(Nutot['cl3d_isobdt_passWP15'].count())*100,2)))

        Tautot = pd.concat([dfTr.query('sgnId==1'), dfVal.query('sgnId==1')], sort=False)
        Tau01 = Tautot.query('cl3d_isobdt_passWP01==True')
        Tau05 = Tautot.query('cl3d_isobdt_passWP05==True')
        Tau10 = Tautot.query('cl3d_isobdt_passWP10==True')
        Tau15 = Tautot.query('cl3d_isobdt_passWP15==True')

        print('\n**INFO: Tau cluster passing the ISO QCD rejection:')
        print('  -- number of Tau events passing WP01: {0}%'.format(round(float(Tau01['cl3d_isobdt_passWP01'].count())/float(Tautot['cl3d_isobdt_passWP01'].count())*100,2)))
        print('  -- number of Tau events passing WP05: {0}%'.format(round(float(Tau05['cl3d_isobdt_passWP05'].count())/float(Tautot['cl3d_isobdt_passWP05'].count())*100,2)))
        print('  -- number of Tau events passing WP10: {0}%'.format(round(float(Tau10['cl3d_isobdt_passWP10'].count())/float(Tautot['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of Tau events passing WP15: {0}%'.format(round(float(Tau15['cl3d_isobdt_passWP15'].count())/float(Tautot['cl3d_isobdt_passWP15'].count())*100,2)))
        print('')
        print('  -- number of Tau pT>30GeV events passing WP01: {0}%'.format(round(float(Tau01.query('cl3d_pt>30')['cl3d_isobdt_passWP01'].count())/float(Tautot.query('cl3d_pt>30')['cl3d_isobdt_passWP01'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP05: {0}%'.format(round(float(Tau05.query('cl3d_pt>30')['cl3d_isobdt_passWP05'].count())/float(Tautot.query('cl3d_pt>30')['cl3d_isobdt_passWP05'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP10: {0}%'.format(round(float(Tau10.query('cl3d_pt>30')['cl3d_isobdt_passWP10'].count())/float(Tautot.query('cl3d_pt>30')['cl3d_isobdt_passWP10'].count())*100,2)))
        print('  -- number of Tau pT>30GeV events passing WP15: {0}%'.format(round(float(Tau15.query('cl3d_pt>30')['cl3d_isobdt_passWP15'].count())/float(Tautot.query('cl3d_pt>30')['cl3d_isobdt_passWP15'].count())*100,2)))


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
        #sel = dfNu_dict[name]['cl3d_isobdt_passWP01'] == True
        #dfNu_dict[name] = dfNu_dict[name][sel]
        #print np.unique(dfNu_dict[name].reset_index()['event']).shape[0]

        print('\n** INFO: finished ISO QCD rejection for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')

      
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    os.system('mkdir -p '+plotdir+'/features/')

    # name : [title, [min, max, step]
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
                     'cl3d_NclIso_dR4'              : [r'Number of clusters inside an isolation cone of dR=0.4',[0.,10.,10]],
                     'cl3d_etIso_dR4'               : [r'Clusters $E_{T}$ inside an isolation cone of dR=0.4',[0.,200.,40]],
                     'tower_etSgn_dRsgn1'           : [r'$E_{T}$ inside a signal cone of dR=0.1',[0.,200.,40]],
                     'tower_eSgn_dRsgn1'            : [r'$E$ inside a signal cone of dR=0.1',[0.,400.,40]],
                     'tower_etSgn_dRsgn2'           : [r'$E_{T}$ inside a signal cone of dR=0.2',[0.,200.,40]],
                     'tower_eSgn_dRsgn2'            : [r'$E$ inside a signal cone of dR=0.2',[0.,400.,40]],
                     'tower_etIso_dRsgn1_dRiso3'    : [r'Towers $E_{T}$ between dR=0.1-0.3 around L1 candidate',[0.,200.,40]],
                     'tower_eIso_dRsgn1_dRiso3'     : [r'Towers $E$ between dR=0.1-0.3 around L1 candidate',[0.,400.,40]],
                     'tower_etEmIso_dRsgn1_dRiso3'  : [r'Towers $E_{T}^{em}$ between dR=0.1-0.3 around L1 candidate',[0.,150.,30]],
                     'tower_etHadIso_dRsgn1_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.1-0.7 around L1 candidate',[0.,200.,40]],
                     'tower_etIso_dRsgn2_dRiso4'    : [r'Towers $E_{T}$ between dR=0.2-0.4 around L1 candidate',[0.,200.,40]],
                     'tower_eIso_dRsgn2_dRiso4'     : [r'Towers $E$ between dR=0.2-0.4 around L1 candidate',[0.,400.,40]],
                     'tower_etEmIso_dRsgn2_dRiso4'  : [r'Towers $E_{T}^{em}$ between dR=0.2-0.4 around L1 candidate',[0.,150.,30]],
                     'tower_etHadIso_dRsgn2_dRiso7' : [r'Towers $E_{T}^{had}$ between dR=0.2-0.7 around L1 candidate',[0.,200.,40]]
    }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfQCD = dfTr.query('gentau_decayMode==-2')
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
        plt.xlim(0.2,1.001)
        plt.yscale('log')
        plt.ylim(0.001,1)
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
    plt.xlim(0.2,1.001)
    plt.yscale('log')
    plt.ylim(0.001,1)
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
    plt.xlim(0.2,1.001)
    plt.yscale('log')
    plt.ylim(0.001,1)
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

        for ISOWP in [01,05,10,15]:
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
            01 : 'green',
            05 : 'blue',
            10 : 'red',
            15 : 'fuchsia'
        }
        
        x_Tau = effVSpt_Tau_dict[name]['gentau_vis_pt'] # is binned and the value is the mean of the entries per bin
        for threshold in [10,20,30]:
            plt.figure(figsize=(10,10))
            for ISOWP in [01,05,10,15]:
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
            for ISOWP in [01,05,10,15]:
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
            for ISOWP in [01,05,10,15]:
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
