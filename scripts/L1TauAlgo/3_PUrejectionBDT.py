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
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_calibrated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_calibrated.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_PUrejected.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUrejected.hdf5',
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

    dfQCDTraining_dict = {}
    dfQCDValidation_dict = {}
    dfTraining_dict = {}
    dfValidation_dict = {}

    # features for BDT training
    #features = ['cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
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

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()


        ######################### SELECT EVENTS FOR TRAINING #########################
  
        dfQCDTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-2').copy(deep=True)
        dfQCDTraining_dict[name]['gentau_pid'] = dfQCDTraining_dict[name]['gentau_decayMode'] # create pid column to use as training target
        dfQCDTraining_dict[name]['gentau_pid'] = dfQCDTraining_dict[name]['gentau_pid'].replace([-2], 0) # tag pileup

        dfQCDValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-2').copy(deep=True)
        dfQCDValidation_dict[name]['gentau_pid'] = dfQCDValidation_dict[name]['gentau_decayMode'] # create pid column to use as training target
        dfQCDValidation_dict[name]['gentau_pid'] = dfQCDValidation_dict[name]['gentau_pid'].replace([-2], 0) # tag QCD as pileup

        dfTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode>=-1').copy(deep=True)
        dfTraining_dict[name]['gentau_pid'] = dfTraining_dict[name]['gentau_decayMode'] # create pid column to use as training target
        dfTraining_dict[name]['gentau_pid'] = dfTraining_dict[name]['gentau_pid'].replace([0,1,5,6,10,11], 1) # tag signal
        dfTraining_dict[name]['gentau_pid'] = dfTraining_dict[name]['gentau_pid'].replace([-1], 0) # tag pileup

        dfValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=-1').copy(deep=True)
        dfValidation_dict[name]['gentau_pid'] = dfValidation_dict[name]['gentau_decayMode'] # create pid column to use as training target
        dfValidation_dict[name]['gentau_pid'] = dfValidation_dict[name]['gentau_pid'].replace([0,1,5,6,10,11], 1) # tag signal
        dfValidation_dict[name]['gentau_pid'] = dfValidation_dict[name]['gentau_pid'].replace([-1], 0) # tag pileup
        

        ######################### TRAINING OF BDT #########################

        model_dict[name], fpr_train_dict[name], tpr_train_dict[name], threshold_train_dict[name], fpr_test_dict[name], tpr_test_dict[name], threshold_test_dict[name], testAuroc_dict[name], trainAuroc_dict[name] = train_xgb(dfTraining_dict[name], features, output, params_dict, test_fraction=0.3)

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
        print('\n**INFO: bkg efficiency at: 0.99 sgn efficiency {0} -  0.95 sgn efficiency {1} -  0.90 sgn efficiency {2}'.format(np.interp(0.99, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.95, tpr_train_dict[name], fpr_train_dict[name]),np.interp(0.90, tpr_train_dict[name], fpr_train_dict[name])))


        ######################### VALIDATION OF BDT #########################

        full = xgb.DMatrix(data=dfValidation_dict[name][features], label=dfValidation_dict[name][output], feature_names=features)
        dfValidation_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)

        fpr_validation_dict[name], tpr_validation_dict[name], threshold_validation_dict[name] = metrics.roc_curve(dfValidation_dict[name][output], dfValidation_dict[name]['cl3d_pubdt_score'])
        auroc_validation = metrics.roc_auc_score(dfValidation_dict[name]['gentau_pid'],dfValidation_dict[name]['cl3d_pubdt_score'])

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

        full = xgb.DMatrix(data=dfQCDTraining_dict[name][features], label=dfQCDTraining_dict[name][output], feature_names=features)
        dfQCDTraining_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfQCDTraining_dict[name]['cl3d_pubdt_passWP99'] = dfQCDTraining_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfQCDTraining_dict[name]['cl3d_pubdt_passWP95'] = dfQCDTraining_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfQCDTraining_dict[name]['cl3d_pubdt_passWP90'] = dfQCDTraining_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        full = xgb.DMatrix(data=dfQCDValidation_dict[name][features], label=dfQCDValidation_dict[name][output], feature_names=features)
        dfQCDValidation_dict[name]['cl3d_pubdt_score'] = model_dict[name].predict(full)
        dfQCDValidation_dict[name]['cl3d_pubdt_passWP99'] = dfQCDValidation_dict[name]['cl3d_pubdt_score'] > bdtWP99_dict[name]
        dfQCDValidation_dict[name]['cl3d_pubdt_passWP95'] = dfQCDValidation_dict[name]['cl3d_pubdt_score'] > bdtWP95_dict[name]
        dfQCDValidation_dict[name]['cl3d_pubdt_passWP90'] = dfQCDValidation_dict[name]['cl3d_pubdt_score'] > bdtWP90_dict[name]

        QCDtot = pd.concat([dfQCDTraining_dict[name],dfQCDValidation_dict[name]],sort=False)
        QCD99 = QCDtot.query('cl3d_pubdt_passWP99==True')
        QCD95 = QCDtot.query('cl3d_pubdt_passWP95==True')
        QCD90 = QCDtot.query('cl3d_pubdt_passWP90==True')

        print('\n**INFO: QCD cluster passing the PU rejection:')
        print('  -- number of QCD events passing WP99: {0}%'.format(round(float(QCD99['cl3d_pubdt_passWP99'].count())/float(QCDtot['cl3d_pubdt_passWP99'].count())*100,2)))
        print('  -- number of QCD events passing WP95: {0}%'.format(round(float(QCD95['cl3d_pubdt_passWP95'].count())/float(QCDtot['cl3d_pubdt_passWP95'].count())*100,2)))
        print('  -- number of QCD events passing WP90: {0}%'.format(round(float(QCD90['cl3d_pubdt_passWP90'].count())/float(QCDtot['cl3d_pubdt_passWP90'].count())*100,2)))


        ######################### SAVE FILES #########################

        dfTraining_dict[name] = pd.concat([dfTraining_dict[name],dfQCDTraining_dict[name]],sort=False)
        dfValidation_dict[name] = pd.concat([dfValidation_dict[name],dfQCDValidation_dict[name]],sort=False)

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


        
if args.doPlots:
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting plotting')

    features_dict = {#'cl3d_pt_c1'            : [r'cl3d_pt_c1',[0.,200.,40]], 
                     #'cl3d_pt_c2'            : [r'cl3d_pt_c2',[0.,200.,40]],
                     #'cl3d_pt_c3'            : [r'cl3d_pt_c3',[0.,200.,40]],
                     'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'     : [r'3D cluster shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'       : [r'3D cluster first layer',[0.,20.,20]], 
                     'cl3d_maxlayer'         : [r'3D cluster maximum layer',[0.,50.,50]], 
                     'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[0.,60.,20]], 
                     'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[0.,0.15,10]], 
                     'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[0.,0.1,10]], 
                     'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[0.,0.01,10]], 
                     'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[0.,0.01,10]], 
                     'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     'cl3d_meanz'            : [r'3D cluster meanz',[325.,375.,30]], 
                     'cl3d_layer10'          : [r'N layers with 10% E deposit',[0.,15.,30]], 
                     'cl3d_layer50'          : [r'N layers with 50% E deposit',[0.,30.,60]], 
                     'cl3d_layer90'          : [r'N layers with 90% E deposit',[0.,40.,40]], 
                     'cl3d_ntc67'            : [r'Number of 3D clusters with 67% of energy',[0.,50.,10]], 
                     'cl3d_ntc90'            : [r'Number of 3D clusters with 90% of energy',[0.,100.,20]]
    }

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
        
        ######################### PLOT FEATURES #########################        
        print('\n** INFO: plotting features')

        dfNu = dfTraining_dict[name].query('gentau_pid==0')
        dfTau = dfTraining_dict[name].query('gentau_pid==1')

        for var in features_dict:
            plt.figure(figsize=(8,8))
            plt.hist(dfNu[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='PU events',      color='red',    histtype='step', lw=2, density=True)
            plt.hist(dfTau[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Tau signal',   color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfQCDTraining_dict[name][var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background',   color='blue',    histtype='step', lw=2, density=True)
            PUline = mlines.Line2D([], [], color='red',markersize=15, label='PU events',lw=2)
            SGNline = mlines.Line2D([], [], color='limegreen',markersize=15, label='Tau signal',lw=2)
            QCDline = mlines.Line2D([], [], color='blue',markersize=15, label='QCD background',lw=2)
            plt.legend(loc = 'upper right',handles=[PUline,SGNline,QCDline])
            plt.grid(linestyle=':')
            plt.xlabel(features_dict[var][0])
            plt.ylabel(r'Normalized entries')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/'+var+'.pdf')
            plt.close()

        del dfNu, dfTau

        ######################### PLOT SINGLE FE OPTION ROCS #########################

        print('\n** INFO: plotting train-test-validation ROC curves')
        plt.figure(figsize=(10,10))
        plt.plot(tpr_train_dict[name],fpr_train_dict[name],label='Train ROC', color='red',lw=2)
        plt.plot(tpr_test_dict[name],fpr_test_dict[name],label='Test ROC', color='green',lw=2)
        plt.plot(tpr_validation_dict[name],fpr_validation_dict[name],label='Validation ROC', color='blue',lw=2)
        plt.grid(linestyle=':')
        plt.legend(loc = 'upper left', fontsize=22)
        plt.xlim(0.97,1.001)
        #plt.yscale('log')
        #plt.ylim(0.01,1)
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background efficiency')
        plt.title('ROC curves - {0} FE'.format(feNames_dict[name]))
        plt.savefig(plotdir+'/PUbdt_train_test_validation_rocs_{0}.pdf'.format(feNames_dict[name]))
        plt.close()


        ######################### PLOT FEATURE IMPORTANCES #########################

        matplotlib.rcParams.update({'font.size': 12})
        print('\n** INFO: plotting features importance and score')
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

        dfNu = dfValidation_dict[name].query('gentau_pid==0')
        dfTau = dfValidation_dict[name].query('gentau_pid==1')

        matplotlib.rcParams.update({'font.size': 22})
        plt.figure(figsize=(10,10))
        plt.hist(dfNu['cl3d_pubdt_score'], bins=np.arange(-0.0, 1.0, 0.02), density=True, color='red', histtype='step', lw=2, label='Tau PU=200')
        plt.hist(dfTau['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='blue', histtype='step', lw=2, label='Nu PU=200')
        plt.hist(dfQCDValidation_dict[name]['cl3d_pubdt_score'],  bins=np.arange(-0.0, 1.0, 0.02), density=True, color='green', histtype='step', lw=2, label='QCD PU=200')
        plt.legend(loc = 'upper right', fontsize=22)
        plt.xlabel(r'PU BDT score')
        plt.ylabel(r'Entries')
        plt.title('pu rejection BDT score')
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
    plt.legend(loc = 'upper left', fontsize=22)
    plt.xlim(0.97,1.001)
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
    plt.xlim(0.97,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.title('Validation ROC curves')
    plt.savefig(plotdir+'/PUbdt_validation_rocs.pdf')
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
