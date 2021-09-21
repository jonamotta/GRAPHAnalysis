# THIS DM SORTING HAS 3 TAU CATEGORIES AND 1 QCD CATEGORY
#       - 1prong
#       - 1prong+pi0
#       - 3prong(+pi0)
#       - QCD
#       - residual PU NOT tagged as QCD

import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.lines as mlines
import scikitplot as skplt
import argparse


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
    parser.add_argument('--noTraining', dest='doTraining', help='skip training and do only calibration?',  action='store_false', default=True)
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--WP', dest='WP', help='which working point do you want to use (90, 95, 99)?', default='99')
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    PUbdtWP = 'WP'+args.WP
    bdtcut = 'cl3d_pubdt_pass'+PUbdtWP
    print('** INFO: using PU rejection BDT WP: '+args.WP)

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_test/hdf5dataframes/isolated_A'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_test/hdf5dataframes/isolated_A'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_test/plots/isolation_A_{0}'.format(PUbdtWP)
    model_dict_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_test/pklModels/isolation_A'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_dict_outdir)

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
        'threshold'    : outdir+'/Training_PU200_th_PUWP{0}_isolated.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUWP{0}_isolated.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_model_dict = {
        'threshold'    : model_dict_outdir+'/model_isolation_th_PU200_PUWP{0}.pkl'.format(args.WP),
        'supertrigger' : model_dict_outdir+'/',
        'bestchoice'   : model_dict_outdir+'/',
        'bestcoarse'   : model_dict_outdir+'/',
        'mixed'        : model_dict_outdir+'/'
    }

    dfTraining_dict = {}   # dictionary of the merged training dataframes
    dfValidation_dict = {} # dictionary of the merged test dataframes
    dfNuTraining_dict = {}
    dfNuValidation_dict = {}
    dfTrainingQCD_dict = {}
    dfValidationQCD_dict = {}
    dfTrainingTau_dict = {}
    dfValidationTau_dict = {}
    
    model_dict = {}            # dictionary of the model for DM sorting
    DMs_dict = {}
    cm_dict = {}
    ns_dict = {}
    
    # features for BDT training
    #features = ['cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_eSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_eSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_eIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_eIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    features = ['cl3d_pt_c3', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_eSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_eSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_eIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_eIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']

    # name : [title, [min, max, step]
    features_dict = {#'cl3d_pt_c1'            : [r'3D cluster $p_{T}$ after C1',[0.,500.,50]], 
                     #'cl3d_pt_c2'            : [r'3D cluster $p_{T}$ after C2',[0.,500.,50]],
                     'cl3d_pt_c3'            : [r'3D cluster $p_{T}$ after C3',[0.,500.,50]],
                     #'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     #'cl3d_showerlength'     : [r'3D cluster shower length',[0.,35.,15]], 
                     #'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     #'cl3d_firstlayer'       : [r'3D cluster first layer',[0.,20.,20]], 
                     #'cl3d_maxlayer'         : [r'3D cluster maximum layer',[0.,50.,50]], 
                     #'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[0.,60.,20]], 
                     #'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[0.,0.15,10]], 
                     #'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[0.,0.1,10]], 
                     #'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[0.,0.01,10]], 
                     #'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[0.,0.01,10]], 
                     #'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     #'cl3d_meanz'            : [r'3D cluster meanz',[325.,375.,30]], 
                     #'cl3d_layer10'          : [r'N layers with 10% E deposit',[0.,15.,30]], 
                     #'cl3d_layer50'          : [r'N layers with 50% E deposit',[0.,30.,60]], 
                     #'cl3d_layer90'          : [r'N layers with 90% E deposit',[0.,40.,40]], 
                     #'cl3d_ntc67'            : [r'Number of 3D clusters with 67% of energy',[0.,50.,10]], 
                     #'cl3d_ntc90'            : [r'Number of 3D clusters with 90% of energy',[0.,100.,20]],
                     'cl3d_etIso_dR4'               : [r'Clusters $E_{T}$ inside an isolation cone of dR=0.4',[0.,300.,20]],
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
        print('** INFO: starting DM sorting for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()


        ######################### SELECT EVENTS FOR TRAINING #########################  

        dfNuTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-1 and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))
        dfNuValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-1 and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))

        dfTraining_dict[name] = dfTraining_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11 or gentau_decayMode==-2) and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))
        dfValidation_dict[name] = dfValidation_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11 or gentau_decayMode==-2) and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))

        # replace DMs by categories
        dfTraining_dict[name]['iso_pid'] = dfTraining_dict[name]['gentau_decayMode'].copy(deep=True) # tag tau
        dfTraining_dict[name]['iso_pid'] = dfTraining_dict[name]['gentau_decayMode'].copy(deep=True) # tag qcd
        dfTraining_dict[name]['iso_pid'].replace([0,1,10,11], 0, inplace=True)
        dfTraining_dict[name]['iso_pid'].replace([-2], 1, inplace=True)

        dfValidation_dict[name]['iso_pid'] = dfValidation_dict[name]['gentau_decayMode'].copy(deep=True) # tag tau
        dfValidation_dict[name]['iso_pid'] = dfValidation_dict[name]['gentau_decayMode'].copy(deep=True) # tag qcd
        dfValidation_dict[name]['iso_pid'].replace([0,1,10,11], 0, inplace=True)
        dfValidation_dict[name]['iso_pid'].replace([-2], 1, inplace=True)

        dfNuTraining_dict[name]['iso_pid'] = dfNuTraining_dict[name]['gentau_decayMode'].copy(deep=True) # tag residual pileup as qcd
        dfNuValidation_dict[name]['iso_pid'] = dfNuValidation_dict[name]['gentau_decayMode'].copy(deep=True) # tag residual pileup as qcd
        dfNuTraining_dict[name]['iso_pid'].replace([-1], 1, inplace=True)
        dfNuValidation_dict[name]['iso_pid'].replace([-1], 1, inplace=True)

        dfTrainingQCD_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-2').copy(deep=True)
        dfValidationQCD_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-2').copy(deep=True)
        dfTrainingTau_dict[name] = dfTraining_dict[name].query('gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)
        dfValidationTau_dict[name] = dfValidation_dict[name].query('gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)

        ######################### TRAINING OF BDT #########################

        if args.doTraining:
            print('\n** INFO: training random forest')
            inputs = dfTraining_dict[name][features]
            target = dfTraining_dict[name]['iso_pid']
            RFC = RandomForestClassifier(n_jobs=10, random_state=0, class_weight='balanced', max_depth=2, n_estimators=1000)
            model_dict[name] = RFC.fit(inputs,target)
            
            save_obj(model_dict[name], outFile_model_dict[name])

        else:
            model_dict[name] = load_obj(outFile_model_dict[name])


        ######################### APPLICATION OF BDT #########################

        print('\n** INFO: doing K-fold validation')
        dfValidation_dict[name]['cl3d_predIso'] = cross_val_predict(model_dict[name], dfValidation_dict[name][features], dfValidation_dict[name]['iso_pid'], cv=5)

        print('\n** INFO: doing probability prediction')
        dfTraining_dict[name]['cl3d_predIso'] = model_dict[name].predict(dfTraining_dict[name][features])
        probas_train = model_dict[name].predict_proba(dfTraining_dict[name][features])
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfTraining_dict[name]['iso_pid'], probas_train)
        plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'_training.pdf')
        plt.close()
        dfTraining_dict[name]['cl3d_probIso0'] = probas_train[:,0]
        dfTraining_dict[name]['cl3d_probIso1'] = probas_train[:,1]

        dfValidation_dict[name]['cl3d_predIso'] = model_dict[name].predict(dfValidation_dict[name][features])
        probas_valid = model_dict[name].predict_proba(dfValidation_dict[name][features])
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfValidation_dict[name]['iso_pid'], probas_valid)
        plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'_validation.pdf')
        plt.close()
        dfValidation_dict[name]['cl3d_probIso0'] = probas_valid[:,0]
        dfValidation_dict[name]['cl3d_probIso1'] = probas_valid[:,1]

        dfNuTraining_dict[name]['cl3d_predIso'] = model_dict[name].predict(dfNuTraining_dict[name][features])
        probasNu = model_dict[name].predict_proba(dfNuTraining_dict[name][features])
        dfNuTraining_dict[name]['cl3d_probIso0'] = probasNu[:,0]
        dfNuTraining_dict[name]['cl3d_probIso1'] = probasNu[:,1]
        del probasNu

        dfNuValidation_dict[name]['cl3d_predIso'] = model_dict[name].predict(dfNuValidation_dict[name][features])
        probasNu = model_dict[name].predict_proba(dfNuValidation_dict[name][features])
        dfNuValidation_dict[name]['cl3d_probIso0'] = probasNu[:,0]
        dfNuValidation_dict[name]['cl3d_probIso1'] = probasNu[:,1]


        ######################### MAKE CONFUSION MATRICES #########################

        DMs_dict[name] = dfValidation_dict[name].groupby('iso_pid')['cl3d_eta'].count()
        cm_dict[name] = confusion_matrix(dfValidation_dict[name]['iso_pid'], dfValidation_dict[name]['cl3d_predIso'])
        cm_dict[name] = cm_dict[name].astype('float') / cm_dict[name].sum(axis=1)[:, np.newaxis]
        ns_dict[name] = DMs_dict[name]
        print('\n** INFO: fraction of correctly identified DMs - FE {0}: {1}'.format(feNames_dict[name], (cm_dict[name][0,0]*ns_dict[name][0]+cm_dict[name][1,1]*ns_dict[name][1])/np.sum(ns_dict[name])))

        print('\n** INFO: finished DM sorting for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')


        ######################### PLOTTING #########################

        if args.doPlots:
            ######################### PLOT FEATURES #########################
            for var in features_dict:
                plt.figure(figsize=(8,8))
                plt.hist(dfTrainingTau_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='Tau signal', color='green',  histtype='step', lw=2, alpha=0.8)
                plt.hist(dfTrainingQCD_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background', color='cyan',  histtype='step', lw=2, alpha=0.8)
                plt.hist(dfNuValidation_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Residual PU', color='red',  histtype='step', lw=2, alpha=0.8)
                plt.legend(loc = 'upper right')
                plt.grid(linestyle=':')
                plt.xlabel(features_dict[var][0])
                plt.ylabel(r'Normalized events')
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/'+var+'_'+name+'_DMs.pdf')
                plt.close()

            ######################### FEATURE IMPORTANCE #########################
            
            print('\n** INFO: plotting feature importance')
            feature_importance = model_dict[name].feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            fig = plt.figure(figsize=(10, 10))
            plt.gcf().subplots_adjust(left=0.2)
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            plt.yticks(pos, np.array(features)[sorted_idx])
            plt.title('Feature Importance (MDI)')
            plt.savefig(plotdir+'/DMbdt_importances_'+name+'.pdf')
            plt.close()

            ######################### CONFUSION MATRIX #########################

            print('\n** INFO: plotting confusion matrix')
            plt.figure(figsize=(8,8))
            fig, ax = plt.subplots()
            im = ax.imshow(cm_dict[name], interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            plt.xticks([0,1], ('Tau', 'QCD'))
            plt.yticks([0,1], ('Tau', 'QCD'))
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5),
            plt.ylabel('Generated decay mode')
            plt.xlabel('Predicted decay mode')
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")
            fmt = '.2f'
            thresh = cm_dict[name].max() / 2.
            for i in range(cm_dict[name].shape[0]):
                for j in range(cm_dict[name].shape[1]):
                    ax.text(j, i, format(cm_dict[name][i, j], fmt), ha="center", va="center", fontsize=18, color="white" if cm_dict[name][i, j] > thresh else "black")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.21)
            plt.subplots_adjust(top=0.9)
            plt.subplots_adjust(left=0.01)
            plt.savefig(plotdir+'/DMconfMatrix_'+PUbdtWP+'_'+name+'.pdf')
            plt.close()


            ######################### PREDICTED PROBABILITIES #########################

            matplotlib.rcParams.update({'font.size': 22})
            print('\n** INFO: plotting predicted probabilities')
            plt.figure(figsize=(15,10))
            plt.hist(dfValidation_dict[name][dfValidation_dict[name].iso_pid==0].cl3d_probIso0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='green', label='Tau signal', density=True)
            plt.hist(dfValidation_dict[name][dfValidation_dict[name].iso_pid==1].cl3d_probIso0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='cyan', label=r'QCD background', density=True)
            plt.hist(dfNuValidation_dict[name].cl3d_probIso0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='red', label=r'Residual PU', density=True)
            plt.legend(fontsize=22)
            plt.title(r'Probabilities 1-prong category')
            plt.ylabel(r'Entries')
            plt.savefig(plotdir+'/DM0_prob_'+feNames_dict[name]+'_'+PUbdtWP+'.pdf')
            plt.close()

            plt.figure(figsize=(15,10))
            plt.hist(dfValidation_dict[name][dfValidation_dict[name].iso_pid==0].cl3d_probIso1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='green', label='Tau signal', density=True)
            plt.hist(dfValidation_dict[name][dfValidation_dict[name].iso_pid==1].cl3d_probIso1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='cyan', label=r'QCD background', density=True)
            plt.hist(dfNuValidation_dict[name].cl3d_probIso1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, color='red', label=r'Residual PU', density=True)
            plt.legend(fontsize=22)
            plt.title(r'Probabilities 1-prong$+\pi_0$ category')
            plt.ylabel(r'Entries')
            plt.savefig(plotdir+'/DM1_prob_'+PUbdtWP+'_'+name+'.pdf')
            plt.close()

            ######################### ROC CURVE #########################
            
            # print('\n** INFO: plotting ROC curve')
            # plt.figure(figsize=(15,10))
            # skplt.metrics.plot_roc(dfHH_dict[name]['iso_pid'], probas_dict[name])
            # plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'.pdf')
            # plt.close()

        
        ######################### SAVE FILES #########################

        dfTraining_dict[name] = pd.concat([dfTraining_dict[name],dfNuTraining_dict[name]],sort=False)
        dfValidation_dict[name] = pd.concat([dfValidation_dict[name],dfNuValidation_dict[name]],sort=False)

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()







