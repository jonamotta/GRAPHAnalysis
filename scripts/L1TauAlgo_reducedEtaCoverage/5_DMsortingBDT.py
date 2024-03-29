# THIS DM SORTING HAS 3 TAU CATEGORIEs
#       - 1prong
#       - 1prong+pi0
#       - 3prong(+pi0)

import os
import sys
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
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.lines as mlines
import scikitplot as skplt
import argparse
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
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--PUWP', dest='PUWP', help='which working point do you want to use (90, 95, 99)?', default='90')
    parser.add_argument('--ISOWP', dest='ISOWP', help='which working point do you want to use (10, 15, 20, 25)?', default='25')
    parser.add_argument('--hardPUrej', dest='hardPUrej', help='apply hard PU rejection and do not consider PU categorized clusters for Iso variables? (99, 95, 90)', default='NO')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    tag1 = "Rscld" if args.doRescale else ""
    tag2 = "{0}hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_skimPUnoPt{0}_skimISO{0}{1}'.format(tag1, tag2)
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/DMsorted_skimPUnoPt{0}_skimISO{0}{1}'.format(tag1, tag2)
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/DMsorting_skimPUnoPt{0}_PUWP{2}_skimISO{0}{1}_ISOWP{3}'.format(tag1, tag2, args.PUWP, args.ISOWP)
    model_dict_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/DMsorting_skimPUnoPt{0}_skimISO{0}{1}'.format(tag1, tag2)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_dict_outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/DMsorting_skimPUnoPt{0}_skimISO{0}{1}/performance_PUWP{2}_ISOWP{3}.log".format(tag1, tag2, args.PUWP,args.ISOWP))

    print('** INFO: using PU rejection BDT WP: '+args.PUWP)
    print('** INFO: using ISO BDT WP: '+args.ISOWP)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_PUWP{0}_isoQCDrejected.hdf5'.format(args.PUWP),
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_PUWP{0}_ISOWP{1}_DMsorted.hdf5'.format(args.PUWP,args.ISOWP),
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_PUWP{0}_ISOWP{1}_DMsorted.hdf5'.format(args.PUWP,args.ISOWP),
        'mixed'        : outdir+'/'
    }

    outFileNu_dict = {
        'threshold'    : outdir+'/Nu_PU200_th_PUWP{0}_ISOWP{1}_DMsorted.hdf5'.format(args.PUWP,args.ISOWP),
        'mixed'        : outdir+'/'
    }

    outFile_model_dict = {
        'threshold'    : model_dict_outdir+'/model_DMsorting_th_PU200_PUWP{0}_ISOWP{1}.pkl'.format(args.PUWP,args.ISOWP),
        'mixed'        : model_dict_outdir+'/'
    }

    dfTraining_dict = {}   # dictionary of the merged training dataframes
    dfValidation_dict = {} # dictionary of the merged test dataframes
    dfNu_dict = {}

    dfTrDM0_dict = {}
    dfTrDM1_dict = {}
    dfTrDM10_dict = {}
    dfTrDM11_dict = {}
    dfTrDM2_dict = {}
    dfTrainingQCD_dict = {}
    dfValDM0_dict = {}
    dfValDM1_dict = {}
    dfValDM10_dict = {}
    dfValDM11_dict = {}
    dfValDM2_dict = {}
    dfValidationQCD_dict = {}
    
    model_dict = {}            # dictionary of the model for DM sorting
    DMs_dict = {}
    cm_dict = {}
    ns_dict = {}
    
    # features from ISO to be possibly used
    # 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_eSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_eSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_eIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_eIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7'

    # features used for the sorting step - FULL AVAILABLE
    features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

    #  the saturation and shifting values are calculated in the "features_reshaping" JupyScript
    features2shift = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',]
    features2saturate = ['cl3d_abseta', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    saturation_dict = {'cl3d_abseta': [1.45, 3.2],
                       'cl3d_seetot': [0, 0.17],
                       'cl3d_seemax': [0, 0.6],
                       'cl3d_spptot': [0, 0.17],
                       'cl3d_sppmax': [0, 0.53],
                       'cl3d_szz': [0, 141.09],
                       'cl3d_srrtot': [0, 0.02],
                       'cl3d_srrmax': [0, 0.02],
                       'cl3d_srrmean': [0, 0.01],
                       'cl3d_hoe': [0, 63],
                       'cl3d_meanz': [305, 535]
                    }

    features_dict = {'cl3d_pt_tr'            : [r'3D cluster $p_{T}$',[0.,500.,50]],
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
    if args.doRescale:
        features_dict = {'cl3d_pt_tr'            : [r'3D cluster $p_{T}$',[-33.,33.,66]],
                         'cl3d_abseta'           : [r'3D cluster |$\eta$|',[-33.,33.,66]], 
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

        dfNuTr = dfTraining_dict[name].query('gentau_decayMode==-1 and cl3d_pubdt_passWP{0}==True and cl3d_isobdt_passWP{1}==True'.format(args.PUWP,args.ISOWP)).copy(deep=True)
        dfNuVal = dfValidation_dict[name].query('gentau_decayMode==-1 and cl3d_pubdt_passWP{0}==True and cl3d_isobdt_passWP{1}==True'.format(args.PUWP,args.ISOWP)).copy(deep=True)
        dfNu_dict[name] = pd.concat([dfNuTr,dfNuVal], sort=False)

        dfTr = dfTraining_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11) and cl3d_pubdt_passWP{0}==True and cl3d_isobdt_passWP{1}==True'.format(args.PUWP,args.ISOWP)).copy(deep=True)
        dfVal = dfValidation_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==10 or gentau_decayMode==11) and cl3d_pubdt_passWP{0}==True and cl3d_isobdt_passWP{1}==True'.format(args.PUWP,args.ISOWP)).copy(deep=True)

        # fill all the DM dataframes
        dfTrDM0_dict[name] = dfTraining_dict[name].query('gentau_decayMode==0').copy(deep=True)
        dfTrDM1_dict[name] = dfTraining_dict[name].query('gentau_decayMode==1').copy(deep=True)
        dfTrDM10_dict[name] = dfTraining_dict[name].query('gentau_decayMode==10').copy(deep=True)
        dfTrDM11_dict[name] = dfTraining_dict[name].query('gentau_decayMode==11').copy(deep=True)
        dfTrDM2_dict[name] = dfTraining_dict[name].query('gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)

        dfValDM0_dict[name] = dfValidation_dict[name].query('gentau_decayMode==0').copy(deep=True)
        dfValDM1_dict[name] = dfValidation_dict[name].query('gentau_decayMode==1').copy(deep=True)
        dfValDM10_dict[name] = dfValidation_dict[name].query('gentau_decayMode==10').copy(deep=True)
        dfValDM11_dict[name] = dfValidation_dict[name].query('gentau_decayMode==11').copy(deep=True)
        dfValDM2_dict[name] = dfValidation_dict[name].query('gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)

        # replace DMs by categories
        dfTraining_dict[name]['gentau_decayMode'] = dfTraining_dict[name]['gentau_decayMode'].replace([10,11], 2) # tag 3-prong(+pi)
        dfValidation_dict[name]['gentau_decayMode'] = dfValidation_dict[name]['gentau_decayMode'].replace([10,11], 2) # tag 3-prong(+pi)  

        dfTr['gentau_decayMode'] = dfTr['gentau_decayMode'].replace([10,11], 2) # tag 3-prong(+pi)
        dfVal['gentau_decayMode'] = dfVal['gentau_decayMode'].replace([10,11], 2) # tag 3-prong(+pi)  


        ######################### TRAINING OF BDT #########################

        print('\n** INFO: training random forest')
        inputs = dfTr[features]
        target = dfTr['gentau_decayMode']
        RFC = RandomForestClassifier(n_jobs=10, random_state=0, class_weight='balanced', max_depth=2, n_estimators=300)
        model_dict[name] = RFC.fit(inputs,target)
        
        save_obj(model_dict[name], outFile_model_dict[name])

        probas_tr = model_dict[name].predict_proba(dfTr[features])
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfTr['gentau_decayMode'], probas_tr)
        plt.savefig(plotdir+'/DM_roc_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'_training.pdf')
        plt.close()

        ######################### APPLICATION OF BDT #########################

        print('\n** INFO: doing K-fold validation')
        dfVal['cl3d_predDM'] = cross_val_predict(model_dict[name], dfVal[features], dfVal['gentau_decayMode'], cv=3)
        probas_val = model_dict[name].predict_proba(dfVal[features])
        dfVal['cl3d_probDM0'] = probas_val[:,0]
        dfVal['cl3d_probDM1'] = probas_val[:,1]
        dfVal['cl3d_probDM2'] = probas_val[:,2]
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfVal['gentau_decayMode'], probas_val)
        plt.savefig(plotdir+'/DM_roc_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'_validation.pdf')
        plt.close()

        print('\n** INFO: doing probability prediction')
        dfTraining_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfTraining_dict[name][features])
        probas_train = model_dict[name].predict_proba(dfTraining_dict[name][features])
        dfTraining_dict[name]['cl3d_probDM0'] = probas_train[:,0]
        dfTraining_dict[name]['cl3d_probDM1'] = probas_train[:,1]
        dfTraining_dict[name]['cl3d_probDM2'] = probas_train[:,2]

        dfValidation_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfValidation_dict[name][features])
        probas_valid = model_dict[name].predict_proba(dfValidation_dict[name][features])
        dfValidation_dict[name]['cl3d_probDM0'] = probas_valid[:,0]
        dfValidation_dict[name]['cl3d_probDM1'] = probas_valid[:,1]
        dfValidation_dict[name]['cl3d_probDM2'] = probas_valid[:,2]

        dfNu_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfNu_dict[name][features])
        probas_nu = model_dict[name].predict_proba(dfNu_dict[name][features])
        dfNu_dict[name]['cl3d_probDM0'] = probas_nu[:,0]
        dfNu_dict[name]['cl3d_probDM1'] = probas_nu[:,1]
        dfNu_dict[name]['cl3d_probDM2'] = probas_nu[:,2]

        ######################### MAKE CONFUSION MATRICES #########################

        DMs_dict[name] = dfVal.groupby('gentau_decayMode')['cl3d_eta'].count()
        cm_dict[name] = confusion_matrix(dfVal.gentau_decayMode, dfVal.cl3d_predDM)
        cm_dict[name] = cm_dict[name].astype('float') / cm_dict[name].sum(axis=1)[:, np.newaxis]
        ns_dict[name] = DMs_dict[name]
        print('\n** INFO: fraction of correctly identified DMs - FE {0}: {1}'.format(feNames_dict[name], (cm_dict[name][0,0]*ns_dict[name][0]+cm_dict[name][1,1]*ns_dict[name][1]+cm_dict[name][2,2]*ns_dict[name][2])/np.sum(ns_dict[name])))

        print('\n** INFO: finished DM sorting for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')


        ######################### PLOTTING #########################

        if args.doPlots:
            os.system('mkdir -p '+plotdir+'/features/')

            ######################### PLOT FEATURES #########################
            for var in features_dict:
                plt.figure(figsize=(8,8))
                plt.hist(dfTrDM0_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'1-prong', color='limegreen',  histtype='step', lw=2, alpha=0.8)
                plt.hist(dfTrDM1_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'1-prong + $\pi^{0}$', color='darkorange',  histtype='step', lw=2, alpha=0.8)
                plt.hist(dfTrDM2_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'3-prong (+ $\pi^{0}$)', color='fuchsia',  histtype='step', lw=2, alpha=0.8)
                plt.legend(loc = 'upper right')
                plt.grid(linestyle=':')
                plt.xlabel(features_dict[var][0])
                plt.ylabel(r'Normalized events')
                plt.subplots_adjust(bottom=0.12)
                plt.savefig(plotdir+'/features/'+var+'_'+name+'_DMs.pdf')
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
            plt.xticks([0,1,2], ('1-prong', r'1-prong$+\pi_0$', r'3-prong($+\pi_0$)'))
            plt.yticks([0,1,2], ('1-prong', r'1-prong$+\pi_0$', r'3-prong($+\pi_0$)'))
            plt.xlim(-0.5, 2.5)
            plt.ylim(-0.5, 2.5),
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
            plt.savefig(plotdir+'/DMconfMatrix_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'.pdf')
            plt.close()


            ######################### PREDICTED PROBABILITIES #########################

            print('\n** INFO: plotting predicted probabilities')
            plt.figure(figsize=(15,10))
            plt.hist(dfVal[dfVal.gentau_decayMode==0].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong', color='limegreen', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==1].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', color='darkorange', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==2].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', color='fuchsia', density=True)
            plt.legend(loc = 'upper right')
            plt.title(r'Probabilities 1-prong category')
            plt.ylabel(r'Entries')
            plt.savefig(plotdir+'/DM0_prob_'+feNames_dict[name]+'_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'.pdf')
            plt.close()

            plt.figure(figsize=(15,10))
            plt.hist(dfVal[dfVal.gentau_decayMode==0].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong', color ='limegreen', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==1].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', color ='darkorange', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==2].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', color ='fuchsia', density=True)
            plt.legend(loc = 'upper right')
            plt.title(r'Probabilities 1-prong$+\pi_0$ category')
            plt.ylabel(r'Entries')
            plt.savefig(plotdir+'/DM1_prob_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'.pdf')
            plt.close()

            plt.figure(figsize=(15,10))
            plt.hist(dfVal[dfVal.gentau_decayMode==0].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong', color='limegreen', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==1].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', color='darkorange', density=True)
            plt.hist(dfVal[dfVal.gentau_decayMode==2].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', color='fuchsia', density=True)              
            plt.legend(loc = 'upper right')
            plt.title(r'Probabilities 3-prong($+\pi_0$) category')
            plt.ylabel(r'Entries')
            plt.savefig(plotdir+'/DM2_prob_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'.pdf')
            plt.close()


            ######################### ROC CURVE #########################
            
            # print('\n** INFO: plotting ROC curve')
            # plt.figure(figsize=(15,10))
            # skplt.metrics.plot_roc(dfHH_dict[name]['gentau_decayMode'], probas_dict[name])
            # plt.savefig(plotdir+'/DM_roc_PUWP'+args.PUWP+'_ISOWP'+args.ISOWP+'_'+name+'.pdf')
            # plt.close()

        
        ######################### SAVE FILES #########################

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()

        store_nu = pd.HDFStore(outFileNu_dict[name], mode='w')
        store_nu[name] = dfNu_dict[name]
        store_nu.close()

# restore normal output
sys.stdout = sys.__stdout__





