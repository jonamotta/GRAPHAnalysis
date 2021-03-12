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
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/PUrejected'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/DMsorted'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/DMsorting'
    model_dict_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/DMsorting'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_dict_outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTau_dict = {
        'threshold'    : indir+'/RelValTenTau_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileNu_dict = {
        'threshold'    : indir+'/RelValNu_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileHH_dict = {
        'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileQCD_dict = {
        'threshold'    : indir+'/QCD_PU200_th_PUrejected.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_dict = {
        'threshold'    : outdir+'/RelValTenTau_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }


    outFileNu_dict = {
        'threshold'    : outdir+'/RelValNu_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileHH_dict = {
        'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileQCD_dict = {
        'threshold'    : outdir+'/QCD_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_model_dict = {
        'threshold'    : model_dict_outdir+'/model_DMsorting_th_PU200_PUWP{0}.pkl'.format(args.WP),
        'supertrigger' : model_dict_outdir+'/',
        'bestchoice'   : model_dict_outdir+'/',
        'bestcoarse'   : model_dict_outdir+'/',
        'mixed'        : model_dict_outdir+'/'
    }


    dfTau_dict = {}            # dictionary of the skim level tau dataframes
    dfNu_dict = {}             # dictionary of the skim level nu dataframes
    dfHH_dict = {}             # dictionary of the skim level HH dataframes
    dfQCD_dict = {}            # dictionary of the skim level QCD dataframes
    dfTauTraining_dict = {}    # dictionary of the training tau dataframes
    dfQCDTraining_dict = {}    # dictionary of the training QCD dataframes
    dfHHValidation_dict = {}
    dfQCDValidation_dict = {}  # dictionary of the validation QCD dataframes
    dfMergedTraining_dict = {}   # dictionary of the merged training dataframes
    dfMergedValidation_dict = {} # dictionary of the merged test dataframes

    dfTau_DM0_dict = {}        # dictionary of the DM0 tau dataframes
    dfTau_DM1_dict = {}        # dictionary of the DM1 tau dataframes
    dfTau_DM10_dict = {}       # dictionary of the DM10 tau dataframes
    dfTau_DM11_dict = {}       # dictionary of the DM11 tau dataframes
    dfTau_DM5_dict = {}        # dictionary of the DM5 tau dataframes
    dfTau_DM2_dict = {}        # dictionary of the DM=5,6 tau dataframes
    dfTau_DM3_dict = {}        # dictionary of the DM=10,11 tau dataframes
    dfHH_DM0_dict = {}         # dictionary of the DM0 HH dataframes
    dfHH_DM1_dict = {}         # dictionary of the DM1 HH dataframes
    dfHH_DM10_dict = {}        # dictionary of the DM10 HH dataframes
    dfHH_DM11_dict = {}        # dictionary of the DM11 HH dataframes
    dfHH_DM5_dict = {}         # dictionary of the DM5 HH dataframes
    dfHH_DM2_dict = {}         # dictionary of the DM=5,6 HH dataframes
    dfHH_DM3_dict = {}         # dictionary of the DM=5,6 HH dataframes
    model_dict = {}            # dictionary of the model fro DM sorting
    DMs_dict = {}
    cm_dict = {}
    ns_dict = {}
    
    # features for BDT training
    features = ['cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']

    # name : [title, [min, max, step]
    features_dict = {'cl3d_pt_c3'            : [r'3D cluster $p_{T}$',[0.,500.,50]],
                     'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'     : [r'3D cluster shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'       : [r'3D cluster first layer',[0.,20.,20]], 
                     'cl3d_maxlayer'         : [r'3D cluster maximum layer',[0.,50.,50]], 
                     'cl3d_szz'              : [r'3D cluster szz',[0.,60.,20]], 
                     'cl3d_seetot'           : [r'3D cluster seetot',[0.,0.15,10]], 
                     'cl3d_spptot'           : [r'3D cluster spptot',[0.,0.1,10]], 
                     'cl3d_srrtot'           : [r'3D cluster srrtot',[0.,0.01,10]], 
                     'cl3d_srrmean'          : [r'3D cluster srrmean',[0.,0.01,10]], 
                     'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     'cl3d_meanz'            : [r'3D cluster meanz',[325.,375.,30]], 
                     'cl3d_layer10'          : [r'N layers with 10% E deposit',[0.,15.,30]], 
                     'cl3d_layer50'          : [r'N layers with 50% E deposit',[0.,30.,60]], 
                     'cl3d_layer90'          : [r'N layers with 90% E deposit',[0.,40.,40]], 
                     'cl3d_ntc67'            : [r'Number of 3D clusters with 67% of energy',[0.,50.,10]], 
                     'cl3d_ntc90'            : [r'Number of 3D clusters with 90% of energy',[0.,100.,20]]
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

        # fill tau dataframes and dictionaries -> training 
        store_tau = pd.HDFStore(inFileTau_dict[name], mode='r')
        dfTau_dict[name]  = store_tau[name]
        store_tau.close()

        # fill nu pileup dataframes and dictionaries
        store_nu = pd.HDFStore(inFileNu_dict[name], mode='r')
        dfNu_dict[name]  = store_nu[name]
        store_nu.close()   

        # fill HH dataframes and dictionaries -> validation
        store_hh = pd.HDFStore(inFileHH_dict[name], mode='r')
        dfHH_dict[name]  = store_hh[name]
        store_hh.close()

        # fill QCD dataframes and dictionaries -> 1/2 training + 1/2 validation 
        store_qcd= pd.HDFStore(inFileQCD_dict[name], mode='r')
        dfQCD_dict[name]  = store_qcd[name]
        store_qcd.close() 
        dfQCD_dict[name]['gentau_decayMode'] = 4 # tag as QCD background


        ######################### SELECT EVENTS FOR TRAINING #########################  

        # SIGNAL
        dfTauTraining_dict[name] = dfTau_dict[name]
        # define selections for the training dataset
        genPt_sel  = dfTauTraining_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfTauTraining_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfTauTraining_dict[name]['gentau_vis_eta']) < 2.9
        cl3dBest_sel = dfTauTraining_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfTauTraining_dict[name]['cl3d_pt'] > 4
        PUWP_sel = dfTauTraining_dict[name][bdtcut] == True
        DMsel = dfTauTraining_dict[name]['gentau_decayMode'] >= 0 # it should already be the case from skim level, but beter be safe
        # apply slections for the training dataset
        dfTauTraining_dict[name] = dfTauTraining_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel & PUWP_sel & DMsel]

        # BACKGROUND
        dfQCDTraining_dict[name] = dfQCD_dict[name].sample(frac=0.5,random_state=10)
        dfQCDValidation_dict[name] = dfQCD_dict[name].drop(dfQCDTraining_dict[name].index)
        # define selections for the training dataset
        genPt_sel  = dfQCDTraining_dict[name]['genjet_pt'] > 20
        eta_sel1   = np.abs(dfQCDTraining_dict[name]['genjet_eta']) > 1.6
        eta_sel2   = np.abs(dfQCDTraining_dict[name]['genjet_eta']) < 2.9
        cl3dBest_sel = dfQCDTraining_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfQCDTraining_dict[name]['cl3d_pt'] > 4
        PUWP_sel = dfQCDTraining_dict[name][bdtcut] == True
        # apply slections for the training dataset
        dfQCDTraining_dict[name] = dfQCDTraining_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel & PUWP_sel]

        # VALIDATION
        dfHHValidation_dict[name] = dfHH_dict[name]
        eta_sel1   = np.abs(dfHHValidation_dict[name]['cl3d_eta']) > 1.6
        eta_sel2   = np.abs(dfHHValidation_dict[name]['cl3d_eta']) < 2.9
        cl3dPt_sel = dfHHValidation_dict[name]['cl3d_pt_c3'] > 4
        # apply slections for the validation dataset
        dfHHValidation_dict[name] = dfHHValidation_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]

        eta_sel1   = np.abs(dfQCDValidation_dict[name]['cl3d_eta']) > 1.6
        eta_sel2   = np.abs(dfQCDValidation_dict[name]['cl3d_eta']) < 2.9
        cl3dPt_sel = dfQCDValidation_dict[name]['cl3d_pt_c3'] > 4
        # apply slections for the validation dataset
        dfQCDValidation_dict[name] = dfQCDValidation_dict[name][eta_sel1 & eta_sel2 & cl3dPt_sel]
        
        # MERGE
        dfMergedTraining_dict[name] = pd.concat([dfTauTraining_dict[name],dfQCDTraining_dict[name]],sort=False)
        dfMergedValidation_dict[name] = pd.concat([dfHHValidation_dict[name],dfQCDValidation_dict[name]],sort=False)

        ######################### SELECT SEPARATE DMs #########################

        # fill all the DM dataframes
        dfTau_DM0_dict[name] = dfTau_dict[name].query('gentau_decayMode==0')
        dfTau_DM1_dict[name] = dfTau_dict[name].query('gentau_decayMode==1')
        dfTau_DM10_dict[name] = dfTau_dict[name].query('gentau_decayMode==10')
        dfTau_DM11_dict[name] = dfTau_dict[name].query('gentau_decayMode==11')
        dfTau_DM5_dict[name] = dfTau_dict[name].query('gentau_decayMode==5')
        dfTau_DM2_dict[name] = dfTau_dict[name].query('gentau_decayMode==5' or 'gentau_decayMode==6')
        dfTau_DM3_dict[name] = dfTau_dict[name].query('gentau_decayMode==10' or 'gentau_decayMode==11')
        dfHH_DM0_dict[name] = dfHH_dict[name].query('gentau_decayMode==0')
        dfHH_DM1_dict[name] = dfHH_dict[name].query('gentau_decayMode==1')
        dfHH_DM10_dict[name] = dfHH_dict[name].query('gentau_decayMode==10')
        dfHH_DM11_dict[name] = dfHH_dict[name].query('gentau_decayMode==11')
        dfHH_DM5_dict[name] = dfHH_dict[name].query('gentau_decayMode==5')
        dfHH_DM2_dict[name] = dfHH_dict[name].query('gentau_decayMode==5' or 'gentau_decayMode==6')
        dfHH_DM3_dict[name] = dfHH_dict[name].query('gentau_decayMode==10' or 'gentau_decayMode==11')

        # replace DM>=5 with category numbers 2/3 (REMEMBER: category number 4 is QCD)
        dfMergedTraining_dict[name]['gentau_decayMode'].replace([5,6], 2, inplace=True)
        dfMergedValidation_dict[name]['gentau_decayMode'].replace([5,6], 2, inplace=True)
        dfMergedTraining_dict[name]['gentau_decayMode'].replace([10,11], 3, inplace=True)
        dfMergedValidation_dict[name]['gentau_decayMode'].replace([10,11], 3, inplace=True)


        ######################### TRAINING OF BDT #########################

        if args.doTraining:
            print('\n** INFO: starting training BDT')
            inputs = dfMergedTraining_dict[name][features]
            target = dfMergedTraining_dict[name]['gentau_decayMode']
            RFC = RandomForestClassifier(n_jobs=10, random_state=0, class_weight='balanced', max_depth=2, n_estimators=1000)
            model_dict[name] = RFC.fit(inputs,target)
            print('** INFO: finished training BDT')
            
            save_obj(model_dict[name], outFile_model_dict[name])

        else:
            model_dict[name] = load_obj(outFile_model_dict[name])


        ######################### APPLICATION OF BDT #########################

        # replace DM>=5 with category number 2
        dfTau_dict[name]['gentau_decayMode'].replace([5,6], 2, inplace=True)
        dfHH_dict[name]['gentau_decayMode'].replace([5,6], 2, inplace=True)
        dfQCD_dict[name]['gentau_decayMode'].replace([5,6], 2, inplace=True) # this should be completely useless, no DM 5,10,11 are stored in this files
        dfTau_dict[name]['gentau_decayMode'].replace([10,11], 3, inplace=True)
        dfHH_dict[name]['gentau_decayMode'].replace([10,11], 3, inplace=True)
        dfQCD_dict[name]['gentau_decayMode'].replace([10,11], 3, inplace=True) # this should be completely useless, no DM 5,10,11 are stored in this files

        print('\n** INFO: starting K-fold validation')
        dfTau_dict[name]['cl3d_predDM'] = cross_val_predict(model_dict[name], dfTau_dict[name][features], dfTau_dict[name]['gentau_decayMode'], cv=5)
        dfHH_dict[name]['cl3d_predDM'] = cross_val_predict(model_dict[name], dfHH_dict[name][features], dfHH_dict[name]['gentau_decayMode'], cv=5)
        print('** INFO: finished K-fold validation')

        print('\n** INFO: starting probability prediction')
        probasTau = model_dict[name].predict_proba(dfTau_dict[name][features])
        dfTau_dict[name]['cl3d_probDM0'] = probasTau[:,0]
        dfTau_dict[name]['cl3d_probDM1'] = probasTau[:,1]
        dfTau_dict[name]['cl3d_probDM2'] = probasTau[:,2]
        dfTau_dict[name]['cl3d_probDM3'] = probasTau[:,3]
        dfTau_dict[name]['cl3d_probDM4'] = probasTau[:,4]

        dfNu_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfNu_dict[name][features])
        probasNu = model_dict[name].predict_proba(dfNu_dict[name][features])
        dfNu_dict[name]['cl3d_probDM0'] = probasNu[:,0]
        dfNu_dict[name]['cl3d_probDM1'] = probasNu[:,1]
        dfNu_dict[name]['cl3d_probDM2'] = probasNu[:,2]
        dfNu_dict[name]['cl3d_probDM3'] = probasNu[:,3]
        dfNu_dict[name]['cl3d_probDM4'] = probasNu[:,4]

        dfHH_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfHH_dict[name][features])
        probasHH = model_dict[name].predict_proba(dfHH_dict[name][features])
        dfHH_dict[name]['cl3d_probDM0'] = probasHH[:,0]
        dfHH_dict[name]['cl3d_probDM1'] = probasHH[:,1]
        dfHH_dict[name]['cl3d_probDM2'] = probasHH[:,2]
        dfHH_dict[name]['cl3d_probDM3'] = probasHH[:,3]
        dfHH_dict[name]['cl3d_probDM4'] = probasHH[:,4]

        dfQCD_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfQCD_dict[name][features])
        probasQCD = model_dict[name].predict_proba(dfQCD_dict[name][features])
        dfQCD_dict[name]['cl3d_probDM0'] = probasQCD[:,0]
        dfQCD_dict[name]['cl3d_probDM1'] = probasQCD[:,1]
        dfQCD_dict[name]['cl3d_probDM2'] = probasQCD[:,2]
        dfQCD_dict[name]['cl3d_probDM3'] = probasQCD[:,3]
        dfQCD_dict[name]['cl3d_probDM4'] = probasQCD[:,4]

        dfMergedTraining_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfMergedTraining_dict[name][features])
        probas_train = model_dict[name].predict_proba(dfMergedTraining_dict[name][features])
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfMergedTraining_dict[name]['gentau_decayMode'], probas_train)
        plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'_training.pdf')
        plt.close()
        dfMergedTraining_dict[name]['cl3d_probDM0'] = probas_train[:,0]
        dfMergedTraining_dict[name]['cl3d_probDM1'] = probas_train[:,1]
        dfMergedTraining_dict[name]['cl3d_probDM2'] = probas_train[:,2]
        dfMergedTraining_dict[name]['cl3d_probDM3'] = probas_train[:,3]
        dfMergedTraining_dict[name]['cl3d_probDM4'] = probas_train[:,4]

        dfMergedValidation_dict[name]['cl3d_predDM'] = model_dict[name].predict(dfMergedValidation_dict[name][features])
        probas_valid = model_dict[name].predict_proba(dfMergedValidation_dict[name][features])
        plt.figure(figsize=(15,10))
        skplt.metrics.plot_roc(dfMergedValidation_dict[name]['gentau_decayMode'], probas_valid)
        plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'_validation.pdf')
        plt.close()
        dfMergedValidation_dict[name]['cl3d_probDM0'] = probas_valid[:,0]
        dfMergedValidation_dict[name]['cl3d_probDM1'] = probas_valid[:,1]
        dfMergedValidation_dict[name]['cl3d_probDM2'] = probas_valid[:,2]
        dfMergedValidation_dict[name]['cl3d_probDM3'] = probas_valid[:,3]
        dfMergedValidation_dict[name]['cl3d_probDM4'] = probas_valid[:,4]

        print('** INFO: finished probability prediction')


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

        store_qcd = pd.HDFStore(outFileQCD_dict[name], mode='w')
        store_qcd[name] = dfQCD_dict[name]
        store_qcd.close()


        ######################### MAKE CONFUSION MATRICES #########################

        DMs_dict[name] = dfMergedTraining_dict[name].groupby('gentau_decayMode')['cl3d_eta'].count()
        cm_dict[name] = confusion_matrix(dfMergedValidation_dict[name].gentau_decayMode, dfMergedValidation_dict[name].cl3d_predDM)
        cm_dict[name] = cm_dict[name].astype('float') / cm_dict[name].sum(axis=1)[:, np.newaxis]
        ns_dict[name] = DMs_dict[name]
        print('\n** INFO: fraction of correctly identified DMs - FE {0}: {1}'.format(feNames_dict[name], (cm_dict[name][0,0]*ns_dict[name][0]+cm_dict[name][1,1]*ns_dict[name][1]+cm_dict[name][2,2]*ns_dict[name][2]+cm_dict[name][3,3]*ns_dict[name][3]+cm_dict[name][4,4]*ns_dict[name][4])/np.sum(ns_dict[name])))

        print('\n** INFO: finished DM sorting for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        

        ######################### PLOTTING #########################

        if args.doPlots:
            for name in feNames_dict:
                if not name in args.FE: continue # skip the front-end options that we do not want to do

                ######################### PLOT FEATURES #########################
                for var in features_dict:
                    plt.figure(figsize=(8,8))
                    plt.hist(dfTau_DM0_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='1-prong', color='limegreen',  histtype='step', lw=2, alpha=0.8)
                    plt.hist(dfTau_DM1_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'1-prong + $\pi^{0}$\'s', color='blue',  histtype='step', lw=2, alpha=0.8)
                    plt.hist(dfTau_DM5_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='2-prong', color='red',  histtype='step', lw=2, alpha=0.8)                    
                    #plt.hist(dfTau_DM6_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='2-prong + $\pi^{0}$', color='red',  histtype='step', lw=2, alpha=0.8)
                    #plt.hist(dfTau_DM10_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='3-prong', color='blue',  histtype='step', lw=2, alpha=0.8)
                    plt.hist(dfTau_DM11_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'3-prong + $\pi^{0}$', color='fuchsia',  histtype='step', lw=2, alpha=0.8)
                    plt.hist(dfQCD_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background', color='cyan',  histtype='step', lw=2, alpha=0.8)
                    DM0_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='1-prong',lw=2)
                    DM1_line = mlines.Line2D([], [], color='blue',markersize=15, label=r'1-prong + $\pi^{0}$',lw=2)
                    DM5_line = mlines.Line2D([], [], color='red',markersize=15, label='2-prong',lw=2)
                    #DM6_line = mlines.Line2D([], [], color='red',markersize=15, label='2-prong',lw=2)
                    #DM10_line = mlines.Line2D([], [], color='red',markersize=15, label='3-prong',lw=2)
                    DM11_line = mlines.Line2D([], [], color='fuchsia',markersize=15, label=r'3-prong + $\pi^{0}$',lw=2)
                    QCD_line = mlines.Line2D([], [], color='cyan', markersize=15, label='QCD background', lw=2)
                    plt.legend(loc = 'upper right',handles=[DM0_line,DM1_line,DM5_line,DM11_line,QCD_line])
                    plt.grid(linestyle=':')
                    plt.xlabel(features_dict[var][0])
                    plt.ylabel(r'Normalized events')
                    plt.subplots_adjust(bottom=0.12)
                    plt.savefig(plotdir+'/'+var+'_'+name+'_DMs.pdf')
                    plt.close()

                    # plt.figure(figsize=(8,8))
                    # plt.hist(dfTau_DM0_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label='1-prong', color='limegreen',  histtype='step', lw=2, alpha=0.8)
                    # plt.hist(dfTau_DM1_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'1-prong + $\pi^{0}$\'s', color='blue',  histtype='step', lw=2, alpha=0.8)
                    # plt.hist(dfTau_DM2_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'2-prong (+ $\pi^{0}$)', color='red',  histtype='step', lw=2, alpha=0.8)
                    # plt.hist(dfTau_DM3_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]),  label=r'3-prong (+ $\pi^{0}$)', color='fuchsia',  histtype='step', lw=2, alpha=0.8)
                    # plt.hist(dfQCD_dict[name][var], density=True, bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='QCD background', color='cyan',  histtype='step', lw=2, alpha=0.8)
                    # DM0_line = mlines.Line2D([], [], color='limegreen',markersize=15, label='1-prong',lw=2)
                    # DM1_line = mlines.Line2D([], [], color='blue',markersize=15, label=r'1-prong + $\pi^{0}$',lw=2)
                    # DM2_line = mlines.Line2D([], [], color='red',markersize=15, label=r'2-prong (+ $\pi^{0})$',lw=2)
                    # DM3_line = mlines.Line2D([], [], color='fuchsia',markersize=15, label=r'3-prong (+ $\pi^{0})$',lw=2)
                    # QCD_line = mlines.Line2D([], [], color='cyan', markersize=15, label='QCD background', lw=2)
                    # plt.legend(loc = 'upper right',handles=[DM0_line,DM1_line,DM2_line,DM3_line,QCD_line])
                    # plt.grid(linestyle=':')
                    # plt.xlabel(features_dict[var][0])
                    # plt.ylabel(r'Normalized events')
                    # plt.subplots_adjust(bottom=0.12)
                    # plt.savefig(plotdir+'/'+var+'_'+name+'_DMmerged.pdf')
                    # plt.close()


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
                plt.xticks([0,1,2,3,4], ('1-prong', r'1-prong$+\pi_0$',r'2-prong($+\pi_0$)' , r'3-prong($+\pi_0$)', 'QCD'))
                plt.yticks([0,1,2,3,4], ('1-prong', r'1-prong$+\pi_0$',r'2-prong($+\pi_0$)' , r'3-prong($+\pi_0$)', 'QCD'))
                plt.xlim(-0.5, 4.5)
                plt.ylim(-0.5, 4.5),
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

                plt.figure(figsize=(8,8))
                skplt.metrics.plot_confusion_matrix(dfMergedValidation_dict[name].gentau_decayMode, dfMergedValidation_dict[name].cl3d_predDM, normalize=True)
                plt.savefig(plotdir+'/test.pdf')
                plt.close()


                ######################### PREDICTED PROBABILITIES #########################

                print('\n** INFO: plotting predicted probabilities')
                plt.figure(figsize=(15,10))
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==0].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label='1-prong', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==1].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==2].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'2-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==3].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==4].cl3d_probDM0, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'QCD background', density=True)
                plt.legend(fontsize=22)
                plt.xlabel(r'Probabilities 1-prong category')
                plt.ylabel(r'Entries')
                plt.savefig(plotdir+'/DM0_prob_'+feNames_dict[name]+'_'+PUbdtWP+'.pdf')
                plt.close()

                plt.figure(figsize=(15,10))
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==0].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label='1-prong', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==1].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==2].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'2-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==3].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==4].cl3d_probDM1, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'QCD background', density=True)
                plt.legend(fontsize=22)
                plt.xlabel(r'Probabilities 1-prong$+\pi_0$ category')
                plt.ylabel(r'Entries')
                plt.savefig(plotdir+'/DM1_prob_'+PUbdtWP+'_'+name+'.pdf')
                plt.close()

                plt.figure(figsize=(15,10))
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==0].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label='1-prong', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==1].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==2].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'2-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==3].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==4].cl3d_probDM2, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'QCD background', density=True)
                plt.legend(fontsize=22)
                plt.xlabel(r'Probabilities 2-prong($+\pi_0$) category')
                plt.ylabel(r'Entries')
                plt.savefig(plotdir+'/DM2_prob_'+PUbdtWP+'_'+name+'.pdf')
                plt.close()

                plt.figure(figsize=(15,10))
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==0].cl3d_probDM3, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label='1-prong', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==1].cl3d_probDM3, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==2].cl3d_probDM3, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'2-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==3].cl3d_probDM3, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==4].cl3d_probDM3, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'QCD background', density=True)
                plt.legend(fontsize=22)
                plt.xlabel(r'Probabilities 3-prong($+\pi_0$) category')
                plt.ylabel(r'Entries')
                plt.savefig(plotdir+'/DM3_prob_'+PUbdtWP+'_'+name+'.pdf')
                plt.close()

                plt.figure(figsize=(15,10))
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==0].cl3d_probDM4, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label='1-prong', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==1].cl3d_probDM4, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'1-prong$+\pi_0$', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==2].cl3d_probDM4, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'2-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==3].cl3d_probDM4, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'3-prongs($+\pi_0$)', density=True)
                plt.hist(dfMergedValidation_dict[name][dfMergedValidation_dict[name].gentau_decayMode==4].cl3d_probDM4, bins=np.arange(0., 1., 0.05), histtype='step', lw=2, label=r'QCD background', density=True)
                plt.legend(fontsize=22)
                plt.xlabel(r'Probabilities QCD category')
                plt.ylabel(r'Entries')
                plt.savefig(plotdir+'/DMqcd_prob_'+PUbdtWP+'_'+name+'.pdf')
                plt.close()


                ######################### ROC CURVE #########################
                
                # print('\n** INFO: plotting ROC curve')
                # plt.figure(figsize=(15,10))
                # skplt.metrics.plot_roc(dfHH_dict[name]['gentau_decayMode'], probas_dict[name])
                # plt.savefig(plotdir+'/DM_roc_'+PUbdtWP+'_'+name+'.pdf')
                # plt.close()
                











