import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import argparse


# binned responses and resolutions
def effrms(df, c=0.68):
    """ Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out_dict = {}
    for col in df:
        x = df[col]
        x = np.sort(x, kind="mergesort")
        m = int(c * len(x)) + 1
        if len(x) > 3: out_dict[col] = [np.min(x[m:] - x[:-m]) / 2.0] 
        elif len(x) > 0: out_dict[col] = [np.min(x[len(x)-1] - x[0]) / 2.0]
        else: out_dict[col] = [0]
    return pd.DataFrame(out_dict).iloc[0]

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'r') as f:
        pickle.load(f)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doPlots', dest='doPlots', help='do plots?',  action='store_true', default=False)
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--ptcut', dest='ptcut', help='baseline 3D cluster pT cut to use', default='0')
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    if args.ptcut == '0':
        print('** INFO: no baseline pT cut specified to be used for training and validation')
        print('** INFO: using default cut pT>4GeV')
        args.ptcut = '4'


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
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/calibration'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_matched.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_matched.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_calibrated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_calibrated.hdf5',
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_modelC1_dict = {
        'threshold'    : model_outdir+'/model_c1_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC2_dict = {
        'threshold'    : model_outdir+'/model_c2_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC3_dict = {
        'threshold'    : model_outdir+'/model_c3_th_PU200.pkl',
        'supertrigger' : model_outdir+'/',
        'bestchoice'   : model_outdir+'/',
        'bestcoarse'   : model_outdir+'/',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}
    dfQCDTraining_dict = {}
    dfNuTraining_dict = {}
    dfQCDValidation_dict = {}
    dfNuValidation_dict = {}
    C1model_dict = {}       # dictionary of models from C1 calibration step
    C2model_dict = {}       # dictionary of models from C2 calibration step
    C3model_dict = {}       # dictionary of models from C3 calibration step    
    
    meansTrainPt_dict = {}       # dictionary of
    rmssTrainPt_dict = {}        # dictionary of
    
    etameansTrain_dict = {}      # dictionary of eta means - used for plotting
    etarmssTrain_dict = {}       # dictionary of eta rms(std) - used for plotting
    etaeffrmssTrain_dict = {}    # dictionary of
    ptmeansTrain_dict = {}       # dictionary of pt means - used for plotting
    ptrmssTrain_dict = {}        # dictionary of pt rms(std) - used for plotting
    pteffrmssTrain_dict = {}     # dictionary of

    etameansValid_dict = {}      # dictionary of eta means - used for plotting
    etarmssValid_dict = {}       # dictionary of eta rms(std) - used for plotting
    etaeffrmssValid_dict = {}    # dictionary of
    ptmeansValid_dict = {}       # dictionary of pt means - used for plotting
    ptrmssValid_dict = {}        # dictionary of pt rms(std) - used for plotting
    pteffrmssValid_dict = {}     # dictionary of

    dfRealApplication_dict = {}
    dfRealApplication_dict = {}
    dfRealApplication1_dict = {}
    dfRealApplication1_dict = {}
    dfRealApplication2_dict = {}
    dfRealApplication2_dict = {}
    dfRealApplication3_dict = {}
    dfRealApplication3_dict = {}

    dfTrainingDM0_dict = {}
    dfTrainingDM1_dict = {}
    dfTrainingDM10_dict = {}
    dfTrainingDM11_dict = {}
    dfValidationDM0_dict = {}
    dfValidationDM1_dict = {}
    dfValidationDM10_dict = {}
    dfValidationDM11_dict = {}
    
    # minimum pt and eta requirements 
    ptmin = 20
    etamin = 1.6

    # features used for the C2 calibration step
    features = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean','cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']

    # features used for the C3 calibration step
    vars = ['gentau_vis_pt', 'gentau_vis_bin_pt', 'cl3d_pt_c2', 'cl3d_response_c2']

    # variables to plot
    plot_var = ['gentau_vis_pt', 'gentau_vis_abseta', 'gentau_vis_bin_eta', 'gentau_vis_bin_pt', 'cl3d_pt', 'cl3d_response', 'cl3d_abseta', 'cl3d_pt_c1', 'cl3d_response_c1', 'cl3d_pt_c2', 'cl3d_response_c2', 'cl3d_pt_c3', 'cl3d_response_c3']

    # colors to use
    colors_dict = {
        'threshold'    : 'blue',
        'supertrigger' : 'red',
        'bestchoice'   : 'olive',
        'bestcoarse'   : 'orange',
        'mixed'        : 'fuchsia'
    }

    # legend to use
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'supertrigger' : 'STC4+16',
        'bestchoice'   : 'BC Decentral',
        'bestcoarse'   : 'BC Coarse 2x2 TC',
        'mixed'        : 'Mixed BC + STC'
    }

    matplotlib.rcParams.update({'font.size': 16})


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end option that we do not want to do
        
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting energy calibration for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        dfTraining_dict[name]['cl3d_abseta'] = np.abs(dfTraining_dict[name]['cl3d_eta'])
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()
        dfValidation_dict[name]['cl3d_abseta'] = np.abs(dfValidation_dict[name]['cl3d_eta'])


        ######################### SELECT EVENTS FOR TRAINING AND VALIDATION #########################

        dfQCDTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-2 and cl3d_pt>{0}  and genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9)) and cl3d_isbestmatch==True'.format(args.ptcut))
        dfNuTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-1 and cl3d_pt>{0} and cl3d_abseta>1.6 and cl3d_abseta<2.9'.format(args.ptcut))

        dfTraining_dict[name] = dfTraining_dict[name].query('gentau_decayMode>=0 and cl3d_pt>{0} and gentau_vis_pt>20 and  ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)) and cl3d_isbestmatch==True'.format(args.ptcut))

        dfQCDValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-2 and cl3d_pt>{0} and((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9))'.format(args.ptcut))
        dfNuValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-1 and cl3d_pt>{0} and cl3d_abseta>1.6 and cl3d_abseta<2.9'.format(args.ptcut))

        dfRealApplication_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9 and cl3d_isbestmatch==True')
        dfRealApplication1_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_pt>8 and cl3d_abseta>1.6 and cl3d_abseta<2.9 and cl3d_isbestmatch==True')
        dfRealApplication2_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_pt>12 and cl3d_abseta>1.6 and cl3d_abseta<2.9 and cl3d_isbestmatch==True')
        dfRealApplication3_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_pt>16 and cl3d_abseta>1.6 and cl3d_abseta<2.9 and cl3d_isbestmatch==True')

        dfValidation_dict[name] = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_pt>{0} and gentau_vis_pt>20 and  ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)) and cl3d_isbestmatch==True'.format(args.ptcut))

        # fix Nu dataframes content
        dfNuTraining_dict[name]['n_matched_cl3d'] = 0 # we have to set this because the matching is not done as no taus are present
        dfNuValidation_dict[name]['n_matched_cl3d'] = 0
        dfNuTraining_dict[name].dropna(axis=1,inplace=True) # drop fake columns created at concatenation
        dfNuValidation_dict[name].dropna(axis=1,inplace=True)

        # calculate responses
        dfTraining_dict[name]['cl3d_response'] = dfTraining_dict[name]['cl3d_pt']/dfTraining_dict[name]['gentau_vis_pt']
        dfValidation_dict[name]['cl3d_response'] = dfValidation_dict[name]['cl3d_pt']/dfValidation_dict[name]['gentau_vis_pt']
        dfRealApplication_dict[name]['cl3d_response'] = dfRealApplication_dict[name]['cl3d_pt']/dfRealApplication_dict[name]['gentau_vis_pt']
        dfRealApplication1_dict[name]['cl3d_response'] = dfRealApplication1_dict[name]['cl3d_pt']/dfRealApplication1_dict[name]['gentau_vis_pt']
        dfRealApplication2_dict[name]['cl3d_response'] = dfRealApplication2_dict[name]['cl3d_pt']/dfRealApplication2_dict[name]['gentau_vis_pt']
        dfRealApplication3_dict[name]['cl3d_response'] = dfRealApplication3_dict[name]['cl3d_pt']/dfRealApplication3_dict[name]['gentau_vis_pt']


        ######################### C1 CALIBRATION TRAINING (PU eta dependent calibration) #########################

        print('\n** INFO: training calibration C1')

        input_c1 = dfTraining_dict[name][['cl3d_abseta']]
        target_c1 = dfTraining_dict[name].gentau_vis_pt - dfTraining_dict[name].cl3d_pt
        C1model_dict[name] = LinearRegression().fit(input_c1, target_c1)

        save_obj(C1model_dict[name], outFile_modelC1_dict[name])

        dfTraining_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfTraining_dict[name][['cl3d_abseta']])
        dfTraining_dict[name]['cl3d_pt_c1'] = dfTraining_dict[name].cl3d_c1 + dfTraining_dict[name].cl3d_pt
        dfTraining_dict[name]['cl3d_response_c1'] = dfTraining_dict[name].cl3d_pt_c1 / dfTraining_dict[name].gentau_vis_pt


        ######################### C2 CALIBRATION TRAINING (DM dependent calibration) #########################

        print('\n** INFO: training calibration C2')

        input_c2 = dfTraining_dict[name][features]
        target_c2 = dfTraining_dict[name].gentau_vis_pt / dfTraining_dict[name].cl3d_pt_c1
        C2model_dict[name] = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=0, loss='huber').fit(input_c2, target_c2)

        save_obj(C2model_dict[name], outFile_modelC2_dict[name])

        dfTraining_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfTraining_dict[name][features])
        dfTraining_dict[name]['cl3d_pt_c2'] = dfTraining_dict[name].cl3d_c2 * dfTraining_dict[name].cl3d_pt_c1
        dfTraining_dict[name]['cl3d_response_c2'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].gentau_vis_pt

        # print importance of the features used for training
        feature_importance = C2model_dict[name].feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(8, 6))
        plt.gcf().subplots_adjust(left=0.3)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(features)[sorted_idx])
        plt.title('Feature Importance (MDI)')
        plt.savefig(plotdir+'/'+name+'_featureImportance_modelC2.pdf')


        ######################### C3 CALIBRATION TRAINING (E dependent calibration) #########################

        print('\n** INFO: training calibration C3')

        dfTraining_dict[name]['gentau_vis_abseta'] = np.abs(dfTraining_dict[name]['gentau_vis_eta'])
        dfTraining_dict[name]['gentau_vis_bin_eta'] = ((dfTraining_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTraining_dict[name]['gentau_vis_bin_pt']  = ((dfTraining_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

        meansTrainPt_dict[name] = dfTraining_dict[name][vars].groupby('gentau_vis_bin_pt').mean() 
        rmssTrainPt_dict[name] = dfTraining_dict[name][vars].groupby('gentau_vis_bin_pt').std() 

        meansTrainPt_dict[name]['logpt1'] = np.log(meansTrainPt_dict[name]['cl3d_pt_c2'])
        meansTrainPt_dict[name]['logpt2'] = meansTrainPt_dict[name].logpt1**2
        meansTrainPt_dict[name]['logpt3'] = meansTrainPt_dict[name].logpt1**3
        meansTrainPt_dict[name]['logpt4'] = meansTrainPt_dict[name].logpt1**4

        input_c3 = meansTrainPt_dict[name][['logpt1', 'logpt2', 'logpt3', 'logpt4']]
        target_c3 = meansTrainPt_dict[name]['cl3d_response_c2']
        C3model_dict[name] = LinearRegression().fit(input_c3, target_c3)

        save_obj(C3model_dict[name], outFile_modelC3_dict[name])

        logpt1 = np.log(abs(dfTraining_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4

        dfTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTraining_dict[name]['cl3d_pt_c3'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].cl3d_c3
        dfTraining_dict[name]['cl3d_response_c3'] = dfTraining_dict[name].cl3d_pt_c3 / dfTraining_dict[name].gentau_vis_pt


        ######################### C1+C2+C3 CALIBRATION APPLICATION #########################

        print('\n** INFO: applying calibration C1+C2+C3')
        
        # VALIDATION DATASET
        # application calibration 1
        dfValidation_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfValidation_dict[name][['cl3d_abseta']])
        dfValidation_dict[name]['cl3d_pt_c1'] = dfValidation_dict[name].cl3d_c1 + dfValidation_dict[name].cl3d_pt
        dfValidation_dict[name]['cl3d_response_c1'] = dfValidation_dict[name].cl3d_pt_c1 / dfValidation_dict[name].gentau_vis_pt
        # application calibration 2
        dfValidation_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfValidation_dict[name][features])
        dfValidation_dict[name]['cl3d_pt_c2'] = dfValidation_dict[name].cl3d_c2 * dfValidation_dict[name].cl3d_pt_c1
        dfValidation_dict[name]['cl3d_response_c2'] = dfValidation_dict[name].cl3d_pt_c2 / dfValidation_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfValidation_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfValidation_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfValidation_dict[name]['cl3d_pt_c3'] = dfValidation_dict[name].cl3d_pt_c2 / dfValidation_dict[name].cl3d_c3
        dfValidation_dict[name]['cl3d_response_c3'] = dfValidation_dict[name].cl3d_pt_c3 / dfValidation_dict[name].gentau_vis_pt

        # TRAINING DATASET
        # application calibration 1
        dfTraining_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfTraining_dict[name][['cl3d_abseta']])
        dfTraining_dict[name]['cl3d_pt_c1'] = dfTraining_dict[name].cl3d_c1 + dfTraining_dict[name].cl3d_pt
        dfTraining_dict[name]['cl3d_response_c1'] = dfTraining_dict[name].cl3d_pt_c1 / dfTraining_dict[name].gentau_vis_pt
        # application calibration 2
        dfTraining_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfTraining_dict[name][features])
        dfTraining_dict[name]['cl3d_pt_c2'] = dfTraining_dict[name].cl3d_c2 * dfTraining_dict[name].cl3d_pt_c1
        dfTraining_dict[name]['cl3d_response_c2'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfTraining_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTraining_dict[name]['cl3d_pt_c3'] = dfTraining_dict[name].cl3d_pt_c2 / dfTraining_dict[name].cl3d_c3
        dfTraining_dict[name]['cl3d_response_c3'] = dfTraining_dict[name].cl3d_pt_c3 / dfTraining_dict[name].gentau_vis_pt

        # REAL APPLICATION DATASET
        dfRealApplication_dict[name]['cl3d_response'] = dfRealApplication_dict[name]['cl3d_pt']/dfRealApplication_dict[name]['gentau_vis_pt']
        # application calibration 1
        dfRealApplication_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfRealApplication_dict[name][['cl3d_abseta']])
        dfRealApplication_dict[name]['cl3d_pt_c1'] = dfRealApplication_dict[name].cl3d_c1 + dfRealApplication_dict[name].cl3d_pt
        dfRealApplication_dict[name]['cl3d_response_c1'] = dfRealApplication_dict[name].cl3d_pt_c1 / dfRealApplication_dict[name].gentau_vis_pt
        # application calibration 2
        dfRealApplication_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfRealApplication_dict[name][features])
        dfRealApplication_dict[name]['cl3d_pt_c2'] = dfRealApplication_dict[name].cl3d_c2 * dfRealApplication_dict[name].cl3d_pt_c1
        dfRealApplication_dict[name]['cl3d_response_c2'] = dfRealApplication_dict[name].cl3d_pt_c2 / dfRealApplication_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfRealApplication_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfRealApplication_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfRealApplication_dict[name]['cl3d_pt_c3'] = dfRealApplication_dict[name].cl3d_pt_c2 / dfRealApplication_dict[name].cl3d_c3
        dfRealApplication_dict[name]['cl3d_response_c3'] = dfRealApplication_dict[name].cl3d_pt_c3 / dfRealApplication_dict[name].gentau_vis_pt

        # REAL APPLICATION DATASET 1
        dfRealApplication1_dict[name]['cl3d_response'] = dfRealApplication1_dict[name]['cl3d_pt']/dfRealApplication1_dict[name]['gentau_vis_pt']
        # application calibration 1
        dfRealApplication1_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfRealApplication1_dict[name][['cl3d_abseta']])
        dfRealApplication1_dict[name]['cl3d_pt_c1'] = dfRealApplication1_dict[name].cl3d_c1 + dfRealApplication1_dict[name].cl3d_pt
        dfRealApplication1_dict[name]['cl3d_response_c1'] = dfRealApplication1_dict[name].cl3d_pt_c1 / dfRealApplication1_dict[name].gentau_vis_pt
        # application calibration 2
        dfRealApplication1_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfRealApplication1_dict[name][features])
        dfRealApplication1_dict[name]['cl3d_pt_c2'] = dfRealApplication1_dict[name].cl3d_c2 * dfRealApplication1_dict[name].cl3d_pt_c1
        dfRealApplication1_dict[name]['cl3d_response_c2'] = dfRealApplication1_dict[name].cl3d_pt_c2 / dfRealApplication1_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfRealApplication1_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfRealApplication1_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfRealApplication1_dict[name]['cl3d_pt_c3'] = dfRealApplication1_dict[name].cl3d_pt_c2 / dfRealApplication1_dict[name].cl3d_c3
        dfRealApplication1_dict[name]['cl3d_response_c3'] = dfRealApplication1_dict[name].cl3d_pt_c3 / dfRealApplication1_dict[name].gentau_vis_pt

        # REAL APPLICATION DATASET 2
        dfRealApplication2_dict[name]['cl3d_response'] = dfRealApplication2_dict[name]['cl3d_pt']/dfRealApplication2_dict[name]['gentau_vis_pt']
        # application calibration 1
        dfRealApplication2_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfRealApplication2_dict[name][['cl3d_abseta']])
        dfRealApplication2_dict[name]['cl3d_pt_c1'] = dfRealApplication2_dict[name].cl3d_c1 + dfRealApplication2_dict[name].cl3d_pt
        dfRealApplication2_dict[name]['cl3d_response_c1'] = dfRealApplication2_dict[name].cl3d_pt_c1 / dfRealApplication2_dict[name].gentau_vis_pt
        # application calibration 2
        dfRealApplication2_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfRealApplication2_dict[name][features])
        dfRealApplication2_dict[name]['cl3d_pt_c2'] = dfRealApplication2_dict[name].cl3d_c2 * dfRealApplication2_dict[name].cl3d_pt_c1
        dfRealApplication2_dict[name]['cl3d_response_c2'] = dfRealApplication2_dict[name].cl3d_pt_c2 / dfRealApplication2_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfRealApplication2_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfRealApplication2_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfRealApplication2_dict[name]['cl3d_pt_c3'] = dfRealApplication2_dict[name].cl3d_pt_c2 / dfRealApplication2_dict[name].cl3d_c3
        dfRealApplication2_dict[name]['cl3d_response_c3'] = dfRealApplication2_dict[name].cl3d_pt_c3 / dfRealApplication2_dict[name].gentau_vis_pt

        # REAL APPLICATION DATASET 3
        dfRealApplication3_dict[name]['cl3d_response'] = dfRealApplication3_dict[name]['cl3d_pt']/dfRealApplication3_dict[name]['gentau_vis_pt']
        # application calibration 1
        dfRealApplication3_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfRealApplication3_dict[name][['cl3d_abseta']])
        dfRealApplication3_dict[name]['cl3d_pt_c1'] = dfRealApplication3_dict[name].cl3d_c1 + dfRealApplication3_dict[name].cl3d_pt
        dfRealApplication3_dict[name]['cl3d_response_c1'] = dfRealApplication3_dict[name].cl3d_pt_c1 / dfRealApplication3_dict[name].gentau_vis_pt
        # application calibration 2
        dfRealApplication3_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfRealApplication3_dict[name][features])
        dfRealApplication3_dict[name]['cl3d_pt_c2'] = dfRealApplication3_dict[name].cl3d_c2 * dfRealApplication3_dict[name].cl3d_pt_c1
        dfRealApplication3_dict[name]['cl3d_response_c2'] = dfRealApplication3_dict[name].cl3d_pt_c2 / dfRealApplication3_dict[name].gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfRealApplication3_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfRealApplication3_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfRealApplication3_dict[name]['cl3d_pt_c3'] = dfRealApplication3_dict[name].cl3d_pt_c2 / dfRealApplication3_dict[name].cl3d_c3
        dfRealApplication3_dict[name]['cl3d_response_c3'] = dfRealApplication3_dict[name].cl3d_pt_c3 / dfRealApplication3_dict[name].gentau_vis_pt

        # QCD VALIDATION DATASET
        dfQCDValidation_dict[name]['cl3d_response'] = dfQCDValidation_dict[name]['cl3d_pt']/dfQCDValidation_dict[name]['genjet_pt']
        # application calibration 1
        dfQCDValidation_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfQCDValidation_dict[name][['cl3d_abseta']])
        dfQCDValidation_dict[name]['cl3d_pt_c1'] = dfQCDValidation_dict[name].cl3d_c1 + dfQCDValidation_dict[name].cl3d_pt
        dfQCDValidation_dict[name]['cl3d_response_c1'] = dfQCDValidation_dict[name].cl3d_pt_c1 / dfQCDValidation_dict[name].genjet_pt
        # application calibration 2
        dfQCDValidation_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfQCDValidation_dict[name][features])
        dfQCDValidation_dict[name]['cl3d_pt_c2'] = dfQCDValidation_dict[name].cl3d_c2 * dfQCDValidation_dict[name].cl3d_pt_c1
        dfQCDValidation_dict[name]['cl3d_response_c2'] = dfQCDValidation_dict[name].cl3d_pt_c2 / dfQCDValidation_dict[name].genjet_pt
        # application calibration 3
        logpt1 = np.log(abs(dfQCDValidation_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDValidation_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDValidation_dict[name]['cl3d_pt_c3'] = dfQCDValidation_dict[name].cl3d_pt_c2 / dfQCDValidation_dict[name].cl3d_c3
        dfQCDValidation_dict[name]['cl3d_response_c3'] = dfQCDValidation_dict[name].cl3d_pt_c3 / dfQCDValidation_dict[name].genjet_pt

        # QCD TRAINING DATASET
        dfQCDTraining_dict[name]['cl3d_response'] = dfQCDTraining_dict[name]['cl3d_pt']/dfQCDTraining_dict[name]['genjet_pt']
        # application calibration 1
        dfQCDTraining_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfQCDTraining_dict[name][['cl3d_abseta']])
        dfQCDTraining_dict[name]['cl3d_pt_c1'] = dfQCDTraining_dict[name].cl3d_c1 + dfQCDTraining_dict[name].cl3d_pt
        dfQCDTraining_dict[name]['cl3d_response_c1'] = dfQCDTraining_dict[name].cl3d_pt_c1 / dfQCDTraining_dict[name].genjet_pt
        # application calibration 2
        dfQCDTraining_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfQCDTraining_dict[name][features])
        dfQCDTraining_dict[name]['cl3d_pt_c2'] = dfQCDTraining_dict[name].cl3d_c2 * dfQCDTraining_dict[name].cl3d_pt_c1
        dfQCDTraining_dict[name]['cl3d_response_c2'] = dfQCDTraining_dict[name].cl3d_pt_c2 / dfQCDTraining_dict[name].genjet_pt
        # application calibration 3
        logpt1 = np.log(abs(dfQCDTraining_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDTraining_dict[name]['cl3d_pt_c3'] = dfQCDTraining_dict[name].cl3d_pt_c2 / dfQCDTraining_dict[name].cl3d_c3
        dfQCDTraining_dict[name]['cl3d_response_c3'] = dfQCDTraining_dict[name].cl3d_pt_c3 / dfQCDTraining_dict[name].genjet_pt

        # NU VALIDATION DATASET
        # application calibration 1
        dfNuValidation_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfNuValidation_dict[name][['cl3d_abseta']])
        dfNuValidation_dict[name]['cl3d_pt_c1'] = dfNuValidation_dict[name].cl3d_c1 + dfNuValidation_dict[name].cl3d_pt
        # application calibration 2
        dfNuValidation_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfNuValidation_dict[name][features])
        dfNuValidation_dict[name]['cl3d_pt_c2'] = dfNuValidation_dict[name].cl3d_c2 * dfNuValidation_dict[name].cl3d_pt_c1
        # application calibration 3
        logpt1 = np.log(abs(dfNuValidation_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfNuValidation_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfNuValidation_dict[name]['cl3d_pt_c3'] = dfNuValidation_dict[name].cl3d_pt_c2 / dfNuValidation_dict[name].cl3d_c3

        # NU TRAINING DATASET
        # application calibration 1
        dfNuTraining_dict[name]['cl3d_c1'] = C1model_dict[name].predict(dfNuTraining_dict[name][['cl3d_abseta']])
        dfNuTraining_dict[name]['cl3d_pt_c1'] = dfNuTraining_dict[name].cl3d_c1 + dfNuTraining_dict[name].cl3d_pt
        # application calibration 2
        dfNuTraining_dict[name]['cl3d_c2'] = C2model_dict[name].predict(dfNuTraining_dict[name][features])
        dfNuTraining_dict[name]['cl3d_pt_c2'] = dfNuTraining_dict[name].cl3d_c2 * dfNuTraining_dict[name].cl3d_pt_c1
        # application calibration 3
        logpt1 = np.log(abs(dfNuTraining_dict[name]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfNuTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfNuTraining_dict[name]['cl3d_pt_c3'] = dfNuTraining_dict[name].cl3d_pt_c2 / dfNuTraining_dict[name].cl3d_c3


        ######################### EVALUATE MEAN AND RMS OF RESPONSE AND SAVE VALUES #########################

        print('\n** INFO: calculation of response MEAN and RMS')

        dfTraining_dict[name]['gentau_vis_bin_eta'] = ((dfTraining_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTraining_dict[name]['gentau_vis_bin_pt']  = ((dfTraining_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

        etameansTrain_dict[name]   = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').mean()
        etarmssTrain_dict[name]    = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').std()
        etaeffrmssTrain_dict[name] = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_eta').apply(effrms)
        ptmeansTrain_dict[name]    = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').mean()
        ptrmssTrain_dict[name]     = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').std()
        pteffrmssTrain_dict[name]  = dfTraining_dict[name][plot_var].groupby('gentau_vis_bin_pt').apply(effrms)

        dfValidation_dict[name]['gentau_vis_abseta'] = np.abs(dfValidation_dict[name]['gentau_vis_eta'])
        dfValidation_dict[name]['gentau_vis_bin_eta'] = ((dfValidation_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfValidation_dict[name]['gentau_vis_bin_pt']  = ((dfValidation_dict[name]['gentau_vis_pt'] - ptmin)/5).astype('int32')

        etameansValid_dict[name]   = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_eta').mean()
        etarmssValid_dict[name]    = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_eta').std()
        etaeffrmssValid_dict[name] = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_eta').apply(effrms)
        ptmeansValid_dict[name]    = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_pt').mean()
        ptrmssValid_dict[name]     = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_pt').std()
        pteffrmssValid_dict[name]  = dfValidation_dict[name][plot_var].groupby('gentau_vis_bin_pt').apply(effrms)

        original_stdout = sys.stdout # Save a reference to the original standard output
        # save response to file
        with open(plotdir+'/responses.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print('TRAINING DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response'].mean(), dfTraining_dict[name]['cl3d_response'].std(), dfTraining_dict[name]['cl3d_response'].std()/dfTraining_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c1'].mean(), dfTraining_dict[name]['cl3d_response_c1'].std(), dfTraining_dict[name]['cl3d_response_c1'].std()/dfTraining_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c2'].mean(), dfTraining_dict[name]['cl3d_response_c2'].std(), dfTraining_dict[name]['cl3d_response_c2'].std()/dfTraining_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfTraining_dict[name]['cl3d_response_c3'].mean(), dfTraining_dict[name]['cl3d_response_c3'].std(), dfTraining_dict[name]['cl3d_response_c3'].std()/dfTraining_dict[name]['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('VALIDATION DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfValidation_dict[name]['cl3d_response'].mean(), dfValidation_dict[name]['cl3d_response'].std(), dfValidation_dict[name]['cl3d_response'].std()/dfValidation_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfValidation_dict[name]['cl3d_response_c1'].mean(), dfValidation_dict[name]['cl3d_response_c1'].std(), dfValidation_dict[name]['cl3d_response_c1'].std()/dfValidation_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfValidation_dict[name]['cl3d_response_c2'].mean(), dfValidation_dict[name]['cl3d_response_c2'].std(), dfValidation_dict[name]['cl3d_response_c2'].std()/dfValidation_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfValidation_dict[name]['cl3d_response_c3'].mean(), dfValidation_dict[name]['cl3d_response_c3'].std(), dfValidation_dict[name]['cl3d_response_c3'].std()/dfValidation_dict[name]['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('REALISTIC APPLICATION pT>4 DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication_dict[name]['cl3d_response'].mean(), dfRealApplication_dict[name]['cl3d_response'].std(), dfRealApplication_dict[name]['cl3d_response'].std()/dfRealApplication_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication_dict[name]['cl3d_response_c1'].mean(), dfRealApplication_dict[name]['cl3d_response_c1'].std(), dfRealApplication_dict[name]['cl3d_response_c1'].std()/dfRealApplication_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication_dict[name]['cl3d_response_c2'].mean(), dfRealApplication_dict[name]['cl3d_response_c2'].std(), dfRealApplication_dict[name]['cl3d_response_c2'].std()/dfRealApplication_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication_dict[name]['cl3d_response_c3'].mean(), dfRealApplication_dict[name]['cl3d_response_c3'].std(), dfRealApplication_dict[name]['cl3d_response_c3'].std()/dfRealApplication_dict[name]['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('REALISTIC APPLICATION pT>8 DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication1_dict[name]['cl3d_response'].mean(), dfRealApplication1_dict[name]['cl3d_response'].std(), dfRealApplication1_dict[name]['cl3d_response'].std()/dfRealApplication1_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication1_dict[name]['cl3d_response_c1'].mean(), dfRealApplication1_dict[name]['cl3d_response_c1'].std(), dfRealApplication1_dict[name]['cl3d_response_c1'].std()/dfRealApplication1_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication1_dict[name]['cl3d_response_c2'].mean(), dfRealApplication1_dict[name]['cl3d_response_c2'].std(), dfRealApplication1_dict[name]['cl3d_response_c2'].std()/dfRealApplication1_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication1_dict[name]['cl3d_response_c3'].mean(), dfRealApplication1_dict[name]['cl3d_response_c3'].std(), dfRealApplication1_dict[name]['cl3d_response_c3'].std()/dfRealApplication1_dict[name]['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('REALISTIC APPLICATION pT>12 DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication2_dict[name]['cl3d_response'].mean(), dfRealApplication2_dict[name]['cl3d_response'].std(), dfRealApplication2_dict[name]['cl3d_response'].std()/dfRealApplication2_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication2_dict[name]['cl3d_response_c1'].mean(), dfRealApplication2_dict[name]['cl3d_response_c1'].std(), dfRealApplication2_dict[name]['cl3d_response_c1'].std()/dfRealApplication2_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication2_dict[name]['cl3d_response_c2'].mean(), dfRealApplication2_dict[name]['cl3d_response_c2'].std(), dfRealApplication2_dict[name]['cl3d_response_c2'].std()/dfRealApplication2_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication2_dict[name]['cl3d_response_c3'].mean(), dfRealApplication2_dict[name]['cl3d_response_c3'].std(), dfRealApplication2_dict[name]['cl3d_response_c3'].std()/dfRealApplication2_dict[name]['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('REALISTIC APPLICATION pT>16 DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication3_dict[name]['cl3d_response'].mean(), dfRealApplication3_dict[name]['cl3d_response'].std(), dfRealApplication3_dict[name]['cl3d_response'].std()/dfRealApplication3_dict[name]['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication3_dict[name]['cl3d_response_c1'].mean(), dfRealApplication3_dict[name]['cl3d_response_c1'].std(), dfRealApplication3_dict[name]['cl3d_response_c1'].std()/dfRealApplication3_dict[name]['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication3_dict[name]['cl3d_response_c2'].mean(), dfRealApplication3_dict[name]['cl3d_response_c2'].std(), dfRealApplication3_dict[name]['cl3d_response_c2'].std()/dfRealApplication3_dict[name]['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfRealApplication3_dict[name]['cl3d_response_c3'].mean(), dfRealApplication3_dict[name]['cl3d_response_c3'].std(), dfRealApplication3_dict[name]['cl3d_response_c3'].std()/dfRealApplication3_dict[name]['cl3d_response_c3'].mean()))
                        

            sys.stdout = original_stdout # Reset the standard output to its original value


        ######################### FILL DM DATAFRAMES #########################

        dfTrainingDM0_dict[name] = dfTraining_dict[name].query('gentau_decayMode==0')
        dfTrainingDM1_dict[name] = dfTraining_dict[name].query('gentau_decayMode==1')
        dfTrainingDM10_dict[name] = dfTraining_dict[name].query('gentau_decayMode==10')
        dfTrainingDM11_dict[name] = dfTraining_dict[name].query('gentau_decayMode==11')

        dfValidationDM0_dict[name] = dfValidation_dict[name].query('gentau_decayMode==0')
        dfValidationDM1_dict[name] = dfValidation_dict[name].query('gentau_decayMode==1')
        dfValidationDM10_dict[name] = dfValidation_dict[name].query('gentau_decayMode==10')
        dfValidationDM11_dict[name] = dfValidation_dict[name].query('gentau_decayMode==11')

        ######################### PLOT REULTS #########################
        
        if args.doPlots:    
            print('\n** INFO: plotting response curves')
            ######################### TRAINING RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfTraining_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response'].mean(),3), round(dfTraining_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c1'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c2'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfTraining_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTraining_dict[name]['cl3d_response_c3'].mean(),3), round(dfTraining_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Training dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D TRAINING RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfTraining_dict[name]['cl3d_response'], np.abs(dfTraining_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfTraining_dict[name]['cl3d_response_c3'], np.abs(dfTraining_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfTraining_dict[name]['cl3d_response'], dfTraining_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfTraining_dict[name]['cl3d_response_c3'], dfTraining_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### VALIDATION RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidation_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response'].mean(),3), round(dfValidation_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfValidation_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidation_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfValidation_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c3'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], dfValidation_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c3'], dfValidation_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION C1 RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c1'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset C1 calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], dfValidation_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c1'], dfValidation_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset C1 calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION C2 RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c2'], np.abs(dfValidation_dict[name]['gentau_vis_eta']), label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset C2 calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfValidation_dict[name]['cl3d_response'], dfValidation_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfValidation_dict[name]['cl3d_response_c2'], dfValidation_dict[name]['gentau_vis_pt'], label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset C2 calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### REALISTIC APPLICATION RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfRealApplication_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfRealApplication_dict[name]['cl3d_response'].mean(),3), round(dfRealApplication_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfRealApplication_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfRealApplication_dict[name]['cl3d_response_c1'].mean(),3), round(dfRealApplication_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfRealApplication_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfRealApplication_dict[name]['cl3d_response_c2'].mean(),3), round(dfRealApplication_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfRealApplication_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfRealApplication_dict[name]['cl3d_response_c3'].mean(),3), round(dfRealApplication_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>4'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt4_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfRealApplication1_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfRealApplication1_dict[name]['cl3d_response'].mean(),3), round(dfRealApplication1_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfRealApplication1_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfRealApplication1_dict[name]['cl3d_response_c1'].mean(),3), round(dfRealApplication1_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfRealApplication1_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfRealApplication1_dict[name]['cl3d_response_c2'].mean(),3), round(dfRealApplication1_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfRealApplication1_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfRealApplication1_dict[name]['cl3d_response_c3'].mean(),3), round(dfRealApplication1_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>8'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt8_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfRealApplication2_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfRealApplication2_dict[name]['cl3d_response'].mean(),3), round(dfRealApplication2_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfRealApplication2_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfRealApplication2_dict[name]['cl3d_response_c1'].mean(),3), round(dfRealApplication2_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfRealApplication2_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfRealApplication2_dict[name]['cl3d_response_c2'].mean(),3), round(dfRealApplication2_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfRealApplication2_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfRealApplication2_dict[name]['cl3d_response_c3'].mean(),3), round(dfRealApplication2_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>12'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt12_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfRealApplication3_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfRealApplication3_dict[name]['cl3d_response'].mean(),3), round(dfRealApplication3_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfRealApplication3_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfRealApplication3_dict[name]['cl3d_response_c1'].mean(),3), round(dfRealApplication3_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfRealApplication3_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfRealApplication3_dict[name]['cl3d_response_c2'].mean(),3), round(dfRealApplication3_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfRealApplication3_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfRealApplication3_dict[name]['cl3d_response_c3'].mean(),3), round(dfRealApplication3_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>16'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt16_'+name+'_PU200.pdf')
            plt.close()

            ######################### QCD RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfQCDValidation_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfQCDValidation_dict[name]['cl3d_response'].mean(),3), round(dfQCDValidation_dict[name]['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfQCDValidation_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfQCDValidation_dict[name]['cl3d_response_c1'].mean(),3), round(dfQCDValidation_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfQCDValidation_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfQCDValidation_dict[name]['cl3d_response_c2'].mean(),3), round(dfQCDValidation_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfQCDValidation_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfQCDValidation_dict[name]['cl3d_response_c3'].mean(),3), round(dfQCDValidation_dict[name]['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('QCD calibration response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_qcd_'+name+'_PU200.pdf')
            plt.close()

            ######################### SEPARATE DMs RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response'].std(),3)), color='blue',    histtype='step', lw=2)
            #plt.hist(dfValidationDM10_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='3-prong, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response'].mean(),3), round(dfValidation_dict[name]['cl3d_response'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset raw response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 650)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c0Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c1'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            #plt.hist(dfValidationDM10_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='3-prong, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c1'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c1'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation  dataset C1 response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 650)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c1Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c2'].std(),3)), color='blue',    histtype='step', lw=2)
            #plt.hist(dfValidationDM10_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='3-prong, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c2'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset C2 response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 650)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c2Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c3'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c3'].std(),3)), color='blue',    histtype='step', lw=2)
            #plt.hist(dfValidationDM10_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='3-prong, mean: {0}, RMS: {1}'.format(round(dfValidation_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidation_dict[name]['cl3d_response_c3'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c3'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset C3 response \n Training pT>{0}'.format(args.ptcut))
            plt.ylim(0, 650)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c3Response_validation_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D REALISTIC RESPONSE-ETA #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication_dict[name]['cl3d_response'], np.abs(dfRealApplication_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication_dict[name]['cl3d_response_c3'], np.abs(dfRealApplication_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>4'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt4_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response'], np.abs(dfRealApplication1_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response_c3'], np.abs(dfRealApplication1_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>8'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt8_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response'], np.abs(dfRealApplication2_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response_c3'], np.abs(dfRealApplication2_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>12'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt12_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response'], np.abs(dfRealApplication3_dict[name]['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response_c3'], np.abs(dfRealApplication3_dict[name]['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>16'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt16_2Deta_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D REALISTIC RESPONSE-pT #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication_dict[name]['cl3d_response'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication_dict[name]['cl3d_response_c3'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>4'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt4_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response_c3'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>8'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt8_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response_c3'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>12'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt12_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response_c3'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application calibration response \n Training pT>{0} - Application pT>16'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_pt16_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            '''
            ######################### 2D REALISTIC C1 RESPONSE-pT #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication_dict[name]['cl3d_response'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication_dict[name]['cl3d_response_c1'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C1 calibration response \n Training pT>{0} - Application pT>4'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_pt4_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response_c1'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C1 calibration response \n Training pT>{0} - Application pT>8'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_pt8_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response_c1'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C1 calibration response \n Training pT>{0} - Application pT>12'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_pt12_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response_c1'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C1 calibration response \n Training pT>{0} - Application pT>16'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_pt16_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D REALISTIC C2 RESPONSE-pT #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication_dict[name]['cl3d_response'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication_dict[name]['cl3d_response_c2'], dfRealApplication_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C2 calibration response \n Training pT>{0} - Application pT>4'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_pt4_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication1_dict[name]['cl3d_response_c2'], dfRealApplication1_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C2 calibration response \n Training pT>{0} - Application pT>8'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_pt8_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication2_dict[name]['cl3d_response_c2'], dfRealApplication2_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C2 calibration response \n Training pT>{0} - Application pT>12'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_pt12_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfRealApplication3_dict[name]['cl3d_response_c2'], dfRealApplication3_dict[name]['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Realistic application C2 calibration response \n Training pT>{0} - Application pT>16'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_pt16_2Dpt_'+name+'_PU200.pdf')
            plt.close()
            '''
        
        ######################### RE-MERGE AND SAVE FILES #########################

        dfTraining_dict[name] = pd.concat([dfTraining_dict[name],dfQCDTraining_dict[name],dfNuTraining_dict[name]],sort=False)
        dfValidation_dict[name] = pd.concat([dfValidation_dict[name],dfQCDValidation_dict[name],dfNuValidation_dict[name]],sort=False)

        store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
        store_tr[name] = dfTraining_dict[name]
        store_tr.close()

        store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
        store_val[name] = dfValidation_dict[name]
        store_val.close()

        
        print('\n** INFO: finished energy calibration for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')




###########

# Model c3
'''
matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(8,8))
pt = np.arange(25, 100, 1)

logpt1 = np.log(pt)
logpt2 = logpt1**2
logpt3 = logpt1**3
logpt4 = logpt1**4

for name in dfTau_dict:

  ptmeansTrain_dict[name]['logpt1'] = np.log(ptmeansTrain_dict[name]['cl3d_pt_c2'])
  ptmeansTrain_dict[name]['logpt2'] = ptmeansTrain_dict[name].logpt1**2
  ptmeansTrain_dict[name]['logpt3'] = ptmeansTrain_dict[name].logpt1**3
  ptmeansTrain_dict[name]['logpt4'] = ptmeansTrain_dict[name].logpt1**4

for name in dfTau_dict:

  plt.plot(ptmeansTrain_dict[name]['cl3d_pt_c2'], ptmeansTrain_dict[name]['cl3d_response_c2'], marker='s', markersize=9, ls='None', color=colors_dict[name], label='Observed')
  plt.plot(pt, C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T), ls='--', color=colors_dict[name], label='Predicted')

plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\langle E_{T,calib2}^{L1}\rangle\, [GeV]$')
plt.ylabel(r'$\langle E_{T,calib2}^{L1}/p_{T}^{gen}\rangle$')
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
t.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.savefig(plotdir+'calib_modelc3_TDRv2.png')
plt.savefig(plotdir+'calib_modelc3_TDRv2.pdf')
#plt.show()
'''

###########

# Responses and resolutions
'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = etameansTrain_dict[name]
  plt.plot(df['gentau_vis_abseta'], etaeffrmssTrain_dict[name]['cl3d_response']/df['cl3d_response'], label="Raw",color='red',lw=2)
  plt.plot(df['gentau_vis_abseta'], etaeffrmssTrain_dict[name]['cl3d_response_c3']/df['cl3d_response_c3'], label="Calibrated",color='blue',lw=2)
  y_array_raw = (etaeffrmssTrain_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (etaeffrmssTrain_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.errorbar(x_array,y_array_raw,xerr=2.5,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=2.5,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
plt.xlabel(r'$|\eta|$')
plt.ylabel(r'$RMS_{eff}(E_{T}^{L1}/p_{T}^{gen})$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig(plotdir+'calib_mean_vs_eta.png')
plt.savefig(plotdir+'calib_mean_vs_eta.pdf')
'''

'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = etameansTrain_dict[name]
  axs[0].plot(df['gentau_vis_abseta'], etarmssTrain_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = etarmssTrain_dict[name]['cl3d_response_c3']/etarmssTrain_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_abseta'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(1.6, 3.0)
axs[0].set_ylim(0.10,0.35)
axs[0].grid()
#axs[0].set_xlabel(r'$|\eta|$')
axs[0].set_ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(1.6, 3.0)
axs[1].set_ylim(0.95,1.20)
axs[1].set_xlabel(r'$|\eta|$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_rms_vs_eta_ratio.png')
plt.savefig(plotdir+'calib_rms_vs_eta_ratio.pdf')
'''

'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = etameansTrain_dict[name]
  axs[0].plot(df['gentau_vis_abseta'], etaeffrmssTrain_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = etaeffrmssTrain_dict[name]['cl3d_response_c3']/etaeffrmssTrain_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_abseta'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(1.6, 3.0)
axs[0].set_ylim(0.10,0.30)
axs[0].grid()
#axs[0].set_xlabel(r'$|\eta|$')
axs[0].set_ylabel(r'$RMS_{eff}(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(1.6, 3.0)
axs[1].set_ylim(0.95,1.25)
axs[1].set_xlabel(r'$|\eta|$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_effrms_vs_eta_ratio.png')
plt.savefig(plotdir+'calib_effrms_vs_eta_ratio.pdf')
'''

'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = ptmeansTrain_dict[name]
  plt.plot(df['gentau_vis_pt'], df['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)

plt.ylim(0.9, 1.1)
plt.xlabel(r'$p_{T}\, [GeV]$', fontsize=18)
plt.ylabel(r'$\langle p_{T}^{L1}/p_{T}^{gen}\rangle$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.ylim(0.95, 1.05)
plt.savefig(plotdir+'calib_mean_vs_pt.png')
plt.savefig(plotdir+'calib_mean_vs_pt.pdf')
'''

'''
plt.figure(figsize=(15,10))

for name in dfTau_dict:

  df = ptmeansTrain_dict[name]
  plt.plot(df['gentau_vis_pt'], ptrmssTrain_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)

plt.ylim(0., 0.3)
plt.xlabel(r'$p_{T}\, [GeV]$')
plt.ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
plt.grid()
plt.legend(loc = 'upper right', fontsize=16)
plt.ylim(0.0, 0.4)
plt.savefig(plotdir+'calib_rms_vs_pt.png')
plt.savefig(plotdir+'calib_rms_vs_pt.pdf')
'''
'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = ptmeansTrain_dict[name]
  axs[0].plot(df['gentau_vis_pt'], ptrmssTrain_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = ptrmssTrain_dict[name]['cl3d_response_c3']/ptrmssTrain_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_pt'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(20., 100.)
axs[0].set_ylim(0.0, 0.4)
axs[0].grid()
#axs[0].set_xlabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[0].set_ylabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(20., 100.)
axs[1].set_ylim(0.95,1.40)
axs[1].set_xlabel(r'$p_{T}\, [GeV]$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_rms_vs_pt_ratio.png')
plt.savefig(plotdir+'calib_rms_vs_pt_ratio.pdf')
'''
'''
plt.figure(figsize=(8,9))

for name in dfTau_dict:

  df = ptmeansTrain_dict[name]
  plt.plot(df['gentau_vis_pt'], pteffrmssTrain_dict[name]['cl3d_response']/df['cl3d_response'], label="Raw",color='red',lw=2)
  plt.plot(df['gentau_vis_pt'], pteffrmssTrain_dict[name]['cl3d_response_c3']/df['cl3d_response_c3'], label="Calibrated",color='blue',lw=2)

plt.ylim(0.01, 0.45)
plt.legend(loc = 'lower left')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$RMS_{eff}\ /\ <p_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'calib_effrms_vs_pt_TDR.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_TDR.pdf')
plt.show()
'''

'''
x_array = [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5]

for name in dfTau_dict:
  df = ptmeansTrain_dict[name]
  y_array_raw = (pteffrmssTrain_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (pteffrmssTrain_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.figure(figsize=(8,8))
plt.errorbar(x_array,y_array_raw,xerr=2.5,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=2.5,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
plt.ylim(0.01, 0.45)
uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Raw',lw=2)
cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calibrated',lw=2)
plt.legend(loc = 'lower left',handles=[uncal_line,cal1_line])
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$RMS_{eff}\ /\ <E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'calib_effrms_vs_pt_TDRv1.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_TDRv1.pdf')
plt.show()
'''
'''
x_array = [1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85]

for name in dfTau_dict:
  df = etameansTrain_dict[name]
  y_array_raw = (etaeffrmssTrain_dict[name]['cl3d_response']/df['cl3d_response']).values
  y_array_calib = (etaeffrmssTrain_dict[name]['cl3d_response_c3']/df['cl3d_response_c3']).values

plt.figure(figsize=(8,8))
plt.errorbar(x_array,y_array_raw,xerr=0.05,yerr=None,marker="o",mec='red',ls='None',label='Raw',color='red',lw=2)
plt.errorbar(x_array,y_array_calib,xerr=0.05,yerr=None,marker="o",mec='blue',ls='None',label='Calibrated',color='blue',lw=2)
uncal_line = mlines.Line2D([], [], color='red',markersize=15, label='Raw',lw=2)
cal1_line = mlines.Line2D([], [], color='blue',markersize=15, label='Calibrated',lw=2)
plt.legend(loc = 'lower left',handles=[uncal_line,cal1_line])
plt.xlabel(r'$|\eta^{gen,\tau}|$')
plt.ylabel(r'$RMS_{eff}\ /\ <E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}>$')
plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(70,0.32, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.savefig(plotdir+'calib_effrms_vs_eta_TDRv1.png')
plt.savefig(plotdir+'calib_effrms_vs_eta_TDRv1.pdf')
plt.show()
'''
'''
matplotlib.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(2, 1, figsize=(15,15))

for name in dfTau_dict:

  df = ptmeansTrain_dict[name]
  axs[0].plot(df['gentau_vis_pt'], pteffrmssTrain_dict[name]['cl3d_response_c3'],color=colors_dict[name], label=legends_dict[name],lw=2)
  ratio = pteffrmssTrain_dict[name]['cl3d_response_c3']/pteffrmssTrain_dict[0]['cl3d_response_c3']
  axs[1].plot(df['gentau_vis_pt'], ratio, color=colors_dict[name], label=legends_dict[name],lw=2)

axs[0].legend(loc = 'upper right', fontsize=18)
axs[0].set_xlim(20., 100.)
axs[0].set_ylim(0., 0.35)
axs[0].grid()
#axs[0].set_xlabel(r'$RMS(p_{T}^{L1}/p_{T}^{gen})$')
axs[0].set_ylabel(r'$RMS_{eff}(p_{T}^{L1}/p_{T}^{gen})$')
axs[1].set_xlim(20., 100.)
axs[1].set_ylim(0.95,1.40)
axs[1].set_xlabel(r'$p_{T}\, [GeV]$')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'calib_effrms_vs_pt_ratio.png')
plt.savefig(plotdir+'calib_effrms_vs_pt_ratio.pdf')
'''
'''
plt.figure(figsize=(8,7))

for name in dfTau_dict:
  plt.hist(dfTau_dict[name]['gentau_vis_eta']-dfTau_dict[name]['cl3d_eta'], bins=np.arange(-0.15, 0.15, 0.005), color='blue',lw=2,histtype='step')

#plt.ylim(0.01, 0.4)
plt.xlabel(r'$\eta_{gen,\tau}$ - $\eta_{L1,\tau}$')
plt.ylabel(r'a. u. ')
#plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
plt.grid()
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(-0.13,1400, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#txt2 = ('Mean: 0.00, RMS: 0.02')
#t2 = plt.text(0.025,1700, txt2, ha='left', wrap=True, fontsize=16)
#t2.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'res_eta_TDRv1.png')
plt.savefig(plotdir+'res_eta_TDRv1.pdf')

plt.figure(figsize=(8,7))

for name in dfTau_dict:
  plt.hist(dfTau_dict[name]['gentau_vis_phi']-dfTau_dict[name]['cl3d_phi'], bins=np.arange(-0.15, 0.15, 0.005), color='blue',lw=2,histtype='step')

#plt.ylim(0.01, 0.4)
plt.xlabel(r'$\phi_{gen,\tau}$ - $\phi_{L1,\tau}$')
plt.ylabel(r'a. u. ')
#plt.grid()
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.14)
plt.grid()
#txt = (r'1.6 < | $\eta_{gen,\tau}$ | < 2.9')
#t = plt.text(-0.13,1400, txt, ha='left', wrap=True, fontsize=16)
#t.set_bbox(dict(facecolor='white', edgecolor='white'))
#txt2 = ('Mean: 0.00, RMS: 0.02')
#t2 = plt.text(0.025,1700, txt2, ha='left', wrap=True, fontsize=16)
#t2.set_bbox(dict(facecolor='white', edgecolor='white'))
#plt.ylim(0.0, 0.5)
plt.savefig(plotdir+'res_phi_TDRv1.png')
plt.savefig(plotdir+'res_phi_TDRv1.pdf')
'''
