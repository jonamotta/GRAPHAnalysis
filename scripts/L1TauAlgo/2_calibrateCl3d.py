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
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated_C1fullC2C3'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/calibration_C1fullC2C3'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration_C1fullC2C3'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir+'; mkdir -p '+model_outdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_matched.hdf5',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_matched.hdf5',
        'mixed'        : indir+'/'
    }

    outFileTraining_dict = {
        'threshold'    : outdir+'/Training_PU200_th_calibrated.hdf5',
        'mixed'        : outdir+'/'
    }

    outFileValidation_dict = {
        'threshold'    : outdir+'/Validation_PU200_th_calibrated.hdf5',
        'mixed'        : outdir+'/'
    }

    outFile_modelC1_dict = {
        'threshold'    : model_outdir+'/model_c1_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC2_dict = {
        'threshold'    : model_outdir+'/model_c2_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    outFile_modelC3_dict = {
        'threshold'    : model_outdir+'/model_c3_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}
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

    dfTrainingDM0_dict = {}
    dfTrainingDM1_dict = {}
    dfTrainingDM10_dict = {}
    dfTrainingDM11_dict = {}
    dfTrainingPU_dict = {}
    dfTrainingQCD_dict = {}
    dfValidationDM0_dict = {}
    dfValidationDM1_dict = {}
    dfValidationDM10_dict = {}
    dfValidationDM11_dict = {}
    dfValidationPU_dict = {}
    dfValidationQCD_dict = {}
    
    # minimum pt and eta requirements 
    ptmin = 20
    etamin = 1.6

    # features used for the C2 calibration step - FULL AVAILABLE
    features = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

    # features used for the C3 calibration step
    vars = ['gentau_vis_pt', 'gentau_vis_bin_pt', 'cl3d_pt_c2', 'cl3d_response_c2']

    # variables to plot
    plot_var = ['gentau_vis_pt', 'gentau_vis_abseta', 'gentau_vis_bin_eta', 'gentau_vis_bin_pt', 'cl3d_pt', 'cl3d_response', 'cl3d_abseta', 'cl3d_pt_c1', 'cl3d_response_c1', 'cl3d_pt_c2', 'cl3d_response_c2', 'cl3d_pt_c3', 'cl3d_response_c3']

    # colors to use
    colors_dict = {
        'threshold'    : 'blue',
        'mixed'        : 'fuchsia'
    }

    # legend to use
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'mixed'        : 'Mixed BC + STC'
    }


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

        # here we apply minimal requirements so that we are working in a fiducial region slightly smaller than the full HGCAL acceptance
        # we apply the OR between tau, jet, and cluster requirements
        dfTraining_dict[name].query('(cl3d_pt>{0} and cl3d_abseta>1.6 and cl3d_abseta<2.9) or ((gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9))) or (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9))))'.format(args.ptcut), inplace=True)
        dfValidation_dict[name].query('(cl3d_pt>{0} and cl3d_abseta>1.6 and cl3d_abseta<2.9) or ((gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9))) or (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9))))'.format(args.ptcut), inplace=True)

        # for the actual training and validation we enforce the AND between jet and cluster requirements
        dfQCDTr = dfTraining_dict[name].query('gentau_decayMode==-2 and cl3d_isbestmatch==True and (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9)))').copy(deep=True)
        dfQCDVal = dfValidation_dict[name].query('gentau_decayMode==-2 and cl3d_isbestmatch==True and (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9)))').copy(deep=True)

        # for the actual training and validation we enforce the AND between tau and cluster requirements
        dfTr = dfTraining_dict[name].query('gentau_decayMode>=0 and cl3d_isbestmatch==True and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))').copy(deep=True)
        dfVal = dfValidation_dict[name].query('gentau_decayMode>=0 and cl3d_isbestmatch==True and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))').copy(deep=True)

        # calculate responses
        dfTr['cl3d_response'] = dfTr['cl3d_pt']/dfTr['gentau_vis_pt']
        dfVal['cl3d_response'] = dfVal['cl3d_pt']/dfVal['gentau_vis_pt']


        ######################### C1 CALIBRATION TRAINING (PU eta dependent calibration) #########################

        print('\n** INFO: training calibration C1')

        input_c1 = dfTr[['cl3d_abseta']]
        target_c1 = dfTr.gentau_vis_pt - dfTr.cl3d_pt
        C1model_dict[name] = LinearRegression().fit(input_c1, target_c1)

        save_obj(C1model_dict[name], outFile_modelC1_dict[name])

        dfTr['cl3d_c1'] = C1model_dict[name].predict(dfTr[['cl3d_abseta']])
        dfTr['cl3d_pt_c1'] = dfTr.cl3d_c1 + dfTr.cl3d_pt
        dfTr['cl3d_response_c1'] = dfTr.cl3d_pt_c1 / dfTr.gentau_vis_pt


        ######################### C2 CALIBRATION TRAINING (DM dependent calibration) #########################

        print('\n** INFO: training calibration C2')

        input_c2 = dfTr[features]
        target_c2 = dfTr.gentau_vis_pt / dfTr.cl3d_pt_c1
        C2model_dict[name] = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=0, loss='huber').fit(input_c2, target_c2)

        save_obj(C2model_dict[name], outFile_modelC2_dict[name])

        dfTr['cl3d_c2'] = C2model_dict[name].predict(dfTr[features])
        dfTr['cl3d_pt_c2'] = dfTr.cl3d_c2 * dfTr.cl3d_pt_c1
        dfTr['cl3d_response_c2'] = dfTr.cl3d_pt_c2 / dfTr.gentau_vis_pt

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

        dfTr['gentau_vis_abseta'] = np.abs(dfTr['gentau_vis_eta'])
        dfTr['gentau_vis_bin_eta'] = ((dfTr['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTr['gentau_vis_bin_pt']  = ((dfTr['gentau_vis_pt'] - ptmin)/5).astype('int32')

        meansTrainPt_dict[name] = dfTr[vars].groupby('gentau_vis_bin_pt').mean() 
        rmssTrainPt_dict[name] = dfTr[vars].groupby('gentau_vis_bin_pt').std() 

        meansTrainPt_dict[name]['logpt1'] = np.log(meansTrainPt_dict[name]['cl3d_pt_c2'])
        meansTrainPt_dict[name]['logpt2'] = meansTrainPt_dict[name].logpt1**2
        meansTrainPt_dict[name]['logpt3'] = meansTrainPt_dict[name].logpt1**3
        meansTrainPt_dict[name]['logpt4'] = meansTrainPt_dict[name].logpt1**4

        input_c3 = meansTrainPt_dict[name][['logpt1', 'logpt2', 'logpt3', 'logpt4']]
        target_c3 = meansTrainPt_dict[name]['cl3d_response_c2']
        C3model_dict[name] = LinearRegression().fit(input_c3, target_c3)

        save_obj(C3model_dict[name], outFile_modelC3_dict[name])

        logpt1 = np.log(abs(dfTr['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4

        dfTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr.cl3d_pt_c2 / dfTr.cl3d_c3
        dfTr['cl3d_response_c3'] = dfTr.cl3d_pt_c3 / dfTr.gentau_vis_pt


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

        # VALIDATION DATASET
        # application calibration 1
        dfVal['cl3d_c1'] = C1model_dict[name].predict(dfVal[['cl3d_abseta']])
        dfVal['cl3d_pt_c1'] = dfVal.cl3d_c1 + dfVal.cl3d_pt
        dfVal['cl3d_response_c1'] = dfVal.cl3d_pt_c1 / dfVal.gentau_vis_pt
        # application calibration 2
        dfVal['cl3d_c2'] = C2model_dict[name].predict(dfVal[features])
        dfVal['cl3d_pt_c2'] = dfVal.cl3d_c2 * dfVal.cl3d_pt_c1
        dfVal['cl3d_response_c2'] = dfVal.cl3d_pt_c2 / dfVal.gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfVal['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfVal['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfVal['cl3d_pt_c3'] = dfVal.cl3d_pt_c2 / dfVal.cl3d_c3
        dfVal['cl3d_response_c3'] = dfVal.cl3d_pt_c3 / dfVal.gentau_vis_pt

        # TRAINING DATASET
        # application calibration 1
        dfTr['cl3d_c1'] = C1model_dict[name].predict(dfTr[['cl3d_abseta']])
        dfTr['cl3d_pt_c1'] = dfTr.cl3d_c1 + dfTr.cl3d_pt
        dfTr['cl3d_response_c1'] = dfTr.cl3d_pt_c1 / dfTr.gentau_vis_pt
        # application calibration 2
        dfTr['cl3d_c2'] = C2model_dict[name].predict(dfTr[features])
        dfTr['cl3d_pt_c2'] = dfTr.cl3d_c2 * dfTr.cl3d_pt_c1
        dfTr['cl3d_response_c2'] = dfTr.cl3d_pt_c2 / dfTr.gentau_vis_pt
        # application calibration 3
        logpt1 = np.log(abs(dfTr['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr.cl3d_pt_c2 / dfTr.cl3d_c3
        dfTr['cl3d_response_c3'] = dfTr.cl3d_pt_c3 / dfTr.gentau_vis_pt

        # QCD VALIDATION DATASET
        dfQCDVal['cl3d_response'] = dfQCDVal['cl3d_pt']/dfQCDVal['genjet_pt']
        # application calibration 1
        dfQCDVal['cl3d_c1'] = C1model_dict[name].predict(dfQCDVal[['cl3d_abseta']])
        dfQCDVal['cl3d_pt_c1'] = dfQCDVal.cl3d_c1 + dfQCDVal.cl3d_pt
        dfQCDVal['cl3d_response_c1'] = dfQCDVal.cl3d_pt_c1 / dfQCDVal.genjet_pt
        # application calibration 2
        dfQCDVal['cl3d_c2'] = C2model_dict[name].predict(dfQCDVal[features])
        dfQCDVal['cl3d_pt_c2'] = dfQCDVal.cl3d_c2 * dfQCDVal.cl3d_pt_c1
        dfQCDVal['cl3d_response_c2'] = dfQCDVal.cl3d_pt_c2 / dfQCDVal.genjet_pt
        # application calibration 3
        logpt1 = np.log(abs(dfQCDVal['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDVal['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDVal['cl3d_pt_c3'] = dfQCDVal.cl3d_pt_c2 / dfQCDVal.cl3d_c3
        dfQCDVal['cl3d_response_c3'] = dfQCDVal.cl3d_pt_c3 / dfQCDVal.genjet_pt

        # QCD TRAINING DATASET
        dfQCDTr['cl3d_response'] = dfQCDTr['cl3d_pt']/dfQCDTr['genjet_pt']
        # application calibration 1
        dfQCDTr['cl3d_c1'] = C1model_dict[name].predict(dfQCDTr[['cl3d_abseta']])
        dfQCDTr['cl3d_pt_c1'] = dfQCDTr.cl3d_c1 + dfQCDTr.cl3d_pt
        dfQCDTr['cl3d_response_c1'] = dfQCDTr.cl3d_pt_c1 / dfQCDTr.genjet_pt
        # application calibration 2
        dfQCDTr['cl3d_c2'] = C2model_dict[name].predict(dfQCDTr[features])
        dfQCDTr['cl3d_pt_c2'] = dfQCDTr.cl3d_c2 * dfQCDTr.cl3d_pt_c1
        dfQCDTr['cl3d_response_c2'] = dfQCDTr.cl3d_pt_c2 / dfQCDTr.genjet_pt
        # application calibration 3
        logpt1 = np.log(abs(dfQCDTr['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDTr['cl3d_pt_c3'] = dfQCDTr.cl3d_pt_c2 / dfQCDTr.cl3d_c3
        dfQCDTr['cl3d_response_c3'] = dfQCDTr.cl3d_pt_c3 / dfQCDTr.genjet_pt


        ######################### EVALUATE MEAN AND RMS OF RESPONSE AND SAVE VALUES #########################

        print('\n** INFO: calculation of response MEAN and RMS')

        dfTr['gentau_vis_bin_eta'] = ((dfTr['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTr['gentau_vis_bin_pt']  = ((dfTr['gentau_vis_pt'] - ptmin)/5).astype('int32')

        etameansTrain_dict[name]   = dfTr[plot_var].groupby('gentau_vis_bin_eta').mean()
        etarmssTrain_dict[name]    = dfTr[plot_var].groupby('gentau_vis_bin_eta').std()
        etaeffrmssTrain_dict[name] = dfTr[plot_var].groupby('gentau_vis_bin_eta').apply(effrms)
        ptmeansTrain_dict[name]    = dfTr[plot_var].groupby('gentau_vis_bin_pt').mean()
        ptrmssTrain_dict[name]     = dfTr[plot_var].groupby('gentau_vis_bin_pt').std()
        pteffrmssTrain_dict[name]  = dfTr[plot_var].groupby('gentau_vis_bin_pt').apply(effrms)

        dfVal['gentau_vis_abseta'] = np.abs(dfVal['gentau_vis_eta'])
        dfVal['gentau_vis_bin_eta'] = ((dfVal['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfVal['gentau_vis_bin_pt']  = ((dfVal['gentau_vis_pt'] - ptmin)/5).astype('int32')

        etameansValid_dict[name]   = dfVal[plot_var].groupby('gentau_vis_bin_eta').mean()
        etarmssValid_dict[name]    = dfVal[plot_var].groupby('gentau_vis_bin_eta').std()
        etaeffrmssValid_dict[name] = dfVal[plot_var].groupby('gentau_vis_bin_eta').apply(effrms)
        ptmeansValid_dict[name]    = dfVal[plot_var].groupby('gentau_vis_bin_pt').mean()
        ptrmssValid_dict[name]     = dfVal[plot_var].groupby('gentau_vis_bin_pt').std()
        pteffrmssValid_dict[name]  = dfVal[plot_var].groupby('gentau_vis_bin_pt').apply(effrms)

        original_stdout = sys.stdout # Save a reference to the original standard output
        # save response to file
        with open(plotdir+'/responses.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print('TRAINING DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfTr['cl3d_response'].mean(), dfTr['cl3d_response'].std(), dfTr['cl3d_response'].std()/dfTr['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfTr['cl3d_response_c1'].mean(), dfTr['cl3d_response_c1'].std(), dfTr['cl3d_response_c1'].std()/dfTr['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfTr['cl3d_response_c2'].mean(), dfTr['cl3d_response_c2'].std(), dfTr['cl3d_response_c2'].std()/dfTr['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfTr['cl3d_response_c3'].mean(), dfTr['cl3d_response_c3'].std(), dfTr['cl3d_response_c3'].std()/dfTr['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('VALIDATION DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response'].mean(), dfVal['cl3d_response'].std(), dfVal['cl3d_response'].std()/dfVal['cl3d_response'].mean()))
            print('  -- CALIBRATED 1: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response_c1'].mean(), dfVal['cl3d_response_c1'].std(), dfVal['cl3d_response_c1'].std()/dfVal['cl3d_response_c1'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response_c2'].mean(), dfVal['cl3d_response_c2'].std(), dfVal['cl3d_response_c2'].std()/dfVal['cl3d_response_c2'].mean()))
            print('  -- CALIBRATED 3: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response_c3'].mean(), dfVal['cl3d_response_c3'].std(), dfVal['cl3d_response_c3'].std()/dfVal['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')

            sys.stdout = original_stdout # Reset the standard output to its original value


        ######################### FILL DM DATAFRAMES #########################

        dfTrainingDM0_dict[name] = dfTr.query('gentau_decayMode==0')
        dfTrainingDM1_dict[name] = dfTr.query('gentau_decayMode==1')
        dfTrainingDM10_dict[name] = dfTr.query('gentau_decayMode==10')
        dfTrainingDM11_dict[name] = dfTr.query('gentau_decayMode==11')
        dfTrainingPU_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-1')
        dfTrainingQCD_dict[name] = dfTraining_dict[name].query('gentau_decayMode==-2')

        dfValidationDM0_dict[name] = dfVal.query('gentau_decayMode==0')
        dfValidationDM1_dict[name] = dfVal.query('gentau_decayMode==1')
        dfValidationDM10_dict[name] = dfVal.query('gentau_decayMode==10')
        dfValidationDM11_dict[name] = dfVal.query('gentau_decayMode==11')
        dfValidationPU_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-1')
        dfValidationQCD_dict[name] = dfValidation_dict[name].query('gentau_decayMode==-2')

        ######################### PLOT REULTS #########################
        
        if args.doPlots:    
            print('\n** INFO: plotting response curves')
            ######################### TRAINING RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfTr['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfTr['cl3d_response'].mean(),3), round(dfTr['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfTr['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfTr['cl3d_response_c1'].mean(),3), round(dfTr['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfTr['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfTr['cl3d_response_c2'].mean(),3), round(dfTr['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfTr['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTr['cl3d_response_c3'].mean(),3), round(dfTr['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Training dataset calibration response'.format(args.ptcut))
            plt.ylim(0,1750)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D TRAINING RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfTr['cl3d_response'], np.abs(dfTr['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfTr['cl3d_response_c3'], np.abs(dfTr['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfTr['cl3d_response'], dfTr['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfTr['cl3d_response_c3'], dfTr['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_training_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### VALIDATION RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfVal['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfVal['cl3d_response'].mean(),3), round(dfVal['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfVal['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfVal['cl3d_response_c1'].mean(),3), round(dfVal['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfVal['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfVal['cl3d_response_c2'].mean(),3), round(dfVal['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfVal['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfVal['cl3d_response_c3'].mean(),3), round(dfVal['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset calibration response'.format(args.ptcut))
            plt.ylim(0, 800)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c3'], np.abs(dfVal['gentau_vis_eta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], dfVal['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c3'], dfVal['gentau_vis_pt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION C1 RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c1'], np.abs(dfVal['gentau_vis_eta']), label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset C1 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], dfVal['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c1'], dfVal['gentau_vis_pt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset C1 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC1_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### 2D VALIDATION C2 RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c2'], np.abs(dfVal['gentau_vis_eta']), label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset C2 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], dfVal['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c2'], dfVal['gentau_vis_pt'], label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset C2 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC2_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### QCD RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfQCDVal['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response'].mean(),3), round(dfQCDVal['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfQCDVal['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label='Calib. 1, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response_c1'].mean(),3), round(dfQCDVal['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfQCDVal['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label='Calib. 2, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response_c2'].mean(),3), round(dfQCDVal['cl3d_response_c2'].std(),3)), color='limegreen',  histtype='step', lw=2)
            plt.hist(dfQCDVal['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response_c3'].mean(),3), round(dfQCDVal['cl3d_response_c3'].std(),3)), color='black',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('QCD calibration response'.format(args.ptcut))
            plt.ylim(0, 1250)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponse_qcd_'+name+'_PU200.pdf')
            plt.close()

            ######################### SEPARATE DMs RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidationDM10_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'3-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM10_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM10_dict[name]['cl3d_response'].std(),3)), color='red',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset raw response'.format(args.ptcut))
            plt.ylim(0, 450)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c0Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c1'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c1'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidationDM10_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'3-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM10_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM10_dict[name]['cl3d_response_c1'].std(),3)), color='red',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c1'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c1'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c1'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation  dataset C1 response'.format(args.ptcut))
            plt.ylim(0, 450)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c1Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c2'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c2'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidationDM10_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'3-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM10_dict[name]['cl3d_response_c2'].std(),3)), color='red',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c2'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c2'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c2'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset C2 response'.format(args.ptcut))
            plt.ylim(0, 450)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c2Response_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c3'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c3'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidationDM10_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'3-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM10_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM10_dict[name]['cl3d_response_c3'].std(),3)), color='red',  histtype='step', lw=2)
            plt.hist(dfValidationDM11_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'3-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM11_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM11_dict[name]['cl3d_response_c3'].std(),3)), color='fuchsia',   histtype='step', lw=2)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset C3 response'.format(args.ptcut))
            plt.ylim(0, 450)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c3Response_validation_'+name+'_PU200.pdf')
            plt.close()

            ######################### C1 DISTRIBUTIONS #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM1_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM10_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM11_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationPU_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationQCD_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C1 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c1_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfTrainingDM0_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM1_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM10_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM11_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingPU_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingQCD_dict[name]['cl3d_c1'], bins=np.arange(0.5, 5., 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C1 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Training dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c1_training_'+name+'_PU200.pdf')
            plt.close()

            ######################### C2 DISTRIBUTIONS #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM1_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM10_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM11_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationPU_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationQCD_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C2 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c2_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfTrainingDM0_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM1_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM10_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM11_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingPU_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingQCD_dict[name]['cl3d_c2'], bins=np.arange(0.5, 2.5, 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C2 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Training dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c2_training_'+name+'_PU200.pdf')
            plt.close()

            ######################### C3 DISTRIBUTIONS #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM1_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM10_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM11_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationPU_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationQCD_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C3 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c3_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfTrainingDM0_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM1_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM10_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM11_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingPU_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingQCD_dict[name]['cl3d_c3'], bins=np.arange(0.75, 2., 0.05),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C3 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Training dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c3_training_'+name+'_PU200.pdf')
            plt.close()

        
        ######################### RE-MERGE AND SAVE FILES #########################

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
