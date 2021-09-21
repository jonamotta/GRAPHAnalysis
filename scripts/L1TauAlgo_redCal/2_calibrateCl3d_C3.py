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
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/calibrated_C3'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/calibration_C3'
    model_outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration_C3'
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
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFile_modelC3_dict = {
        'threshold'    : model_outdir+'/model_c3_th_PU200.pkl',
        'mixed'        : model_outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}
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

    # features used for the C3 calibration step
    vars = ['gentau_vis_pt', 'gentau_vis_bin_pt', 'cl3d_pt', 'cl3d_response']

    # variables to plot
    plot_var = ['gentau_vis_pt', 'gentau_vis_abseta', 'gentau_vis_bin_eta', 'gentau_vis_bin_pt', 'cl3d_pt', 'cl3d_response', 'cl3d_abseta', 'cl3d_pt_c3', 'cl3d_response_c3']

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


        ######################### C3 CALIBRATION TRAINING (E dependent calibration) #########################

        print('\n** INFO: training calibration C3')

        dfTr['gentau_vis_abseta'] = np.abs(dfTr['gentau_vis_eta'])
        dfTr['gentau_vis_bin_eta'] = ((dfTr['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTr['gentau_vis_bin_pt']  = ((dfTr['gentau_vis_pt'] - ptmin)/5).astype('int32')

        meansTrainPt_dict[name] = dfTr[vars].groupby('gentau_vis_bin_pt').mean() 
        rmssTrainPt_dict[name] = dfTr[vars].groupby('gentau_vis_bin_pt').std() 

        meansTrainPt_dict[name]['logpt1'] = np.log(meansTrainPt_dict[name]['cl3d_pt'])
        meansTrainPt_dict[name]['logpt2'] = meansTrainPt_dict[name].logpt1**2
        meansTrainPt_dict[name]['logpt3'] = meansTrainPt_dict[name].logpt1**3
        meansTrainPt_dict[name]['logpt4'] = meansTrainPt_dict[name].logpt1**4

        input_c3 = meansTrainPt_dict[name][['logpt1', 'logpt2', 'logpt3', 'logpt4']]
        target_c3 = meansTrainPt_dict[name]['cl3d_response']
        C3model_dict[name] = LinearRegression().fit(input_c3, target_c3)

        save_obj(C3model_dict[name], outFile_modelC3_dict[name])

        logpt1 = np.log(abs(dfTr['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4

        dfTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr.cl3d_pt / dfTr.cl3d_c3
        dfTr['cl3d_response_c3'] = dfTr.cl3d_pt_c3 / dfTr.gentau_vis_pt



        ######################### C3 CALIBRATION APPLICATION #########################

        print('\n** INFO: applying calibration C3')
        
        # VALIDATION DATASET
        # application calibration 3
        logpt1 = np.log(abs(dfValidation_dict[name]['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfValidation_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfValidation_dict[name]['cl3d_pt_c3'] = dfValidation_dict[name].cl3d_pt / dfValidation_dict[name].cl3d_c3
        dfValidation_dict[name]['cl3d_response_c3'] = dfValidation_dict[name].cl3d_pt_c3 / dfValidation_dict[name].gentau_vis_pt

        # TRAINING DATASET
        # application calibration 2
        logpt1 = np.log(abs(dfTraining_dict[name]['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTraining_dict[name]['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTraining_dict[name]['cl3d_pt_c3'] = dfTraining_dict[name].cl3d_pt / dfTraining_dict[name].cl3d_c3
        dfTraining_dict[name]['cl3d_response_c3'] = dfTraining_dict[name].cl3d_pt_c3 / dfTraining_dict[name].gentau_vis_pt

        # VALIDATION DATASET
        # application calibration 2
        logpt1 = np.log(abs(dfVal['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfVal['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfVal['cl3d_pt_c3'] = dfVal.cl3d_pt / dfVal.cl3d_c3
        dfVal['cl3d_response_c3'] = dfVal.cl3d_pt_c3 / dfVal.gentau_vis_pt

        # TRAINING DATASET
        # application calibration 2
        logpt1 = np.log(abs(dfTr['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr.cl3d_pt / dfTr.cl3d_c3
        dfTr['cl3d_response_c3'] = dfTr.cl3d_pt_c3 / dfTr.gentau_vis_pt

        # QCD VALIDATION DATASET
        dfQCDVal['cl3d_response'] = dfQCDVal['cl3d_pt']/dfQCDVal['genjet_pt']
        # application calibration 2
        logpt1 = np.log(abs(dfQCDVal['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDVal['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDVal['cl3d_pt_c3'] = dfQCDVal.cl3d_pt / dfQCDVal.cl3d_c3
        dfQCDVal['cl3d_response_c3'] = dfQCDVal.cl3d_pt_c3 / dfQCDVal.genjet_pt

        # QCD TRAINING DATASET
        dfQCDTr['cl3d_response'] = dfQCDTr['cl3d_pt']/dfQCDTr['genjet_pt']
        # application calibration 2
        logpt1 = np.log(abs(dfQCDTr['cl3d_pt']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfQCDTr['cl3d_c3'] = C3model_dict[name].predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfQCDTr['cl3d_pt_c3'] = dfQCDTr.cl3d_pt / dfQCDTr.cl3d_c3
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
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfTr['cl3d_response_c3'].mean(), dfTr['cl3d_response_c3'].std(), dfTr['cl3d_response_c3'].std()/dfTr['cl3d_response_c3'].mean()))
            print('\n--------------------------------------------------------------------------------------\n')
            print('VALIDATION DATASET:')
            print('  -- RAW: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response'].mean(), dfVal['cl3d_response'].std(), dfVal['cl3d_response'].std()/dfVal['cl3d_response'].mean()))
            print('  -- CALIBRATED 2: mean={0}, rms={1}, rms/mean={2}'.format(dfVal['cl3d_response_c3'].mean(), dfVal['cl3d_response_c3'].std(), dfVal['cl3d_response_c3'].std()/dfVal['cl3d_response_c3'].mean()))
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
            plt.hist(dfTr['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfTr['cl3d_response_c3'].mean(),3), round(dfTr['cl3d_response_c3'].std(),3)), color='limegreen',  histtype='step', lw=2)
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
            plt.hist(dfVal['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfVal['cl3d_response_c3'].mean(),3), round(dfVal['cl3d_response_c3'].std(),3)), color='limegreen',  histtype='step', lw=2)
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

            ######################### 2D VALIDATION C3 RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['gentau_vis_eta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c3'], np.abs(dfVal['gentau_vis_eta']), label='Calib. 3', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$|\eta^{gen,\tau}|$')
            plt.title('Validation dataset C3 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC3_validation_2Deta_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.scatter(dfVal['cl3d_response'], dfVal['gentau_vis_pt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
            plt.scatter(dfVal['cl3d_response_c3'], dfVal['gentau_vis_pt'], label='Calib. 3', color='limegreen', marker='.', alpha=0.3)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
            plt.ylabel(r'$p_{T}^{gen,\tau}$')
            plt.title('Validation dataset C3 calibration response'.format(args.ptcut))
            plt.xlim(0, 2)
            plt.ylim(0, 200)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/calibResponseC3_validation_2Dpt_'+name+'_PU200.pdf')
            plt.close()

            ######################### QCD RESPONSE #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfQCDVal['cl3d_response'], bins=np.arange(0., 2., 0.03),  label='Uncalibrated, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response'].mean(),3), round(dfQCDVal['cl3d_response'].std(),3)), color='red',    histtype='step', lw=2)
            plt.hist(dfQCDVal['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label='Calib. 3, mean: {0}, RMS: {1}'.format(round(dfQCDVal['cl3d_response_c3'].mean(),3), round(dfQCDVal['cl3d_response_c3'].std(),3)), color='limegreen',  histtype='step', lw=2)
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
            plt.hist(dfValidationDM0_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM0_dict[name]['cl3d_response_c3'].std(),3)), color='limegreen',    histtype='step', lw=2)
            plt.hist(dfValidationDM1_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'1-prong + $\pi^0$, mean: {0}, RMS: {1}'.format(round(dfValidationDM1_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM1_dict[name]['cl3d_response_c3'].std(),3)), color='blue',    histtype='step', lw=2)
            plt.hist(dfValidationDM10_dict[name]['cl3d_response_c3'], bins=np.arange(0., 2., 0.03),  label=r'3-prong, mean: {0}, RMS: {1}'.format(round(dfValidationDM0_dict[name]['cl3d_response_c3'].mean(),3), round(dfValidationDM10_dict[name]['cl3d_response_c3'].std(),3)), color='red',  histtype='step', lw=2)
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


            ######################### C3 DISTRIBUTIONS #########################
            plt.figure(figsize=(8,8))
            plt.hist(dfValidationDM0_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM1_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM10_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfValidationDM11_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationPU_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfValidationQCD_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'C3 factor value')
            plt.ylabel(r'a. u.')
            plt.title('Validation dataset')
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(plotdir+'/DM_c3_validation_'+name+'_PU200.pdf')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.hist(dfTrainingDM0_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'1-prong', color='limegreen',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM1_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'1-prong + $\pi^0$', color='blue',    histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM10_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'3-prong', color='red',  histtype='step', lw=2, density=True)
            plt.hist(dfTrainingDM11_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'3-prong + $\pi^0$', color='fuchsia',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingPU_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'PU', color='orange',   histtype='step', lw=2, density=True)
            plt.hist(dfTrainingQCD_dict[name]['cl3d_c3'], bins=np.arange(0.75, 1., 0.005),  label=r'QCD', color='black',   histtype='step', lw=2, density=True)
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
