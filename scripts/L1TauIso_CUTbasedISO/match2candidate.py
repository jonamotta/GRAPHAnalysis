import os
import numpy as np
import pandas as pd
import root_pandas
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import argparse
import time


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
    parser.add_argument('--doHH', dest='doHH', help='match the HH samples?',  action='store_true', default=False)
    parser.add_argument('--doTenTau', dest='doTenTau', help='match the TenTau samples?',  action='store_true', default=False)
    parser.add_argument('--doSingleTau', dest='doSingleTau', help='match the SingleTau samples?',  action='store_true', default=False)
    parser.add_argument('--doAllTau', dest='doAllTau', help='match all the Tau samples',  action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='match the QCD samples?',  action='store_true', default=False)
    parser.add_argument('--doMinbias', dest='doMinbias', help='match the Minbias samples?',  action='store_true', default=False)
    parser.add_argument('--doNu', dest='doNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--testRun', dest='testRun', help='do test run with reduced number of events?',  action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    start = time.time()

    if not args.doHH and not args.doTenTau and not args.doSingleTau and not args.doAllTau and not args.doQCD and not args.doMinbias and not args.doNu:
        print('** WARNING: no matching dataset specified. What do you want to do (doHH, doTenTau, doSingleTau doAllTau, doQCD, doMinbias, doNu)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    if args.doAllTau:
        args.doSingleTau = True
        args.doTenTau = True
        args.doHH = True

    print('** INFO: using front-end option: '+args.FE)

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolation_application_dRsgn2'

    # DICTIONARIES FOR THE MATCHING
    if args.doHH:
        inFileHH_match = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        inFileTenTau_match = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        inFileSingleTau_match = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        inFileQCD_match = {
            'threshold'    : indir+'/QCD_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        inFileMinbias_match = {
            'threshold'    : indir+'/Minbias_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        inFileNu_match = {
            'threshold'    : indir+'/RelValNu_PU200_th_matched.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE CALIBRATION
    calib_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/calibration'

    modelC1_calib = {
        'threshold'    : calib_model_indir+'/model_c1_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    modelC2_calib = {
        'threshold'    : calib_model_indir+'/model_c2_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    modelC3_calib = {
        'threshold'    : calib_model_indir+'/model_c3_th_PU200.pkl',
        'supertrigger' : calib_model_indir+'/',
        'bestchoice'   : calib_model_indir+'/',
        'bestcoarse'   : calib_model_indir+'/',
        'mixed'        : calib_model_indir+'/'
    }

    if args.doHH:
        outFileHH_calib = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_calib = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_calib = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_calib = {
            'threshold'    : indir+'/QCD_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_calib = {
            'threshold'    : indir+'/Minbias_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_calib = {
            'threshold'    : indir+'/RelValNu_PU200_th_calibrated.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE PILEUP REJECTION
    PUrej_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/PUrejection'

    model_PUrej = {
        'threshold'    : PUrej_model_indir+'/model_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP90_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP90_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP95_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP95_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    WP99_PUrej = {
        'threshold'    : PUrej_model_indir+'/WP99_PUrejection_th_PU200.pkl',
        'supertrigger' : PUrej_model_indir+'/',
        'bestchoice'   : PUrej_model_indir+'/',
        'bestcoarse'   : PUrej_model_indir+'/',
        'mixed'        : PUrej_model_indir+'/'
    }

    if args.doHH:
        outFileHH_PUrej = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_PUrej = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_PUrej = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_PUrej = {
            'threshold'    : indir+'/QCD_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_PUrej = {
            'threshold'    : indir+'/Minbias_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_PUrej = {
            'threshold'    : indir+'/RelValNu_PU200_th_PUrejected.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    # DICTIONARIES FOR THE DM sorting
    DMsort_model_indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/DMsorting'

    model_DMsort = {
        'threshold'    : DMsort_model_indir+'/model_DMsorting_th_PU200_PUWP{0}.pkl',
        'supertrigger' : DMsort_model_indir+'/',
        'bestchoice'   : DMsort_model_indir+'/',
        'bestcoarse'   : DMsort_model_indir+'/',
        'mixed'        : DMsort_model_indir+'/'
    }

    if args.doHH:
        outFileHH_DMsort = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doTenTau:
        outFileTenTau_DMsort = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doSingleTau:
        outFileSingleTau_DMsort = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doQCD:
        outFileQCD_DMsort = {
            'threshold'    : indir+'/QCD_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doMinbias:
        outFileMinbias_DMsort = {
            'threshold'    : indir+'/Minbias_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    if args.doNu:
        outFileNu_DMsort = {
            'threshold'    : indir+'/RelValNu_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

    ###########################################
    # CREATE LISTS OF FILES TO CREATE AND SAVE

    dfs4Iso = []
    outFiles4Iso_calib = []
    outFiles4Iso_PUrej = []
    outFiles4Iso_DMsort = []

    if args.doHH:
        store = pd.HDFStore(inFileHH_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE].query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileHH_calib)
        outFiles4Iso_PUrej.append(outFileHH_PUrej)
        outFiles4Iso_DMsort.append(outFileHH_DMsort)
    if args.doTenTau:
        store = pd.HDFStore(inFileTenTau_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE].query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileTenTau_calib)
        outFiles4Iso_PUrej.append(outFileTenTau_PUrej)
        outFiles4Iso_DMsort.append(outFileTenTau_DMsort)
    if args.doSingleTau:
        store = pd.HDFStore(inFileSingleTau_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE].query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileSingleTau_calib)
        outFiles4Iso_PUrej.append(outFileSingleTau_PUrej)
        outFiles4Iso_DMsort.append(outFileSingleTau_DMsort)
    if args.doQCD:
        store = pd.HDFStore(inFileQCD_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE]dfs4Iso.query('(cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9) and (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9)))', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileQCD_calib)
        outFiles4Iso_PUrej.append(outFileQCD_PUrej)
        outFiles4Iso_DMsort.append(outFileQCD_DMsort)
    if args.doMinbias:
        store = pd.HDFStore(inFileMinbias_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE].query('cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileMinbias_calib)
        outFiles4Iso_PUrej.append(outFileMinbias_PUrej)
        outFiles4Iso_DMsort.append(outFileMinbias_DMsort)
    if args.doNu:
        store = pd.HDFStore(inFileNu_match[args.FE], mode='r')
        dfs4Iso.append(store[args.FE].query('cl3d_pt>4 and cl3d_abseta>1.6 and cl3d_abseta<2.9', inplace=True))
        store.close()

        outFiles4Iso_calib.append(outFileNu_calib)
        outFiles4Iso_PUrej.append(outFileNu_PUrej)
        outFiles4Iso_DMsort.append(outFileNu_DMsort)

    ####################################################################
    # LOAD MODELS FOR CALIBRATION, PU REJECTION, AND DECAY MODE SORTING

    # features used for the C2 calibration step
    features_calib = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean','cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    modelC1 = load_obj(modelC1_calib[args.FE])
    modelC2 = load_obj(modelC2_calib[args.FE])
    modelC3 = load_obj(modelC3_calib[args.FE])

    # features for PU rejection BDT
    features_PUrej = ['cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    model_PU = load_obj(model_PUrej[args.FE])
    bdtWP99 = load_obj(WP99_PUrej[args.FE])
    bdtWP95 = load_obj(WP95_PUrej[args.FE])
    bdtWP90 = load_obj(WP90_PUrej[args.FE])

    # features for DM sorting BDT
    features = ['cl3d_pt_c3', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']

    ##################################################
    # LOOP OVER THE LISTS OF FILES TO CREATE AND SAVE

    for k in range(len(dfs4Iso)):

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting cluster calibration, PU rejection, and DM sorting to produce '+outFiles4Iso_DMsort[k][args.FE])

        # CLUSTERS CALIBRATION
        dfs4Iso[k]['cl3d_abseta'] = np.abs(dfs4Iso[k]['cl3d_eta'])
        # application of calibration 1
        print('\n** INFO: applying calibration C1')    
        dfs4Iso[k]['cl3d_c1'] = modelC1.predict(dfs4Iso[k][['cl3d_abseta']])
        dfs4Iso[k]['cl3d_pt_c1'] = dfs4Iso[k].cl3d_c1 + dfs4Iso[k].cl3d_pt
        # application of calibration 2
        print('** INFO: applying calibration C2')
        dfs4Iso[k]['cl3d_c2'] = modelC2.predict(dfs4Iso[k][features_calib])
        dfs4Iso[k]['cl3d_pt_c2'] = dfs4Iso[k].cl3d_c2 * dfs4Iso[k].cl3d_pt_c1
        # application of calibration 3
        print('** INFO: applying calibration C3')
        logpt1 = np.log(abs(dfs4Iso[k]['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfs4Iso[k]['cl3d_c3'] = modelC3.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfs4Iso[k]['cl3d_pt_c3'] = dfs4Iso[k].cl3d_pt_c2 / dfs4Iso[k].cl3d_c3
        # save file
        print('** INFO: saving file ' + outFiles4Iso_calib[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_calib[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        # PILE UP REJECTION
        print('\n** INFO: applying PU rejection BDT')
        full = xgb.DMatrix(data=dfs4Iso[k][features_PUrej], feature_names=features_PUrej)
        dfs4Iso[k]['cl3d_pubdt_score'] = model_PU.predict(full)
        dfs4Iso[k]['cl3d_pubdt_passWP99'] = dfs4Iso[k]['cl3d_pubdt_score'] > bdtWP99
        dfs4Iso[k]['cl3d_pubdt_passWP95'] = dfs4Iso[k]['cl3d_pubdt_score'] > bdtWP95
        dfs4Iso[k]['cl3d_pubdt_passWP90'] = dfs4Iso[k]['cl3d_pubdt_score'] > bdtWP90
        # save file
        print('** INFO: saving file ' + outFiles4Iso_PUrej[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_PUrej[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        # DM SORTING
        print('\n** INFO: starting DM sorting')
        for WP in [99, 95, 90]:
            print('    - PUWP = {0}'.format(WP))
            model_DM = load_obj(model_DMsort[args.FE].format(WP))
            dfs4Iso[k]['cl3d_predDM_PUWP{0}'.format(WP)] = model_DM.predict(dfs4Iso[k][features])
            probas_DM = model_DM.predict_proba(dfs4Iso[k][features])
            dfs4Iso[k]['cl3d_probDM0_PUWP{0}'.format(WP)] = probas_DM[:,0]
            dfs4Iso[k]['cl3d_probDM1_PUWP{0}'.format(WP)] = probas_DM[:,1]
            dfs4Iso[k]['cl3d_probDM2_PUWP{0}'.format(WP)] = probas_DM[:,2]
            dfs4Iso[k]['cl3d_probDM3_PUWP{0}'.format(WP)] = probas_DM[:,3]
            del probas_DM, model_DM 
        
        # save file
        print('\n** INFO: saving file ' + outFiles4Iso_DMsort[k][args.FE])
        store = pd.HDFStore(outFiles4Iso_DMsort[k][args.FE], mode='w')
        store[args.FE] = dfs4Iso[k]
        store.close()

        print('\n** INFO: finished producing and saving '+outFiles4Iso_DMsort[k][args.FE]+' and all intermediate files')
        print('---------------------------------------------------------------------------------------')

    end = time.time()
    print '\nRunning time = %02dh %02dm %02ds'%((end-start)/3600, ((end-start)%3600)/60, (end-start)% 60)
