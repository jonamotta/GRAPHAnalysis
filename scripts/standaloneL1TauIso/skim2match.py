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


def deltar( df, isTau ):
    if isTau:
        df['deta_cl3d_gentau'] = np.abs(df['cl3d_eta'] - df['gentau_vis_eta'])
        df['dphi_cl3d_gentau'] = np.abs(df['cl3d_phi'] - df['gentau_vis_phi'])
        sel = df['dphi_cl3d_gentau'] > np.pi
        df['dphi_cl3d_gentau'] = np.abs(sel*(2*np.pi) - df['dphi_cl3d_gentau'])
        return ( np.sqrt(df['dphi_cl3d_gentau']*df['dphi_cl3d_gentau']+df['deta_cl3d_gentau']*df['deta_cl3d_gentau']) )
    else:
        df['deta_cl3d_genjet'] = np.abs(df['cl3d_eta'] - df['genjet_eta'])
        df['dphi_cl3d_genjet'] = np.abs(df['cl3d_phi'] - df['genjet_phi'])
        sel = df['dphi_cl3d_genjet'] > np.pi
        df['dphi_cl3d_genjet'] = np.abs(sel*(2*np.pi) - df['dphi_cl3d_genjet'])
        return ( np.sqrt(df['dphi_cl3d_genjet']*df['dphi_cl3d_genjet']+df['deta_cl3d_genjet']*df['deta_cl3d_genjet']) )

def taumatching( df_cl3d, df_gen, deta, dphi, dR ):
    df_cl3d.set_index('event', inplace=True)
    df_gen.set_index('event', inplace=True)

    # join dataframes to create all the possible combinations
    df_joined  = df_gen.join(df_cl3d, on='event', how='left', rsuffix='_cl3d', sort=False)
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False

    # calculate distances and select geometrical matches
    df_joined['deltar_cl3d_gentau']  = deltar(df_joined, 1)
    df_joined['geom_match'] = df_joined['deltar_cl3d_gentau'] < dR
    # df_joined['geom_match']  = (df_joined['deta_cl3d_gentau'] < deta/2) & (df_joined['dphi_cl3d_gentau'] < dphi/2)
    
    # if the geometrical match is false then we can take away the info about the tau in the row
    df_joined_PU = df_joined.query('geom_match==False').copy(deep=True)
    df_joined_PU.drop_duplicates(['cl3d_pt','cl3d_eta'], keep='first', inplace=True)
    df_joined_PU[['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']] = -99.9
    df_joined_PU.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined_match = df_joined.query('geom_match==True').copy(deep=True)
    df_joined_match.reset_index(inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between tau and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined_match.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined_match.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('gentau_vis_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # insert the information of the gen taus in the rows of the best matches
    for i in df_joined_match.index.values:
        if not i % 10000: print('         reading index {0}/{1}'.format(i, len(df_joined_match.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined_match['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined_match['event'][i] and temp['gentau_vis_pt'][idx] == df_joined_match['gentau_vis_pt'][i] and df_joined_match['geom_match'][i] == True:
                df_joined_match.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined_match['cl3d_pt'][i]: df_joined_match.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    df_joined_match.set_index('event', inplace=True)
    df_joined_PU.set_index('event', inplace=True)

    dfOut = pd.concat([df_joined_match, df_joined_PU], sort=False)
    dfOut.sort_values('event', inplace=True)

    return dfOut

def jetmatching( df_cl3d, df_gen, deta, dphi, dR ):
    df_cl3d.set_index('event', inplace=True)
    df_gen.set_index('event', inplace=True)

    # join dataframes to create all the possible combinations
    df_joined  = df_gen.join(df_cl3d, on='event', how='left', rsuffix='_cl3d', sort=False)
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False

    # calculate distances and select geometrical matches
    df_joined['deltar_cl3d_genjet']  = deltar(df_joined, 0)
    df_joined['geom_match'] = df_joined['deltar_cl3d_genjet'] < dR
    # df_joined['geom_match']  = (df_joined['deta_cl3d_genjet'] < deta/2) & (df_joined['dphi_cl3d_genjet'] < dphi/2)
    
    # if the geometrical match is false then we can take away the info about the jet in the row
    df_joined_PU = df_joined.query('geom_match==False').copy(deep=True)
    df_joined_PU.drop_duplicates(['cl3d_pt','cl3d_eta'], keep='first', inplace=True)
    df_joined_PU[['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']] = 0.0
    df_joined_PU.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined_match = df_joined.query('geom_match==True').copy(deep=True)
    df_joined_match.reset_index(inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between jet and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined.query('geom_match==True').groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined.query('geom_match==True').groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('genjet_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # insert the information of the gen jets in the rows of the best matches
    for i in df_joined_match.index.values:
        if not i % 10000: print('         reading index {0}/{1}'.format(i, len(df_joined_match.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined_match['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined_match['event'][i] and temp['genjet_pt'][idx] == df_joined_match['genjet_pt'][i] and df_joined_match['geom_match'][i] == True:
                df_joined_match.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined_match['cl3d_pt'][i]: df_joined_match.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    df_joined_match.set_index('event', inplace=True)
    df_joined_PU.set_index('event', inplace=True)

    dfOut = pd.concat([df_joined_match, df_joined_PU], sort=False)
    dfOut.sort_values('event', inplace=True)

    return dfOut


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
    indir   = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_12May2021'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_rateEvaluation/hdf5dataframes'
    os.system('mkdir -p '+outdir)

    # DICTIONARIES FOR THE MATCHING
    if args.doHH:
        inFileHH_match = {
            'threshold'    : indir+'/SKIM_GluGluHHTo2b2Tau_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }        

        outFileHH_match = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileHH_towers = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doTenTau:
        inFileTenTau_match = {
            'threshold'    : indir+'/SKIM_RelValTenTau_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileTenTau_match = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileTenTau_towers = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doSingleTau:
        inFileSingleTau_match = {
            'threshold'    : indir+'/SKIM_RelValSingleTau_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileSingleTau_match = {
            'threshold'    : outdir+'/RelValSingleTau_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileSingleTau_towers = {
            'threshold'    : outdir+'/RelValSingleTau_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doQCD:
        inFileQCD_match = {
            'threshold'    : indir+'/SKIM_RelValQCD_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileQCD_match = {
            'threshold'    : outdir+'/QCD_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileQCD_towers = {
            'threshold'    : outdir+'/QCD_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doMinbias:
        inFileMinbias_match = {
            'threshold'    : indir+'/SKIM_Minbias_PU200/mergedOutput{0}.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileMinbias_match = {
            'threshold'    : outdir+'/Minbias_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileMinbias_towers = {
            'threshold'    : outdir+'/Minbias_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doNu:
        inFileNu_match = {
            'threshold'    : indir+'/SKIM_RelValNu_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileNu_match = {
            'threshold'    : outdir+'/RelValNu_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileNu_towers = {
            'threshold'    : outdir+'/RelValNu_PU200_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }


    ##################### READ TTREES AND MATCH EVENTS ####################

    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting cluster matching')

    # TTree to be read
    treename = 'HGCALskimmedTree'
    #TBranches to be stored containing the 3D clusters' info
    branches_event_cl3d = ['event','cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_spptot','cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    branches_cl3d       = ['cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_spptot','cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    # TBranches to be stored containing the gen taus' info
    branches_event_gentau = ['event', 'gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    branches_gentau       = ['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    # TBranches to be stored containing the gen jets' info
    branches_event_genjet = ['event', 'genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    branches_genjet       = ['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    #TBranches to be stored containing the towers' info
    branches_event_towers = ['event','tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']
    branches_towers       = ['tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']

    if args.doHH:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.1
        dphi_matching = 0.2
        dr_matching = 0.1

        # fill the dataframes with the needed info from the branches defined above for the HH sample
        print('\n** INFO: creating dataframes for ' + inFileHH_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileHH_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_gentau = root_pandas.read_root(inFileHH_match[args.FE], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
        dfHH_towers = root_pandas.read_root(inFileHH_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)
        
        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_gentau.drop('__array_index', inplace=True, axis=1)
        dfHH_towers.drop('__array_index', inplace=True, axis=1)

        df_cl3d['event'] += 18000    # here we need to make sure we are not overlapping the 'events' columns of HH, TenTau, and SingleTau!!!
        df_gentau['event'] += 18000  # "
        dfHH_towers['event'] += 18000  # "

        if args.testRun:
            df_cl3d.query('event<18010', inplace=True)
            df_gentau.query('event<18010', inplace=True)
            dfHH_towers.query('event<18010', inplace=True)

        print('** INFO: matching gentaus for ' + inFileHH_match[args.FE])
        dfHH = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
        dfHH['dataset'] = 0 # tag the dataset it came from

        print('** INFO: saving file ' + outFileHH_match[args.FE])
        store_hh = pd.HDFStore(outFileHH_match[args.FE], mode='w')
        store_hh[args.FE] = dfHH
        store_hh.close()

        print('** INFO: saving file ' + outFileHH_towers[args.FE])
        dfHH_towers.set_index('event', inplace=True)
        dfHH_towers.sort_values('event', inplace=True)
        store_towers = pd.HDFStore(outFileHH_towers[args.FE], mode='w')
        store_towers[args.FE] = dfHH_towers
        store_towers.close()

        del  df_cl3d, df_gentau

    if args.doTenTau:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.1
        dphi_matching = 0.2
        dr_matching = 0.1

        # fill the dataframes with the needed info from the branches defined above for the TENTAU
        print('\n** INFO: creating dataframes for ' + inFileTenTau_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileTenTau_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_gentau = root_pandas.read_root(inFileTenTau_match[args.FE], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
        dfTenTau_towers = root_pandas.read_root(inFileTenTau_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_gentau.drop('__array_index', inplace=True, axis=1)
        dfTenTau_towers.drop('__array_index', inplace=True, axis=1)

        if args.testRun:
            df_cl3d.query('event<10', inplace=True)
            df_gentau.query('event<10', inplace=True)
            dfTenTau_towers.query('event<10', inplace=True)

        print('** INFO: matching gentaus for ' + inFileTenTau_match[args.FE])
        dfTenTau = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
        dfTenTau['dataset'] = 1 # tag the dataset it came from

        print('** INFO: saving file ' + outFileTenTau_match[args.FE])
        store_tau = pd.HDFStore(outFileTenTau_match[args.FE], mode='w')
        store_tau[args.FE] = dfTenTau
        store_tau.close()

        print('** INFO: saving file ' + outFileTenTau_towers[args.FE])
        dfTenTau_towers.set_index('event', inplace=True)
        dfTenTau_towers.sort_values('event', inplace=True)
        store_towers = pd.HDFStore(outFileTenTau_towers[args.FE], mode='w')
        store_towers[args.FE] = dfTenTau_towers
        store_towers.close()

        del  df_cl3d, df_gentau

    if args.doSingleTau:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.1
        dphi_matching = 0.2
        dr_matching = 0.1

        # fill the dataframes with the needed info from the branches defined above for the SINGLETAU
        print('\n** INFO: creating dataframes for ' + inFileSingleTau_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileSingleTau_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_gentau = root_pandas.read_root(inFileSingleTau_match[args.FE], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
        dfSingleTau_towers = root_pandas.read_root(inFileSingleTau_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_gentau.drop('__array_index', inplace=True, axis=1)
        dfSingleTau_towers.drop('__array_index', inplace=True, axis=1)

        df_cl3d['event'] += 9000    # here we need to make sure we are not overlapping the 'events' columns of HH, TenTau, and SingleTau!!!
        df_gentau['event'] += 9000  # "
        dfSingleTau_towers['event'] += 9000  # "

        if args.testRun:
            df_cl3d.query('event<9050', inplace=True)
            df_gentau.query('event<9050', inplace=True)
            dfSingleTau_towers.query('event<9050', inplace=True)

        print('** INFO: matching gentaus for ' + inFileSingleTau_match[args.FE])
        dfSingleTau = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
        dfSingleTau['dataset'] = 2 # tag the dataset it came from

        print('** INFO: saving file ' + outFileSingleTau_match[args.FE])
        store_tau = pd.HDFStore(outFileSingleTau_match[args.FE], mode='w')
        store_tau[args.FE] = dfSingleTau
        store_tau.close()

        print('** INFO: saving file ' + outFileSingleTau_towers[args.FE])
        dfSingleTau_towers.set_index('event', inplace=True)
        dfSingleTau_towers.sort_values('event', inplace=True)
        store_towers = pd.HDFStore(outFileSingleTau_towers[args.FE], mode='w')
        store_towers[args.FE] = dfSingleTau_towers
        store_towers.close()

        del  df_cl3d, df_gentau

    if args.doQCD:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.2
        dphi_matching = 0.3
        dr_matching = 0.4

        # fill the dataframes with the needed info from the branches defined above for the QCD
        print('\n** INFO: creating dataframes for ' + inFileQCD_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_genjet = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_genjet, flatten=branches_genjet)
        dfQCD_towers = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_genjet.drop('__array_index', inplace=True, axis=1)
        dfQCD_towers.drop('__array_index', inplace=True, axis=1)

        df_cl3d['event'] += 38000    # here we need to make sure we are not overlapping the 'events' columns of HH, TenTau, and SingleTau!!!
        df_genjet['event'] += 38000  # "
        dfQCD_towers['event'] += 38000  # "

        if args.testRun:
            df_cl3d.query('event<88100', inplace=True)
            df_genjet.query('event<88100', inplace=True)
            dfQCD_towers.query('event<88100', inplace=True)

        print('** INFO: matching genjets for ' + inFileQCD_match[args.FE])
        dfQCD = jetmatching(df_cl3d, df_genjet, deta_matching, dphi_matching, dr_matching)
        dfQCD['gentau_decayMode'] = -2 # tag as QCD
        dfQCD['dataset'] = 3 # tag the dataset it came from

        print('** INFO: saving file ' + outFileQCD_match[args.FE])
        store_qcd = pd.HDFStore(outFileQCD_match[args.FE], mode='w')
        store_qcd[args.FE] = dfQCD
        store_qcd.close()

        print('** INFO: saving file ' + outFileQCD_towers[args.FE])
        dfQCD_towers.set_index('event', inplace=True)
        dfQCD_towers.sort_values('event', inplace=True)
        store_towers = pd.HDFStore(outFileQCD_towers[args.FE], mode='w')
        store_towers[args.FE] = dfQCD_towers
        store_towers.close()

        del  df_cl3d, df_genjet

    if args.doNu:
        # fill the dataframes with the needed info from the branches defined above for the NU
        print('\n** INFO: creating dataframes for ' + inFileNu_match[args.FE]) 
        dfNu = root_pandas.read_root(inFileNu_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        dfNu_towers = root_pandas.read_root(inFileNu_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)
        
        dfNu.drop('__array_index', inplace=True, axis=1)
        dfNu['gentau_decayMode'] = -1 # tag as PU
        dfNu['geom_match'] = False
        dfNu['cl3d_isbestmatch'] = False
        dfNu['dataset'] = 4 # tag the dataset it came from

        dfNu_towers.drop('__array_index', inplace=True, axis=1)
        dfNu_towers.set_index('event', inplace=True)
        dfNu_towers.sort_values('event', inplace=True)

        print('** INFO: saving file ' + outFileNu_match[args.FE])
        store_nu = pd.HDFStore(outFileNu_match[args.FE], mode='w')
        store_nu[args.FE] = dfNu
        store_nu.close()

        print('** INFO: saving file ' + outFileNu_towers[args.FE])
        store_towers = pd.HDFStore(outFileNu_towers[args.FE], mode='w')
        store_towers[args.FE] = dfNu_towers
        store_towers.close()

        del dfNu, dfNu_towers

    if args.doMinbias:
        dfMinbias = pd.DataFrame()
        dfMinbias_towers = pd.DataFrame()

        # fill the dataframes with the needed info from the branches defined above for the NU
        print('\n** INFO: creating dataframes for ' + inFileMinbias_match[args.FE].format('')) 
        for i in ['', '_1', '_2']:     
            df_cl3d = root_pandas.read_root(inFileMinbias_match[args.FE].format(i), key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            towers = root_pandas.read_root(inFileMinbias_match[args.FE].format(i), key=treename, columns=branches_event_towers, flatten=branches_towers)

            dfMinbias = pd.concat([dfMinbias,df_cl3d], sort=False)
            dfMinbias_towers = pd.concat([dfMinbias_towers,towers], sort=False)

        dfMinbias.drop('__array_index', inplace=True, axis=1)
        dfMinbias['gentau_decayMode'] = -1 # tag as PU
        dfMinbias['geom_match'] = False
        dfMinbias['cl3d_isbestmatch'] = False
        dfMinbias['dataset'] = 5 # tag the dataset it came from
        dfMinbias.set_index('event', inplace=True)
        dfMinbias.sort_values('event', inplace=True)

        dfMinbias_towers.drop('__array_index', inplace=True, axis=1)
        dfMinbias_towers.set_index('event', inplace=True)
        dfMinbias_towers.sort_values('event', inplace=True)

        print('** INFO: saving file ' + outFileMinbias_match[args.FE])
        store = pd.HDFStore(outFileMinbias_match[args.FE], mode='w')
        store[args.FE] = dfMinbias
        store.close()

        print('** INFO: saving file ' + outFileMinbias_towers[args.FE])
        store_towers = pd.HDFStore(outFileMinbias_towers[args.FE], mode='w')
        store_towers[args.FE] = dfMinbias_towers
        store_towers.close()

        del df_cl3d, towers

    print('\n** INFO: finished cluster matching')
    print('---------------------------------------------------------------------------------------')

    end = time.time()
    print '\nRunning time = %02dh %02dm %02ds'%((end-start)/3600, ((end-start)%3600)/60, (end-start)% 60)
