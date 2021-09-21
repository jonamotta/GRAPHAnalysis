import os
import sys
import numpy as np
import pandas as pd
import root_pandas
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import argparse


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

    # calculate distances and select geometrical matches
    df_joined['deltar_cl3d_gentau'] = deltar(df_joined, 1)
    df_joined['geom_match'] = df_joined['deltar_cl3d_gentau'] < dR
    # df_joined['geom_match']  = (df_joined['deta_cl3d_gentau'] < deta/2) & (df_joined['dphi_cl3d_gentau'] < dphi/2)
    df_joined.query('geom_match==True', inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between tau and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('gentau_vis_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined.drop_duplicates(['cl3d_pt', 'cl3d_eta', 'cl3d_phi'], keep='last',inplace=True)
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False
    df_joined.reset_index(inplace=True)

    # insert the information of the gen taus in the rows of the best matches
    for i in df_joined.index.values:
        if not i % 5000: print('         reading index {0}/{1}'.format(i, len(df_joined.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined['event'][i] and temp['gentau_vis_pt'][idx] == df_joined['gentau_vis_pt'][i]:
                df_joined.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined['cl3d_pt'][i]: df_joined.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    df_joined.set_index('event', inplace=True)
    df_joined.sort_values('event', inplace=True)

    return df_joined

def jetmatching( df_cl3d, df_gen, deta, dphi, dR ):
    df_cl3d.set_index('event', inplace=True)
    df_gen.set_index('event', inplace=True)

    # join dataframes to create all the possible combinations
    # here we use how='inner' because we have selected events with only one jet --> when we do the joining we want to
    # keep only those events and to have all the combinations for those events
    df_joined  = df_gen.join(df_cl3d, on='event', how='inner', rsuffix='_cl3d', sort=False)

    # calculate distances and select geometrical matches
    df_joined['deltar_cl3d_genjet']  = deltar(df_joined, 0)
    df_joined['geom_match']  = df_joined['deltar_cl3d_genjet'] < dR
    # df_joined['geom_match']  = (df_joined['deta_cl3d_genjet'] < deta/2) & (df_joined['dphi_cl3d_genjet'] < dphi/2)
    df_joined.query('geom_match==True', inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between jet and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined.query('geom_match==True').groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined.query('geom_match==True').groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('genjet_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined.drop_duplicates(['cl3d_pt', 'cl3d_eta', 'cl3d_phi'], keep='last',inplace=True)
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False
    df_joined.reset_index(inplace=True)

    # insert the information of the gen taus in the rows of the best matches
    for i in df_joined.index.values:
        if not i % 5000: print('         reading index {0}/{1}'.format(i, len(df_joined.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined['event'][i] and temp['genjet_pt'][idx] == df_joined['genjet_pt'][i]:
                df_joined.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined['cl3d_pt'][i]: df_joined.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    df_joined.set_index('event', inplace=True)
    df_joined.sort_values('event', inplace=True)

    return df_joined

def deltar2tower ( df ):
    delta_eta = np.abs(df['cl3d_eta'] - df['tower_eta'])
    delta_phi = np.abs(df['cl3d_phi'] - df['tower_phi'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def L1TowerEtIso ( dfL1Candidates, dfL1Towers, dRsgn, dRiso ):
    df_joined  = dfL1Candidates.join(dfL1Towers, on='event', how='left', rsuffix='_tow', sort=False)

    df_joined['deltar2tower'] = deltar2tower(df_joined)
    sel_sgn = (df_joined['deltar2tower'] <= dRsgn)
    sel_iso = (df_joined['deltar2tower'] <= dRiso) & (df_joined['deltar2tower'] > dRsgn)
    df_joined_sgn = df_joined[sel_sgn].copy(deep=True)
    df_joined_iso = df_joined[sel_iso].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt'], inplace=True)

    dfL1Candidates['tower_etSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined_sgn.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
    dfL1Candidates['tower_eSgn_dRsgn{0}'.format(int(dRsgn*10))] = df_joined_sgn.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()

    dfL1Candidates['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_pt'].sum()
    dfL1Candidates['tower_eIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_energy'].sum()
    dfL1Candidates['tower_etEmIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_etEm'].sum()
    dfL1Candidates['tower_etHadIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))] = df_joined_iso.groupby(['event', 'cl3d_pt'])['tower_etHad'].sum() 

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)

    del df_joined

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
    parser.add_argument('--doTau', dest='doTau', help='match the TenTau samples?',  action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='match all the QCD samples',  action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    if not args.doQCD and not args.doTau:
        print('** WARNING: no matching dataset specified. What do you want to do (doHH, doTau, doAll)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    print('** INFO: using front-end option: '+args.FE)

    # create needed folders
    indir   = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_12May2021'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/towers_studies_noPU'
    os.system('mkdir -p '+outdir)


    # DICTIONARIES FOR THE MATCHING
    if args.doTau:
        inFileTau_match = {
            'threshold'    : indir+'/SKIM_RelValSingleTau_noPU/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileTau_match = {
            'threshold'    : outdir+'/RelValSingleTau_noPU_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileTau_towers = {
            'threshold'    : outdir+'/RelValSingleTau_noPU_th_towers.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doQCD:
        inFileQCD_match = {
            'threshold'    : indir+'/SKIM_RelValQCD_noPU/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileQCD_match = {
            'threshold'    : outdir+'/RelValQCD_noPU_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

        outFileQCD_towers = {
            'threshold'    : outdir+'/RelValQCD_noPU_th_towers.hdf5',
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

    # TBranches to be stored containing the gen jets' info
    branches_event_genjet = ['event', 'genjet_pt', 'genjet_eta', 'genjet_phi']
    branches_genjet       = ['genjet_pt', 'genjet_eta', 'genjet_phi']

    #TBranches to be stored containing the 3D clusters' info
    branches_event_cl3d = ['event','cl3d_pt','cl3d_eta','cl3d_phi']
    branches_cl3d       = ['cl3d_pt','cl3d_eta','cl3d_phi']
    
    # TBranches to be stored containing the gen taus' info
    branches_event_gentau = ['event', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi']
    branches_gentau       = ['gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi']

    #TBranches to be stored containing the towers' info
    branches_event_towers = ['event','tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']
    branches_towers       = ['tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']


    if args.doTau:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.1
        dphi_matching = 0.2
        dr_matching = 0.1

        # fill the dataframes with the needed info from the branches defined above for the TENTAU
        print('\n** INFO: creating dataframes for ' + inFileTau_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileTau_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_gentau = root_pandas.read_root(inFileTau_match[args.FE], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
        df_towers = root_pandas.read_root(inFileTau_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_gentau.drop('__array_index', inplace=True, axis=1)
        df_towers.set_index('event', inplace=True)

        print('** INFO: matching gentaus for ' + inFileTau_match[args.FE])
        dfTau = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching) 

        print('** INFO: calculating iso and sgn pt from towers for '+ inFileTau_match[args.FE])
        dfTau.query('cl3d_isbestmatch==True', inplace=True)
        print('dRsgn=0.1 - dRiso=0.8')
        L1TowerEtIso(dfTau, df_towers, 0.1, 0.8)
        print('dRsgn=0.2 - dRiso=0.8')
        L1TowerEtIso(dfTau, df_towers, 0.2, 0.8)
        print('dRsgn=0.3 - dRiso=0.8')
        L1TowerEtIso(dfTau, df_towers, 0.3, 0.8)
        print('dRsgn=0.4 - dRiso=0.8')
        L1TowerEtIso(dfTau, df_towers, 0.4, 0.8)
        print('dRsgn=0.5 - dRiso=0.8')
        L1TowerEtIso(dfTau, df_towers, 0.5, 0.8)

        print('** INFO: saving file ' + outFileTau_match[args.FE])
        store_tau = pd.HDFStore(outFileTau_match[args.FE], mode='w')
        store_tau[args.FE] = dfTau
        store_tau.close()

        print('** INFO: saving file ' + outFileTau_towers[args.FE])
        store_towers = pd.HDFStore(outFileTau_towers[args.FE], mode='w')
        store_towers[args.FE] = df_towers
        store_towers.close()

    if args.doQCD:
        # define delta eta and delta phi to be used for the matching
        deta_matching = 0.1
        dphi_matching = 0.2
        dr_matching = 0.5

        # fill the dataframes with the needed info from the branches defined above for the TENTAU
        print('\n** INFO: creating dataframes for ' + inFileQCD_match[args.FE]) 
        df_cl3d = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
        df_genjet = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_genjet, flatten=branches_genjet)
        df_towers = root_pandas.read_root(inFileQCD_match[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

        df_cl3d.drop('__array_index', inplace=True, axis=1)
        df_genjet.drop('__array_index', inplace=True, axis=1)
        df_towers.set_index('event', inplace=True)

        # keep only QCD events with one single jet --> this implies that we will use how='inner' for the joining
        # so that only for these events that have been selected we do the cobination with clusters!
        df_genjet.drop_duplicates(['event'], inplace=True, keep=False)

        print('** INFO: joining genjets-cl3d for ' + inFileQCD_match[args.FE])
        dfQCD = jetmatching(df_cl3d, df_genjet, deta_matching, dphi_matching, dr_matching)

        print('** INFO: calculating iso and sgn pt from towers for '+ inFileQCD_match[args.FE])
        dfQCD.query('cl3d_isbestmatch==True', inplace=True)
        print('dRsgn=0.1 - dRiso=0.8')
        L1TowerEtIso(dfQCD, df_towers, 0.1, 0.8)
        print('dRsgn=0.2 - dRiso=0.8')
        L1TowerEtIso(dfQCD, df_towers, 0.2, 0.8)
        print('dRsgn=0.3 - dRiso=0.8')
        L1TowerEtIso(dfQCD, df_towers, 0.3, 0.8)
        print('dRsgn=0.4 - dRiso=0.8')
        L1TowerEtIso(dfQCD, df_towers, 0.4, 0.8)
        print('dRsgn=0.5 - dRiso=0.8')
        L1TowerEtIso(dfQCD, df_towers, 0.5, 0.8)

        print('** INFO: saving file ' + outFileQCD_match[args.FE])
        store_QCD = pd.HDFStore(outFileQCD_match[args.FE], mode='w')
        store_QCD[args.FE] = dfQCD
        store_QCD.close()

        print('** INFO: saving file ' + outFileQCD_towers[args.FE])
        store_towers = pd.HDFStore(outFileQCD_towers[args.FE], mode='w')
        store_towers[args.FE] = df_towers
        store_towers.close()





































































































































