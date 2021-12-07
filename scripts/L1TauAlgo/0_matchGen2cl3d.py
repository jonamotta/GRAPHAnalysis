import os
import sys
import numpy as np
import pandas as pd
import root_pandas
import argparse
from sklearn.model_selection import train_test_split

class Logger(object):
    def __init__(self, file):
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
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False

    # if the geometrical match is false then we can take away the info about the tau in the row
    df_joined_PU = df_joined.query('geom_match==False').copy(deep=True)
    df_joined_PU.drop_duplicates(['cl3d_pt','cl3d_eta', 'cl3d_phi'], keep='first', inplace=True)
    df_joined_PU[['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass']] = -99.9
    df_joined_PU['gentau_decayMode'] = -3
    df_joined_PU.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined.query('geom_match==True', inplace=True)
    df_joined.reset_index(inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between tau and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined.groupby(['event', 'gentau_vis_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('gentau_vis_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # insert the information of the gen taus in the rows of the best matches
    for i in df_joined.index.values:
        if not i % 10000: print('         reading index {0}/{1}'.format(i, len(df_joined.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined['event'][i] and temp['gentau_vis_pt'][idx] == df_joined['gentau_vis_pt'][i]:
                df_joined.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined['cl3d_pt'][i]: df_joined.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    dfOut = pd.concat([df_joined, df_joined_PU], sort=False)

    return dfOut

def jetmatching( df_cl3d, df_gen, deta, dphi, dR ):
    df_cl3d.set_index('event', inplace=True)
    df_gen.set_index('event', inplace=True)

    # join dataframes to create all the possible combinations
    df_joined  = df_gen.join(df_cl3d, on='event', how='left', rsuffix='_cl3d', sort=False)

    # calculate distances and select geometrical matches
    df_joined['deltar_cl3d_genjet']  = deltar(df_joined, 0)
    df_joined['geom_match'] = df_joined['deltar_cl3d_genjet'] < dR
    # df_joined['geom_match']  = (df_joined['deta_cl3d_genjet'] < deta/2) & (df_joined['dphi_cl3d_genjet'] < dphi/2)
    df_joined['n_matched_cl3d'] = 0
    df_joined['cl3d_isbestmatch'] = False
    
    # if the geometrical match is false then we can take away the info about the jet in the row
    df_joined_PU = df_joined.query('geom_match==False').copy(deep=True)
    df_joined_PU.drop_duplicates(['cl3d_pt','cl3d_eta'], keep='first', inplace=True)
    df_joined_PU[['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']] = 0.0
    df_joined_PU.reset_index(inplace=True)

    # prepare for the best matches insertion
    df_joined.query('geom_match==True', inplace=True)
    df_joined.reset_index(inplace=True)

    # create a temporary dataframe containing the information of the sole best matches between jet and clusters
    df_best_matches = pd.DataFrame()
    df_best_matches['cl3d_pt'] = df_joined.groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].max()
    df_best_matches['n_matched_cl3d'] = df_joined.groupby(['event', 'genjet_pt'], sort=False)['cl3d_pt'].size()
    df_best_matches.sort_values('genjet_pt',inplace=True)
    df_best_matches.reset_index(inplace=True)

    # insert the information of the gen jets in the rows of the best matches
    for i in df_joined.index.values:
        if not i % 10000: print('         reading index {0}/{1}'.format(i, len(df_joined.index.values)))
        temp = df_best_matches.query('event=={0}'.format(df_joined['event'][i])).copy(deep=True)
        for idx in temp.index:
            if temp['event'][idx] == df_joined['event'][i] and temp['genjet_pt'][idx] == df_joined['genjet_pt'][i] and df_joined['geom_match'][i] == True:
                df_joined.loc[i,'n_matched_cl3d'] = temp.loc[idx,'n_matched_cl3d']
                if temp['cl3d_pt'][idx] == df_joined['cl3d_pt'][i]: df_joined.loc[i,'cl3d_isbestmatch'] = True
        del temp # for safety delete temp every time

    dfOut = pd.concat([df_joined, df_joined_PU], sort=False)

    return dfOut

def prepareCat(row):
    if row['cl3d_isbestmatch'] == True and row['gentau_decayMode']>=0:
        return 1
    else:
        return 0 

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doHH', dest='doHH', help='match the HH samples?',  action='store_true', default=False)
    parser.add_argument('--doSingleTau', dest='doSingleTau', help='match the SingleTau samples?',  action='store_true', default=False)
    parser.add_argument('--doTenTau', dest='doTenTau', help='match the TenTau samples?',  action='store_true', default=False)
    parser.add_argument('--doNu', dest='doNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='match the QCD samples?',  action='store_true', default=False)
    parser.add_argument('--doZprime', dest='doZprime', help="match the Z' samples?",  action='store_true', default=False)
    parser.add_argument('--doVBFH', dest='doVBFH', help='match the VBF H samples?',  action='store_true', default=False)
    parser.add_argument('--doTrainValid', dest='doTrainValid', help='build merged training/validation datasets?',  action='store_true', default=False)
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--ptcut', dest='ptcut', help='baseline 3D cluster pT cut to use', default='-99')
    parser.add_argument('--testRun', dest='testRun', help='do test run with reduced number of events?',  action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    if not args.doHH and not args.doTenTau and not args.doSingleTau and not args.doNu and not args.doQCD and not args.doTrainValid:
        print('** WARNING: no matching dataset specified. What do you want to do (doHH, doTenTau, doSingleTau, doNu, doQCD, doTrainValid)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    if args.ptcut == '-99':
        print('** INFO: no baseline pT cut specified to be used for training and validation')
        print('** INFO: using default cut pT>4GeV')
        args.ptcut = '4'

    if args.doTrainValid:
        args.doTenTau = True
        args.doSingleTau = True
        args.doHH = True
        args.doZprime = True
        args.doVBFH = True
        args.doNu = True
        args.doQCD = True

    ##################### DEFINE HANDLING DICTIONARIES ####################

    # define the input and output dictionaries for the handling of different datasets
    indir   = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_2021_05_12'
    indir2  = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_2021_10_11'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    os.system('mkdir -p '+outdir)

    # set output to go both to terminal and to file
    sys.stdout = Logger("/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched/info.log")
    print('** INFO: using baseline cut pT>{0}GeV'.format(args.ptcut))

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'mixed'        : 'Mixed BC+STC',  
    }

    if args.doHH:
        inFileHH_dict = {
            'threshold'    : indir+'/SKIM_GluGluHHTo2b2Tau_PU200/mergedOutput.root',
            'mixed'        : indir+'/'
        }        

        outFileHH_dict = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileHH_towers = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doTenTau:
        inFileTenTau_dict = {
            'threshold'    : indir+'/SKIM_RelValTenTau_PU200/mergedOutput.root',
            'mixed'        : indir+'/'
        }

        outFileTenTau_dict = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileTenTau_towers = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doSingleTau:
        inFileSingleTau_dict = {
            'threshold'    : indir+'/SKIM_RelValSingleTau_PU200/mergedOutput.root',
            'mixed'        : indir+'/'
        }

        outFileSingleTau_dict = {
            'threshold'    : outdir+'/RelValSingleTau_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileSingleTau_towers = {
            'threshold'    : outdir+'/RelValSingleTau_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doNu:
        inFileNu_dict = {
            'threshold'    : indir+'/SKIM_RelValNu_PU200/mergedOutput.root',
            'mixed'        : indir+'/'
        }   

        outFileNu_dict = {
            'threshold'    : outdir+'/RelValNu_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileNu_towers = {
            'threshold'    : outdir+'/RelValNu_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }
 
    if args.doQCD:
        inFileQCD_dict = {
            'threshold'    : indir+'/SKIM_RelValQCD_PU200/mergedOutput.root',
            'mixed'        : indir+'/'
        }

        outFileQCD_dict = {
            'threshold'    : outdir+'/QCD_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileQCD_towers = {
            'threshold'    : outdir+'/QCD_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doZprime:
        inFileZprime_dict = {
            'threshold'    : indir2+'/SKIM_ZprimeTauTau_PU200/mergedOutput.root',
            'mixed'        : indir2+'/'
        }

        outFileZprime_dict = {
            'threshold'    : outdir+'/ZprimeToTauTau_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileZprime_towers = {
            'threshold'    : outdir+'/ZprimeToTauTau_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doVBFH:
        inFileVBFH_dict = {
            'threshold'    : indir2+'/SKIM_VBFHToTauTau_PU200/mergedOutput.root',
            'mixed'        : indir2+'/'
        }

        outFileVBFH_dict = {
            'threshold'    : outdir+'/VBFHToTauTau_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileVBFH_towers = {
            'threshold'    : outdir+'/VBFHToTauTau_PU200_th_towers.hdf5',
            'mixed'        : outdir+'/'
        }

    if args.doTrainValid:
        outFileTraining_dict = {
            'threshold'    : outdir+'/Training_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }

        outFileValidation_dict = {
            'threshold'    : outdir+'/Validation_PU200_th_matched.hdf5',
            'mixed'        : outdir+'/'
        }


    ##################### READ TTREES AND MATCH EVENTS ####################

    # TTree to be read
    treename = 'HGCALskimmedTree'
    #TBranches to be stored containing the 3D clusters' info
    branches_event_cl3d = ['event','cl3d_pt','cl3d_eta','cl3d_phi', 'cl3d_energy', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz', 'cl3d_srrtot','cl3d_srrmax','cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    branches_cl3d       = ['cl3d_pt','cl3d_eta','cl3d_phi', 'cl3d_energy', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz', 'cl3d_srrtot','cl3d_srrmax','cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    # TBranches to be stored containing the gen taus' info
    branches_event_gentau = ['event', 'gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    branches_gentau       = ['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    # TBranches to be stored containing the gen jets' info
    branches_event_genjet = ['event', 'genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    branches_genjet       = ['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    #TBranches to be stored containing the towers' info
    branches_event_towers = ['event','tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']
    branches_towers       = ['tower_pt','tower_eta','tower_phi', 'tower_energy' , 'tower_etEm', 'tower_etHad']
    
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end option that we do not want to do

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting cluster matching for the front-end option '+feNames_dict[name])

        if args.doHH:
            # define delta eta, delta phi, amd delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.1

            # fill the dataframes with the needed info from the branches defined above for the HH sample
            print('\n** INFO: creating dataframes for ' + inFileHH_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileHH_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_gentau = root_pandas.read_root(inFileHH_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            dfHH_towers = root_pandas.read_root(inFileHH_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_gentau.drop('__array_index', inplace=True, axis=1)
            dfHH_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<100', inplace=True)
                df_gentau.query('event<100', inplace=True)
                dfHH_towers.query('event<100', inplace=True)

            print('** INFO: matching gentaus for ' + inFileHH_dict[name])
            dfHH = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
            dfHH['sgnId'] = dfHH.apply(lambda row: prepareCat(row), axis=1)
            dfHH['cl3d_abseta'] = np.abs(dfHH['cl3d_eta'])
            dfHH.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfHH['event'] = 'hh' + dfHH['event'].astype(str)
            dfHH.set_index('event', inplace=True)
            dfHH.sort_values('event', inplace=True)
            dfHH_towers['event'] = 'hh' + dfHH_towers['event'].astype(str)
            dfHH_towers.set_index('event', inplace=True)
            dfHH_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileHH_dict[name])
            store_hh = pd.HDFStore(outFileHH_dict[name], mode='w')
            store_hh[name] = dfHH
            store_hh.close()

            print('** INFO: saving file ' + outFileHH_towers[args.FE])
            store_towers = pd.HDFStore(outFileHH_towers[args.FE], mode='w')
            store_towers[args.FE] = dfHH_towers
            store_towers.close()

            del df_cl3d, df_gentau, dfHH_towers

        if args.doTenTau:
            # define delta eta, delta phi, amd delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.1

            # fill the dataframes with the needed info from the branches defined above for the TENTAU
            print('\n** INFO: creating dataframes for ' + inFileTenTau_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileTenTau_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_gentau = root_pandas.read_root(inFileTenTau_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            dfTenTau_towers = root_pandas.read_root(inFileTenTau_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_gentau.drop('__array_index', inplace=True, axis=1)
            dfTenTau_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<50', inplace=True)
                df_gentau.query('event<50', inplace=True)
                dfTenTau_towers.query('event<50', inplace=True)

            print('** INFO: matching gentaus for ' + inFileTenTau_dict[name])
            dfTenTau = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
            dfTenTau['sgnId'] = dfTenTau.apply(lambda row: prepareCat(row), axis=1)
            dfTenTau['cl3d_abseta'] = np.abs(dfTenTau['cl3d_eta'])
            dfTenTau.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfTenTau['event'] = 'tentau' + dfTenTau['event'].astype(str)
            dfTenTau.set_index('event', inplace=True)
            dfTenTau.sort_values('event', inplace=True)
            dfTenTau_towers['event'] = 'tentau' + dfTenTau_towers['event'].astype(str)
            dfTenTau_towers.set_index('event', inplace=True)
            dfTenTau_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileTenTau_dict[name])
            store_tau = pd.HDFStore(outFileTenTau_dict[name], mode='w')
            store_tau[name] = dfTenTau
            store_tau.close()

            print('** INFO: saving file ' + outFileTenTau_towers[args.FE])
            store_towers = pd.HDFStore(outFileTenTau_towers[args.FE], mode='w')
            store_towers[args.FE] = dfTenTau_towers
            store_towers.close()

            del df_cl3d, df_gentau, dfTenTau_towers

        if args.doSingleTau:
            # define delta eta, delta phi, amd delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.1

            # fill the dataframes with the needed info from the branches defined above for the SingleTau
            print('\n** INFO: creating dataframes for ' + inFileSingleTau_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileSingleTau_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_gentau = root_pandas.read_root(inFileSingleTau_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            dfSingleTau_towers = root_pandas.read_root(inFileSingleTau_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_gentau.drop('__array_index', inplace=True, axis=1)
            dfSingleTau_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<100', inplace=True)
                df_gentau.query('event<100', inplace=True)
                dfSingleTau_towers.query('event<100', inplace=True)

            print('** INFO: matching gentaus for ' + inFileSingleTau_dict[name])
            dfSingleTau = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
            dfSingleTau['sgnId'] = dfSingleTau.apply(lambda row: prepareCat(row), axis=1)
            dfSingleTau['cl3d_abseta'] = np.abs(dfSingleTau['cl3d_eta'])
            dfSingleTau.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfSingleTau['event'] = 'singletau' + dfSingleTau['event'].astype(str)
            dfSingleTau.set_index('event', inplace=True)
            dfSingleTau.sort_values('event', inplace=True)
            dfSingleTau_towers['event'] = 'singletau' + dfSingleTau_towers['event'].astype(str)
            dfSingleTau_towers.set_index('event', inplace=True)
            dfSingleTau_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileSingleTau_dict[name])
            store_tau = pd.HDFStore(outFileSingleTau_dict[name], mode='w')
            store_tau[name] = dfSingleTau
            store_tau.close()

            print('** INFO: saving file ' + outFileSingleTau_towers[args.FE])
            store_towers = pd.HDFStore(outFileSingleTau_towers[args.FE], mode='w')
            store_towers[args.FE] = dfSingleTau_towers
            store_towers.close()

            del df_cl3d, df_gentau, dfSingleTau_towers

        if args.doVBFH:
            # define delta eta, delta phi, amd delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.1

            # fill the dataframes with the needed info from the branches defined above for the VBFH
            print('\n** INFO: creating dataframes for ' + inFileVBFH_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileVBFH_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_gentau = root_pandas.read_root(inFileVBFH_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            dfVBFH_towers = root_pandas.read_root(inFileVBFH_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_gentau.drop('__array_index', inplace=True, axis=1)
            dfVBFH_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<100', inplace=True)
                df_gentau.query('event<100', inplace=True)
                dfVBFH_towers.query('event<100', inplace=True)

            print('** INFO: matching gentaus for ' + inFileVBFH_dict[name])
            dfVBFH = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
            dfVBFH['sgnId'] = dfVBFH.apply(lambda row: prepareCat(row), axis=1)
            #dfVBFH.query('sgnId==1', inplace=True)
            dfVBFH['cl3d_abseta'] = np.abs(dfVBFH['cl3d_eta'])
            dfVBFH.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfVBFH['event'] = 'vbfh' + dfVBFH['event'].astype(str)
            dfVBFH.set_index('event', inplace=True)
            dfVBFH.sort_values('event', inplace=True)
            dfVBFH_towers['event'] = 'vbfh' + dfVBFH_towers['event'].astype(str)
            dfVBFH_towers.set_index('event', inplace=True)
            dfVBFH_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileVBFH_dict[name])
            store_tau = pd.HDFStore(outFileVBFH_dict[name], mode='w')
            store_tau[name] = dfVBFH
            store_tau.close()

            print('** INFO: saving file ' + outFileVBFH_towers[args.FE])
            store_towers = pd.HDFStore(outFileVBFH_towers[args.FE], mode='w')
            store_towers[args.FE] = dfVBFH_towers
            store_towers.close()

            del df_cl3d, df_gentau, dfVBFH_towers

        if args.doZprime:
            # define delta eta, delta phi, amd delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.1

            # fill the dataframes with the needed info from the branches defined above for the Zprime
            print('\n** INFO: creating dataframes for ' + inFileZprime_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileZprime_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_gentau = root_pandas.read_root(inFileZprime_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            dfZprime_towers = root_pandas.read_root(inFileZprime_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_gentau.drop('__array_index', inplace=True, axis=1)
            dfZprime_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<100', inplace=True)
                df_gentau.query('event<100', inplace=True)
                dfZprime_towers.query('event<100', inplace=True)

            print('** INFO: matching gentaus for ' + inFileZprime_dict[name])
            dfZprime = taumatching(df_cl3d, df_gentau, deta_matching, dphi_matching, dr_matching)
            dfZprime['sgnId'] = dfZprime.apply(lambda row: prepareCat(row), axis=1)
            #dfZprime.query('sgnId==1', inplace=True)
            dfZprime['cl3d_abseta'] = np.abs(dfZprime['cl3d_eta'])
            dfZprime.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfZprime['event'] = 'zprime' + dfZprime['event'].astype(str)
            dfZprime.set_index('event', inplace=True)
            dfZprime.sort_values('event', inplace=True)
            dfZprime_towers['event'] = 'zprime' + dfZprime_towers['event'].astype(str)
            dfZprime_towers.set_index('event', inplace=True)
            dfZprime_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileZprime_dict[name])
            store_tau = pd.HDFStore(outFileZprime_dict[name], mode='w')
            store_tau[name] = dfZprime
            store_tau.close()

            print('** INFO: saving file ' + outFileZprime_towers[args.FE])
            store_towers = pd.HDFStore(outFileZprime_towers[args.FE], mode='w')
            store_towers[args.FE] = dfZprime_towers
            store_towers.close()

            del df_cl3d, df_gentau, dfZprime_towers

        if args.doQCD:
            # define delta eta, delta phi, and delta R to be used for the matching
            deta_matching = 0.1
            dphi_matching = 0.2
            dr_matching = 0.4

            # fill the dataframes with the needed info from the branches defined above for the QCD
            print('\n** INFO: creating dataframes for ' + inFileQCD_dict[name]) 
            df_cl3d = root_pandas.read_root(inFileQCD_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_genjet = root_pandas.read_root(inFileQCD_dict[name], key=treename, columns=branches_event_genjet, flatten=branches_genjet)
            dfQCD_towers = root_pandas.read_root(inFileQCD_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            df_cl3d.drop('__array_index', inplace=True, axis=1)
            df_genjet.drop('__array_index', inplace=True, axis=1)
            dfQCD_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                df_cl3d.query('event<30', inplace=True)
                df_genjet.query('event<30', inplace=True)
                dfQCD_towers.query('event<30', inplace=True)

            print('** INFO: matching gentaus for ' + inFileQCD_dict[name])
            dfQCD = jetmatching(df_cl3d, df_genjet, deta_matching, dphi_matching, dr_matching)
            dfQCD['gentau_decayMode'] = -2 # tag as QCD
            dfQCD['sgnId'] = 0
            dfQCD['cl3d_abseta'] = np.abs(dfQCD['cl3d_eta'])
            dfQCD.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut

            # here we need to make sure we are not overlapping the 'events' columns
            dfQCD['event'] = 'qcd' + dfQCD['event'].astype(str)
            dfQCD.set_index('event', inplace=True)
            dfQCD.sort_values('event', inplace=True)
            dfQCD_towers['event'] = 'qcd' + dfQCD_towers['event'].astype(str)
            dfQCD_towers.set_index('event', inplace=True)
            dfQCD_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileQCD_dict[name])
            store_qcd = pd.HDFStore(outFileQCD_dict[name], mode='w')
            store_qcd[name] = dfQCD
            store_qcd.close()

            print('** INFO: saving file ' + outFileQCD_towers[args.FE])
            store_towers = pd.HDFStore(outFileQCD_towers[args.FE], mode='w')
            store_towers[args.FE] = dfQCD_towers
            store_towers.close()

            del df_cl3d, df_genjet, dfQCD_towers

        if args.doNu:
            # fill the dataframes with the needed info from the branches defined above for the NU
            print('\n** INFO: creating dataframes for ' + inFileNu_dict[name]) 
            dfNu = root_pandas.read_root(inFileNu_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            dfNu_towers = root_pandas.read_root(inFileNu_dict[args.FE], key=treename, columns=branches_event_towers, flatten=branches_towers)

            dfNu.drop('__array_index', inplace=True, axis=1)
            dfNu_towers.drop('__array_index', inplace=True, axis=1)

            if args.testRun:
                dfNu.query('event<100', inplace=True)
                dfNu_towers.query('event<100', inplace=True)

            dfNu['gentau_decayMode'] = -1 # tag as PU
            dfNu['geom_match'] = False
            dfNu['cl3d_isbestmatch'] = False
            dfNu['sgnId'] = 0
            dfNu['cl3d_abseta'] = np.abs(dfNu['cl3d_eta'])
            dfNu.query('cl3d_pt>{0}'.format(args.ptcut), inplace=True) # apply baseline pT cut
            
            # here we need to make sure we are not overlapping the 'events' columns
            dfNu['event'] = 'nu' + dfNu['event'].astype(str)
            dfNu.set_index('event', inplace=True)
            dfNu.sort_values('event', inplace=True)
            dfNu_towers['event'] = 'nu' + dfNu_towers['event'].astype(str)
            dfNu_towers.set_index('event', inplace=True)
            dfNu_towers.sort_values('event', inplace=True)

            print('** INFO: saving file ' + outFileNu_dict[name])
            store_nu = pd.HDFStore(outFileNu_dict[name], mode='w')
            store_nu[name] = dfNu
            store_nu.close()

            print('** INFO: saving file ' + outFileNu_towers[args.FE])
            store_towers = pd.HDFStore(outFileNu_towers[args.FE], mode='w')
            store_towers[args.FE] = dfNu_towers
            store_towers.close()

            del dfNu_towers

        if args.doTrainValid:
            print('\n** INFO: creating dataframes for Training/Validation')
            print('** INFO: RelValTenTau Training/Validation')
            dfTenTauTraining, dfTenTauValidation = train_test_split(dfTenTau, test_size=0.3)

            print('** INFO: RelValSingleTau Training/Validation')
            dfSingleTauTraining, dfSingleTauValidation = train_test_split(dfSingleTau, test_size=0.3)

            print('** INFO: HH Training/Validation')
            dfHHTraining, dfHHValidation = train_test_split(dfHH, test_size=0.3)

            print('** INFO: VBFH Training/Validation')
            dfVBFHTraining, dfVBFHValidation = train_test_split(dfVBFH, test_size=0.3)

            print('** INFO: Zprime Training/Validation')
            dZprimeTraining, dZprimeValidation = train_test_split(dfZprime, test_size=0.3)

            print('** INFO: QCD Training/Validation')
            dfQCDTraining, dfQCDValidation = train_test_split(dfQCD, test_size=0.3)

            print('** INFO: RelValNu Training/Validation')
            dfNuTraining, dfNuValidation = train_test_split(dfNu, test_size=0.3)

            # MERGE
            print('** INFO: merging Training/Validation')
            dfMergedTraining = pd.concat([dfTenTauTraining,dfSingleTauTraining,dfHHTraining,dfVBFHTraining,dZprimeTraining,dfNuTraining,dfQCDTraining],sort=False)
            dfMergedValidation = pd.concat([dfTenTauValidation,dfSingleTauValidation,dfHHValidation,dfVBFHValidation,dZprimeValidation,dfNuValidation,dfQCDValidation],sort=False)
            dfMergedTraining.fillna(0.0, inplace=True)
            dfMergedValidation.fillna(0.0, inplace=True)

            # BIN PT AND ETA OF THE GENTAUS
            ptcut = 1
            etamin = 1.6
            pt_binwidth = 3
            eta_binwidth = 0.1
            dfMergedTraining['gentau_vis_abseta'] = np.abs(dfMergedTraining['gentau_vis_eta'])
            dfMergedValidation['gentau_vis_abseta'] = np.abs(dfMergedValidation['gentau_vis_eta'])
            dfMergedTraining['gentau_bin_eta'] = ((dfMergedTraining['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
            dfMergedTraining['gentau_bin_pt']  = ((dfMergedTraining['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')
            dfMergedValidation['gentau_bin_eta'] = ((dfMergedValidation['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
            dfMergedValidation['gentau_bin_pt']  = ((dfMergedValidation['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')

            # here we apply minimal requirements so that we are working in a fiducial region slightly smaller than the full HGCAL acceptance
            # we apply the OR between tau, jet, and cluster requirements
            dfMergedTraining.query('(cl3d_abseta>1.6 and cl3d_abseta<2.9) or ((gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9))) or (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9))))', inplace=True)
            dfMergedValidation.query('(cl3d_abseta>1.6 and cl3d_abseta<2.9) or ((gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9))) or (genjet_pt>20 and ((genjet_eta>1.6 and genjet_eta<2.9) or (genjet_eta<-1.6 and genjet_eta>-2.9))))', inplace=True)

            # minimal queries to reject weird lines
            # the first three 3 requirements actually always select the same clusters
            dfMergedTraining.query('cl3d_showerlength>0 and cl3d_firstlayer<=50 and cl3d_meanz>250 and cl3d_hoe>=0', inplace=True)
            dfMergedValidation.query('cl3d_showerlength>0 and cl3d_firstlayer<=50 and cl3d_meanz>250 and cl3d_hoe>=0', inplace=True)

            # SAVE
            print('** INFO: saving file ' + outFileTraining_dict[name])
            store_tr = pd.HDFStore(outFileTraining_dict[name], mode='w')
            store_tr[name] = dfMergedTraining
            store_tr.close()

            print('** INFO: saving file ' + outFileValidation_dict[name])
            store_val = pd.HDFStore(outFileValidation_dict[name], mode='w')
            store_val[name] = dfMergedValidation
            store_val.close()

        # delete variables before next iteration with different FE option
        if args.doHH: del dfHH
        if args.doQCD: del dfQCD
        if args.doTenTau: del dfTenTau
        if args.doSingleTau: del dfSingleTau
        if args.doNu: del dfNu
        if args.doTrainValid: del dfMergedTraining, dfMergedValidation, dfHHTraining, dfHHValidation, dfTenTauTraining, dfTenTauValidation, dfSingleTauTraining, dfSingleTauValidation, dfVBFHTraining, dZprimeTraining, dfVBFHValidation, dZprimeValidation, dfQCDTraining, dfQCDValidation, dfNuTraining, dfNuValidation

        print('\n** INFO: finished cluster matching for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')

# restore normal output
sys.stdout = sys.__stdout__
