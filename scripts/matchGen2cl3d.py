import os
import numpy as np
import pandas as pd
import root_pandas
import argparse


# define delta eta and delta phi to be used for the matchong
deta_matching = 0.1
dphi_matching = 0.2

def deltar( df, isTau ):
    
    if isTau:
        df['deta_cl3d_gentau'] = df['cl3d_eta'] - df['gentau_vis_eta']
        df['dphi_cl3d_gentau'] = np.abs(df['cl3d_phi'] - df['gentau_vis_phi'])
        sel = df['dphi_cl3d_gentau'] > np.pi
        df['dphi_cl3d_gentau'] -= sel*(2*np.pi)
        return ( np.sqrt(df['dphi_cl3d_gentau']*df['dphi_cl3d_gentau']+df['deta_cl3d_gentau']*df['deta_cl3d_gentau']) )
    else:
        df['deta_cl3d_genjet'] = df['cl3d_eta'] - df['genjet_eta']
        df['dphi_cl3d_genjet'] = np.abs(df['cl3d_phi'] - df['genjet_phi'])
        sel = df['dphi_cl3d_genjet'] > np.pi
        df['dphi_cl3d_genjet'] -= sel*(2*np.pi)
        return ( np.sqrt(df['dphi_cl3d_genjet']*df['dphi_cl3d_genjet']+df['deta_cl3d_genjet']*df['deta_cl3d_genjet']) )
    

def taumatching( df_cl3d, df_gen, deta, dphi ):
    df_cl3d_plus    = df_cl3d.query('cl3d_eta>0')
    df_cl3d_minus   = df_cl3d.query('cl3d_eta<=0')

    df_gen_plus     = df_gen.query('gentau_vis_eta>0')  
    df_gen_minus    = df_gen.query('gentau_vis_eta<=0')

    df_cl3d_plus.set_index('event', inplace=True)
    df_cl3d_minus.set_index('event', inplace=True)

    df_gen_plus.set_index('event', inplace=True)
    df_gen_minus.set_index('event', inplace=True)

    df_merged_plus  = df_gen_plus.join(df_cl3d_plus, how='left', rsuffix='_cl3d')
    df_merged_minus = df_gen_minus.join(df_cl3d_minus, how='left', rsuffix='_cl3d')

    df_merged_plus['deltar_cl3d_gentau']    = deltar(df_merged_plus, isTau=True)
    df_merged_minus['deltar_cl3d_gentau']   = deltar(df_merged_minus, isTau=True)

    sel_plus = np.abs(df_merged_plus['deta_cl3d_gentau']) < (deta/2)
    sel_minus = np.abs(df_merged_minus['deta_cl3d_gentau']) < (deta/2)

    df_merged_plus  = df_merged_plus[sel_plus]
    df_merged_minus = df_merged_minus[sel_minus]

    sel_plus    = np.abs(df_merged_plus['dphi_cl3d_gentau']) < (dphi/2)
    sel_minus   = np.abs(df_merged_minus['dphi_cl3d_gentau']) < (dphi/2)

    df_merged_plus  = df_merged_plus[sel_plus]
    df_merged_minus = df_merged_minus[sel_minus]

    group_plus  = df_merged_plus.groupby('event')
    group_minus = df_merged_minus.groupby('event')

    n_cl3d_plus     = group_plus['cl3d_pt'].size()
    n_cl3d_minus    = group_minus['cl3d_pt'].size()

    df_merged_plus['n_matched_cl3d']    = n_cl3d_plus
    df_merged_minus['n_matched_cl3d']   = n_cl3d_minus

    df_merged_plus['bestmatch_pt']  = group_plus['cl3d_pt'].max()
    df_merged_minus['bestmatch_pt'] = group_minus['cl3d_pt'].max()

    df_merged_plus['cl3d_isbestmatch']  = df_merged_plus['bestmatch_pt'] == df_merged_plus['cl3d_pt']
    df_merged_minus['cl3d_isbestmatch'] = df_merged_minus['bestmatch_pt'] == df_merged_minus['cl3d_pt']

    df_merged = pd.concat([df_merged_plus, df_merged_minus], sort=False).sort_values('event')

    return df_merged

def jetmatching( df_cl3d, df_gen, deta, dphi ):
    df_cl3d_plus    = df_cl3d.query('cl3d_eta>0')
    df_cl3d_minus   = df_cl3d.query('cl3d_eta<=0')

    df_gen_plus     = df_gen.query('genjet_eta>0')  
    df_gen_minus    = df_gen.query('genjet_eta<=0')

    df_cl3d_plus.set_index('event', inplace=True)
    df_cl3d_minus.set_index('event', inplace=True)

    df_gen_plus.set_index('event', inplace=True)
    df_gen_minus.set_index('event', inplace=True)

    df_merged_plus  = df_gen_plus.join(df_cl3d_plus, how='left', rsuffix='_cl3d')
    df_merged_minus = df_gen_minus.join(df_cl3d_minus, how='left', rsuffix='_cl3d')

    df_merged_plus['deltar_cl3d_genjet']    = deltar(df_merged_plus, isTau=False)
    df_merged_minus['deltar_cl3d_genjet']   = deltar(df_merged_minus, isTau=False)

    sel_plus = np.abs(df_merged_plus['deta_cl3d_genjet']) < (deta/2)
    sel_minus = np.abs(df_merged_minus['deta_cl3d_genjet']) < (deta/2)

    df_merged_plus  = df_merged_plus[sel_plus]
    df_merged_minus = df_merged_minus[sel_minus]

    sel_plus    = np.abs(df_merged_plus['dphi_cl3d_genjet']) < (dphi/2)
    sel_minus   = np.abs(df_merged_minus['dphi_cl3d_genjet']) < (dphi/2)

    df_merged_plus  = df_merged_plus[sel_plus]
    df_merged_minus = df_merged_minus[sel_minus]

    group_plus  = df_merged_plus.groupby('event')
    group_minus = df_merged_minus.groupby('event')

    n_cl3d_plus     = group_plus['cl3d_pt'].size()
    n_cl3d_minus    = group_minus['cl3d_pt'].size()

    df_merged_plus['n_matched_cl3d']    = n_cl3d_plus
    df_merged_minus['n_matched_cl3d']   = n_cl3d_minus

    df_merged_plus['bestmatch_pt']  = group_plus['cl3d_pt'].max()
    df_merged_minus['bestmatch_pt'] = group_minus['cl3d_pt'].max()

    df_merged_plus['cl3d_isbestmatch']  = df_merged_plus['bestmatch_pt'] == df_merged_plus['cl3d_pt']
    df_merged_minus['cl3d_isbestmatch'] = df_merged_minus['bestmatch_pt'] == df_merged_minus['cl3d_pt']

    df_merged = pd.concat([df_merged_plus, df_merged_minus], sort=False).sort_values('event')

    return df_merged


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doHH', dest='doHH', help='match the HH samples?',  action='store_true', default=False)
    parser.add_argument('--doTau', dest='doTau', help='match the Tau samples?',  action='store_true', default=False)
    parser.add_argument('--doNu', dest='doNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--doQCD', dest='doQCD', help='match the QCD samples?',  action='store_true', default=False)
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    # store parsed options
    args = parser.parse_args()

    if not args.doHH and not args.doTau and not args.doNu and not args.doQCD:
        print('** WARNING: no matching dataset specified. What do you want to do (doHH, doTau, doNu, doQCD)?')
        print('** EXITING')
        exit()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    ##################### DEFINE HANDLING DICTIONARIES ####################

    # define the input and output dictionaries for the handling of different datasets
    indir   = '/data_CMS_upgrade/motta/HGCAL_SKIMS/SKIM_18Feb2021'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    os.system('mkdir -p '+outdir)

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    if args.doHH:
        inFileHH_dict = {
            'threshold'    : indir+'/SKIM_GluGluHHTo2b2Tau_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }        

        outFileHH_dict = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doTau:
        inFileTau_dict = {
            'threshold'    : indir+'/SKIM_RelValTenTau_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileTau_dict = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doNu:
        inFileNu_dict = {
            'threshold'    : indir+'/SKIM_RelValNu_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }   

        outFileNu_dict = {
            'threshold'    : outdir+'/RelValNu_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }
 
    if args.doQCD:
        inFileQCD_dict = {
            'threshold'    : indir+'/SKIM_QCD_PU200/mergedOutput.root',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileQCD_dict = {
            'threshold'    : outdir+'/QCD_PU200_th_matched.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    # dictionaries used for saving skim level info
    dfHH_dict = {}
    dfQCD_dict = {}
    dfTau_dict = {}
    dfNu_dict = {}


    ##################### READ TTREES AND MATCH EVENTS ####################

    # TTree to be read
    treename = 'SkimmedTree'
    #TBranches to be stored containing the 3D clusters' info
    branches_event_cl3d = ['event','cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_spptot','cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    branches_cl3d       = ['cl3d_pt','cl3d_eta','cl3d_phi','cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_spptot','cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
    # TBranches to be stored containing the gen taus' info
    branches_event_gentau = ['event', 'gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    branches_gentau       = ['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    # TBranches to be stored containing the gen jets' info
    branches_event_genjet = ['event', 'genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    branches_genjet       = ['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']

    
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end option that we do not want to do

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting cluster matching for the front-end option '+feNames_dict[name])

        if args.doHH:
            # fill the dataframes with the needed info from the branches defined above for the HH sample
            print('\n** INFO: creating dataframes for ' + inFileHH_dict[name]) 
            df_hh_cl3d = root_pandas.read_root(inFileHH_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_hh_gentau = root_pandas.read_root(inFileHH_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            
            print('** INFO: matching gentaus for ' + inFileHH_dict[name])
            dfHH_dict[name] = taumatching(df_hh_cl3d, df_hh_gentau, deta_matching, dphi_matching)

            print('** INFO: saving file ' + outFileHH_dict[name])
            store_hh = pd.HDFStore(outFileHH_dict[name], mode='w')
            store_hh[name] = dfHH_dict[name]
            store_hh.close()

        if args.doQCD:
            # fill the dataframes with the needed info from the branches defined above for the BKG
            print('\n** INFO: creating dataframes for ' + inFileQCD_dict[name]) 
            df_qcd_cl3d = root_pandas.read_root(inFileQCD_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_qcd_genjet = root_pandas.read_root(inFileQCD_dict[name], key=treename, columns=branches_event_genjet, flatten=branches_genjet)
            
            print('** INFO: matching gentaus for ' + inFileQCD_dict[name])
            dfQCD_dict[name] = jetmatching(df_qcd_cl3d, df_qcd_genjet, deta_matching, dphi_matching)

            print('** INFO: saving file ' + outFileQCD_dict[name])
            store_qcd = pd.HDFStore(outFileQCD_dict[name], mode='w')
            store_qcd[name] = dfQCD_dict[name]
            store_qcd.close()

        if args.doTau:
            # fill the dataframes with the needed info from the branches defined above for the TENTAU
            print('\n** INFO: creating dataframes for ' + inFileTau_dict[name]) 
            df_tau_cl3d = root_pandas.read_root(inFileTau_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            df_tau_gentau = root_pandas.read_root(inFileTau_dict[name], key=treename, columns=branches_event_gentau, flatten=branches_gentau)
            
            print('** INFO: matching gentaus for ' + inFileTau_dict[name])
            dfTau_dict[name] = taumatching(df_tau_cl3d, df_tau_gentau, deta_matching, dphi_matching)

            print('** INFO: saving file ' + outFileTau_dict[name])
            store_tau = pd.HDFStore(outFileTau_dict[name], mode='w')
            store_tau[name] = dfTau_dict[name]
            store_tau.close()

        if args.doNu:
            # fill the dataframes with the needed info from the branches defined above for the NU
            print('\n** INFO: creating dataframes for ' + inFileNu_dict[name]) 
            df_nu_cl3d = root_pandas.read_root(inFileNu_dict[name], key=treename, columns=branches_event_cl3d, flatten=branches_cl3d)
            dfNu_dict[name] = df_nu_cl3d

            print('** INFO: saving file ' + outFileNu_dict[name])
            store_nu = pd.HDFStore(outFileNu_dict[name], mode='w')
            store_nu[name] = dfNu_dict[name]
            store_nu.close()

        print('\n** INFO: finished cluster matching for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        






















