import os
import numpy as np
import pandas as pd
import root_pandas
import argparse


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
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1GNN/hdf5dataframes/skimLevel'
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
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th',
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
            'threshold'    : outdir+'/RelValTenTau_PU200_th',
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
            'threshold'    : outdir+'/RelValNu_PU200_th',
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
            'threshold'    : outdir+'/QCD_PU200_th',
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
    #TBranches to be stored containing the 3D TCs' info
    branches_tc = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_waferu', 'tc_waferv', 'tc_wafertype', 'tc_panel_number', 'tc_panel_sector', 'tc_cellu', 'tc_cellv', 'tc_data', 'tc_uncompressedCharge', 'tc_compressedCharge',  'tc_pt', 'tc_energy', 'tc_eta', 'tc_phi', 'tc_x', 'tc_y', 'tc_z', 'tc_mipPt', 'tc_cluster_id', 'tc_multiuclaster_id', 'tc_multicluster_pt']
    branches_tc_flatten = ['tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_waferu', 'tc_waferv', 'tc_wafertype', 'tc_panel_number', 'tc_panel_sector', 'tc_cellu', 'tc_cellv', 'tc_data', 'tc_uncompressedCharge', 'tc_compressedCharge',  'tc_pt', 'tc_energy', 'tc_eta', 'tc_phi', 'tc_x', 'tc_y', 'tc_z', 'tc_mipPt', 'tc_cluster_id', 'tc_multiuclaster_id', 'tc_multicluster_pt']
    #TBranches to be stored containing the 3D clusters' info
    branches_cl3d = ['event', 'cl3_id', 'cl3d_pt', 'cl3d_energy', 'cl3d_eta', 'cl3d_phi', 'cl3d_clusters_n', 'cl3d_showerlength', 'cl3d_coreshowerlength',  'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean',  'cled_emaxe', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90',  'cl3d_ntc67', 'cl3d_ntc90', 'cl3d_bdteg', 'cl3d_uality']
    branches_cl3d_flatten = ['cl3_id', 'cl3d_pt', 'cl3d_energy', 'cl3d_eta', 'cl3d_phi', 'cl3d_clusters_n', 'cl3d_showerlength', 'cl3d_coreshowerlength',  'cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean',  'cled_emaxe', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90',  'cl3d_ntc67', 'cl3d_ntc90', 'cl3d_bdteg', 'cl3d_uality']
    # TBranches to be stored containing the gen taus' info
    branches_gentau = ['event', 'gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    branches_gentau_flatten = ['gentau_pt', 'gentau_eta', 'gentau_phi', 'gentau_energy', 'gentau_mass', 'gentau_vis_pt', 'gentau_vis_eta', 'gentau_vis_phi', 'gentau_vis_energy', 'gentau_vis_mass', 'gentau_decayMode']
    # TBranches to be stored containing the gen jets' info
    branches_genjet = ['event', 'genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']
    branches_genjet_flatten = ['genjet_pt', 'genjet_eta', 'genjet_phi', 'genjet_energy', 'genjet_mass']


    
    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end option that we do not want to do

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting cluster matching for the front-end option '+feNames_dict[name])

        if args.doHH:
            # fill tau dataframes and dictionaries -> training 
            df_hh_cl3d = root_pandas.read_root(inFileHH_dict[FE], key=treename, columns=branches_cl3d, flatten=branches_cl3d_flatten)
            df_hh_gentau = root_pandas.read_root(inFileHH_dict[FE], key=treename, columns=branches_gentau, flatten=branches_gentau_flatten)
            df_hh_tc = root_pandas.read_root(inFileHH_dict[FE], key=treename, columns=branches_tc, flatten=branches_tc_flatten)

            store_hh_tc = pd.HDFStore(outFileHH_dict[FE]+'_TC.hdf5', mode='w')
            store_hh_tc[FE] = df_hh_tc
            store_hh_tc.close()

            store_hh_cl3d = pd.HDFStore(outFileHH_dict[FE]+'_CL3D.hdf5', mode='w')
            store_hh_cl3d[FE] = df_hh_cl3d
            store_hh_cl3d.close()

            store_hh_gen = pd.HDFStore(outFileHH_dict[FE]+'_GEN.hdf5', mode='w')
            store_hh_gen[FE] = df_hh_gen
            store_hh_gen.close()

        if args.doQCD:
            # fill tau dataframes and dictionaries -> training 
            df_qcd_cl3d = root_pandas.read_root(inFileQCD_dict[FE], key=treename, columns=branches_cl3d, flatten=branches_cl3d_flatten)
            df_qcd_gen = root_pandas.read_root(inFileQCD_dict[FE], key=treename, columns=branches_genjet, flatten=branches_genjet_flatten)
            df_qcd_tc = root_pandas.read_root(inFileQCD_dict[FE], key=treename, columns=branches_tc, flatten=branches_tc_flatten)

            store_qcd_tc = pd.HDFStore(outFileQCD_dict[FE]+'_TC.hdf5', mode='w')
            store_qcd_tc[FE] = df_qcd_tc
            store_qcd_tc.close()

            store_qcd_cl3d = pd.HDFStore(outFileQCD_dict[FE]+'_CL3D.hdf5', mode='w')
            store_qcd_cl3d[FE] = df_qcd_cl3d
            store_qcd_cl3d.close()

            store_qcd_gen = pd.HDFStore(outFileQCD_dict[FE]+'_GEN.hdf5', mode='w')
            store_qcd_gen[FE] = df_qcd_gen
            store_qcd_gen.close()

        if args.doTau:
            # fill tau dataframes and dictionaries -> training 
            df_tau_cl3d = root_pandas.read_root(inFileTau_dict[FE], key=treename, columns=branches_cl3d, flatten=branches_cl3d_flatten)
            df_tau_gen = root_pandas.read_root(inFileTau_dict[FE], key=treename, columns=branches_gentau, flatten=branches_gentau_flatten)
            df_tau_tc = root_pandas.read_root(inFileTau_dict[FE], key=treename, columns=branches_tc, flatten=branches_tc_flatten)

            store_tau_tc = pd.HDFStore(outFileTau_dict[FE]+'_TC.hdf5', mode='w')
            store_tau_tc[FE] = df_tau_tc
            store_tau_tc.close()

            store_tau_cl3d = pd.HDFStore(outFileTau_dict[FE]+'_CL3D.hdf5', mode='w')
            store_tau_cl3d[FE] = df_tau_cl3d
            store_tau_cl3d.close()

            store_tau_gen = pd.HDFStore(outFileTau_dict[FE]+'_GEN.hdf5', mode='w')
            store_tau_gen[FE] = df_tau_gen
            store_tau_gen.close()

        if args.doNu:
            # fill tau dataframes and dictionaries -> training 
            df_nu_cl3d = root_pandas.read_root(inFileNu_dict[FE], key=treename, columns=branches_cl3d, flatten=branches_cl3d_flatten)
            df_nu_tc = root_pandas.read_root(inFileNu_dict[FE], key=treename, columns=branches_tc, flatten=branches_tc_flatten)

            store_nu_tc = pd.HDFStore(outFileNu_dict[FE]+'_TC.hdf5', mode='w')
            store_nu_tc[FE] = df_nu_tc
            store_nu_tc.close()

            store_nu_cl3d = pd.HDFStore(outFileNu_dict[FE]+'_CL3D.hdf5', mode='w')
            store_nu_cl3d[FE] = df_nu_cl3d
            store_nu_cl3d.close()


        print('\n** INFO: finished cluster matching for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        


























