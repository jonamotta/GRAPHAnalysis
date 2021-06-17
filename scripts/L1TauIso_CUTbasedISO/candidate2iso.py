import os
import numpy as np
import pandas as pd
import argparse
import time


def deltar2cluster ( df ):
    delta_eta = np.abs(df['cl3d_eta']-df['cl3d_eta_ass'])
    delta_phi = np.abs(df['cl3d_phi']-df['cl3d_phi_ass'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def L1Cl3dEtIso ( dfL1Candidates, dfL1associated2Candidates, dR, weighted=False ):
    df_joined  = dfL1Candidates.join(dfL1associated2Candidates, on='event', how='left', rsuffix='_ass', sort=False)

    df_joined['deltar2cluster'] = deltar2cluster(df_joined)
    sel = (df_joined['deltar2cluster'] <= dR) & (df_joined['deltar2cluster'] > 0.0001)
    df_joined = df_joined[sel].copy(deep=True)

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index(['event', 'cl3d_pt_c3'], inplace=True)
    
    dfL1Candidates['cl3d_etIso_dR{0}'.format(int(dR*10))] = df_joined.groupby(['event', 'cl3d_pt_c3'])['cl3d_pt_c3_ass'].sum()

    dfL1Candidates.reset_index(inplace=True)
    dfL1Candidates.set_index('event',inplace=True)
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

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
    dfL1Candidates.fillna(0.0,inplace=True)

    del df_joined

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
    parser.add_argument('--doAll', dest='doAll', help='match all the samples',  action='store_true', default=False)
    parser.add_argument('--doMinbias', dest='doMinbias', help='match the Minbias samples?',  action='store_true', default=False)
    parser.add_argument('--doNu', dest='doNu', help='match the Nu samples?',  action='store_true', default=False)
    parser.add_argument('--testRun', dest='testRun', help='do test run with reduced number of events?',  action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    start = time.time()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    if args.doAll:
        args.doAllTau = True
        args.doQCD = True
        args.doMinbias = True
        args.doNu = True

    if args.doAllTau:
        args.doSingleTau = True
        args.doTenTau = True
        args.doHH = True

    dRsgn = 0.2
    dRiso_list = [0.3,0.4, 0.5, 0.6]

    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolation_application_dRsgn{0}_test/'.format(int(dRsgn*10))
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolation_application_dRsgn{0}_test/'.format(int(dRsgn*10))
    os.system('mkdir -p '+outdir)

    # DICTIONARIES FOR THE ISOLATION 
    if args.doHH:
        inFileHH_DMsort = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileHH_towers = {
            'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileHH_iso = {
            'threshold'    : outdir+'/GluGluHHTo2b2Tau_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }  

    if args.doTenTau:
        inFileTenTau_DMsort = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileTenTau_towers = {
            'threshold'    : indir+'/RelValTenTau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileTenTau_iso = {
            'threshold'    : outdir+'/RelValTenTau_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doSingleTau:
        inFileSingleTau_DMsort = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileSingleTau_towers = {
            'threshold'    : indir+'/RelValSingleTau_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileSingleTau_iso = {
            'threshold'    : outdir+'/RelValSingleTau_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doQCD:
        inFileQCD_DMsort = {
            'threshold'    : indir+'/QCD_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileQCD_towers = {
            'threshold'    : indir+'/QCD_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileQCD_iso = {
            'threshold'    : outdir+'/QCD_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doMinbias:
        inFileMinbias_DMsort = {
            'threshold'    : indir+'/Minbias_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileMinbias_towers = {
            'threshold'    : indir+'/Minbias_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileMinbias_iso = {
            'threshold'    : outdir+'/Minbias_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    if args.doNu:
        inFileRelValNu_DMsort = {
            'threshold'    : indir+'/RelValNu_PU200_th_DMsorted.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        inFileRelValNu_towers = {
            'threshold'    : indir+'/RelValNu_PU200_th_towers.hdf5',
            'supertrigger' : indir+'/',
            'bestchoice'   : indir+'/',
            'bestcoarse'   : indir+'/',
            'mixed'        : indir+'/'
        }

        outFileRelValNu_iso = {
            'threshold'    : outdir+'/RelValNu_PU200_th_isolation.hdf5',
            'supertrigger' : outdir+'/',
            'bestchoice'   : outdir+'/',
            'bestcoarse'   : outdir+'/',
            'mixed'        : outdir+'/'
        }

    inFiles4Iso = []
    inFiles4Iso_towers = []
    outFiles4Iso = []

    if args.doHH:
        inFiles4Iso.append(inFileHH_DMsort)
        inFiles4Iso_towers.append(inFileHH_towers)
        outFiles4Iso.append(outFileHH_iso)
    if args.doTenTau:
        inFiles4Iso.append(inFileTenTau_DMsort)
        inFiles4Iso_towers.append(inFileTenTau_towers)
        outFiles4Iso.append(outFileTenTau_iso)
    if args.doSingleTau:
        inFiles4Iso.append(inFileSingleTau_DMsort)
        inFiles4Iso_towers.append(inFileSingleTau_towers)
        outFiles4Iso.append(outFileSingleTau_iso)
    if args.doQCD:
        inFiles4Iso.append(inFileQCD_DMsort)
        inFiles4Iso_towers.append(inFileQCD_towers)
        outFiles4Iso.append(outFileQCD_iso)
    if args.doMinbias:
        inFiles4Iso.append(inFileMinbias_DMsort)
        inFiles4Iso_towers.append(inFileMinbias_towers)
        outFiles4Iso.append(outFileMinbias_iso)
    if args.doNu:
        inFiles4Iso.append(inFileRelValNu_DMsort)
        inFiles4Iso_towers.append(inFileRelValNu_towers)
        outFiles4Iso.append(outFileRelValNu_iso)

    #toKeep = ['cl3d_isbestmatch', 'cl3d_pt_c3', 'cl3d_abseta', 'cl3d_eta', 'cl3d_phi', 'cl3d_pubdt_score', 'cl3d_pubdt_passWP99', 'cl3d_pubdt_passWP95', 'cl3d_pubdt_passWP90', 'cl3d_predDM', 'cl3d_probDM0', 'cl3d_probDM1', 'cl3d_probDM2', 'cl3d_probDM3']
    
    for k in range(len(outFiles4Iso)):

        print('---------------------------------------------------------------------------------------')
        print('** INFO: start dataframe creation for isolation studies for '+outFiles4Iso[k][args.FE])
        
        store = pd.HDFStore(inFiles4Iso[k][args.FE], mode='r')
        df4Iso = store[args.FE]
        store.close()

        store = pd.HDFStore(inFiles4Iso_towers[k][args.FE], mode='r')
        df4Iso_towers = store[args.FE]
        store.close()

        #df4Iso = df4Iso.loc[:,toKeep]

        if args.testRun:
            df4Iso = df4Iso.head(1).copy(deep=True)
            df4Iso_towers = df4Iso_towers.head(10).copy(deep=True)

        if ('Minbias' in outFiles4Iso[k][args.FE]) or ('Nu' in outFiles4Iso[k][args.FE]):
            print('\n** INFO: doing minbias/nu special L1 candidate selection')
            df4Iso.reset_index(inplace=True)
            dfL1Candidates = df4Iso.query('cl3d_pubdt_passWP99==True').copy(deep=True) # selecting WP 99 we select also 95 and 90
            dfL1Candidates.sort_values('cl3d_pt_c3', inplace=True)
            dfL1Candidates.drop_duplicates('event', keep='last', inplace=True) # keep only highest pt cluster
            sel = df4Iso['cl3d_pt_c3'].isin(dfL1Candidates['cl3d_pt_c3'])
            dfL1ass2cand = df4Iso.drop(df4Iso[sel].index)
            dfL1Candidates.set_index('event', inplace=True)
            dfL1Candidates.sort_values('event', inplace=True)
            dfL1ass2cand.set_index('event', inplace=True)
            dfL1ass2cand.sort_values('event', inplace=True)
        else:
            dfL1Candidates = df4Iso.query('cl3d_isbestmatch==True').copy(deep=True)
            dfL1ass2cand = df4Iso.query('cl3d_isbestmatch==False').copy(deep=True)

        # split the two endcaps to make the loops over the rows faster
        dfL1Candidates_p = dfL1Candidates.query('cl3d_eta>=0').copy(deep=True)
        dfL1Candidates_m = dfL1Candidates.query('cl3d_eta<0').copy(deep=True)
        dfL1ass2cand_p = dfL1ass2cand.query('cl3d_eta>=0').copy(deep=True)
        dfL1ass2cand_m = dfL1ass2cand.query('cl3d_eta<0').copy(deep=True)
        dfL1Towers_p = df4Iso_towers.query('tower_eta>=0').copy(deep=True)
        dfL1Towers_m = df4Iso_towers.query('tower_eta<0').copy(deep=True)

        print('\n** INFO: calculating eT for L1Tau candidates')
        for dRiso in dRiso_list:
            print('         calculating iso in cone of dR={0}'.format(dRiso))
            print('             positive z encap')
            L1Cl3dEtIso(dfL1Candidates_p, dfL1ass2cand_p, dRiso)
            L1Cl3dEtIso(dfL1Candidates_p, dfL1ass2cand_p, dRiso, weighted=True)
            L1TowerEtIso(dfL1Candidates_p, dfL1Towers_p, dRsgn, dRiso)
            print('             negative z encap')
            L1Cl3dEtIso(dfL1Candidates_m, dfL1ass2cand_m, dRiso)
            L1Cl3dEtIso(dfL1Candidates_m, dfL1ass2cand_m, dRiso, weighted=True)
            L1TowerEtIso(dfL1Candidates_m, dfL1Towers_m, dRsgn, dRiso)

        # put together again the two endcaps
        dfOut = pd.concat([dfL1Candidates_p, dfL1Candidates_m], sort=False)
        dfOut.sort_values('event')

        print('** INFO: saving file ' + outFiles4Iso[k][args.FE])
        store = pd.HDFStore(outFiles4Iso[k][args.FE], mode='w')
        store[args.FE] = dfOut
        store.close()

        print('\n** INFO: finished dataframe creation for isolation studies for '+outFiles4Iso[k][args.FE])
        print('---------------------------------------------------------------------------------------')

    end = time.time()
    print '\nRunning time = %02dh %02dm %02ds'%((end-start)/3600, ((end-start)%3600)/60, (end-start)% 60)
    