import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import argparse


def PTcompressor( row, ptBins ):
    for i in range(len(ptBins)):
        if row['cl3d_pt_c3'] < ptBins[i]:
            compressedL1tauPt = i
            break
    return compressedL1tauPt

def ETAcompressor( row, etaBins ):
    for i in range(len(etaBins)):
        if row['cl3d_abseta'] < etaBins[i]:
            compressedL1tauEta = i
            break
    return compressedL1tauEta

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)

    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    dRsgn = 0.2
    list_dRiso = [0.2,0.3,0.4,0.5,0.6,0.7]

    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolation_application_dRsgn{0}_test/'.format(int(dRsgn*10))
    #plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/isolation/'
    #for dRiso in list_dRiso:
    #    os.system('mkdir -p '+plotdir+'/dRsgn{0}/dRiso{1}/iEta'.format(int(dRsgn*10),int(dRiso*10)))
    #    os.system('mkdir -p '+plotdir+'/dRsgn{0}/dRiso{1}/iPt'.format(int(dRsgn*10),int(dRiso*10)))
    #    os.system('mkdir -p '+plotdir+'/dRsgn{0}/dRiso{1}/iEtaPt'.format(int(dRsgn*10),int(dRiso*10)))

    inFileHH = {
        'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileTenTau = {
        'threshold'    : indir+'/RelValTenTau_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileSingleTau = {
        'threshold'    : indir+'/RelValSingleTau_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileQCD = {
        'threshold'    : indir+'/QCD_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileMinbias = {
        'threshold'    : indir+'/Minbias_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileRelValNu = {
        'threshold'    : indir+'/RelValNu_PU200_th_isolation.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileIso_tau = {
        'threshold'    : indir+'/AllTau_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileIso_QCD = {
        'threshold'    : indir+'/QCD_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileIso_minbias = {
        'threshold'    : indir+'/Minbias_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileIso_nu = {
        'threshold'    : indir+'/RelValNu_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    store = pd.HDFStore(inFileTenTau[args.FE], mode='r')
    dfL1_tentau = store[args.FE]
    store.close()

    store = pd.HDFStore(inFileSingleTau[args.FE], mode='r')
    dfL1_singletau = store[args.FE]
    store.close()

    store = pd.HDFStore(inFileHH[args.FE], mode='r')
    dfL1_HH = store[args.FE]
    store.close()

    store = pd.HDFStore(inFileQCD[args.FE], mode='r')
    dfL1_qcd = store[args.FE]
    store.close()

    #store = pd.HDFStore(inFileMinbias[args.FE], mode='r')
    #dfL1_minbias = store[args.FE]
    #store.close()

    #store = pd.HDFStore(inFileRelValNu[args.FE], mode='r')
    #dfL1_nu = store[args.FE]
    #store.close()

    dfL1Taus = pd.concat([dfL1_tentau,dfL1_singletau,dfL1_HH], sort=False)
    dfL1Jets = dfL1_qcd.copy(deep=True)
    #dfL1Minbias = dfL1_minbias.copy(deep=True)
    #dfL1Nu = dfL1_nu.copy(deep=True)

    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting eta and pT compression')

    etaBins = [1.0, 1.5, 1.9, 2.2, 2.4, 2.6, 2.8, 3.0, 4.0]
    ptBins  = [0, 15, 19, 25, 33, 45, 62, 88, 99999]
    NetaBins = len(etaBins)
    NptBins = len(ptBins)

    dfL1Taus['compressedEta'] = dfL1Taus.apply(lambda row : ETAcompressor(row, etaBins), axis=1)
    dfL1Taus['compressedPt']  = dfL1Taus.apply(lambda row : PTcompressor(row, ptBins), axis=1)
    dfL1Jets['compressedEta'] = dfL1Jets.apply(lambda row : ETAcompressor(row, etaBins), axis=1)
    dfL1Jets['compressedPt']  = dfL1Jets.apply(lambda row : PTcompressor(row, ptBins), axis=1)
    #dfL1Minbias['compressedEta'] = dfL1Minbias.apply(lambda row : ETAcompressor(row, etaBins), axis=1)
    #dfL1Minbias['compressedPt']  = dfL1Minbias.apply(lambda row : PTcompressor(row, ptBins), axis=1)
    #dfL1Nu['compressedEta'] = dfL1Nu.apply(lambda row : ETAcompressor(row, etaBins), axis=1)
    #dfL1Nu['compressedPt']  = dfL1Nu.apply(lambda row : PTcompressor(row, ptBins), axis=1)

    print('** INFO: saving file ' + outFileIso_tau[args.FE])
    store = pd.HDFStore(outFileIso_tau[args.FE], mode='w')
    store[args.FE] = dfL1Taus
    store.close()

    print('** INFO: saving file ' + outFileIso_QCD[args.FE])
    store = pd.HDFStore(outFileIso_QCD[args.FE], mode='w')
    store[args.FE] = dfL1Jets
    store.close() 

    #print('** INFO: saving file ' + outFileIso_minbias[args.FE])
    #store = pd.HDFStore(outFileIso_minbias[args.FE], mode='w')
    #store[args.FE] = dfL1Minbias
    #store.close()

    #print('** INFO: saving file ' + outFileIso_nu[args.FE])
    #store = pd.HDFStore(outFileIso_nu[args.FE], mode='w')
    #store[args.FE] = dfL1Nu
    #store.close() 

    print('** INFO: finished eta and pT compression')
    print('---------------------------------------------------------------------------------------')
    
'''
    print('---------------------------------------------------------------------------------------')
    print('** INFO: starting eta and pT dependent etIso plots')

    plot_bins = np.arange(0,250,5)
    
    for dRiso in list_dRiso:
        print('dRiso{0}'.format(dRiso))
        
        plt.figure(figsize=(18,8))
        plt.hist(dfL1Taus['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 $\tau$', color='blue', histtype='step', lw=2, density=1, bins=plot_bins)
        plt.hist(dfL1Jets['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 QCD jet', color='red', histtype='step', lw=2, density=1, bins=plot_bins)
        plt.legend(loc = 'upper right', fontsize=15)
        plt.grid(linestyle=':')
        plt.xlabel(r'$E_{T}^{Iso}$')
        plt.ylabel(r'$a.u.$')
        plt.title(r'Inclusive L1 candidate')
        plt.savefig(plotdir+'/dRsgn{0}/dRiso{1}/inclusive.pdf'.format(int(dRsgn*10),int(dRiso*10)))
        plt.close()

        for iEta in range(1,NetaBins):
            print('iEta{0}'.format(iEta))
            temp_taus = dfL1Taus.query('compressedEta=={0}'.format(iEta)).copy(deep=True)
            temp_jets = dfL1Jets.query('compressedEta=={0}'.format(iEta)).copy(deep=True)

            plt.figure(figsize=(18,8))
            plt.hist(temp_taus['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 $\tau$', color='blue', histtype='step', lw=2, density=1, bins=plot_bins)
            plt.hist(temp_jets['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 QCD jet', color='red', histtype='step', lw=2, density=1, bins=plot_bins)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{Iso}$ ciao')
            plt.ylabel(r'$a.u.$')
            plt.title(r'$\eta^{L1 candidate}$ bin [%.2f,%.2f])'%(etaBins[iEta-1],etaBins[iEta]))
            plt.savefig(plotdir+'/dRsgn{0}/dRiso{1}/iEta/iEta{2}.pdf'.format(int(dRsgn*10),int(dRiso*10), iEta))
            plt.close()

            del temp_taus, temp_jets

            for iPt in range(1,NptBins):
                print('iEta{0} - iPt{1}'.format(iEta, iPt))
                temp_taus = dfL1Taus.query('compressedPt=={0} and compressedEta=={0}'.format(iPt, iEta)).copy(deep=True)
                temp_jets = dfL1Jets.query('compressedPt=={0} and compressedEta=={0}'.format(iPt, iEta)).copy(deep=True)

                plt.figure(figsize=(18,8))
                plt.hist(temp_taus['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 $\tau$', color='blue', histtype='step', lw=2, density=1, bins=plot_bins)
                plt.hist(temp_jets['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 QCD jet', color='red', histtype='step', lw=2, density=1, bins=plot_bins)
                plt.legend(loc = 'upper right', fontsize=15)
                plt.grid(linestyle=':')
                plt.xlabel(r'$E_{T}^{Iso}$')
                plt.ylabel(r'$a.u.$')
                plt.title(r'$\eta^{L1 candidate}$ bin [%.2f,%.2f] - $p_{T}^{L1 candidate}$ bin [%.2f,%.2f])'%(etaBins[iEta-1],etaBins[iEta], ptBins[iPt-1],ptBins[iPt]))
                plt.savefig(plotdir+'/dRsgn{0}/dRiso{1}/iEtaPt/iEta{2}Pt{3}.pdf'.format(int(dRsgn*10),int(dRiso*10),iEta,iPt))
                plt.close()

                del temp_taus, temp_jets

        for iPt in range(1,NptBins):
            print('iPt{0}'.format(iPt))
            temp_taus = dfL1Taus.query('compressedPt=={0}'.format(iPt)).copy(deep=True)
            temp_jets = dfL1Jets.query('compressedPt=={0}'.format(iPt)).copy(deep=True)

            plt.figure(figsize=(18,8))
            plt.hist(temp_taus['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 $\tau$', color='blue', histtype='step', lw=2, density=1, bins=plot_bins)
            plt.hist(temp_jets['tower_etIso_dRsgn{0}_dRiso{1}'.format(int(dRsgn*10),int(dRiso*10))], label=r'L1 QCD jet', color='red', histtype='step', lw=2, density=1, bins=plot_bins)
            plt.legend(loc = 'upper right', fontsize=15)
            plt.grid(linestyle=':')
            plt.xlabel(r'$E_{T}^{Iso}$')
            plt.ylabel(r'$a.u.$')
            plt.title(r'$p_{T}^{L1 candidate}$ bin [%.2f,%.2f])'%(ptBins[iPt-1],ptBins[iPt]))
            plt.savefig(plotdir+'/dRsgn{0}/dRiso{1}/iPt/iPt{2}.pdf'.format(int(dRsgn*10),int(dRiso*10), iPt))
            plt.close()

            del temp_taus, temp_jets

    print('** INFO: finished eta and pT dependent etIso plots')
    print('---------------------------------------------------------------------------------------')
'''

    