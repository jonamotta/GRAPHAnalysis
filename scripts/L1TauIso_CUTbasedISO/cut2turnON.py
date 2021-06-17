import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle
import matplotlib.lines as mlines
import argparse


def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'r') as f:
        return pickle.load(f)

def efficiency(group, threshold):
    tot = group.shape[0]
    sel = group[(group.cl3d_pt_c3 > threshold)].shape[0]
    return float(sel)/float(tot)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--WP', dest='WP', help='which working point do you want to use (90, 95, 99)?', default='99')
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    print('** INFO: using PU rejection BDT WP: '+args.WP)

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    dRsgn = 0.1 

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolation_application_dRsgn{0}'.format(int(dRsgn*10))
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/mapping'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/turnONs/dRsgn{0}'.format(int(dRsgn*10))
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileIso_tau = {
        'threshold'    : indir+'/AllTau_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_mapping_dict = {
        'threshold'    : outdir+'/AllTau_PU200_th_isolation_PUWP{0}_dRsgn{1}_mapping.pkl'.format(args.WP,int(dRsgn*10)),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    dfTau_dict = {}
    dfTauDM0_dict = {}
    dfTauDM1_dict = {}
    dfTauDM2_dict = {}

    effVSpt_Tau_dict = {}
    effVSpt_TauDM0_dict = {}
    effVSpt_TauDM1_dict = {}
    effVSpt_TauDM2_dict = {}
    
    pt95s_Tau_dict = {}
    mappingTau_dict = {}
    mappingHH_dict = {}

    ptcut = 1
    etamin = 1.6

    online_thresholds = range(1, 100, 1)

    # colors to use for plotting
    colors_dict = {
        'threshold'    : 'blue',
        'supertrigger' : 'red',
        'bestchoice'   : 'olive',
        'bestcoarse'   : 'orange',
        'mixed'        : 'fuchsia'
    }

    # legend to use for plotting
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'supertrigger' : 'STC4+16',
        'bestchoice'   : 'BC Decentral',
        'bestcoarse'   : 'BC Coarse 2x2 TC',
        'mixed'        : 'Mixed BC + STC'
    }


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    if dRsgn == 0.1:
        dRiso = 0.3
        dRisoEm = 0.3
        dRcl3d = 0.4
        twEtiso = 70 #GeV
        twEtEmiso = 50 #GeV
        clEtIso = 80 #GeV
    elif dRsgn == 0.2:
        dRiso = 0.4
        dRisoEm = 0.4
        dRcl3d = 0.4
        twEtiso = 80 #GeV
        twEtEmiso = 60 #GeV
        clEtIso = 80 #GeV

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
            
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting efficiency evaluation for the front-end option '+feNames_dict[name])

        store = pd.HDFStore(inFileIso_tau[name], mode='r')
        dfTau_dict[name] = store[name]
        store.close()

        ######################### SELECT EVENTS #########################   
        
        dfTau_dict[name].query('cl3d_pubdt_passWP{0}==True and cl3d_predDM_PUWP{0}!=3 and (tower_etIso_dRsgn{1}_dRiso{2}<={3} and tower_etEmIso_dRsgn{1}_dRiso{4}<={5} and cl3d_etIso_dR{6}<={7})'.format(args.WP, int(dRsgn*10), int(dRiso*10), twEtiso, int(dRisoEm*10), twEtEmiso, int(dRcl3d*10), clEtIso), inplace=True)
        #dfTau_dict[name].query('cl3d_pubdt_passWP{0}==True'.format(args.WP), inplace=True)

        # fill all the DM dataframes
        dfTauDM0_dict[name] = dfTau_dict[name].query('gentau_decayMode==0').copy(deep=True)
        dfTauDM1_dict[name] = dfTau_dict[name].query('gentau_decayMode==1').copy(deep=True)
        dfTauDM2_dict[name] = dfTau_dict[name].query('gentau_decayMode==10 or gentau_decayMode==11').copy(deep=True)


        ######################### BIN pt AND eta IN THE DATAFRAMES #########################

        pt_binwidth = 3
        eta_binwidth = 0.1

        dfTau_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_dict[name]['gentau_vis_eta'])
        dfTau_dict[name]['gentau_bin_eta'] = ((dfTau_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
        dfTau_dict[name]['gentau_bin_pt']  = ((dfTau_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')

        dfTauDM0_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM0_dict[name]['gentau_vis_eta'])
        dfTauDM0_dict[name]['gentau_bin_eta'] = ((dfTauDM0_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
        dfTauDM0_dict[name]['gentau_bin_pt']  = ((dfTauDM0_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')

        dfTauDM1_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM1_dict[name]['gentau_vis_eta'])
        dfTauDM1_dict[name]['gentau_bin_eta'] = ((dfTauDM1_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
        dfTauDM1_dict[name]['gentau_bin_pt']  = ((dfTauDM1_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')

        dfTauDM2_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM2_dict[name]['gentau_vis_eta'])
        dfTauDM2_dict[name]['gentau_bin_eta'] = ((dfTauDM2_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
        dfTauDM2_dict[name]['gentau_bin_pt']  = ((dfTauDM2_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')


        ######################### CALCULATE EFFICIENCIES & SAVE MAPPINGS #########################
        
        print('\n** INFO: calculating efficiency')

        effVSpt_Tau_dict[name] = {}
        effVSpt_TauDM0_dict[name] = {}
        effVSpt_TauDM1_dict[name] = {}
        effVSpt_TauDM2_dict[name] = {}

        effVSpt_Tau_dict[name] = dfTau_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_TauDM0_dict[name] = dfTauDM0_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_TauDM1_dict[name] = dfTauDM1_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_TauDM2_dict[name] = dfTauDM2_dict[name].groupby('gentau_bin_pt').mean()

        for threshold in online_thresholds:
            # calculate efficiency for the TAU datasets --> calculated per bin that will be plotted
            eff = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold)) # --> the output of this will be a series with idx=bin and entry=efficiency
            eff_DM0 = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM1 = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM2 = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            
            # calculate the mean every 7 df entries -> this allows to smooth out the entries 
            eff_smooth = eff.rolling(window=7, center=True).mean()
            eff_smooth_DM0 = eff_DM0.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM1 = eff_DM1.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM2 = eff_DM2.rolling(window=7, win_type='triang', center=True).mean()
            # fill NaN entries with 0.
            eff_smooth.fillna(0., inplace=True)
            eff_smooth_DM0.fillna(0., inplace=True)
            eff_smooth_DM1.fillna(0., inplace=True)
            eff_smooth_DM2.fillna(0., inplace=True)
            
            # fill the dataframes with the efficiency values --> every efficiency_at{threshold} contains the value of the efficiency when applying the specific threshold 
            effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)] = eff
            effVSpt_TauDM0_dict[name]['efficiency_at{0}GeV'.format(threshold)] = eff_DM0
            effVSpt_TauDM1_dict[name]['efficiency_at{0}GeV'.format(threshold)] = eff_DM1
            effVSpt_TauDM2_dict[name]['efficiency_at{0}GeV'.format(threshold)] = eff_DM2
            effVSpt_Tau_dict[name]['efficiency_smooth_at{0}GeV'.format(threshold)] = eff_smooth
            effVSpt_TauDM0_dict[name]['efficiency_smooth_at{0}GeV'.format(threshold)] = eff_smooth_DM0
            effVSpt_TauDM1_dict[name]['efficiency_smooth_at{0}GeV'.format(threshold)] = eff_smooth_DM1
            effVSpt_TauDM2_dict[name]['efficiency_smooth_at{0}GeV'.format(threshold)] = eff_smooth_DM2

        mappingTau_dict[name] = {'threshold':[], 'pt95':[]}
        for threshold in online_thresholds:
            # fixed a certain threshold, find the gentau_vis_pt value that has a 95% selection efficiency when applying that threshold on cl3d_pt_c3
            # this values of pT_95 represent the offline threshold because 'a posteriori' when doing analysis we ask for a tau with pT>x and that pT for us now is just gentau_vis_pt
            pt_95 = np.interp(0.95, effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)], effVSpt_Tau_dict[name]['gentau_vis_pt'],right=-99,left=-98)
            mappingTau_dict[name]['threshold'].append(threshold)
            mappingTau_dict[name]['pt95'].append(pt_95)
            #print(threshold, pt_95)
        pt95s_Tau_dict[name] = pd.DataFrame(mappingTau_dict[name])

        save_obj(pt95s_Tau_dict[name],outFileTau_mapping_dict[name])    


        ######################### STORE VALUES FOR TURN-ON CURVES #########################

        y_eff_20_Tau = effVSpt_Tau_dict[name]['efficiency_at20GeV']
        y_eff_30_Tau = effVSpt_Tau_dict[name]['efficiency_at30GeV']
        y_eff_40_Tau = effVSpt_Tau_dict[name]['efficiency_at40GeV']
        y_eff_smooth_20_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_at20GeV']
        y_eff_smooth_30_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_at30GeV']
        y_eff_smooth_40_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_at40GeV']
        x_Tau = effVSpt_Tau_dict[name].gentau_vis_pt # is binned and the value is the mean of the entries per bin

        dfTauDM0 = effVSpt_TauDM0_dict[name]
        dfTauDM1 = effVSpt_TauDM1_dict[name]
        dfTauDM2 = effVSpt_TauDM2_dict[name]
        effTauDM0 = dfTauDM0['efficiency_at30GeV']
        effTauDM1 = dfTauDM1['efficiency_at30GeV']
        effTauDM2 = dfTauDM2['efficiency_at30GeV']
        eff_smooth_DM0_Tau = dfTauDM0['efficiency_smooth_at30GeV']
        eff_smooth_DM1_Tau = dfTauDM1['efficiency_smooth_at30GeV']
        eff_smooth_DM2_Tau = dfTauDM2['efficiency_smooth_at30GeV']
        x_DM0_Tau = effVSpt_TauDM0_dict[name].gentau_vis_pt
        x_DM1_Tau = effVSpt_TauDM1_dict[name].gentau_vis_pt
        x_DM2_Tau = effVSpt_TauDM2_dict[name].gentau_vis_pt

        print('\n** INFO: finished efficiency evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        

######################### MAKE PLOTS #########################

print('\n** INFO: plotting turnon curves')

matplotlib.rcParams.update({'font.size': 22})

lab_20 = r"$E_{T}^{L1,\tau}$ > 20 GeV"
lab_30 = r"$E_{T}^{L1,\tau}$ > 30 GeV"
lab_40 = r"$E_{T}^{L1,\tau}$ > 40 GeV"
plt.rcParams['legend.numpoints'] = 1

plt.figure(figsize=(8,8))
plt.errorbar(x_Tau,y_eff_20_Tau,xerr=1,ls='None',label=lab_20,color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_Tau,y_eff_30_Tau,xerr=1,ls='None',label=lab_30,color='green',lw=2,marker='o',mec='green')
plt.errorbar(x_Tau,y_eff_40_Tau,xerr=1,ls='None',label=lab_40,color='red',lw=2,marker='o',mec='red')
plt.plot(x_Tau,y_eff_smooth_20_Tau,label=lab_20,color='blue',lw=1.5)
plt.plot(x_Tau,y_eff_smooth_30_Tau,label=lab_30,color='green',lw=1.5)
plt.plot(x_Tau,y_eff_smooth_40_Tau,label=lab_40,color='red',lw=1.5)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlim(10, 75)
plt.ylim(0., 1.10)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title('Efficiency vs pT - PUWP{0} \n dRsgn={1} dRiso={2} dRcl3d={3} twEtIso={4} clEtIso={5}'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso), fontsize=15)
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_isolated_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()

plt.figure(figsize=(8,8))
plt.errorbar(x_DM0_Tau,effTauDM0,xerr=1,ls='None',label=r'1-prong',color='limegreen',lw=2,marker='o',mec='limegreen')
plt.errorbar(x_DM1_Tau,effTauDM1,xerr=1,ls='None',label=r'1-prong + $\pi^{0}$',color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_DM2_Tau,effTauDM2,xerr=1,ls='None',label=r'3-prong (+ $\pi^{0}$)',color='fuchsia',lw=2,marker='o',mec='fuchsia')
plt.legend(loc = 'lower right', fontsize=18)
plt.plot(x_DM0_Tau,eff_smooth_DM0_Tau,color='limegreen',lw=1.5)
plt.plot(x_DM1_Tau,eff_smooth_DM1_Tau,color='blue',lw=1.5)
plt.plot(x_DM2_Tau,eff_smooth_DM2_Tau,color='fuchsia',lw=1.5)
# txt = (r'Gen. $\tau$ decay mode:')
# t = plt.text(63,0.20, txt, ha='left', wrap=True, fontsize=18)
# t.set_bbox(dict(facecolor='white', edgecolor='white'))
txt2 = (r'$E_{T}^{L1,\tau}$ > 30 GeV')
t2 = plt.text(5,0.9, txt2, ha='left', wrap=True, fontsize=18)
t2.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title('Efficiency vs pT - PUWP{0} \n dRsgn={1} dRiso={2} dRcl3d={3} twEtIso={4} clEtIso={5}'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso), fontsize=15)
plt.grid()
plt.xlim(10, 75)
plt.ylim(0., 1.10)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_DM_isolated_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()


plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at20GeV']
    eff_smooth = df['efficiency_smooth_at20GeV']
    plt.plot(df.gentau_vis_pt, eff_smooth, label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.01)
plt.xlim(0, 90)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_20_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()

plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at30GeV']
    eff_smooth = df['efficiency_smooth_at30GeV']
    plt.plot(df.gentau_vis_pt, eff_smooth, label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.01)
plt.xlim(10, 90)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_30_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()

plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at40GeV']
    eff_smooth = df['efficiency_smooth_at40GeV']
    plt.plot(df.gentau_vis_pt, eff_smooth, label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.01)
plt.xlim(20, 90)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_40_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()

plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue 
    df = pt95s_Tau_dict[name]
    plt.plot(df.threshold, df.pt95, label=legends_dict[name], linewidth=2, color=colors_dict[name])
#plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(10, 80)
plt.ylim(10, 100)
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/L1_to_offline_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()
