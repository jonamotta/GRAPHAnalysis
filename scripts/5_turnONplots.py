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
    parser.add_argument('--doPlots', dest='doPlots', help='do you want to produce the plots?', action='store_true', default=False)
    parser.add_argument('--WP', dest='WP', help='which working point do you want to use (90, 95, 99)?', default='99')
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    PUbdtWP = 'WP'+args.WP
    bdtcut = 'cl3d_pubdt_pass'+PUbdtWP
    print('** INFO: using PU rejection BDT WP: '+args.WP)

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/DMsorted'
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/mapping'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/turnONs_{0}'.format(PUbdtWP)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_mapping_dict = {
        'threshold'    : outdir+'/TauInclusive_th_PU200_PUWP{0}_mapping.pkl'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}
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

    turnon_thresholds = range(1, 100, 1)

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

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
            
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting efficiency evaluation for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()

        ######################### SELECT EVENTS #########################  

        dfTraining_dict[name] = dfTraining_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==2) and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))
        dfValidation_dict[name] = dfValidation_dict[name].query('(gentau_decayMode==0 or gentau_decayMode==1 or gentau_decayMode==2) and cl3d_pubdt_pass{0}==True'.format(PUbdtWP))

        dfTau_dict[name] = pd.concat([dfTraining_dict[name],dfValidation_dict[name]],sort=False)
        
        # fill all the DM dataframes
        dfTauDM0_dict[name] = dfTau_dict[name].query('gentau_decayMode==0')
        dfTauDM1_dict[name] = dfTau_dict[name].query('gentau_decayMode==1')
        dfTauDM2_dict[name] = dfTau_dict[name].query('gentau_decayMode==2')


        ######################### BIN pt AND eta IN THE DATAFRAMES #########################

        dfTau_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_dict[name]['gentau_vis_eta'])
        dfTau_dict[name]['gentau_bin_eta'] = ((dfTau_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTau_dict[name]['gentau_bin_pt']  = ((dfTau_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfTauDM0_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM0_dict[name]['gentau_vis_eta'])
        dfTauDM0_dict[name]['gentau_bin_eta'] = ((dfTauDM0_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTauDM0_dict[name]['gentau_bin_pt']  = ((dfTauDM0_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfTauDM1_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM1_dict[name]['gentau_vis_eta'])
        dfTauDM1_dict[name]['gentau_bin_eta'] = ((dfTauDM1_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTauDM1_dict[name]['gentau_bin_pt']  = ((dfTauDM1_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfTauDM2_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM2_dict[name]['gentau_vis_eta'])
        dfTauDM2_dict[name]['gentau_bin_eta'] = ((dfTauDM2_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTauDM2_dict[name]['gentau_bin_pt']  = ((dfTauDM2_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')


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

        for threshold in turnon_thresholds:
            # calculate efficiency for the TAU datasets
            eff = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
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
            # fill the dataframes withe the efficiency values
            effVSpt_Tau_dict[name]['efficiency_{}'.format(threshold)] = eff
            effVSpt_TauDM0_dict[name]['efficiency_{}'.format(threshold)] = eff_DM0
            effVSpt_TauDM1_dict[name]['efficiency_{}'.format(threshold)] = eff_DM1
            effVSpt_TauDM2_dict[name]['efficiency_{}'.format(threshold)] = eff_DM2
            effVSpt_Tau_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth
            effVSpt_TauDM0_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM0
            effVSpt_TauDM1_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM1
            effVSpt_TauDM2_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM2

        mappingTau_dict[name] = {'threshold':[], 'pt95':[]}
        for threshold in turnon_thresholds:
            pt_95 = np.interp(0.95, effVSpt_Tau_dict[name]['efficiency_{}'.format(threshold)], effVSpt_Tau_dict[name].gentau_vis_pt,right=-99,left=-98)
            mappingTau_dict[name]['threshold'].append(threshold)
            mappingTau_dict[name]['pt95'].append(pt_95)
            #print(threshold, pt_95)
        pt95s_Tau_dict[name] = pd.DataFrame(mappingTau_dict[name])

        save_obj(pt95s_Tau_dict[name],outFileTau_mapping_dict[name])    


        ######################### STORE VALUES FOR TURN-ON CURVES #########################

        y_eff_40_Tau = effVSpt_Tau_dict[name]['efficiency_40']
        y_eff_50_Tau = effVSpt_Tau_dict[name]['efficiency_50']
        y_eff_60_Tau = effVSpt_Tau_dict[name]['efficiency_60']
        y_eff_smooth_40_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_40']
        y_eff_smooth_50_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_50']
        y_eff_smooth_60_Tau = effVSpt_Tau_dict[name]['efficiency_smooth_60']
        x_Tau = effVSpt_Tau_dict[name].gentau_vis_pt

        dfTauDM0 = effVSpt_TauDM0_dict[name]
        dfTauDM1 = effVSpt_TauDM1_dict[name]
        dfTauDM2 = effVSpt_TauDM2_dict[name]
        effTauDM0 = dfTauDM0['efficiency_50']
        effTauDM1 = dfTauDM1['efficiency_50']
        effTauDM2 = dfTauDM2['efficiency_50']
        eff_smooth_DM0_Tau = dfTauDM0['efficiency_smooth_50']
        eff_smooth_DM1_Tau = dfTauDM1['efficiency_smooth_50']
        eff_smooth_DM2_Tau = dfTauDM2['efficiency_smooth_50']
        x_DM0_Tau = effVSpt_TauDM0_dict[name].gentau_vis_pt
        x_DM1_Tau = effVSpt_TauDM1_dict[name].gentau_vis_pt
        x_DM2_Tau = effVSpt_TauDM2_dict[name].gentau_vis_pt

        print('\n** INFO: finished efficiency evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        

######################### MAKE PLOTS #########################

print('\n** INFO: plotting turnon curves')

matplotlib.rcParams.update({'font.size': 22})

lab_40 = r"$E_{T}^{L1,\tau}$ > 40 GeV"
lab_50 = r"$E_{T}^{L1,\tau}$ > 50 GeV"
lab_60 = r"$E_{T}^{L1,\tau}$ > 60 GeV"
plt.rcParams['legend.numpoints'] = 1

plt.figure(figsize=(8,8))
plt.errorbar(x_Tau,y_eff_40_Tau,xerr=1,ls='None',label=lab_40,color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_Tau,y_eff_50_Tau,xerr=1,ls='None',label=lab_50,color='green',lw=2,marker='o',mec='green')
plt.errorbar(x_Tau,y_eff_60_Tau,xerr=1,ls='None',label=lab_60,color='red',lw=2,marker='o',mec='red')
plt.plot(x_Tau,y_eff_smooth_40_Tau,label=lab_40,color='blue',lw=1.5)
plt.plot(x_Tau,y_eff_smooth_50_Tau,label=lab_50,color='green',lw=1.5)
plt.plot(x_Tau,y_eff_smooth_60_Tau,label=lab_60,color='red',lw=1.5)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT - PUWP{0}'.format(args.WP))
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_Tau_PUWP{0}.pdf'.format(args.WP))
plt.close()

plt.figure(figsize=(8,8))
plt.errorbar(x_DM0_Tau,effTauDM0,xerr=1,ls='None',label=r'1-prong',color='limegreen',lw=2,marker='o',mec='limegreen')
plt.errorbar(x_DM1_Tau,effTauDM1,xerr=1,ls='None',label=r'1-prong + $\pi^{0}$',color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_DM2_Tau,effTauDM2,xerr=1,ls='None',label=r'3-prong (+ $\pi^{0}$)',color='fuchsia',lw=2,marker='o',mec='fuchsia')
plt.plot(x_DM0_Tau,eff_smooth_DM0_Tau,color='limegreen',lw=1.5)
plt.plot(x_DM1_Tau,eff_smooth_DM1_Tau,color='blue',lw=1.5)
plt.plot(x_DM2_Tau,eff_smooth_DM2_Tau,color='fuchsia',lw=1.5)
plt.legend(loc = 'lower right', fontsize=18)
# txt = (r'Gen. $\tau$ decay mode:')
# t = plt.text(63,0.20, txt, ha='left', wrap=True, fontsize=18)
# t.set_bbox(dict(facecolor='white', edgecolor='white'))
txt2 = (r'$E_{T}^{L1,\tau}$ > 50 GeV')
t2 = plt.text(26,0.83, txt2, ha='left', wrap=True, fontsize=18)
t2.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT  - PUWP{0}'.format(args.WP))
plt.grid()
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_DM_Tau_PUWP{0}.pdf'.format(args.WP))
plt.close()




'''
matplotlib.rcParams.update({'font.size': 22})

plt.figure(figsize=(10,10))
for name in dfTau_dict: 
  df = effVSpt_Tau_dict[name]
  eff = df['efficiency_40']
  eff_smooth = df['efficiency_smooth_40']
  plt.plot(df.gentau_vis_pt, eff_smooth, label=legends[name], linewidth=2, color=colors[name])
plt.ylim(0., 1.01)
plt.xlim(28, 90)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'eff_vs_pt_L1_40_WP_algo'+algo+'_'+PUBDTWP+'.png')
plt.savefig(plotdir+'eff_vs_pt_L1_40_WP_algo'+algo+'_'+PUBDTWP+'.pdf')

plt.figure(figsize=(10,10))
for name in dfTau_dict: 
  df = effVSpt_Tau_dict[name]
  eff = df['efficiency_60']
  eff_smooth = df['efficiency_smooth_60']
  plt.plot(df.gentau_vis_pt, eff_smooth, label=legends[name], linewidth=2, color=colors[name])
plt.ylim(0., 1.01)
plt.xlim(28, 90)
plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'eff_vs_pt_L1_60_WP_algo'+algo+'_'+PUBDTWP+'.png')
plt.savefig(plotdir+'eff_vs_pt_L1_60_WP_algo'+algo+'_'+PUBDTWP+'.pdf')
'''

'''
plt.figure(figsize=(8,8))
for name in dfTau_dict:
  df = pt95s_Tau_dict[name]
  plt.plot(df.threshold, df.pt95, label=legends[name], linewidth=2, color=colors[name])
#plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(10, 80)
plt.ylim(10, 100)
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.cmstext("CMS"," Phase-2 Simulation")
plt.lumitext("PU=200","HGCAL")
plt.savefig(plotdir+'L1_to_offline_WP_algo'+algo+'_'+PUBDTWP+'.png')
plt.savefig(plotdir+'L1_to_offline_WP_algo'+algo+'_'+PUBDTWP+'.pdf')
plt.show()
'''