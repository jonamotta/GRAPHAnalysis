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
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/turnONs'
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir)


    inFileTau_dict = {
        'threshold'    : indir+'/RelValTenTau_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileHH_dict = {
        'threshold'    : indir+'/GluGluHHTo2b2Tau_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_mapping_dict = {
        'threshold'    : outdir+'/RelValTenTau_mapping_th_PU200_PUWP{0}'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    outFileHH_mapping_dict = {
        'threshold'    : outdir+'/GluGluHHTo2b2Tau_mapping_th_PU200_PUWP{0}'.format(args.WP),
        'supertrigger' : outdir+'/',
        'bestchoice'   : outdir+'/',
        'bestcoarse'   : outdir+'/',
        'mixed'        : outdir+'/'
    }

    dfTau_dict = {}
    dfTau_DM01_dict = {}
    dfTau_DM45_dict = {}
    dfHH_dict = {}
    dfHH_DM01_dict = {}
    dfHH_DM45_dict = {}

    effVSpt_Tau_dict = {}
    effVSpt_Tau_DM01_dict = {}
    effVSpt_Tau_DM45_dict = {}
    effVSpt_HH_dict = {}
    effVSpt_HH_DM01_dict = {}
    effVSpt_HH_DM45_dict = {}
    
    pt95s_Tau_dict = {}
    pt95s_HH_dict = {}
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

        # fill signal dataframes and dictionaries
        store_tau = pd.HDFStore(inFileTau_dict[name], mode='r')
        dfTau_dict[name]  = store_tau[name]
        store_tau.close()

        # fill HH dataframes and dictionaries
        store_hh = pd.HDFStore(inFileHH_dict[name], mode='r')
        dfHH_dict[name]  = store_hh[name]
        store_hh.close() 

        ######################### SELECT EVENTS #########################  

        genPt_sel  = dfTau_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfTau_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfTau_dict[name]['gentau_vis_eta']) < 2.9
        cl3dBest_sel = dfTau_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfTau_dict[name]['cl3d_pt'] > 4
        PUWP_sel = dfTau_dict[name][bdtcut] == True
        DMsel = dfTau_dict[name]['gentau_decayMode'] >= 0 # it should already be the case from skim level, but beter be safe
        dfTau_dict[name] = dfTau_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel & PUWP_sel & DMsel]

        genPt_sel  = dfHH_dict[name]['gentau_vis_pt'] > 20
        eta_sel1   = np.abs(dfHH_dict[name]['gentau_vis_eta']) > 1.6
        eta_sel2   = np.abs(dfHH_dict[name]['gentau_vis_eta']) < 2.9
        cl3dBest_sel = dfHH_dict[name]['cl3d_isbestmatch'] == True
        cl3dPt_sel = dfHH_dict[name]['cl3d_pt'] > 4
        PUWP_sel = dfHH_dict[name][bdtcut] == True
        DMsel = dfHH_dict[name]['gentau_decayMode'] >= 0 # it should already be the case from skim level, but beter be safe
        dfHH_dict[name] = dfHH_dict[name][genPt_sel & eta_sel1 & eta_sel2 & cl3dBest_sel & cl3dPt_sel & PUWP_sel & DMsel]

        DM0sel = (dfTau_dict[name]['gentau_decayMode'] == 0) | (dfTau_dict[name]['gentau_decayMode'] == 1)
        DM45sel = (dfTau_dict[name]['gentau_decayMode'] == 2)
        dfTau_DM01_dict[name] = dfTau_dict[name][DM0sel]
        dfTau_DM45_dict[name] = dfTau_dict[name][DM45sel]

        DM0sel = (dfHH_dict[name]['gentau_decayMode'] == 0) | (dfHH_dict[name]['gentau_decayMode'] == 1)
        DM45sel = (dfHH_dict[name]['gentau_decayMode'] == 2)
        dfHH_DM01_dict[name] = dfHH_dict[name][DM0sel]
        dfHH_DM45_dict[name] = dfHH_dict[name][DM45sel]


        ######################### BIN pt AND eta IN THE DATAFRAMES #########################

        dfTau_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_dict[name]['gentau_vis_eta'])
        dfTau_dict[name]['gentau_bin_eta'] = ((dfTau_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTau_dict[name]['gentau_bin_pt']  = ((dfTau_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfTau_DM01_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_DM01_dict[name]['gentau_vis_eta'])
        dfTau_DM01_dict[name]['gentau_bin_eta'] = ((dfTau_DM01_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTau_DM01_dict[name]['gentau_bin_pt']  = ((dfTau_DM01_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfTau_DM45_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_DM45_dict[name]['gentau_vis_eta'])
        dfTau_DM45_dict[name]['gentau_bin_eta'] = ((dfTau_DM45_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfTau_DM45_dict[name]['gentau_bin_pt']  = ((dfTau_DM45_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfHH_dict[name]['gentau_vis_abseta'] = np.abs(dfHH_dict[name]['gentau_vis_eta'])
        dfHH_dict[name]['gentau_bin_eta'] = ((dfHH_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfHH_dict[name]['gentau_bin_pt']  = ((dfHH_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfHH_DM01_dict[name]['gentau_vis_abseta'] = np.abs(dfHH_DM01_dict[name]['gentau_vis_eta'])
        dfHH_DM01_dict[name]['gentau_bin_eta'] = ((dfHH_DM01_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfHH_DM01_dict[name]['gentau_bin_pt']  = ((dfHH_DM01_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')

        dfHH_DM45_dict[name]['gentau_vis_abseta'] = np.abs(dfHH_DM45_dict[name]['gentau_vis_eta'])
        dfHH_DM45_dict[name]['gentau_bin_eta'] = ((dfHH_DM45_dict[name]['gentau_vis_abseta'] - etamin)/0.1).astype('int32')
        dfHH_DM45_dict[name]['gentau_bin_pt']  = ((dfHH_DM45_dict[name]['gentau_vis_pt'] - ptcut)/2).astype('int32')


        ######################### CALCULATE EFFICIENCIES & SAVE MAPPINGS #########################
        print('\n** INFO: starting efficiency calculation')

        effVSpt_Tau_dict[name] = {}
        effVSpt_Tau_DM01_dict[name] = {}
        effVSpt_Tau_DM45_dict[name] = {}

        effVSpt_Tau_dict[name] = dfTau_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_Tau_DM01_dict[name] = dfTau_DM01_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_Tau_DM45_dict[name] = dfTau_DM45_dict[name].groupby('gentau_bin_pt').mean()

        effVSpt_HH_dict[name] = dfHH_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_HH_DM01_dict[name] = dfHH_DM01_dict[name].groupby('gentau_bin_pt').mean()
        effVSpt_HH_DM45_dict[name] = dfHH_DM45_dict[name].groupby('gentau_bin_pt').mean()

        for threshold in turnon_thresholds:
            # calculate efficiency for the TAU datasets
            eff = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM01 = dfTau_DM01_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM45 = dfTau_DM45_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            # calculate the mean every 7 df entries -> this allows to smooth out the entries 
            eff_smooth = eff.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM01 = eff_DM01.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM45 = eff_DM45.rolling(window=7, win_type='triang', center=True).mean()
            # fill NaN entries with 0.
            eff_smooth.fillna(0., inplace=True)
            eff_smooth_DM01.fillna(0., inplace=True)
            eff_smooth_DM45.fillna(0., inplace=True)
            # fill the dataframes withe the efficiency values
            effVSpt_Tau_dict[name]['efficiency_{}'.format(threshold)] = eff
            effVSpt_Tau_DM01_dict[name]['efficiency_{}'.format(threshold)] = eff_DM01
            effVSpt_Tau_DM45_dict[name]['efficiency_{}'.format(threshold)] = eff_DM45
            effVSpt_Tau_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth
            effVSpt_Tau_DM01_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM01
            effVSpt_Tau_DM45_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM45

        mappingTau_dict[name] = {'threshold':[], 'pt95':[]}
        for threshold in turnon_thresholds:
            eff_smooth = effVSpt_Tau_dict[name]['efficiency_smooth_{}'.format(threshold)]
            pt_95 = np.interp(0.95, eff_smooth, effVSpt_Tau_dict[name].gentau_vis_pt)
            mappingTau_dict[name]['threshold'].append(threshold)
            mappingTau_dict[name]['pt95'].append(pt_95)
            #print threshold, pt_95
        pt95s_Tau_dict[name] = pd.DataFrame(mappingTau_dict[name])

        save_obj(pt95s_Tau_dict[name],outFileTau_mapping_dict[name])    

        for threshold in turnon_thresholds:
            # calculate efficiency for the HH datasets
            eff = dfHH_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM01 = dfHH_DM01_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            eff_DM45 = dfHH_DM45_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold))
            # calculate the mean every 7 df entries -> this allows to smooth out the entries 
            eff_smooth = eff.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM01 = eff_DM01.rolling(window=7, win_type='triang', center=True).mean()
            eff_smooth_DM45 = eff_DM45.rolling(window=7, win_type='triang', center=True).mean()
            # fill NaN entries with 0.
            eff_smooth.fillna(0., inplace=True)
            eff_smooth_DM01.fillna(0., inplace=True)
            eff_smooth_DM45.fillna(0., inplace=True)
            # fill the dataframes withe the efficiency values
            effVSpt_HH_dict[name]['efficiency_{}'.format(threshold)] = eff
            effVSpt_HH_DM01_dict[name]['efficiency_{}'.format(threshold)] = eff_DM01
            effVSpt_HH_DM45_dict[name]['efficiency_{}'.format(threshold)] = eff_DM45
            effVSpt_HH_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth
            effVSpt_HH_DM01_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM01
            effVSpt_HH_DM45_dict[name]['efficiency_smooth_{}'.format(threshold)] = eff_smooth_DM45

        mappingHH_dict[name] = {'threshold':[], 'pt95':[]}
        for threshold in turnon_thresholds:
            eff_smooth = effVSpt_HH_dict[name]['efficiency_smooth_{}'.format(threshold)]
            pt_95 = np.interp(0.95, eff_smooth, effVSpt_HH_dict[name].gentau_vis_pt)
            mappingHH_dict[name]['threshold'].append(threshold)
            mappingHH_dict[name]['pt95'].append(pt_95)
            #print threshold, pt_95
        pt95s_HH_dict[name] = pd.DataFrame(mappingHH_dict[name])

        save_obj(pt95s_HH_dict[name],outFileHH_mapping_dict[name])

        print('** INFO: finished efficiency calculation')

        ######################### STORE VALUES FOR TURN-ON CURVES #########################
        
        df = effVSpt_Tau_dict[name]
        y_eff_40_Tau = df['efficiency_40']
        y_eff_50_Tau = df['efficiency_50']
        y_eff_60_Tau = df['efficiency_60']
        y_eff_smooth_40_Tau = df['efficiency_smooth_40']
        y_eff_smooth_50_Tau = df['efficiency_smooth_50']
        y_eff_smooth_60_Tau = df['efficiency_smooth_60']
        x_Tau = df.gentau_vis_pt

        df = effVSpt_HH_dict[name]
        y_eff_40_HH = df['efficiency_40']
        y_eff_50_HH = df['efficiency_50']
        y_eff_60_HH = df['efficiency_60']
        y_eff_smooth_40_HH = df['efficiency_smooth_40']
        y_eff_smooth_50_HH = df['efficiency_smooth_50']
        y_eff_smooth_60_HH = df['efficiency_smooth_60']
        x_HH = df.gentau_vis_pt

        df_DM01_Tau = effVSpt_Tau_DM01_dict[name]
        df_DM45_Tau = effVSpt_Tau_DM45_dict[name]
        eff_DM01_Tau = df_DM01_Tau['efficiency_50']
        eff_DM45_Tau = df_DM45_Tau['efficiency_50']
        eff_smooth_DM01_Tau = df_DM01_Tau['efficiency_smooth_50']
        eff_smooth_DM45_Tau = df_DM45_Tau['efficiency_smooth_50']
        x_DM01_Tau = effVSpt_Tau_DM01_dict[name].gentau_vis_pt
        x_DM45_Tau = effVSpt_Tau_DM45_dict[name].gentau_vis_pt

        df_DM01_HH = effVSpt_HH_DM01_dict[name]
        df_DM45_HH = effVSpt_HH_DM45_dict[name]
        eff_DM01_HH = df_DM01_HH['efficiency_50']
        eff_DM45_HH = df_DM45_HH['efficiency_50']
        eff_smooth_DM01_HH = df_DM01_HH['efficiency_smooth_50']
        eff_smooth_DM45_HH = df_DM45_HH['efficiency_smooth_50']
        x_DM01_HH = effVSpt_HH_DM01_dict[name].gentau_vis_pt
        x_DM45_HH = effVSpt_HH_DM45_dict[name].gentau_vis_pt

        print('\n** INFO: finished efficiency evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')
        

######################### MAKE PLOTS #########################

print('\n** INFO: plotting turnon curves')

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
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
eff40Tau_line = mlines.Line2D([], [], color='blue',markersize=15, label=lab_40,lw=2)
eff50Tau_line = mlines.Line2D([], [], color='green',markersize=15, label=lab_50,lw=2)
eff60Tau_line = mlines.Line2D([], [], color='red',markersize=15, label=lab_60,lw=2)
plt.legend(loc = 'lower right', fontsize=16, handles=[eff40Tau_line,eff50Tau_line,eff60Tau_line])
#plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT - PUWP{0} - RelValTenTau'.format(args.WP))
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_Tau_PUWP{0}.pdf'.format(args.WP))
plt.close()


plt.figure(figsize=(8,8))
plt.errorbar(x_HH,y_eff_40_HH,xerr=1,ls='None',label=lab_40,color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_HH,y_eff_50_HH,xerr=1,ls='None',label=lab_50,color='green',lw=2,marker='o',mec='green')
plt.errorbar(x_HH,y_eff_60_HH,xerr=1,ls='None',label=lab_60,color='red',lw=2,marker='o',mec='red')
plt.plot(x_HH,y_eff_smooth_40_HH,label=lab_40,color='blue',lw=1.5)
plt.plot(x_HH,y_eff_smooth_50_HH,label=lab_50,color='green',lw=1.5)
plt.plot(x_HH,y_eff_smooth_60_HH,label=lab_60,color='red',lw=1.5)
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
eff40HH_line = mlines.Line2D([], [], color='blue',markersize=15, label=lab_40,lw=2)
eff50HH_line = mlines.Line2D([], [], color='green',markersize=15, label=lab_50,lw=2)
eff60HH_line = mlines.Line2D([], [], color='red',markersize=15, label=lab_60,lw=2)
plt.legend(loc = 'lower right', fontsize=16, handles=[eff40HH_line,eff50HH_line,eff60HH_line])
#plt.legend(loc = 'lower right', fontsize=16)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT - PUWP{0} - GluGluHHTo2b2Tau'.format(args.WP))
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_HH_PUWP{0}.pdf'.format(args.WP))
plt.close()


plt.figure(figsize=(8,8))
plt.errorbar(x_DM01_Tau,eff_DM01_Tau,xerr=1,ls='None',label=r'1-prong (+ $\pi^{0}$)',color='red',lw=2,marker='o',mec='red')
plt.errorbar(x_DM45_Tau,eff_DM45_Tau,xerr=1,ls='None',label=r'3-prong (+ $\pi^{0}$)',color='blue',lw=2,marker='o',mec='blue')
plt.plot(x_DM01_Tau,eff_smooth_DM01_Tau,color='red',lw=1.5)
plt.plot(x_DM45_Tau,eff_smooth_DM45_Tau,color='blue',lw=1.5)
red_line = mlines.Line2D([], [], color='red',markersize=15, label=r'1-prong (+ $\pi^{0}$)',lw=2)
blue_line = mlines.Line2D([], [], color='blue',markersize=15, label=r'3-prong (+ $\pi^{0}$)',lw=2)
plt.legend(loc = 'lower right', fontsize=18, handles=[red_line,blue_line])
txt = (r'Gen. $\tau$ decay mode:')
t = plt.text(63,0.20, txt, ha='left', wrap=True, fontsize=18)
t.set_bbox(dict(facecolor='white', edgecolor='white'))
txt2 = (r'$E_{T}^{L1,\tau}$ > 50 GeV')
t2 = plt.text(26,0.83, txt2, ha='left', wrap=True, fontsize=18)
t2.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT  - PUWP{0} - RelValTenTau'.format(args.WP))
plt.grid()
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_DM_Tau_PUWP{0}.pdf'.format(args.WP))
plt.close()


plt.figure(figsize=(8,8))
plt.errorbar(x_DM01_HH,eff_DM01_HH,xerr=1,ls='None',label=r'1-prong (+ $\pi^{0}$)',color='red',lw=2,marker='o',mec='red')
plt.errorbar(x_DM45_HH,eff_DM45_HH,xerr=1,ls='None',label=r'3-prong (+ $\pi^{0}$)',color='blue',lw=2,marker='o',mec='blue')
plt.plot(x_DM01_HH,eff_smooth_DM01_HH,color='red',lw=1.5)
plt.plot(x_DM45_HH,eff_smooth_DM45_HH,color='blue',lw=1.5)
red_line = mlines.Line2D([], [], color='red',markersize=15, label=r'1-prong (+ $\pi^{0}$)',lw=2)
blue_line = mlines.Line2D([], [], color='blue',markersize=15, label=r'3-prong (+ $\pi^{0}$)',lw=2)
plt.legend(loc = 'lower right', fontsize=18, handles=[red_line,blue_line])
txt = (r'Gen. $\tau$ decay mode:')
t = plt.text(63,0.20, txt, ha='left', wrap=True, fontsize=18)
t.set_bbox(dict(facecolor='white', edgecolor='white'))
txt2 = (r'$E_{T}^{L1,\tau}$ > 50 GeV')
t2 = plt.text(26,0.83, txt2, ha='left', wrap=True, fontsize=18)
t2.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Efficiency vs pT - PUWP{0} - GluGluHHTo2b2Tau'.format(args.WP))
plt.grid()
plt.xlim(20, 91.5)
plt.ylim(0., 1.10)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_DM_HH_PUWP{0}.pdf'.format(args.WP))
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