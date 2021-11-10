import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
import argparse


def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'r') as f:
        return pickle.load(f)

def efficiency(group, threshold, PUWP, ISOWP):
    tot = group.shape[0]
    
    if PUWP == '99':
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
    elif PUWP == '95':
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
    else:
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]    
    
    return float(sel)/float(tot)

def efficiency_err(group, threshold, PUWP, ISOWP, upper=False):
    tot = group.shape[0]

    if PUWP == '99':
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP99 == True)].shape[0]
    elif PUWP == '95':
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP95 == True)].shape[0]
    else:
        if   ISOWP == '20': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP20 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        elif ISOWP == '25': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP25 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        elif ISOWP == '10': sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP10 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]
        else:             sel = group[(group.cl3d_pt_c3 > threshold) & (group.cl3d_isobdt_passWP15 == True) & (group.cl3d_pubdt_passWP90 == True)].shape[0]    

    # clopper pearson errors --> ppf gives the boundary of the cinfidence interval, therefore for plotting we have to subtract the value of the central value float(sel)/float(tot)!!
    alpha = (1 - 0.9) / 2
    if upper:
        if sel == tot:
             return 0.
        else:
            return abs(btdtri(sel+1, tot-sel, 1-alpha) - float(sel)/float(tot))
    else:
        if sel == 0:
            return 0.
        else:
            return abs(float(sel)/float(tot) - btdtri(sel, tot-sel+1, alpha))

    # naive errors calculated as propagation of the statistical error on sel and tot
    # eff = sel / (sel + not_sel) --> propagate stat error of sel and not_sel to efficiency
    #return np.sqrt( (np.sqrt(tot-sel)+np.sqrt(sel))**2 * sel**2/tot**4 + np.sqrt(sel)**2 * 1./tot**4 )

def sigmoid(x , a, x0, k):
    return a / ( 1 + np.exp(-k*(x-x0)) )

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--FE', dest='FE', help='which front-end option are we using?', default=None)
    parser.add_argument('--PUWP', dest='PUWP', help='which working point do you want to use (90, 95, 99)?', default='99')
    parser.add_argument('--ISOWP', dest='ISOWP', help='which working point do you want to use (10, 15, 20, 25)?', default='25')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_argument('--effFitLimit', dest='effFitLimit', help='how many gentau_pt bins you wnat to consider for the fit of the turnON? (default: 49 bins = <150GeV)', default=49)
    # store parsed options
    args = parser.parse_args()

    if args.FE == None:
        print('** WARNING: no front-end options specified. Which front-end options do you want to use (threshold, supertrigger, bestchoice, bestcoarse, mixed)?')
        print('** WARNING: check which datasets you have available')
        print('** EXITING')
        exit()

    print('** INFO: using PU rejection BDT WP: '+args.PUWP)
    print('** INFO: using ISO BDT WP: '+args.ISOWP)

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'Threshold',
        'supertrigger' : 'Super Trigger Cell',
        'bestchoice'   : 'BestChoice',
        'bestcoarse'   : 'BestChoiceCoarse',
        'mixed'        : 'Mixed BC+STC',  
    }

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/DMsorted_fullPUnoPt{0}_fullISO{0}'.format("Rscld" if args.doRescale else "")
    outdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/mapping_fullPUnoPt{0}_fullISO{0}'.format("Rscld" if args.doRescale else "")
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/turnONs_fullPUnoPt{0}_fullISO{0}/PUWP{1}_ISOWP{2}'.format("Rscld" if args.doRescale else "",args.PUWP,args.ISOWP)
    os.system('mkdir -p '+indir+'; mkdir -p '+outdir+'; mkdir -p '+plotdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : indir+'/Training_PU200_th_PUWP{0}_ISOWP{1}_DMsorted.hdf5'.format(args.PUWP,args.ISOWP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : indir+'/Validation_PU200_th_PUWP{0}_ISOWP{1}_DMsorted.hdf5'.format(args.PUWP,args.ISOWP),
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    outFileTau_mapping_dict = {
        'threshold'    : outdir+'/Tau_PU200_th_PUWP{0}_ISOWP{1}_mapping.pkl'.format(args.PUWP,args.ISOWP),
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
    
    pts_Tau_dict = {}
    mappingTau_dict = {}
    mappingHH_dict = {}

    ptcut = 1
    etamin = 1.6

    online_thresholds = range(1, 100, 1)

    # colors to use for plotting
    colors_dict = {
        'threshold'    : 'blue',
        'mixed'        : 'fuchsia'
    }

    # legend to use for plotting
    legends_dict = {
        'threshold'    : 'Threshold 1.35 mipT',
        'mixed'        : 'Mixed BC + STC'
    }


    #*****************************************************************************#
    #************************ LOOP OVER FRONT-END METHODS ************************#

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do
            
        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting efficiency evaluation for the front-end option '+feNames_dict[name])

        
        store = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store[name]
        store.close()

        store = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store[name]
        store.close()

        ######################### SELECT EVENTS #########################   
        
        dfTau_dict[name] = pd.concat([dfTraining_dict[name],dfValidation_dict[name]], sort=False)
        dfTau_dict[name].query('sgnId==1 and gentau_bin_pt<={0}'.format(args.effFitLimit), inplace=True)

        # fill all the DM dataframes
        dfTauDM0_dict[name] = dfTau_dict[name].query('gentau_decayMode==0').copy(deep=True)
        dfTauDM1_dict[name] = dfTau_dict[name].query('gentau_decayMode==1').copy(deep=True)
        dfTauDM2_dict[name] = dfTau_dict[name].query('gentau_decayMode==2').copy(deep=True)


        ######################### BIN pt AND eta IN THE DATAFRAMES #########################
        # this is currwently done in the matching step --> use this only if you want to change the default binning: pt_binwidth = 3, eta_binwidth = 0.1
#
#        pt_binwidth = 3
#        eta_binwidth = 0.1
#
#        dfTau_dict[name]['gentau_vis_abseta'] = np.abs(dfTau_dict[name]['gentau_vis_eta'])
#        dfTau_dict[name]['gentau_bin_eta'] = ((dfTau_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
#        dfTau_dict[name]['gentau_bin_pt']  = ((dfTau_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')
#
#        dfTauDM0_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM0_dict[name]['gentau_vis_eta'])
#        dfTauDM0_dict[name]['gentau_bin_eta'] = ((dfTauDM0_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
#        dfTauDM0_dict[name]['gentau_bin_pt']  = ((dfTauDM0_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')
#
#        dfTauDM1_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM1_dict[name]['gentau_vis_eta'])
#        dfTauDM1_dict[name]['gentau_bin_eta'] = ((dfTauDM1_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
#        dfTauDM1_dict[name]['gentau_bin_pt']  = ((dfTauDM1_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')
#
#        dfTauDM2_dict[name]['gentau_vis_abseta'] = np.abs(dfTauDM2_dict[name]['gentau_vis_eta'])
#        dfTauDM2_dict[name]['gentau_bin_eta'] = ((dfTauDM2_dict[name]['gentau_vis_abseta'] - etamin)/eta_binwidth).astype('int32')
#        dfTauDM2_dict[name]['gentau_bin_pt']  = ((dfTauDM2_dict[name]['gentau_vis_pt'] - ptcut)/pt_binwidth).astype('int32')
#

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
            #                                           --> every efficiency_at{threshold} contains the value of the efficiency when applying the specific threshold 
            effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, args.PUWP, args.ISOWP)) # --> the output of this will be a series with idx=bin and entry=efficiency
            effVSpt_TauDM0_dict[name]['efficiency_at{0}GeV'.format(threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, args.PUWP, args.ISOWP))
            effVSpt_TauDM1_dict[name]['efficiency_at{0}GeV'.format(threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, args.PUWP, args.ISOWP))
            effVSpt_TauDM2_dict[name]['efficiency_at{0}GeV'.format(threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency(x, threshold, args.PUWP, args.ISOWP))

            effVSpt_Tau_dict[name]['efficiency_err_low_at{0}GeV'.format(threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=False))
            effVSpt_TauDM0_dict[name]['efficiency_err_low_at{0}GeV'.format(threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=False))
            effVSpt_TauDM1_dict[name]['efficiency_err_low_at{0}GeV'.format(threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=False))
            effVSpt_TauDM2_dict[name]['efficiency_err_low_at{0}GeV'.format(threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=False))

            effVSpt_Tau_dict[name]['efficiency_err_up_at{0}GeV'.format(threshold)] = dfTau_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=True))
            effVSpt_TauDM0_dict[name]['efficiency_err_up_at{0}GeV'.format(threshold)] = dfTauDM0_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=True))
            effVSpt_TauDM1_dict[name]['efficiency_err_up_at{0}GeV'.format(threshold)] = dfTauDM1_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=True))
            effVSpt_TauDM2_dict[name]['efficiency_err_up_at{0}GeV'.format(threshold)] = dfTauDM2_dict[name].groupby('gentau_bin_pt').apply(lambda x : efficiency_err(x, threshold, args.PUWP, args.ISOWP, upper=True))

        mappingTau_dict[name] = {'threshold':[], 'pt95':[], 'pt90':[], 'pt50':[]}
        for threshold in online_thresholds:
            # fixed a certain threshold, find the gentau_vis_pt value that has a 95% selection efficiency when applying that threshold on cl3d_pt
            # this values of pT_95 represent the offline threshold because 'a posteriori' when doing analysis we ask for a tau with pT>x and that pT for us now is just gentau_vis_pt
            pt_95 = np.interp(0.95, effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)], effVSpt_Tau_dict[name]['gentau_vis_pt'])#,right=-99,left=-98)
            pt_90 = np.interp(0.90, effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)], effVSpt_Tau_dict[name]['gentau_vis_pt'])#,right=-99,left=-98)
            pt_50 = np.interp(0.50, effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)], effVSpt_Tau_dict[name]['gentau_vis_pt'])#,right=-99,left=-98)
            mappingTau_dict[name]['threshold'].append(threshold)
            mappingTau_dict[name]['pt95'].append(pt_95)
            mappingTau_dict[name]['pt90'].append(pt_90)
            mappingTau_dict[name]['pt50'].append(pt_50)
            #print(threshold, pt_95)
        pts_Tau_dict[name] = pd.DataFrame(mappingTau_dict[name])

        save_obj(pts_Tau_dict[name],outFileTau_mapping_dict[name])    


        ######################### STORE VALUES FOR TURN-ON CURVES #########################

        y_eff_20_Tau = effVSpt_Tau_dict[name]['efficiency_at20GeV']
        y_eff_30_Tau = effVSpt_Tau_dict[name]['efficiency_at30GeV']
        y_eff_40_Tau = effVSpt_Tau_dict[name]['efficiency_at40GeV']
        y_eff_err_low_20_Tau = effVSpt_Tau_dict[name]['efficiency_err_low_at20GeV']
        y_eff_err_low_30_Tau = effVSpt_Tau_dict[name]['efficiency_err_low_at30GeV']
        y_eff_err_low_40_Tau = effVSpt_Tau_dict[name]['efficiency_err_low_at40GeV']
        y_eff_err_up_20_Tau = effVSpt_Tau_dict[name]['efficiency_err_up_at20GeV']
        y_eff_err_up_30_Tau = effVSpt_Tau_dict[name]['efficiency_err_up_at30GeV']
        y_eff_err_up_40_Tau = effVSpt_Tau_dict[name]['efficiency_err_up_at40GeV']
        x_Tau = effVSpt_Tau_dict[name]['gentau_vis_pt'] # is binned and the value is the mean of the entries per bin

        effTauDM0 = effVSpt_TauDM0_dict[name]['efficiency_at30GeV']
        effTauDM1 = effVSpt_TauDM1_dict[name]['efficiency_at30GeV']
        effTauDM2 = effVSpt_TauDM2_dict[name]['efficiency_at30GeV']
        eff_err_low_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_err_low_at30GeV']
        eff_err_low_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_err_low_at30GeV']
        eff_err_low_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_err_low_at30GeV']
        eff_err_up_TauDM0 = effVSpt_TauDM0_dict[name]['efficiency_err_up_at30GeV']
        eff_err_up_TauDM1 = effVSpt_TauDM1_dict[name]['efficiency_err_up_at30GeV']
        eff_err_up_TauDM2 = effVSpt_TauDM2_dict[name]['efficiency_err_up_at30GeV']
        x_DM0_Tau = effVSpt_TauDM0_dict[name]['gentau_vis_pt']
        x_DM1_Tau = effVSpt_TauDM1_dict[name]['gentau_vis_pt']
        x_DM2_Tau = effVSpt_TauDM2_dict[name]['gentau_vis_pt']

        print('\n** INFO: finished efficiency evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')

######################### MAKE PLOTS #########################

print('\n** INFO: plotting turnon curves')

matplotlib.rcParams.update({'font.size': 20})

lab_20 = r"$E_{T}^{L1,\tau}$ > 20 GeV"
lab_30 = r"$E_{T}^{L1,\tau}$ > 30 GeV"
lab_40 = r"$E_{T}^{L1,\tau}$ > 40 GeV"
plt.rcParams['legend.numpoints'] = 1


#*******************************
cmap = matplotlib.cm.get_cmap('tab20c'); i=0
name = 'threshold'
plt.figure(figsize=(10,10))
for threshold in online_thresholds:
    if not threshold%10:
        plt.errorbar(x_Tau,effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)],xerr=1,yerr=[effVSpt_Tau_dict[name]['efficiency_err_low_at{0}GeV'.format(threshold)],effVSpt_Tau_dict[name]['efficiency_err_up_at{0}GeV'.format(threshold)]],ls='None',label='{0}GeV'.format(threshold),lw=2,marker='o', color=cmap(i))

        p0 = [1, threshold, 1] 
        popt, pcov = curve_fit(sigmoid, x_Tau, effVSpt_Tau_dict[name]['efficiency_at{0}GeV'.format(threshold)], p0)
        plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 

plt.legend(loc = 'lower right')
plt.ylim(0., 1.10)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title('Efficiency vs pT - PUWP={0} ISOWP={1}'.format(args.PUWP,args.ISOWP))
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_ALL_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()
#*******************************


plt.figure(figsize=(10,10))
plt.errorbar(x_Tau,y_eff_20_Tau,xerr=1,yerr=[y_eff_err_low_20_Tau,y_eff_err_up_20_Tau],ls='None',label=lab_20,color='blue',lw=2,marker='o',mec='blue')
plt.errorbar(x_Tau,y_eff_30_Tau,xerr=1,yerr=[y_eff_err_low_30_Tau,y_eff_err_up_30_Tau],ls='None',label=lab_30,color='green',lw=2,marker='o',mec='green')
plt.errorbar(x_Tau,y_eff_40_Tau,xerr=1,yerr=[y_eff_err_low_40_Tau,y_eff_err_up_40_Tau],ls='None',label=lab_40,color='red',lw=2,marker='o',mec='red')

#p0 = [np.median(x_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_Tau[x_Tau>0], y_eff_20_Tau[y_eff_20_Tau>0], p0, sigma=y_eff_err_20_Tau[y_eff_err_20_Tau>0], absolute_sigma=True)
#plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='blue', lw=1.5)
#
#p0 = [np.median(x_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_Tau[x_Tau>0], y_eff_30_Tau[y_eff_30_Tau>0], p0, sigma=y_eff_err_30_Tau[y_eff_err_30_Tau>0], absolute_sigma=True)
#plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='green', lw=1.5)
#
#p0 = [np.median(x_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_Tau[x_Tau>0], y_eff_40_Tau[y_eff_40_Tau>0], p0, sigma=y_eff_err_40_Tau[y_eff_err_40_Tau>0], absolute_sigma=True)
#plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='red', lw=1.5)

p0 = [1, 20, 1] 
popt, pcov = curve_fit(sigmoid, x_Tau, y_eff_20_Tau, p0)
plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='blue', lw=1.5)

p0 = [1, 30, 1] 
popt, pcov = curve_fit(sigmoid, x_Tau, y_eff_30_Tau, p0)
plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='green', lw=1.5)

p0 = [1, 40, 1] 
popt, pcov = curve_fit(sigmoid, x_Tau, y_eff_40_Tau, p0)
plt.plot(x_Tau, sigmoid(x_Tau, *popt), '-', label='_', color='red', lw=1.5)

plt.legend(loc = 'lower right')
#plt.xlim(0, args.effFitLimit*3+3)
plt.ylim(0., 1.10)
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title('Efficiency vs pT - PUWP={0} ISOWP={1}'.format(args.PUWP,args.ISOWP))
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_isolated_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()

plt.figure(figsize=(10,10))
plt.errorbar(x_DM0_Tau,effTauDM0,xerr=1,yerr=[eff_err_low_TauDM0,eff_err_up_TauDM0],ls='None',label=r'1-prong',color='limegreen',lw=2,marker='o',mec='limegreen')
plt.errorbar(x_DM1_Tau,effTauDM1,xerr=1,yerr=[eff_err_low_TauDM1,eff_err_up_TauDM1],ls='None',label=r'1-prong + $\pi^{0}$',color='darkorange',lw=2,marker='o',mec='darkorange')
plt.errorbar(x_DM2_Tau,effTauDM2,xerr=1,yerr=[eff_err_low_TauDM2,eff_err_up_TauDM2],ls='None',label=r'3-prong (+ $\pi^{0}$)',color='fuchsia',lw=2,marker='o',mec='fuchsia')

#p0 = [np.median(x_DM0_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_DM0_Tau[x_DM0_Tau>0], effTauDM0[effTauDM0>0], p0, sigma=eff_errTauDM0[eff_errTauDM0>0], absolute_sigma=True)
#plt.plot(x_DM0_Tau, sigmoid(x_DM0_Tau, *popt), '-', label='_', color='limegreen', lw=1.5)
#
#p0 = [np.median(x_DM1_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_DM1_Tau[x_DM1_Tau>0], effTauDM1[effTauDM1>0], p0, sigma=eff_errTauDM1[eff_errTauDM1>0], absolute_sigma=True)
#plt.plot(x_DM1_Tau, sigmoid(x_DM1_Tau, *popt), '-', label='_', color='darkorange', lw=1.5)
#
#p0 = [np.median(x_DM2_Tau), 1] 
#popt, pcov = curve_fit(sigmoid, x_DM2_Tau[x_DM2_Tau>0], effTauDM2[effTauDM2>0], p0, sigma=eff_errTauDM2[eff_errTauDM2>0], absolute_sigma=True)
#plt.plot(x_DM2_Tau, sigmoid(x_DM2_Tau, *popt), '-', label='_', color='fuchsia', lw=1.5)

p0 = [1, 30, 1] 
popt, pcov = curve_fit(sigmoid, x_DM0_Tau, effTauDM0, p0)
plt.plot(x_DM0_Tau, sigmoid(x_DM0_Tau, *popt), '-', label='_', color='limegreen', lw=1.5)

p0 = [1, 30, 1] 
popt, pcov = curve_fit(sigmoid, x_DM1_Tau, effTauDM1, p0)
plt.plot(x_DM1_Tau, sigmoid(x_DM1_Tau, *popt), '-', label='_', color='darkorange', lw=1.5)

p0 = [1, 30, 1] 
popt, pcov = curve_fit(sigmoid, x_DM2_Tau, effTauDM2, p0)
plt.plot(x_DM2_Tau, sigmoid(x_DM2_Tau, *popt), '-', label='_', color='fuchsia', lw=1.5)

plt.legend(loc = 'lower right')
# txt = (r'Gen. $\tau$ decay mode:')
# t = plt.text(63,0.20, txt, ha='left', wrap=True)
# t.set_bbox(dict(facecolor='white', edgecolor='white'))
txt2 = (r'$E_{T}^{L1,\tau}$ > 30 GeV')
t2 = plt.text(55,0.25, txt2, ha='left')
t2.set_bbox(dict(facecolor='white', edgecolor='white'))
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'$\epsilon$')
plt.title('Efficiency vs pT - PUWP={0} ISOWP={1}'.format(args.PUWP,args.ISOWP))
plt.grid()
#plt.xlim(0, args.effFitLimit*3+3)
plt.ylim(0., 1.10)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/eff_vs_pt_DM_isolated_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()


plt.figure(figsize=(10,10))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at20GeV']
    #eff_err = df['efficiency_err_at20GeV']
    p0 = [1, 20, 1] 
    popt, pcov = curve_fit(sigmoid, df['gentau_vis_pt'], eff, p0)#, sigma=eff_err, absolute_sigma=True)
    plt.plot(df['gentau_vis_pt'], sigmoid(df['gentau_vis_pt'], *popt), label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.10)
#plt.xlim(0, args.effFitLimit*3+3)
plt.legend(loc = 'lower right')
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_20_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()

plt.figure(figsize=(10,10))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at30GeV']
    #eff_err = df['efficiency_err_at30GeV']
    p0 = [1, 30, 1] 
    popt, pcov = curve_fit(sigmoid, df['gentau_vis_pt'], eff, p0)#, sigma=eff_err, absolute_sigma=True)
    plt.plot(df['gentau_vis_pt'], sigmoid(df['gentau_vis_pt'], *popt), label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.10)
#plt.xlim(0, args.effFitLimit*3+3)
plt.legend(loc = 'lower right')
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_30_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()

plt.figure(figsize=(10,10))
for name in feNames_dict:
    if not name in args.FE: continue  
    df = effVSpt_Tau_dict[name]
    eff = df['efficiency_at40GeV']
    #eff_err = df['efficiency_err_at30GeV']
    p0 = [1, 40, 1] 
    popt, pcov = curve_fit(sigmoid, df['gentau_vis_pt'], eff, p0)#, sigma=eff_err, absolute_sigma=True)
    plt.plot(df['gentau_vis_pt'], sigmoid(df['gentau_vis_pt'], *popt), label=legends_dict[name], linewidth=2, color=colors_dict[name])
plt.ylim(0., 1.10)
#plt.xlim(0, args.effFitLimit*3+3)
plt.legend(loc = 'lower right')
plt.xlabel(r'Gen. tau $p_{T}\,[GeV]$')
plt.ylabel('Efficiency')
plt.grid()
plt.savefig(plotdir+'/eff_vs_pt_L1_40_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()

plt.figure(figsize=(10,10))
for name in feNames_dict:
    if not name in args.FE: continue 
    plt.plot(mappingTau_dict[name]['threshold'], mappingTau_dict[name]['pt95'], label=legends_dict[name], linewidth=2, color='blue')
    plt.plot(mappingTau_dict[name]['threshold'], mappingTau_dict[name]['pt90'], label=legends_dict[name], linewidth=2, color='red')
    plt.plot(mappingTau_dict[name]['threshold'], mappingTau_dict[name]['pt50'], label=legends_dict[name], linewidth=2, color='green')
#plt.legend(loc = 'lower right')
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
#plt.xlim(0, args.effFitLimit*3+3)
#plt.ylim(0, 130)
plt.grid()
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/L1_to_offline_PUWP{0}_ISOWP{1}.pdf'.format(args.PUWP,args.ISOWP))
plt.close()
