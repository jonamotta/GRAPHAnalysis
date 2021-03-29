import os
import sys
import pandas as pd
import numpy as np
import root_pandas
import pickle
import xgboost as xgb
from matplotlib import pyplot as plt
import matplotlib
import argparse
from decimal import *
getcontext().prec = 30

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
    dfdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/DMsorted'
    mapdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/mapping'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/rates_{0}'.format(PUbdtWP)
    os.system('mkdir -p '+plotdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileTraining_dict = {
        'threshold'    : dfdir+'/Training_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : dfdir+'/',
        'bestchoice'   : dfdir+'/',
        'bestcoarse'   : dfdir+'/',
        'mixed'        : dfdir+'/'
    }

    inFileValidation_dict = {
        'threshold'    : dfdir+'/Validation_PU200_th_PUWP{0}_DMsorted.hdf5'.format(args.WP),
        'supertrigger' : dfdir+'/',
        'bestchoice'   : dfdir+'/',
        'bestcoarse'   : dfdir+'/',
        'mixed'        : dfdir+'/'
    }

    inFileTau_mapping_dict = {
        'threshold'    : mapdir+'/TauInclusive_th_PU200_PUWP{0}_mapping.pkl'.format(args.WP),
        'supertrigger' : mapdir+'/',
        'bestchoice'   : mapdir+'/',
        'bestcoarse'   : mapdir+'/',
        'mixed'        : mapdir+'/'
    }

    dfTraining_dict = {}
    dfValidation_dict = {}
    dfNuTraining_dict = {}
    dfNuValidation_dict = {}
    dfNu_dict = {}

    pt95s_dict = {}

    events_total = {}
    events_pubdtwp = {}

    rates = {}
    rates_rescale = {}

    rates_DM0 = {}
    rates_rescale_DM0 = {}

    rates_DM1 = {}
    rates_rescale_DM1 = {}

    rates_DM2 = {}
    rates_rescale_DM2 = {}

    rates_DM3 = {}
    rates_rescale_DM3 = {}

    myscale=2760*11246./1000  # bunches * frequency , /1000 to pass to kHz

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

        pt95s_dict[name] = load_obj(inFileTau_mapping_dict[name])

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting rate evaluation for the front-end option '+feNames_dict[name])

        store_tr = pd.HDFStore(inFileTraining_dict[name], mode='r')
        dfTraining_dict[name] = store_tr[name]
        store_tr.close()
        
        store_val = pd.HDFStore(inFileValidation_dict[name], mode='r')
        dfValidation_dict[name] = store_val[name]
        store_val.close()


        ######################### SELECT EVENTS #########################  

        dfNuTraining_dict[name] = dfTraining_dict[name].query('dataset==2')
        dfNuValidation_dict[name] = dfValidation_dict[name].query('dataset==2')

        dfNu_dict[name] = pd.concat([dfNuTraining_dict[name],dfNuValidation_dict[name]],sort=False)
        dfNu_dict[name].dropna(axis=1,inplace=True) # drop fake columns created at concatenation

        events_total[name] = np.unique(dfNu_dict[name].reset_index()['event']).shape[0]

        dfNu_dict[name] = dfNu_dict[name].query('cl3d_pubdt_pass{0}==True'.format(PUbdtWP))
        dfNu_dict[name].set_index('event', inplace=True)

        group = dfNu_dict[name].groupby('event')
        dfNu_dict[name]['leading_cl3d_pt_c3'] = group['cl3d_pt_c3'].max()

        dfNu_dict[name]['cl3d_isleading'] = dfNu_dict[name]['leading_cl3d_pt_c3'] == dfNu_dict[name]['cl3d_pt_c3']

        dfNu_dict[name] = dfNu_dict[name].query('cl3d_isleading==True')

        events_pubdtwp[name] = np.unique(dfNu_dict[name].reset_index()['event']).shape[0]

        print('\n** INFO: leading_clusters / total_clusters = {0} '.format(float(events_pubdtwp[name])/float(events_total[name])))

        tmp = dfNu_dict[name].query('cl3d_pt_c3>10')
        tmp['cl3d_pt_95'] = tmp.cl3d_pt_c3.apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        rate = np.arange( float(tmp.shape[0])/events_total[name], 0., (-1.)/events_total[name])
        rate = rate[rate>0]

        rates[name] = ( np.sort(tmp.cl3d_pt_c3), rate )
        rates_rescale[name] = ( np.sort(tmp.cl3d_pt_95), rate )

        # print('** INFO: L1 threshold: {0}'.format(rates[name][0])) #L1 threshold
        # print('** INFO: Rate: {0}'.format(rates[name][1])) #Rate
        # print('** INFO: Offline threshold: {0}'.format(rates_rescale[name][0])) #Offline threshold
        # print('** INFO: Rate: {0}'.format(rates_rescale[name][1])) #Rate

        # DMs

        tmp_DM0 = tmp.query('cl3d_predDM==0')
        tmp_DM0['cl3d_pt_95'] = tmp_DM0.cl3d_pt_c3.apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        rate_DM0 = np.arange( float(tmp_DM0.shape[0])/events_total[name], 0., (-1.)/events_total[name])
        rate_DM0 = rate_DM0[rate_DM0>0]

        rates_DM0[name] = ( np.sort(tmp_DM0.cl3d_pt_c3), rate_DM0 )
        rates_rescale_DM0[name] = ( np.sort(tmp_DM0.cl3d_pt_95), rate_DM0 )

        ########

        tmp_DM1 = tmp.query('cl3d_predDM==1')
        tmp_DM1['cl3d_pt_95'] = tmp_DM1.cl3d_pt_c3.apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        rate_DM1 = np.arange( float(tmp_DM1.shape[0])/events_total[name], 0., (-1.)/events_total[name])
        rate_DM1 = rate_DM1[rate_DM1>0]

        rates_DM1[name] = ( np.sort(tmp_DM1.cl3d_pt_c3), rate_DM1 )
        rates_rescale_DM1[name] = ( np.sort(tmp_DM1.cl3d_pt_95), rate_DM1 )

        ########

        tmp_DM2 = tmp.query('cl3d_predDM==2')
        tmp_DM2['cl3d_pt_95'] = tmp_DM2.cl3d_pt_c3.apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        rate_DM2 = np.arange( float(tmp_DM2.shape[0])/events_total[name], 0., (-1.)/events_total[name])
        rate_DM2 = rate_DM2[rate_DM2>0]

        rates_DM2[name] = ( np.sort(tmp_DM2.cl3d_pt_c3), rate_DM2 )
        rates_rescale_DM2[name] = ( np.sort(tmp_DM2.cl3d_pt_95), rate_DM2 )

        ########

        tmp_DM3 = tmp.query('cl3d_predDM==3')
        tmp_DM3['cl3d_pt_95'] = tmp_DM3.cl3d_pt_c3.apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        rate_DM3 = np.arange( float(tmp_DM3.shape[0])/events_total[name], 0., (-1.)/events_total[name])
        rate_DM3 = rate_DM3[rate_DM3>0]

        rates_DM3[name] = ( np.sort(tmp_DM3.cl3d_pt_c3), rate_DM3 )
        rates_rescale_DM3[name] = ( np.sort(tmp_DM3.cl3d_pt_95), rate_DM3 )


        print('\n** INFO: finished rate evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')



interp_points = np.arange(20, 90, 0.5)

plt.figure(figsize=(8,8))

for name in feNames_dict:
    if not name in args.FE: continue # skip the front-end options that we do not want to do

    rate = rates_rescale[name]
    plt.plot(rate[0], rate[1]*myscale, linewidth=2, color='black', label='All')
    rate_interp = np.interp(interp_points, rate[0], rate[1]*myscale)
  
    rate_DM0 = rates_rescale_DM0[name]
    plt.plot(rate_DM0[0], rate_DM0[1]*myscale, linewidth=2, color='limegreen', label='1-prong')
    rate_interp_DM0 = np.interp(interp_points, rate_DM0[0], rate_DM0[1]*myscale)
  
    rate_DM1 = rates_rescale_DM1[name]
    #myrate = rate_DM1[1]
    #rate_DM1[1].resize(17759, refcheck=False)
    plt.plot(rate_DM1[0], rate_DM1[1]*myscale, linewidth=2, color='blue', label=r'1-prong + $\pi^{0}$')
    rate_interp_DM1 = np.interp(interp_points, rate_DM1[0], rate_DM1[1]*myscale)
  
    rate_DM2 = rates_rescale_DM2[name]
    plt.plot(rate_DM2[0], rate_DM2[1]*myscale, linewidth=2, color='fuchsia', label=r'3-prongs (+ $\pi^{0}$)')
    rate_interp_DM2 = np.interp(interp_points, rate_DM2[0], rate_DM2[1]*myscale)

    #rate_DM3 = rates_rescale_DM3[name]
    #plt.plot(rate_DM3[0], rate_DM3[1]*myscale, linewidth=2, color='cyan', label=r'QCD')
    #rate_interp_DM3 = np.interp(interp_points, rate_DM3[0], rate_DM3[1]*myscale)

plt.yscale("log")
plt.xlim(20, 90)
#plt.ylim(10,15000)
legend = plt.legend(loc = 'upper right')
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel('Rate [kHz]')
txt3 = ('CMS')
t3 = plt.text(22,17000, txt3, ha='left', wrap=True, fontsize=18, fontweight='bold')
txt4 = ('Phase-2 Simulation')
t4 = plt.text(30,17000, txt4, ha='left', wrap=True, fontsize=18, fontstyle='italic')
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/rate_offline_'+PUbdtWP+'_'+'.pdf')





  

  


#####



'''
interp_points = np.arange(10, 80, 0.5)
rate_ref = rates[0]
rate_ref_interp = np.interp(interp_points, rate_ref[0], rate_ref[1]/scale)
fig, axs = plt.figure(figsize=(8,8))
for name in file_in_nu:
  rate = rates[name]
  axs[0].plot(rate[0], rate[1]*myscale, label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1]*myscale)
plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'_v3.png')
plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'_v3.pdf')
axs[0].set_ylabel('Rate [kHz]')
axs[0].set_xlabel('L1 threshold [GeV]')
'''


'''
fig, axs = plt.subplots(2, 1, figsize=(15,20))
for name in file_in_nu:

  rate = rates[name]
  #plt.plot(rate[0],rate[1]*scale,label=legends[name], linewidth=2, color=colors[name])
  #plt.plot(rate[0],rate[1],label=legends[name], linewidth=2, color=colors[name])
  axs[0].plot(rate[0], rate[1]*myscale, label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1]*myscale)
  #ratio =  rate_interp / rate_ref_interp
  #axs[1].plot(interp_points, ratio, label=legends[name], linewidth=2, color=colors[name])

axs[0].legend(loc = 'upper right', fontsize=22)
axs[0].set_yscale("log")
#axs[0].set_xlim(10, 80)
#axs[0].set_ylim(6e-4, 1)
axs[0].grid()
#axs[0].set_xlabel('L1 threshold [GeV]')
axs[0].set_ylabel('Rate [a.u.]')
axs[1].set_ylim(0.6,2.0)
axs[1].set_xlim(10, 80)
axs[1].set_xlabel('L1 threshold [GeV]')
#axs[1].set_ylabel('Ratio to STC')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'_v2.png')
plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'_v2.pdf')
'''


#plt.legend(loc = 'upper right', fontsize=16)
#plt.yscale('log')
#plt.xlabel('L1 threshold [GeV]')
#plt.ylabel('Rate [a.u.]')
#plt.xlim(20, 80)
#plt.ylim(0.0007, 0.3)
#plt.grid()
#plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'.png')
#plt.savefig(plotdir+'rate_L1_algo'+algo+'_WP'+PUBDTWP+'.pdf')
#plt.show()

'''
plt.figure(figsize=(10,10))
for name in file_in_nu:
  rate = rates_rescale[name]
  #plt.plot(rate[0],rate[1]*scale,label=legends[name], linewidth=2, color=colors[name])
  plt.plot(rate[0],rate[1],label=legends[name], linewidth=2, color=colors[name])
plt.legend(loc = 'upper right', fontsize=16)
plt.yscale('log')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel('Rate [a.u.]')
plt.xlim(30, 90)
plt.ylim(0.1, 0.6)
plt.grid()
plt.savefig(plotdir+'rate_offline_algo'+algo+'_WP'+PUBDTWP+'.png')
plt.savefig(plotdir+'rate_offline_algo'+algo+'_WP'+PUBDTWP+'.pdf')
plt.show()
'''
'''
interp_points = np.arange(10, 120, 0.5)
rate_ref = rates_rescale[0]
rate_ref_interp = np.interp(interp_points, rate_ref[0], rate_ref[1]*scale)
fig, axs = plt.subplots(2, 1, figsize=(15,20))

for name in file_in_nu:

  rate = rates_rescale[name]
  #plt.plot(rate[0],rate[1]*scale,label=legends[name], linewidth=2, color=colors[name])
  #plt.plot(rate[0],rate[1],label=legends[name], linewidth=2, color=colors[name])
  axs[0].plot(rate[0], rate[1]*myscale, label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1]*myscale)
  #ratio =  rate_interp / rate_ref_interp
  #axs[1].plot(interp_points, ratio, label=legends[name], linewidth=2, color=colors[name])

axs[0].legend(loc = 'upper right', fontsize=22)
axs[0].set_yscale("log")
axs[0].set_xlim(20, 90)
#axs[0].set_ylim(6e-4, 1)
axs[0].grid()
#axs[0].set_xlabel('Offline threshold [GeV]')
axs[0].set_ylabel('Rate [a.u.]')
#axs[1].set_ylim(0.6,2.0)
#axs[1].set_xlim(20, 90)
axs[1].set_xlabel('Offline threshold [GeV]')
#axs[1].set_ylabel('Ratio to STC')
axs[1].set_ylabel('Ratio to Threshold')
axs[1].grid()
plt.savefig(plotdir+'rate_offline_algo'+algo+'_WP'+PUBDTWP+'_v2.png')
plt.savefig(plotdir+'rate_offline_algo'+algo+'_WP'+PUBDTWP+'_v2.pdf')
'''


