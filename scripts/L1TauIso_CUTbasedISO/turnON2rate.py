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
    parser.add_argument('--WP', dest='WP', help='which working point do you want to use (90, 95, 99)?', default='99')
    parser.add_argument('--useNu', dest='useNu', help='use RelValNu for rate evaluation?', action='store_true', default=False)
    parser.add_argument('--useMinbias', dest='useMinbias', help='use Minbias for rate evaluation?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    if not args.useNu and not args.useMinbias:
        print('** WARNING: no dataset specified. What do you want to do (useNu, useMinbias)?')
        print('** EXITING')
        exit()

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
    mapdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/pklModels/mapping'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/plots/rates'
    os.system('mkdir -p '+plotdir)

    # define the input and output dictionaries for the handling of different datasets
    inFileMinbias_dict = {
        'threshold'    : indir+'/Minbias_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileNu_dict = {
        'threshold'    : indir+'/RelValNu_PU200_th_L1candidates.hdf5',
        'supertrigger' : indir+'/',
        'bestchoice'   : indir+'/',
        'bestcoarse'   : indir+'/',
        'mixed'        : indir+'/'
    }

    inFileTau_mapping_dict = {
        'threshold'    : mapdir+'/AllTau_PU200_th_isolation_PUWP{0}_dRsgn{1}_mapping.pkl'.format(args.WP,int(dRsgn*10)),
        'supertrigger' : mapdir+'/',
        'bestchoice'   : mapdir+'/',
        'bestcoarse'   : mapdir+'/',
        'mixed'        : mapdir+'/'
    }

    pt95s_dict = {}

    events_total = {}
    events_passing = {}

    rates_online = {}
    rates_offline = {}

    rates_online_DM0 = {}
    rates_offline_DM0 = {}

    rates_online_DM1 = {}
    rates_offline_DM1 = {}

    rates_online_DM2 = {}
    rates_offline_DM2 = {}

    rates_online_DM3 = {}
    rates_offline_DM3 = {}

    events_frequency=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf

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
        clEtIso = 10 #GeV
    elif dRsgn == 0.2:
        dRiso = 0.4
        dRisoEm = 0.4
        dRcl3d = 0.4
        twEtiso = 80 #GeV
        twEtEmiso = 60 #GeV
        clEtIso = 10 #GeV

    for name in feNames_dict:
        if not name in args.FE: continue # skip the front-end options that we do not want to do

        pt95s_dict[name] = load_obj(inFileTau_mapping_dict[name])

        print('---------------------------------------------------------------------------------------')
        print('** INFO: starting rate evaluation for the front-end option '+feNames_dict[name])

        if args.useMinbias:
            store_tr = pd.HDFStore(inFileMinbias_dict[name], mode='r')
            df4Rate = store_tr[name]
            store_tr.close()

            df4Rate.reset_index(inplace=True)
            events_total[name] = 258597.0
        if args.useNu:
            store_tr = pd.HDFStore(inFileNu_dict[name], mode='r')
            df4Rate = store_tr[name]
            store_tr.close()

            df4Rate.reset_index(inplace=True)
            events_total[name] = 9000.0

        ######################### SELECT EVENTS #########################  
        
        # the "fake candidates" from minbias/nu are taken as the highest pt cluster passing the PU rejection per event
        # the pt selection at this stage is already done, so now we ask only for the PUbdtWP and the isolation of the candidate
        df4Rate.query('cl3d_pt_c3>10 and cl3d_pubdt_passWP{0}==True and cl3d_predDM_PUWP{0}!=3 and (tower_etIso_dRsgn{1}_dRiso{2}<={3} and tower_etEmIso_dRsgn{1}_dRiso{4}<={5} and cl3d_etIso_dR{6}<={7})'.format(args.WP, int(dRsgn*10), int(dRiso*10), twEtiso, int(dRisoEm*10), twEtEmiso, int(dRcl3d*10), clEtIso), inplace=True)
        #df4Rate.query('cl3d_pt_c3>10 and cl3d_pubdt_passWP{0}==True'.format(args.WP), inplace=True)
        events_passing[name] = np.unique(df4Rate.reset_index()['event']).shape[0]
        print('\n** INFO: selected_clusters / total_clusters = {0} / {1} = {2} '.format(events_passing[name], events_total[name], float(events_passing[name])/float(events_total[name])))

        # find the gentau_vis_pt with 95% efficiency of selection if we were to apply a threshold on cl3d_pt equal to the specific cl3d_pt we are considering
        # this  essentially returns the offline threshold that wen applied corresponds to applying an online threshold equal to cl3d_pt_c3
        df4Rate['cl3d_pt95'] = df4Rate['cl3d_pt_c3'].apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        df4Rate.sort_values('cl3d_pt_c3', inplace=True)
        rate = []
        for thr in df4Rate['cl3d_pt_c3']:
            single_rate = len(df4Rate.query('cl3d_pt_c3>{0}'.format(float(thr))))
            rate.append(float(single_rate))

        rates_online[name] = ( np.sort(df4Rate['cl3d_pt_c3']), np.array(rate)/events_total[name]*events_frequency )
        rates_offline[name] = ( np.sort(df4Rate['cl3d_pt95']), np.array(rate)/events_total[name]*events_frequency )

        # print('** INFO: L1 threshold: {0}'.format(rates[name][0])) #L1 threshold
        # print('** INFO: Rate: {0}'.format(rates[name][1])) #Rate
        # print('** INFO: Offline threshold: {0}'.format(rates_offline[name][0])) #Offline threshold
        # print('** INFO: Rate: {0}'.format(rates_offline[name][1])) #Rate

        # DMs

        tmp_DM0 = df4Rate.query('cl3d_predDM_PUWP{0}==0'.format(args.WP)).copy(deep=True)
        tmp_DM0['cl3d_pt95'] = tmp_DM0['cl3d_pt_c3'].apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        tmp_DM0.sort_values('cl3d_pt_c3', inplace=True)
        rate_DM0 = []
        for thr in tmp_DM0['cl3d_pt_c3']:
            single_rate = len(tmp_DM0.query('cl3d_pt_c3>{0}'.format(float(thr))))
            rate_DM0.append(float(single_rate))

        rates_online_DM0[name] = ( np.sort(tmp_DM0['cl3d_pt_c3']), np.array(rate_DM0)/events_total[name]*events_frequency )
        rates_offline_DM0[name] = ( np.sort(tmp_DM0['cl3d_pt95']), np.array(rate_DM0)/events_total[name]*events_frequency )

        ########

        tmp_DM1 = df4Rate.query('cl3d_predDM_PUWP{0}==1'.format(args.WP)).copy(deep=True)
        tmp_DM1['cl3d_pt95'] = tmp_DM1['cl3d_pt_c3'].apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        tmp_DM1.sort_values('cl3d_pt_c3', inplace=True)
        rate_DM1 = []
        for thr in tmp_DM1['cl3d_pt_c3']:
            single_rate = len(tmp_DM1.query('cl3d_pt_c3>{0}'.format(float(thr))))
            rate_DM1.append(float(single_rate))

        rates_online_DM1[name] = ( np.sort(tmp_DM1['cl3d_pt_c3']), np.array(rate_DM1)/events_total[name]*events_frequency )
        rates_offline_DM1[name] = ( np.sort(tmp_DM1['cl3d_pt95']), np.array(rate_DM1)/events_total[name]*events_frequency )

        ########

        tmp_DM2 = df4Rate.query('cl3d_predDM_PUWP{0}==2'.format(args.WP)).copy(deep=True)
        tmp_DM2['cl3d_pt95'] = tmp_DM2['cl3d_pt_c3'].apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        tmp_DM2.sort_values('cl3d_pt_c3', inplace=True)
        rate_DM2 = []
        for thr in tmp_DM2['cl3d_pt_c3']:
            single_rate = len(tmp_DM2.query('cl3d_pt_c3>{0}'.format(float(thr))))
            rate_DM2.append(float(single_rate))

        rates_online_DM2[name] = ( np.sort(tmp_DM2['cl3d_pt_c3']), np.array(rate_DM2)/events_total[name]*events_frequency )
        rates_offline_DM2[name] = ( np.sort(tmp_DM2['cl3d_pt95']), np.array(rate_DM2)/events_total[name]*events_frequency )

        ########

        #tmp_DM3 = df4Rate.query('cl3d_predDM_PUWP{0}==3'.format(args.WP)).copy(deep=True)
        #tmp_DM3['cl3d_pt95'] = tmp_DM3['cl3d_pt_c3'].apply(lambda x : np.interp(x, pt95s_dict[name].threshold, pt95s_dict[name].pt95))

        #rate_DM3 = np.arange( float(tmp_DM3.shape[0])/events_total[name], 0., -1./events_total[name])
        #rate_DM3 = rate_DM3[rate_DM3>0]

        #rates_online_DM3[name] = ( np.sort(tmp_DM3['cl3d_pt_c3']), rate_DM3 )
        #rates_offline_DM3[name] = ( np.sort(tmp_DM3['cl3d_pt95']), rate_DM3 )


        print('\n** INFO: finished rate evaluation for the front-end option '+feNames_dict[name])
        print('---------------------------------------------------------------------------------------')



interp_points = np.arange(20, 90, 0.5)

plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue # skip the front-end options that we do not want to do

    plt.plot(rates_offline[name][0], rates_offline[name][1], linewidth=2, color='black', label='All')
    rate_interp = np.interp(interp_points, rates_offline[name][0], rates_offline[name][1])
  
    plt.plot(rates_offline_DM0[name][0], rates_offline_DM0[name][1], linewidth=2, color='limegreen', label='1-prong')
    rate_interp_DM0 = np.interp(interp_points, rates_offline_DM0[name][0], rates_offline_DM0[name][1])
  
    plt.plot(rates_offline_DM1[name][0], rates_offline_DM1[name][1], linewidth=2, color='blue', label=r'1-prong + $\pi^{0}$')
    rate_interp_DM1 = np.interp(interp_points, rates_offline_DM1[name][0], rates_offline_DM1[name][1])
  
    plt.plot(rates_offline_DM2[name][0], rates_offline_DM2[name][1], linewidth=2, color='fuchsia', label=r'3-prongs (+ $\pi^{0}$)')
    rate_interp_DM2 = np.interp(interp_points, rates_offline_DM2[name][0], rates_offline_DM2[name][1])

    #rate_DM3 = rates_offline_DM3[name]
    #plt.plot(rate_DM3[0], rate_DM3[1], linewidth=2, color='cyan', label=r'QCD')
    #rate_interp_DM3 = np.interp(interp_points, rate_DM3[0], rate_DM3[1])


plt.yscale("log")
plt.xlim(20, 90)
#plt.ylim(10,15000)
legend = plt.legend(loc = 'upper right')
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel('Rate [kHz]')
plt.title('Rate - PUWP{0} \n dRsgn={1} dRiso={2} dRcl3d={3} twEtIso={4} clEtIso={5}'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso), fontsize=15)
plt.subplots_adjust(bottom=0.12)
if args.useNu: plt.savefig(plotdir+'/rate_offline_Nu_isolated_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
if args.useMinbias: plt.savefig(plotdir+'/rate_offline_Minbias_isolated_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()


plt.figure(figsize=(8,8))
for name in feNames_dict:
    if not name in args.FE: continue # skip the front-end options that we do not want to do
    plt.plot(rates_online[name][0], rates_online[name][1], linewidth=2, color='black', label='All')
    plt.plot(rates_online_DM0[name][0], rates_online_DM0[name][1], linewidth=2, color='limegreen', label='1-prong')
    plt.plot(rates_online_DM1[name][0], rates_online_DM1[name][1], linewidth=2, color='blue', label=r'1-prong + $\pi^{0}$')
    plt.plot(rates_online_DM2[name][0], rates_online_DM2[name][1], linewidth=2, color='fuchsia', label=r'3-prongs (+ $\pi^{0}$)')
plt.yscale("log")
plt.xlim(20, 90)
#plt.ylim(10,15000)
legend = plt.legend(loc = 'upper right')
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel('Rate [kHz]')
plt.title('Rate - PUWP{0} \n dRsgn={1} dRiso={2} dRcl3d={3} twEtIso={4} clEtIso={5}'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso), fontsize=15)
plt.subplots_adjust(bottom=0.12)
plt.savefig(plotdir+'/rate_online_isolated_PUWP{0}_dRsgn{1}_dRiso{2}_dRcl3d{3}_twEtIso{4}_clEtIso{5}.pdf'.format(args.WP,dRsgn,dRiso,dRcl3d,twEtiso,clEtIso))
plt.close()




  

  


#####



'''
interp_points = np.arange(10, 80, 0.5)
rate_ref = rates[0]
rate_ref_interp = np.interp(interp_points, rate_ref[0], rate_ref[1]/scale)
fig, axs = plt.figure(figsize=(8,8))
for name in file_in_nu:
  rate = rates[name]
  axs[0].plot(rate[0], rate[1], label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1])
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
  axs[0].plot(rate[0], rate[1], label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1])
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
  rate = rates_offline[name]
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
rate_ref = rates_offline[0]
rate_ref_interp = np.interp(interp_points, rate_ref[0], rate_ref[1]*scale)
fig, axs = plt.subplots(2, 1, figsize=(15,20))

for name in file_in_nu:

  rate = rates_offline[name]
  #plt.plot(rate[0],rate[1]*scale,label=legends[name], linewidth=2, color=colors[name])
  #plt.plot(rate[0],rate[1],label=legends[name], linewidth=2, color=colors[name])
  axs[0].plot(rate[0], rate[1], label=legends[name], linewidth=2, color=colors[name])
  rate_interp = np.interp(interp_points, rate[0], rate[1])
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


