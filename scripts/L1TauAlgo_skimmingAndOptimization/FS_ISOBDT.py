from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import ModuleFeatureSkimmer
import pandas as pd
import numpy as np
import argparse
import os


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_argument('--metric', dest='metric', help='which metric do you want to use for the evaluation? (train, test)', default="test")
    parser.add_argument('--PUWP', dest='PUWP', help='which PU working point do you want to use (90, 95, 99)?', default='90')
    parser.add_argument('--hardPUrej', dest='hardPUrej', help='apply hard PU rejection and do not consider PU categorized clusters for Iso variables? (99, 95, 90)', default='NO')
    # store parsed options
    # store parsed options
    args = parser.parse_args()

    # create needed folders
    tag1 = "Rscld" if args.doRescale else ""
    tag2 = "_{0}hardPUrej".format(args.hardPUrej) if args.hardPUrej != 'NO' else ""
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_skimPUnoPt{0}{1}'.format(tag1, tag2)
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/ISOrejectionBDT_againstPU_skimmingAndOptimization{0}/FS{1}'.format(tag1, tag2)
    os.system('mkdir -p '+plotdir)

    print('** INFO: prepearing dataset')
    # read training datasets
    store_tr = pd.HDFStore(indir+'/Training_PU200_th_isoCalculated.hdf5', mode='r')
    dfTr = store_tr['threshold']
    store_tr.close()
    # create a copy to cl3d_pt that will be used only for the training of the BDTs
    dfTr['cl3d_pt_tr'] = dfTr['cl3d_pt_c3'].copy(deep=True)
    # select events for the training (just genuine taus and qcd)
    #dfTr = dfTr.query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
    dfTr = dfTr.query('cl3d_pubdt_passWP{0}==True'.format(args.PUWP)).copy(deep=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['sgnId', 'cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    dfTr = dfTr[tokeep]

    # reduce dimension for testing new stuff
    #dfTr = pd.concat([dfTr[dfTr['sgnId']==1].sample(500), dfTr[dfTr['sgnId']==0].sample(500)], sort=False)

    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = ['cl3d_NclIso_dR4']
            features2saturate = ['cl3d_pt_tr', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
            saturation_dict = {'cl3d_pt_tr': [0, 200],
                               'cl3d_etIso_dR4': [0, 58],
                               'tower_etSgn_dRsgn1': [0, 194],
                               'tower_etSgn_dRsgn2': [0, 228],
                               'tower_etIso_dRsgn1_dRiso3': [0, 105],
                               'tower_etEmIso_dRsgn1_dRiso3': [0, 72],
                               'tower_etHadIso_dRsgn1_dRiso7': [0, 43],
                               'tower_etIso_dRsgn2_dRiso4': [0, 129],
                               'tower_etEmIso_dRsgn2_dRiso4': [0, 95],
                               'tower_etHadIso_dRsgn2_dRiso7': [0, 42]
                              }

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            # shift features to be shifted
            for feat in features2shift:
                dfTr[feat] = dfTr[feat] - 25

            # saturate features
            for feat in features2saturate:
                dfTr[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                
                # fill the bounds DF
                bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

            scale_range = [-32,32]
            MMS = MinMaxScaler(scale_range)

            for feat in features2saturate:
                MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
                dfTr[feat] = MMS.transform( np.array(dfTr[feat]).reshape(-1,1) )


    params_dict = {}
    params_dict['eval_metric']        = 'logloss'
    params_dict['nthread']            = 10  # limit number of threads
    params_dict['eta']                = 0.2 # learning rate
    params_dict['max_depth']          = 5   # maximum depth of a tree
    params_dict['subsample']          = 0.6 # fraction of events to train tree on
    params_dict['colsample_bytree']   = 0.7 # fraction of features to train tree on
    params_dict['objective']          = 'binary:logistic' # objective function
    params_dict['alpha']              = 10
    params_dict['lambda']             = 0.3
    num_trees = 30
    

    # features for BDT training - ORDERED BY ANY IMPORTANCE METRIC (from highest to lowest)
    featuresRNDM = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7']
    if args.PUWP == '99':
        featuresSHAP = ['tower_etSgn_dRsgn1', 'cl3d_coreshowerlength', 'cl3d_pt_tr', 'cl3d_abseta', 'cl3d_NclIso_dR4', 'tower_etIso_dRsgn1_dRiso3', 'tower_etSgn_dRsgn2', 'cl3d_seetot', 'cl3d_etIso_dR4', 'cl3d_srrmax', 'tower_etEmIso_dRsgn1_dRiso3', 'cl3d_spptot', 'cl3d_showerlength', 'cl3d_sppmax', 'cl3d_szz', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etIso_dRsgn2_dRiso4', 'cl3d_seemax', 'cl3d_firstlayer', 'tower_etHadIso_dRsgn2_dRiso7', 'cl3d_hoe', 'cl3d_srrmean', 'cl3d_srrtot', 'cl3d_meanz']
        #featuresXGB  = ['cl3d_etIso_dR4', 'tower_etSgn_dRsgn1', 'cl3d_NclIso_dR4', 'tower_etIso_dRsgn1_dRiso3', 'cl3d_srrtot', 'tower_etSgn_dRsgn2', 'cl3d_coreshowerlength', 'tower_etEmIso_dRsgn1_dRiso3', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_pt_tr', 'cl3d_abseta', 'cl3d_meanz', 'cl3d_spptot', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'cl3d_showerlength', 'tower_etHadIso_dRsgn2_dRiso7', 'tower_etEmIso_dRsgn2_dRiso4', 'cl3d_seetot', 'cl3d_sppmax', 'cl3d_firstlayer', 'cl3d_szz', 'cl3d_seemax', 'cl3d_srrmax']

    elif args.PUWP == '95':
        featuresSHAP = ['tower_etSgn_dRsgn1', 'cl3d_pt_tr', 'tower_etIso_dRsgn1_dRiso3', 'cl3d_abseta', 'cl3d_seetot', 'cl3d_coreshowerlength', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn2', 'cl3d_srrmax', 'cl3d_spptot', 'tower_etEmIso_dRsgn1_dRiso3', 'cl3d_szz', 'cl3d_etIso_dR4', 'cl3d_showerlength', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'cl3d_sppmax', 'tower_etEmIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7', 'cl3d_srrmean', 'cl3d_firstlayer', 'cl3d_seemax', 'cl3d_hoe', 'cl3d_srrtot', 'cl3d_meanz']
        #featuresXGB  = ['cl3d_NclIso_dR4', 'tower_etIso_dRsgn1_dRiso3', 'tower_etSgn_dRsgn1', 'tower_etEmIso_dRsgn1_dRiso3', 'cl3d_etIso_dR4', 'cl3d_srrtot', 'cl3d_pt_tr', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etSgn_dRsgn2', 'cl3d_srrmean', 'cl3d_abseta', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_spptot', 'cl3d_coreshowerlength', 'tower_etIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7', 'cl3d_showerlength', 'tower_etEmIso_dRsgn2_dRiso4', 'cl3d_seetot', 'cl3d_srrmax', 'cl3d_firstlayer', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_seemax']


    else:
        featuresSHAP = ['tower_etIso_dRsgn1_dRiso3', 'cl3d_abseta', 'tower_etSgn_dRsgn1', 'cl3d_pt_tr', 'cl3d_seetot', 'cl3d_NclIso_dR4', 'tower_etEmIso_dRsgn1_dRiso3', 'tower_etSgn_dRsgn2', 'cl3d_spptot', 'cl3d_srrmax', 'cl3d_coreshowerlength', 'tower_etHadIso_dRsgn1_dRiso7', 'cl3d_szz', 'cl3d_showerlength', 'tower_etIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn2_dRiso7', 'cl3d_sppmax', 'tower_etEmIso_dRsgn2_dRiso4', 'cl3d_etIso_dR4', 'cl3d_srrmean', 'cl3d_firstlayer', 'cl3d_hoe', 'cl3d_seemax', 'cl3d_srrtot', 'cl3d_meanz']
        #featuresXGB  = ['tower_etIso_dRsgn1_dRiso3', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'cl3d_srrtot', 'cl3d_pt_tr', 'cl3d_etIso_dR4', 'tower_etSgn_dRsgn2', 'cl3d_abseta', 'cl3d_hoe', 'tower_etEmIso_dRsgn1_dRiso3', 'cl3d_srrmean', 'cl3d_spptot', 'cl3d_coreshowerlength', 'cl3d_meanz', 'tower_etHadIso_dRsgn2_dRiso7', 'tower_etIso_dRsgn2_dRiso4', 'tower_etHadIso_dRsgn1_dRiso7', 'tower_etEmIso_dRsgn2_dRiso4', 'cl3d_seetot', 'cl3d_showerlength', 'cl3d_szz', 'cl3d_sppmax', 'cl3d_srrmax', 'cl3d_seemax', 'cl3d_firstlayer']

    
    # cerate train and test 
    X_train = dfTr[featuresSHAP] ; y_train = dfTr['sgnId']

    # do random search
    print('\n** INFO: doing random search')
    FS = ModuleFeatureSkimmer.FeatureSkimmer("ISOBDT", featuresRNDM, params_dict, num_trees, 2, args.metric)
    randomScores, randomStds, randomFeats = FS.RandomSearch(X_train, y_train)
    FS.plotterFctFeats(plotdir, "RNDM_PUWP{0}".format(args.PUWP))

    # do sequential backward search based on SHAP importance
    print('\n** INFO: doing sequential backward search based on SHAP importance')
    FS = ModuleFeatureSkimmer.FeatureSkimmer("ISOBDT", featuresSHAP, params_dict, num_trees, 2, args.metric)
    SHAPsequentialScores, SHAPsequentialStds, SHAPsequentialFeats = FS.SequentialBackwardSearch(X_train, y_train)
    FS.plotterFctFeats(plotdir, "SHAP_PUWP{0}".format(args.PUWP))

    # # do sequential backward search based on XGB importance
    # print('\n** INFO: doing sequential backward search based on XGB importance')
    # FS = ModuleFeatureSkimmer.FeatureSkimmer("ISOBDT", featuresXGB, params_dict, num_trees, 2, args.metric)
    # XGBsequentialScores, XGBsequentialStds, XGBsequentialFeats = FS.SequentialBackwardSearch(X_train, y_train)
    # FS.plotterFctFeats(plotdir, "XGB_PUWP{0}".format(args.PUWP))

    # plot the three methods superimposed
    plt.figure(figsize=(8,8))
    x = np.linspace(2,len(randomScores),len(randomScores))
    plt.errorbar(x, randomScores, randomStds, label='Random skimming', color='red', lw=2, marker="h", elinewidth=2, alpha=0.7)
    plt.errorbar(x, SHAPsequentialScores, SHAPsequentialStds, label='SHAP backward skimming', color='limegreen', lw=2, marker="h", elinewidth=2, alpha=0.7)
    #plt.errorbar(x, XGBsequentialScores, XGBsequentialStds, label='XGB backward skimming', color='blue', lw=2, marker="h", elinewidth=2, alpha=0.7)
    plt.legend(loc = 'lower right')
    plt.grid(linestyle=':')
    plt.xlabel('Number of features used in training')
    plt.ylabel(r'Optimization metric')
    plt.savefig(plotdir+'/FS_ISOBDT_{0}AUROC.pdf'.format(args.metric))
    plt.close()
