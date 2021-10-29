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
    # store parsed options
    args = parser.parse_args()

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/PUrejectionBDTskimmingAndOptimization{0}/FS_test'.format("_Rscld" if args.doRescale else "")
    os.system('mkdir -p '+plotdir)

    print('** INFO: prepearing dataset')
    # read training datasets
    store_tr = pd.HDFStore(indir+'/Training_PU200_th_matched.hdf5', mode='r')
    dfTr = store_tr['threshold']
    store_tr.close()
    # select events for the training
    dfTr = dfTr.query('sgnId==1 or cl3d_isbestmatch==False').copy(deep=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['sgnId', 'cl3d_coreshowerlength', 'cl3d_seetot', 'cl3d_srrmean', 'cl3d_srrtot', 'cl3d_spptot', 'cl3d_abseta', 'cl3d_meanz', 'cl3d_showerlength', 'cl3d_hoe', 'cl3d_szz', 'cl3d_seemax', 'cl3d_firstlayer', 'cl3d_sppmax', 'cl3d_srrmax']
    dfTr = dfTr[tokeep]

    # reduce dimension for testing new stuff
    #dfTr = pd.concat([dfTr[dfTr['sgnId']==1].sample(500), dfTr[dfTr['sgnId']==0].sample(500)], sort=False)
    # reduce dimension to speed up process but still have meaningfull results
    numberOfTaus = dfTr[dfTr['sgnId']==1].shape[0]
    dfTr = pd.concat([ dfTr[dfTr['sgnId']==1] , dfTr[dfTr['sgnId']==0].sample(numberOfTaus*2) ], sort=False)

    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',]
            features2saturate = ['cl3d_abseta', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
            saturation_dict = {'cl3d_abseta': [1.45, 3.2],
                               'cl3d_seetot': [0, 0.17],
                               'cl3d_seemax': [0, 0.6],
                               'cl3d_spptot': [0, 0.17],
                               'cl3d_sppmax': [0, 0.53],
                               'cl3d_szz': [0, 141.09],
                               'cl3d_srrtot': [0, 0.02],
                               'cl3d_srrmax': [0, 0.02],
                               'cl3d_srrmean': [0, 0.01],
                               'cl3d_hoe': [0, 63],
                               'cl3d_meanz': [305, 535]
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
    params_dict['nthread']            = 10   # limit number of threads
    params_dict['eta']                = 0.2 # learning rate
    params_dict['max_depth']          = 5    # maximum depth of a tree
    params_dict['subsample']          = 0.6 # fraction of events to train tree on
    params_dict['colsample_bytree']   = 0.7 # fraction of features to train tree on
    params_dict['objective']          = 'binary:logistic' # objective function
    params_dict['alpha']              = 10
    params_dict['lambda']             = 0.3
    num_trees = 30
    

    # features for BDT training - ORDERED BY ANY IMPORTANCE METRIC (from highest to lowest)
    featuresSHAP = ['cl3d_coreshowerlength', 'cl3d_seetot', 'cl3d_srrmean', 'cl3d_srrtot', 'cl3d_spptot', 'cl3d_abseta', 'cl3d_meanz', 'cl3d_showerlength', 'cl3d_hoe', 'cl3d_szz', 'cl3d_seemax', 'cl3d_firstlayer', 'cl3d_sppmax', 'cl3d_srrmax']
    featuresXGB  = ['cl3d_coreshowerlength', 'cl3d_seetot', 'cl3d_srrtot', 'cl3d_abseta', 'cl3d_srrmean', 'cl3d_spptot', 'cl3d_showerlength', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_szz', 'cl3d_seemax', 'cl3d_firstlayer', 'cl3d_sppmax', 'cl3d_srrmax']
    featuresRNDM = ['cl3d_abseta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

    # cerate train and label 
    X_train = dfTr[featuresSHAP] ; y_train = dfTr['sgnId']

    # do random search
    print('\n** INFO: doing random search')
    FS = ModuleFeatureSkimmer.FeatureSkimmer('PUBDT', featuresRNDM, params_dict, num_trees, 2, args.metric)
    randomScores, randomStds, randomFeats = FS.RandomSearch(X_train, y_train)
    FS.plotterFctFeats(plotdir, "RNDM")

    # do sequential backward search based on SHAP importance
    print('\n** INFO: doing sequential backward search based on SHAP importance')
    FS = ModuleFeatureSkimmer.FeatureSkimmer('PUBDT', featuresSHAP, params_dict, num_trees, 2, args.metric)
    SHAPsequentialScores, SHAPsequentialStds, SHAPsequentialFeats = FS.SequentialBackwardSearch(X_train, y_train)
    FS.plotterFctFeats(plotdir, "SHAP")

    # do sequential backward search based on XGB importance
    print('\n** INFO: doing sequential backward search based on XGB importance')
    FS = ModuleFeatureSkimmer.FeatureSkimmer('PUBDT', featuresXGB, params_dict, num_trees, 2, args.metric)
    XGBsequentialScores, XGBsequentialStds, XGBsequentialFeats = FS.SequentialBackwardSearch(X_train, y_train)
    FS.plotterFctFeats(plotdir, "XGB")

    # plot the three methods superimposed
    plt.figure(figsize=(10,8))
    x = np.linspace(2,len(randomScores),len(randomScores))
    plt.errorbar(x, randomScores, randomStds, label='Random skimming', color='red', lw=2, marker="h", elinewidth=2, alpha=0.7)
    plt.errorbar(x, SHAPsequentialScores, SHAPsequentialStds, label='SHAP backward skimming', color='limegreen', lw=2, marker="h", elinewidth=2, alpha=0.7)
    plt.errorbar(x, XGBsequentialScores, XGBsequentialStds, label='XGB backward skimming', color='blue', lw=2, marker="h", elinewidth=2, alpha=0.7)
    plt.legend(loc = 'lower right')
    plt.grid(linestyle=':')
    plt.xlabel('Number of features used in training')
    plt.ylabel(r'Optimization metric')
    plt.savefig(plotdir+'/FS_PUBDT_{0}AUROC.pdf'.format(args.metric))
    plt.close()
