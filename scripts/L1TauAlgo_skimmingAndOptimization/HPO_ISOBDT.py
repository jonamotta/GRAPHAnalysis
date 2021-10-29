from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import ModuleHyperparameterOptimizer
import pandas as pd
import numpy as np
import argparse
import os


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_argument('--PUWP', dest='PUWP', help='which PU working point do you want to use (90, 95, 99)?', default='90')
    # store parsed options
    args = parser.parse_args()

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/isolated_fullPUnoPt{0}'.format("Rscld" if args.doRescale else "")
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/ISOrejectionBDTskimmingAndOptimization_Rscld{0}/HPO'.format("_Rscld" if args.doRescale else "")
    os.system('mkdir -p '+plotdir)

    print('** INFO: prepearing datasets')
    # read training and validation datasets
    store_tr = pd.HDFStore(indir+'/Training_PU200_th_isoCalculated.hdf5', mode='r')
    dfTr = store_tr['threshold']
    store_tr.close()
    store_val = pd.HDFStore(indir+'/Validation_PU200_th_isoCalculated.hdf5', mode='r')
    dfVal = store_val['threshold']
    store_val.close()
    # select events for the training
    dfTr = dfTr.query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
    dfVal = dfVal.query('cl3d_pubdt_passWP{0}==True and cl3d_isbestmatch==True'.format(args.PUWP)).copy(deep=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['sgnId', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_abseta', 'cl3d_hoe', 'cl3d_srrmean']
    dfTr = dfTr[tokeep]
    dfVal = dfVal[tokeep]

    # reduce dimension for testing new stuff
    dfTr = pd.concat([dfTr[dfTr['sgnId']==1].sample(500), dfTr[dfTr['sgnId']==0].sample(500)], sort=False)
    dfVal = pd.concat([dfVal[dfVal['sgnId']==1].sample(500), dfVal[dfVal['sgnId']==0].sample(500)], sort=False)

    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = 
            features2saturate = 
            saturation_dict = {

                              }

            #define a DF with the bound values of the features to use for the MinMaxScaler fit
            bounds4features = pd.DataFrame(columns=features2saturate)

            # shift features to be shifted
            for feat in features2shift:
                dfTr[feat] = dfTr[feat] - 25
                dfVal[feat] = dfVal[feat] - 25

            # saturate features
            for feat in features2saturate:
                dfTr[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                dfVal[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
                
                # fill the bounds DF
                bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

            scale_range = [-32,32]
            MMS = MinMaxScaler(scale_range)

            for feat in features2saturate:
                MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
                dfTr[feat] = MMS.transform( np.array(dfTr[feat]).reshape(-1,1) )
                dfVal[feat] = MMS.transform( np.array(dfVal[feat]).reshape(-1,1) )


    features = []

    # cerate train, test, and validation datasets
    X_train, X_test, y_train, y_test = train_test_split(dfTr[features], dfTr['sgnId'], stratify=dfTr['sgnId'], test_size=0.3)
    X_val = dfVal[features] ; y_val = dfVal['sgnId']

    min_num_trees = 5
    max_num_trees = 50
    hypar_bounds = {'eta'              : (0.1, 0.4),
                    'max_depth'        : (2, 25), 
                    'subsample'        : (0.1, 0.9),
                    'colsample_bytree' : (0.1, 0.9)
                   }

    print('\n** INFO: doing bayesian optimization')
    HPO = ModuleHyperparameterOptimizer.HyperparametersOptimizer("ISOBDT", features, hypar_bounds, min_num_trees, max_num_trees, X_train, X_test, X_val, y_train, y_test, y_val)
    #best_params, score, report = HPO.RunBayesianOptimization(init_points=2, n_iter=3)

    extended_best_params, extended_train_scores, extended_test_scores, extended_val_scores, extended_reports = HPO.RunExtendedBayesianOptimization(init_points=10, n_iter=40)

    extended_best_params, extended_reports = HPO.RunExtendedBayesianOptimization(init_points=10, n_iter=40)

    HPO.plotterAucVsParams(plotdir, 'train')
    HPO.plotterAucVsParams(plotdir, 'test')
    HPO.plotterAucVsParams(plotdir, 'val')
    HPO.plotterRmsleVsParams(plotdir, 'train')
    HPO.plotterRmsleVsParams(plotdir, 'test')
    HPO.plotterRmsleVsParams(plotdir, 'val')
    HPO.storeExtendedReport(plotdir, "ISOBDT")
    HPO.storeExtendedBestParams(plotdir, "ISOBDT")
