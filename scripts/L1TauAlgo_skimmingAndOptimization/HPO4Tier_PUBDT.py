from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import ModuleHyperparameterGridOptimizer
import pandas as pd
import numpy as np
import argparse
import os


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--output', dest='output', help='output folder', default='none')
    parser.add_argument('--FE', dest='FE', help='front-end', default='threshold')
    parser.add_argument('--num_trees', dest='num_trees', help='number of boosting rounds', default=None)
    parser.add_argument('--tag', dest='tag', help='output folder tag name', default='')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    outdir = args.output

    # dictionary of front-end options
    feNames_dict = {
        'threshold'    : 'th',
        'mixed'        : 'mx',  
    }
    fe = feNames_dict[args.FE]

    print('** INFO: prepearing datasets')
    # read training and validation datasets
    store_tr = pd.HDFStore(indir+'/Training_PU200_{0}_matched.hdf5'.format(fe), mode='r')
    dfTr = store_tr['threshold']
    store_tr.close()
    store_val = pd.HDFStore(indir+'/Validation_PU200_{0}_matched.hdf5'.format(fe), mode='r')
    dfVal = store_val['threshold']
    store_val.close()
    # select events for the training
    dfTr.query('sgnId==1 or cl3d_isbestmatch==False', inplace=True)
    dfVal.query('sgnId==1 or cl3d_isbestmatch==False', inplace=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['sgnId', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_abseta', 'cl3d_hoe', 'cl3d_srrmean']
    dfTr = dfTr[tokeep].copy(deep=True)
    dfVal = dfVal[tokeep].copy(deep=True)

    # reduce dimension for testing new stuff
    #dfTr = pd.concat([dfTr[dfTr['sgnId']==1].sample(500), dfTr[dfTr['sgnId']==0].sample(500)], sort=False)
    #dfVal = pd.concat([dfVal[dfVal['sgnId']==1].sample(500), dfVal[dfVal['sgnId']==0].sample(500)], sort=False)
    # reduce dimension to speed up process but still have meaningfull results
    numberOfTaus = dfTr[dfTr['sgnId']==1].shape[0]
    dfTr = pd.concat([ dfTr[dfTr['sgnId']==1] , dfTr[dfTr['sgnId']==0].sample(numberOfTaus*3) ], sort=False).copy(deep=True)
    numberOfTaus = dfVal[dfVal['sgnId']==1].shape[0]
    dfVal = pd.concat([ dfVal[dfVal['sgnId']==1] , dfVal[dfVal['sgnId']==0].sample(numberOfTaus*3) ], sort=False).copy(deep=True)


    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = ['cl3d_coreshowerlength']
            features2saturate = ['cl3d_abseta', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_meanz']
            saturation_dict = {'cl3d_abseta': [1.45, 3.2],
                               'cl3d_srrtot': [0, 0.02],
                               'cl3d_srrmean': [0, 0.01],
                               'cl3d_meanz': [305, 535]
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


    features = ['cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_abseta', 'cl3d_hoe', 'cl3d_srrmean']

    # cerate train, test, and validation datasets
    X_train, X_test, y_train, y_test = train_test_split(dfTr[features], dfTr['sgnId'], stratify=dfTr['sgnId'], test_size=0.3)
    X_val = dfVal[features] ; y_val = dfVal['sgnId']

    # set to the same number to have only one hypothesis tested at a time
    min_num_trees = int(args.num_trees)
    max_num_trees = int(args.num_trees)

    gridsearch_params = {'max_depth'        : [2, 3, 4, 5],
                         'eta'              : [0.001,0.005,0.01,0.05,0.1],
                         'alpha'            : [0.01, 0.1,1,10],
                         'lambda'           : [0.01, 0.1,1,10],
                         'subsample'        : [0.5,0.7,0.9],
                         'colsample_bytree' : [0.5,0.7,0.9]
                        }

    HPO = ModuleHyperparameterGridOptimizer.HyperparametersGridOptimizer("PUBDT", features, gridsearch_params, min_num_trees, max_num_trees, X_train, X_test, X_val, y_train, y_test, y_val)

    best_params = HPO.RunTierGridOptimization()

    HPO.storeBestParamsScoresTier(outdir)

    
