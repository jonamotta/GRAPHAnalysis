from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import ModuleHyperparameterBayesianOptimizer
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
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/ISOrejectionBDTskimmingAndOptimization{0}/HPO_bayes'.format("_Rscld" if args.doRescale else "")
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
    # create a copy to cl3d_pt that will be used only for the training of the BDTs
    dfTr['cl3d_pt_tr'] = dfTr['cl3d_pt'].copy(deep=True)
    dfVal['cl3d_pt_tr'] = dfVal['cl3d_pt'].copy(deep=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['sgnId', 'cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_srrmean', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
    dfTr = dfTr[tokeep]
    dfVal = dfVal[tokeep]

    # reduce dimension for testing new stuff
    #dfTr = pd.concat([dfTr[dfTr['sgnId']==1].sample(500), dfTr[dfTr['sgnId']==0].sample(500)], sort=False)
    #dfVal = pd.concat([dfVal[dfVal['sgnId']==1].sample(500), dfVal[dfVal['sgnId']==0].sample(500)], sort=False)

    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = ['cl3d_NclIso_dR4']
            features2saturate = ['cl3d_pt_tr', 'cl3d_spptot', 'cl3d_hoe', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']
            saturation_dict = {'cl3d_pt_tr': [0, 200],
                               'cl3d_spptot': [0, 0.17],
                               'cl3d_hoe': [0, 63],
                               'tower_etSgn_dRsgn1': [0, 194],
                               'tower_etSgn_dRsgn2': [0, 228],
                               'tower_etIso_dRsgn1_dRiso3': [0, 105],
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
    if args.PUWP == '99':
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_srrmean', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    elif args.PUWP == '95':
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_srrmean', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']

    else:
        features = ['cl3d_pt_tr', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_NclIso_dR4', 'tower_etSgn_dRsgn1', 'tower_etSgn_dRsgn2', 'tower_etIso_dRsgn1_dRiso3']


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
    HPO = ModuleHyperparameterBayesianOptimizer.HyperparametersBayesianOptimizer("ISOBDT", features, hypar_bounds, min_num_trees, max_num_trees, X_train, X_test, X_val, y_train, y_test, y_val)

    extended_best_params, extended_reports = HPO.RunExtendedBayesianOptimization(init_points=10, n_iter=40)

    HPO.plotterAucVsParams(plotdir, 'train', "_PUWP{0}".format(args.PUWP))
    HPO.plotterAucVsParams(plotdir, 'test', "_PUWP{0}".format(args.PUWP))
    HPO.plotterAucVsParams(plotdir, 'val', "_PUWP{0}".format(args.PUWP))
    HPO.plotterRmsleVsParams(plotdir, 'train', "_PUWP{0}".format(args.PUWP))
    HPO.plotterRmsleVsParams(plotdir, 'test', "_PUWP{0}".format(args.PUWP))
    HPO.plotterRmsleVsParams(plotdir, 'val', "_PUWP{0}".format(args.PUWP))
    HPO.storeExtendedReport(plotdir, "ISOBDT_PUWP{0}".format(args.PUWP))
    HPO.storeExtendedBestParams(plotdir, "ISOBDT_PUWP{0}".format(args.PUWP))
