from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import ModuleRegressorOptimizer
import pandas as pd
import numpy as np
import argparse
import os


if __name__ == "__main__" :

    # parse user's options
    parser = argparse.ArgumentParser(description='Command line parser of plotting options')
    parser.add_argument('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    # store parsed options
    args = parser.parse_args()

    # create needed folders
    indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/hdf5dataframes/matched'
    plotdir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT/C2skimmingAndOptimization{0}/HPO_depth5'.format("_Rscld" if args.doRescale else "")
    os.system('mkdir -p '+plotdir)

    print('** INFO: prepearing dataset')
    # read training datasets
    store_tr = pd.HDFStore(indir+'/Training_PU200_th_matched.hdf5', mode='r')
    dfTr = store_tr['threshold']
    store_tr.close()
    # select events for the training
    dfTr = dfTr.query('gentau_decayMode>=0 and cl3d_isbestmatch==True and (gentau_vis_pt>20 and ((gentau_vis_eta>1.6 and gentau_vis_eta<2.9) or (gentau_vis_eta<-1.6 and gentau_vis_eta>-2.9)))').copy(deep=True)
    # clean the dataframe to reduce memory usage
    tokeep = ['gentau_vis_pt', 'cl3d_pt', 'cl3d_abseta', 'cl3d_coreshowerlength', 'cl3d_srrmean', 'cl3d_spptot', 'cl3d_meanz', 'cl3d_showerlength']
    dfTr = dfTr[tokeep]

    # reduce dimension for testing new stuff
    #dfTr = dfTr.sample(500)

    # apply rescaling if wanted
    if args.doRescale:
            print('\n** INFO: rescaling features to bound their values')

            # the saturation and shifting values are calculated in the "features_reshaping" JupyScript
            features2shift = ['cl3d_showerlength', 'cl3d_coreshowerlength']
            features2saturate = ['cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
            saturation_dict = {'cl3d_abseta': [1.45, 3.2],
                               'cl3d_spptot': [0, 0.17],
                               'cl3d_srrmean': [0, 0.01],
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


    print('\n** INFO: training calibration C1')

    input_c1 = dfTr[['cl3d_abseta']]
    target_c1 = dfTr['gentau_vis_pt'] - dfTr['cl3d_pt']
    C1model = LinearRegression().fit(input_c1, target_c1)
    dfTr['cl3d_c1'] = C1model.predict(dfTr[['cl3d_abseta']])
    dfTr['cl3d_pt_c1'] = dfTr['cl3d_c1'] + dfTr['cl3d_pt']


    features = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']

    # cerate input and regression target
    inputFeats = dfTr[features] ; target = dfTr['gentau_vis_pt'] / dfTr['cl3d_pt_c1']

    # do random search
    print('\n** INFO: doing boosting rounds optimization')
    FS = ModuleRegressorOptimizer.RegressorOptimizer('C2calibration', dfTr, features, target, 2)
    hpo_results, hpo_rounds = FS.HyperparameterOptimizer(5, 100)
    FS.plotterFctBR(plotdir)

