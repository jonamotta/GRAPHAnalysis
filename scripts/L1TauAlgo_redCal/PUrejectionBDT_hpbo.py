import os
import time
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization


def prepareCat(row):
    if row['cl3d_isbestmatch'] == True and row['gentau_decayMode']>=0:
        return 1
    else:
        return 0


indir = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/L1BDT_redCalib/hdf5dataframes/calibrated'

FE = 'threshold'

inFileTraining_dict = {
    'threshold'    : indir+'/Training_PU200_th_calibrated.hdf5',
    'mixed'        : indir+'/'
}

# features for BDT training
#features = ['cl3d_c2', 'cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
features = ['cl3d_c2', 'cl3d_pt_c2', 'cl3d_ntc90', 'cl3d_abseta']
output = 'gentau_pid'

store_tr = pd.HDFStore(inFileTraining_dict[FE], mode='r')
dfTraining = store_tr[FE]
store_tr.close()

dfTraining['gentau_pid'] = dfTraining.apply(lambda row: prepareCat(row), axis=1)
dfTr = dfTraining.query('gentau_pid==1 or (gentau_pid==0 and gentau_decayMode!=-2)').copy(deep=True) # take all the taus and all the PU not coming from QCD sample

del dfTraining

X_train, X_test, y_train, y_test = train_test_split(dfTr[features], dfTr[output], stratify=dfTr[output], test_size=0.3)
dtrain = xgb.DMatrix(data=X_train,label=y_train, feature_names=features)
dtest = xgb.DMatrix(data=X_test,label=y_test,feature_names=features)

def func4optimization(hypar_bounds, min_trees, max_trees, init_points, n_iter):
    start = time.time()
    best_target = -999
    
    for num_trees in range(min_trees,max_trees):
        print("Running optimization with num_trees="+str(num_trees))
        def xgb4bo(eta, max_depth, subsample, colsample_bytree, lambd, alpha):
            hyperparams = {'eval_metric'      : 'logloss',
                           'objective'        : 'binary:logistic', # objective function
                           'nthread'          : 10, # limit number of threads
                           'eta'              : eta, # learning rate
                           'max_depth'        : int(round(max_depth,0)), # maximum depth of a tree
                           'subsample'        : subsample, # fraction of events to train tree on
                           'colsample_bytree' : colsample_bytree,# fraction of features to train tree on
                           'lambda'           : lambd,
                           'alpha'            : alpha
            }

            booster = xgb.train(hyperparams, dtrain, num_boost_round=num_trees)
            X_train['bdt_output'] = booster.predict(dtrain)
            X_test['bdt_output'] = booster.predict(dtest)
            auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
            auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])

            # 10**(auroc_train+auroc_test) - 100**(auroc_train-auroc_test)
            # this function has a maximum for auroc_test=1 and auroc_train=1 which is our ideal goal
            # its shape allows to have more control on the overtraining as the function plummets if auroc_test->0
            # and auroc_train->1
            return auroc_test - 10*abs(auroc_train-auroc_test)
        
        xgb_bo = BayesianOptimization(f = xgb4bo, pbounds = hypar_bounds)
        xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei', alpha=1e-3)
        
        best = xgb_bo.max
        print(best)
        print('\n')
        
        if best['target'] > best_target:
            bestOFbest = best
            best_target = best['target']
            betsNtrees = num_trees
    
    end = time.time()
    print('\nRunning time = %02dh %02dm %02ds'%((end-start)/3600, ((end-start)%3600)/60, (end-start)% 60))
    
    print('\n')
    print()
    print("Best optimization ran for num_trees="+str(betsNtrees))
    print(bestOFbest)


hypar_bounds = {'eta'              : (0.1, 0.3), 
                'max_depth'        : (2, 5),
                'subsample'        : (0.5, 0.8),
                'colsample_bytree' : (0.5, 0.9),
                'alpha'            : (0.01, 10),
                'lambd'            : (0.01, 10)
               }


func4optimization(hypar_bounds, min_trees=30, max_trees=100, init_points=5, n_iter=35)



