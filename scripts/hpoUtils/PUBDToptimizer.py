import os
import time
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
import matplotlib
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
from scipy.special import btdtri # beta quantile function
from bayes_opt import BayesianOptimization
#from bayes_opt import SequentialDomainReductionTransformer

def prepareCat(row):
    if row['cl3d_isbestmatch'] == True and row['gentau_decayMode']>=0:
        return 1
    else:
        return 0

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

    booster = xgb.train(hyperparams, dtrain, num_boost_round=int(args.num_trees))
    X_train['bdt_output'] = booster.predict(dtrain)
    X_test['bdt_output'] = booster.predict(dtest)
    
    auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
    auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])

    # this function has a maximum for auroc_test=1 and auroc_train=1 which is our ideal goal
    # its shape allows to have more control on the overtraining by having 10* the test difference wrt train
    return auroc_test - 10*abs(auroc_train-auroc_test)

def xgb4valid(bets_params_dict):
    hyperparams = {'eval_metric'      : 'logloss',
                   'objective'        : 'binary:logistic',
                   'nthread'          : 10,
                   'eta'              : bets_params_dict['eta'],
                   'max_depth'        : int(round(bets_params_dict['max_depth'],0)),
                   'subsample'        : bets_params_dict['subsample'],
                   'colsample_bytree' : bets_params_dict['colsample_bytree'],
                   'lambda'           : bets_params_dict['lambd'],
                   'alpha'            : bets_params_dict['alpha']
    }

    booster = xgb.train(hyperparams, dtrain, num_boost_round=int(args.num_trees))
    X_train['bdt_output'] = booster.predict(dtrain)
    X_test['bdt_output'] = booster.predict(dtest)
    X_valid['bdt_output'] = booster.predict(dvalid)

    auroc_test = metrics.roc_auc_score(y_test,X_test['bdt_output'])
    auroc_train = metrics.roc_auc_score(y_train,X_train['bdt_output'])
    auroc_valid = metrics.roc_auc_score(y_valid,X_valid['bdt_output'])

    # for the validation we want: 
    # auroc_valid and auroc_test as large as possible --> have good PU rejection
    # auroc_valid and auroc_test as close as possible to auroc_train --> have little overtraining
    # auroc_test the closest possible to auroc_valid --> the overtraining should be of the same order for test and valid
    return auroc_valid + auroc_test - 10*abs(auroc_train-auroc_test) - 10*abs(auroc_train-auroc_valid) - 100*abs(auroc_test-auroc_valid)


# parse user's options
parser = argparse.ArgumentParser(description='Command line parser of plotting options')
parser.add_argument('--indir', dest='indir', help='input folder', default=None)
parser.add_argument('--FE', dest='FE', help='front-end', default='threshold')
parser.add_argument('--num_trees', dest='num_trees', help='number of boosting rounds', default=None)
parser.add_argument('--init_points', dest='init_points',  help='number of initialization points for the bayesian optimization', default=5)
parser.add_argument('--n_iter', dest='n_iter',  help='na,ber of bayesian optimization rounds', default=50)
# store parsed options
args = parser.parse_args()

if args.indir == None:
    print('** WARNING: no folder specified')
    print('** EXITING')
    exit()

if args.num_trees == None:
    print('** WARNING: no number of boosting rounds specified')
    print('** EXITING')
    exit()

indir = args.indir
FE = args.FE
init_points = int(args.init_points)
n_iter = int(args.n_iter)

inFileTraining_dict = {
    'threshold'    : indir+'/Training_PU200_th_calibrated.hdf5',
    'mixed'        : indir+'/'
}

inFileValidation_dict = {
    'threshold'    : indir+'/Validation_PU200_th_calibrated.hdf5',
    'mixed'        : indir+'/'
}

# features for BDT training
#features = ['cl3d_c2', 'cl3d_abseta', 'cl3d_showerlength','cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 'cl3d_ntc67', 'cl3d_ntc90']
features = ['cl3d_c3', 'cl3d_pt_c3', 'cl3d_ntc90', 'cl3d_abseta']
#features = ['cl3d_c2', 'cl3d_pt_c2', 'cl3d_ntc90', 'cl3d_abseta']
output = 'gentau_pid'

store_tr = pd.HDFStore(inFileTraining_dict[FE], mode='r')
dfTraining = store_tr[FE]
store_tr.close()

store_tr = pd.HDFStore(inFileValidation_dict[FE], mode='r')
dfValidation = store_tr[FE]
store_tr.close()

print('creating target column')
# take all the taus and all the PU not coming from QCD sample
dfTr = dfTraining.query('gentau_decayMode!=-2').copy(deep=True)
dfTr['gentau_pid'] = dfTr.apply(lambda row: prepareCat(row), axis=1)

dfVal = dfValidation.query('gentau_decayMode!=-2').copy(deep=True)
dfVal['gentau_pid'] = dfVal.apply(lambda row: prepareCat(row), axis=1)

del dfTraining, dfValidation
print('done creating target column')

X_train, X_test, y_train, y_test = train_test_split(dfTr[features], dfTr[output], stratify=dfTr[output], test_size=0.3)
dtrain = xgb.DMatrix(data=X_train,label=y_train, feature_names=features)
dtest = xgb.DMatrix(data=X_test,label=y_test, feature_names=features)

#bounds_transf = SequentialDomainReductionTransformer()
hypar_bounds = {'eta'              : (0.1, 0.3), 
                'max_depth'        : (2, 5),
                'subsample'        : (0.5, 0.8),
                'colsample_bytree' : (0.5, 0.9),
                'alpha'            : (0.01, 10),
                'lambd'            : (0.01, 10)
               }

xgb_bo = BayesianOptimization(f = xgb4bo, pbounds = hypar_bounds)#, bounds_transformer=bounds_transf)
xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei', alpha=1e-3)

best = xgb_bo.max
print(best['target'])
print(best['params'])

# after having done the optimization of the hyperparameters apply the BDT to the validation dataset to check 
# how it behaves on a completely new dataset that was never used, not even in testing
X_valid = dfVal[features]
y_valid = dfVal['gentau_pid']
dvalid = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=features)

validation_target = xgb4valid(best['params'])
print(validation_target)
