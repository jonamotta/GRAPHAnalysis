from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from itertools import product
import xgboost as xgb
import numpy as np


class HyperparametersGridOptimizer():

    # instantiate 
    def __init__(self, BDT, features, hyperparam_grid_dict, min_num_trees, max_num_trees, X_train, X_test, X_val, y_train, y_test, y_val, tier=False):
        self.BDT = BDT
        self.train = xgb.DMatrix(data=X_train[features], label=y_train, feature_names=features)
        self.test = xgb.DMatrix(data=X_test[features], label=y_test, feature_names=features)
        self.val = xgb.DMatrix(data=X_val[features], label=y_val, feature_names=features)
        self.ytrain = y_train
        self.ytest = y_test
        self.yval = y_val
        self.features = features
        self.min_num_trees = min_num_trees
        self.max_num_trees = max_num_trees
        if min_num_trees == max_num_trees: self.num_trees = self.min_num_trees
        self.hyperparam_keys = hyperparam_grid_dict.keys()
        self.hyperparam_grid = list(product(*(hyperparam_grid_dict[key] for key in self.hyperparam_keys)))
        # set the default values for the yperparameters
        self.testing_params = {'objective'        : 'binary:logistic',
                               'eval_metric'      : 'logloss',
                               'nthread'          : 10,
                               'max_depth'        : 5,
                               'eta'              : 0.1,
                               'alpha'            : 9,
                               'lambda'           : 5,
                               'subsample'        : 0.8,
                               'colsample_bytree' : 0.8,
                              }

        self.train_aucs = []
        self.test_aucs = []
        self.val_aucs = []
        self.train_aucs_stds = []
        self.test_aucs_stds = []
        self.val_aucs_stds = []

        self.train_rmses = []
        self.test_rmses = []
        self.val_rmses = []
        self.train_rmses_stds = []
        self.test_rmses_stds = []
        self.val_rmses_stds = []


    def RunExtendedGridOptimization(self):
        self.ticks = []
        self.best_params = []

        for boosting_rounds in range(self.min_num_trees, self.max_num_trees+1):
            print('         - testing BDT with {0} boosting rounds'.format(boosting_rounds))
            self.num_trees = boosting_rounds

            local_best_params = self.RunGridOptimization()
            self.best_params.append(local_best_params)

            # define the dictionary for the ticks of the summary plot as fct of the parameters
            tick = {'boost rounds' : boosting_rounds,
                    'max depth'    : local_best_params['max_depth'],
                    'learn. rate'  : local_best_params['eta'],
                    'feat. frac.'  : local_best_params['colsample_bytree'],
                    'evts. frac.'  : local_best_params['subsample'],
                    'L1 regul.'    : local_best_params['alpha'],
                    'L2 regul.'    : local_best_params['lambda']
                   }
            self.ticks.append(tick)

        return self.best_params


    def RunGridOptimization(self):
        best_score = -99.0
        local_best_params = {}
        for j in range(len(self.hyperparam_grid)):
            for i, key in enumerate(self.hyperparam_keys):
                self.testing_params[key] = self.hyperparam_grid[j][i]
                print(self.testing_params)
                score = self.fitTrain()

                if score >= best_score:
                    best_score = score
                    local_best_params = self.testing_params

        self.fitVal(local_best_params)

        return local_best_params

    
    def RunTierGridOptimization(self):
        self.best_params = []
        best_score = -99.0
        local_best_params = {}
        for j in range(len(self.hyperparam_grid)):
            for i, key in enumerate(self.hyperparam_keys):
                self.testing_params[key] = self.hyperparam_grid[j][i]
                score = self.fitTrain()

                if score >= best_score:
                    best_score = score
                    local_best_params = self.testing_params

        self.best_params.append(local_best_params)

        self.fitVal(local_best_params)

        return local_best_params

    
    # train model with specific set of hyperparameters and get evaluation metric
    def fitTrain(self):
        booster = xgb.train(self.testing_params, self.train, num_boost_round=self.num_trees)
        auroc_test  = roc_auc_score(self.ytest, booster.predict(self.test))
        auroc_train = roc_auc_score(self.ytrain, booster.predict(self.train))
        
        # this function has a maximum for auroc_test=1 and auroc_train=1 which is our ideal goal
        # its shape allows to have more control on the overtraining by having 10* the test difference wrt train
        return auroc_test - 10*abs(auroc_train-auroc_test)

    
    # apply model to validation dataset using the best parameters got from the grid optimization
    def fitVal(self, hyperparams):
        cv_train = xgb.cv(hyperparams, self.train, metrics=['auc', 'rmse'], nfold=5, num_boost_round=self.num_trees, stratified=True)
        cv_val = xgb.cv(hyperparams, self.val, metrics=['auc', 'rmse'], nfold=5, num_boost_round=self.num_trees, stratified=True)
        
        auc_train = cv_train['train-auc-mean'][self.num_trees-1]
        auc_std_train = cv_train['train-auc-std'][self.num_trees-1]
        auc_test = cv_train['test-auc-mean'][self.num_trees-1]
        auc_std_test = cv_train['test-auc-std'][self.num_trees-1]
        auc_val = (cv_val['test-auc-mean'][self.num_trees-1] + cv_val['train-auc-mean'][self.num_trees-1]) / 2
        auc_std_val = np.sqrt( cv_val['test-auc-std'][self.num_trees-1]**2 + cv_val['train-auc-std'][self.num_trees-1]**2 ) / 2

        self.train_aucs.append(auc_train)
        self.test_aucs.append(auc_test)
        self.val_aucs.append(auc_val)
        self.train_aucs_stds.append(auc_std_train)
        self.test_aucs_stds.append(auc_std_test)
        self.val_aucs_stds.append(auc_std_val)

        rmse_train = cv_train['train-rmse-mean'][self.num_trees-1]
        rmse_std_train = cv_train['train-rmse-std'][self.num_trees-1]
        rmse_test = cv_train['test-rmse-mean'][self.num_trees-1]
        rmse_std_test = cv_train['test-rmse-std'][self.num_trees-1]
        rmse_val = (cv_val['test-rmse-mean'][self.num_trees-1] + cv_val['train-rmse-mean'][self.num_trees-1]) / 2
        rmse_std_val = np.sqrt( cv_val['test-rmse-std'][self.num_trees-1]**2 + cv_val['train-rmse-std'][self.num_trees-1]**2 ) / 2

        self.train_rmses.append(rmse_train)
        self.test_rmses.append(rmse_test)
        self.val_rmses.append(rmse_val)
        self.train_rmses_stds.append(rmse_std_train)
        self.test_rmses_stds.append(rmse_std_test)
        self.val_rmses_stds.append(rmse_std_val)


    # apply model to validation dataset using the best parameters got from the grid optimization
    # def fitValOnTier(self, hyperparams):
    #     cv_train = xgb.cv(hyperparams, self.train, metrics=['auc', 'rmse'], nfold=5, num_boost_round=self.num_trees, stratified=True)
    #     cv_val = xgb.cv(hyperparams, self.val, metrics=['auc', 'rmse'], nfold=5, num_boost_round=self.num_trees, stratified=True)
        
    #     auc_train = cv_train['train-auc-mean'][self.num_trees-1]
    #     auc_std_train = cv_train['train-auc-std'][self.num_trees-1]
    #     auc_test = cv_train['test-auc-mean'][self.num_trees-1]
    #     auc_std_test = cv_train['test-auc-std'][self.num_trees-1]
    #     auc_val = (cv_val['test-auc-mean'][self.num_trees-1] + cv_val['train-auc-mean'][self.num_trees-1]) / 2
    #     auc_std_val = np.sqrt( cv_val['test-auc-std'][self.num_trees-1]**2 + cv_val['train-auc-std'][self.num_trees-1]**2 ) / 2

    #     rmse_train = cv_train['train-rmse-mean'][self.num_trees-1]
    #     rmse_std_train = cv_train['train-rmse-std'][self.num_trees-1]
    #     rmse_test = cv_train['test-rmse-mean'][self.num_trees-1]
    #     rmse_std_test = cv_train['test-rmse-std'][self.num_trees-1]
    #     rmse_val = (cv_val['test-rmse-mean'][self.num_trees-1] + cv_val['train-rmse-mean'][self.num_trees-1]) / 2
    #     rmse_std_val = np.sqrt( cv_val['test-rmse-std'][self.num_trees-1]**2 + cv_val['train-rmse-std'][self.num_trees-1]**2 ) / 2

    #     results = {'train-auc-mean'   : auc_train,
    #                'train-auc-std'    : auc_std_train,
    #                'test-auc-mean'    : auc_test,
    #                'test-auc-std'     : auc_std_test,
    #                'test-auc-mean'    : auc_val,
    #                'test-auc-std'     : auc_std_val,
    #                'train-rmse-mean' : rmse_train,
    #                'train-rmse-std'  : rmse_std_train,
    #                'test-rmse-mean'  : rmse_test,
    #                'test-rmse-std'   : rmse_std_test,
    #                'test-rmse-mean'  : rmse_val,
    #                'test-rmse-std'   : rmse_std_val
    #               }

    #     return results
    

    # plot the results as a function of the hyperparameters chosen at each boosting round
    def plotterAucVsParams(self, outdir, metric="val", tag=""):
        optimiz_metric = []
        metric_std = []
        if metric == "train":
            optimiz_metric = self.train_aucs
            metric_std = self.train_aucs_stds
        
        elif metric == "test":
            optimiz_metric = self.test_aucs
            metric_std = self.test_aucs_stds

        elif metric == "val":
            optimiz_metric = self.val_aucs
            metric_std = self.val_aucs_stds

        fig, ax = plt.subplots(figsize=(self.max_num_trees*2,self.max_num_trees))
        x = np.linspace(self.min_num_trees, self.max_num_trees, self.max_num_trees-self.min_num_trees+1)
        ax.errorbar(x, optimiz_metric, metric_std, lw=6, marker="h", ms=20, elinewidth=4)
        ax.set_xticks(x)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(self.ticks)):
            string = ""
            for par in self.ticks[i]:
                string = string+par+' = '+str(self.ticks[i][par])+'\n'
            labels[i] = string
        ax.set_xticklabels(labels)
        plt.grid(linestyle=':')
        plt.gcf().subplots_adjust(bottom=0.1)
        plt.xlabel('Hyperparameters used in training')
        plt.ylabel(r'Optimization metric')
        plt.savefig(outdir+'/HPO_{0}_{1}_AucVsParameters{2}.pdf'.format(self.BDT,metric,tag))
        plt.close()

    
    # plot the results as a function of the hyperparameters chosen at each boosting round
    def plotterrmseVsParams(self, outdir, metric="val", tag=""):
        optimiz_metric = []
        metric_std = []
        if metric == "train":
            optimiz_metric = self.train_rmses
            metric_std = self.train_rmses_stds
        
        elif metric == "test":
            optimiz_metric = self.test_rmses
            metric_std = self.test_rmses_stds

        elif metric == "val":
            optimiz_metric = self.val_rmses
            metric_std = self.val_rmses_stds

        fig, ax = plt.subplots(figsize=(self.max_num_trees*2,self.max_num_trees))
        x = np.linspace(self.min_num_trees, self.max_num_trees, self.max_num_trees-self.min_num_trees+1)
        ax.errorbar(x, optimiz_metric, metric_std, lw=6, marker="h", ms=20, elinewidth=4)
        ax.set_xticks(x)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(self.ticks)):
            string = ""
            for par in self.ticks[i]:
                string = string+par+' = '+str(self.ticks[i][par])+'\n'
            labels[i] = string
        ax.set_xticklabels(labels)
        plt.grid(linestyle=':')
        plt.gcf().subplots_adjust(bottom=0.1)
        plt.xlabel('Hyperparameters used in training')
        plt.ylabel(r'Optimization metric')
        plt.savefig(outdir+'/HPO_{0}_{1}_rmseVsParameters{2}.pdf'.format(self.BDT,metric,tag))
        plt.close()
    

    def storeExtendedBestParams(self, outdir, tag=""):
        with open(outdir+"/HPO_{0}_extendedBestParams{1}.txt".format(self.BDT,tag), "w") as reportFile:
            for j in range(len(self.best_params)):
                reportFile.write("Number of boosting rounds = {0}\n\n".format(self.min_num_trees+j))
                for i, res in enumerate(self.best_params[j]):
                    reportFile.write("    >>> {}: {}\n".format(res, self.best_params[j][res]))
                reportFile.write("\n")
        reportFile.close()


    def storeExtendedBestScores(self, outdir, tag=""):
        with open(outdir+"/HPO_{0}_extendedBestScore{1}.txt".format(self.BDT,tag), "w") as reportFile:
            for j in range(len(self.train_rmses)):
                reportFile.write("Number of boosting rounds = {0}\n\n".format(self.min_num_trees+j))
                reportFile.write("    >>> train-auc-mean = {0}\n".format(self.train_aucs[j]))
                reportFile.write("    >>> train-auc-std = {0}\n".format(self.train_aucs[j]))
                reportFile.write("    >>> test-auc-mean = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-std = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-mean = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-std = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> train-rmse-mean = {0}\n".format(self.train_rmses[j]))
                reportFile.write("    >>> train-rmse-std = {0}\n".format(self.train_rmses[j]))
                reportFile.write("    >>> test-rmse-mean = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-std = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-mean = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-std = {0}\n".format(self.test_rmses[j]))
        reportFile.close()


    def storeBestParamsScoresTier(self, outdir, tag=""):
        with open(outdir+"/HPO_{0}_BestParamsScores{1}_br{2}.txt".format(self.BDT,tag,self.min_num_trees), "w") as reportFile:
            for j in range(len(self.best_params)):
                reportFile.write("Number of boosting rounds = {0}\n\n".format(self.min_num_trees+j))
                for i, res in enumerate(self.best_params[j]):
                    reportFile.write("    >>> {}: {}\n".format(res, self.best_params[j][res]))
                reportFile.write("\n")
                reportFile.write("Number of boosting rounds = {0}\n\n".format(self.num_trees))
                reportFile.write("    >>> train-auc-mean = {0}\n".format(self.train_aucs[j]))
                reportFile.write("    >>> train-auc-std = {0}\n".format(self.train_aucs[j]))
                reportFile.write("    >>> test-auc-mean = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-std = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-mean = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> test-auc-std = {0}\n".format(self.test_aucs[j]))
                reportFile.write("    >>> train-rmse-mean = {0}\n".format(self.train_rmses[j]))
                reportFile.write("    >>> train-rmse-std = {0}\n".format(self.train_rmses[j]))
                reportFile.write("    >>> test-rmse-mean = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-std = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-mean = {0}\n".format(self.test_rmses[j]))
                reportFile.write("    >>> test-rmse-std = {0}\n".format(self.test_rmses[j]))
        reportFile.close()












