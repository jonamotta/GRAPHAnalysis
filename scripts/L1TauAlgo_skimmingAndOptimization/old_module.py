from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import xgboost as xgb
import numpy as np

class HyperparametersOptimizer():

    # instantiate 
    def __init__(self, BDT, features, hyperparam_bounds, min_num_trees, max_num_trees, X_train, X_test, X_val, y_train, y_test, y_val):
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
        self.hyperparam_bounds = hyperparam_bounds


    def RunExtendedBayesianOptimization(self, init_points=5, n_iter=15):
        self.ticks = []
        extended_best_params = []
        self.extended_train_scores = []
        self.extended_test_scores = []
        self.extended_val_scores = []
        self.extended_reports = []
        for boosting_rounds in range(self.min_num_trees, self.max_num_trees+1):
            print('         - testing BDT with {0} boosting rounds'.format(boosting_rounds))
            self.num_trees = boosting_rounds
            best_params, score_train, score_test ,score_val, report = self.RunBayesianOptimization(init_points, n_iter)

            extended_best_params.append(best_params)
            self.extended_train_scores.append(score_train)
            self.extended_test_scores.append(score_test)
            self.extended_val_scores.append(score_val)
            self.extended_reports.append(report)

            # define the dictionary for the ticks of the summary plot as fct of the parameters
            tick = {'boost rounds' : boosting_rounds,
                    'max depth'    : int(round(best_params['max_depth'],0)),
                    'learn. rate'  : round(best_params['eta'],2),
                    'feat. frac.'  : round(best_params['colsample_bytree'],2),
                    'evts. frac.'  : round(best_params['subsample'],2),
                    'L1 regul.'    : 9,
                    'L2 regul.'    : 5
                   }
            self.ticks.append(tick)
        
        return extended_best_params, self.extended_train_scores, self.extended_test_scores, self.extended_val_scores, self.extended_reports


    def RunBayesianOptimization(self, init_points=5, n_iter=15):
        xgb_bo = BayesianOptimization(f = self.xgb4bo, pbounds = self.hyperparam_bounds, verbose=0, random_state=1996)
        xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei', alpha=1e-3)
        best_params = xgb_bo.max['params']
        report = xgb_bo.res
        
        score_train, score_test ,score_val = self.xgb4valid(best_params)

        return best_params, score_train, score_test ,score_val, report


    def xgb4bo(self, eta, subsample, colsample_bytree, max_depth):
        hyperparams = {'objective'        : 'binary:logistic',
                       'eval_metric'      : 'logloss',
                       'nthread'          : 10,
                       'eta'              : eta, # learning rate
                       'max_depth'        : int(round(max_depth,0)), # maximum depth of a tree
                       'subsample'        : subsample, # fraction of events to train tree on
                       'colsample_bytree' : colsample_bytree, # fraction of features to train tree on
                       'lambda'           : 9,
                       'alpha'            : 5
                      }

        score = self.fitTrain(hyperparams)
        return score


    def xgb4valid(self, best_params):
        # add 'missing parameters' that are not optimized by the bayesian optimization
        hyperparams = {'objective'        : 'binary:logistic',
                       'eval_metric'      : 'logloss',
                       'nthread'          : 10,
                       'eta'              : best_params['eta'],
                       'max_depth'        : int(round(best_params['max_depth'],0)),
                       'subsample'        : best_params['subsample'],
                       'colsample_bytree' : best_params['colsample_bytree'],
                       'lambda'           : 9,
                       'alpha'            : 5
                      }
        
        score = self.fitVal(hyperparams)
        return score


    # train model with specific set of hyperparameters and get evaluation metric
    def fitTrain(self, hyperparams):
        booster = xgb.train(hyperparams, self.train, num_boost_round=self.num_trees)
        auroc_test  = roc_auc_score(self.ytest, booster.predict(self.test))
        auroc_train = roc_auc_score(self.ytrain, booster.predict(self.train))
        
        # this function has a maximum for auroc_test=1 and auroc_train=1 which is our ideal goal
        # its shape allows to have more control on the overtraining by having 10* the test difference wrt train
        return auroc_test - 10*abs(auroc_train-auroc_test)


    # apply model to validation dataset using the best parameters got from the bayesian optimization
    def fitVal(self, hyperparams):
        booster = xgb.train(hyperparams, self.train, num_boost_round=self.num_trees)
        auroc_train = roc_auc_score(self.ytrain, booster.predict(self.train))
        auroc_test = roc_auc_score(self.ytest, booster.predict(self.test))
        auroc_val = roc_auc_score(self.yval, booster.predict(self.val))

        return auroc_train, auroc_test, auroc_val


    # plot the results as a function of the hyperparameters chosen at each boosting round
    def plotterFctParams(self, plotdir, metric="val", tag=""):
        optimiz_metric = []
        if metric == "train":
            optimiz_metric = self.extended_train_scores
        
        elif metric == "test":
            optimiz_metric = self.extended_test_scores

        elif metric == "val":
            optimiz_metric = self.extended_val_scores

        elif metric == "testMtrain":
            zipObj = zip(self.extended_train_scores, self.extended_test_scores)
            for train, test in zipObj:
                optimiz_metric.append(test - 10*abs(test-train))
        
        elif metric == "valMtrain":
            zipObj = zip(self.extended_train_scores, self.extended_val_scores)
            for train, val in zipObj:
                optimiz_metric.append(val - 10*abs(val-train))
        
        elif metric == "valPtestMtrain":
            zipObj = zip(self.extended_train_scores, self.extended_test_scores, self.extended_val_scores)
            for train, test, val in zipObj:
                optimiz_metric.append(val + test - 10*abs(test-train) - 10*abs(val-train) - 10*abs(test-val))


        fig, ax = plt.subplots(figsize=(60,30))
        x = np.linspace(self.min_num_trees, self.max_num_trees, self.max_num_trees-self.min_num_trees+1)
        ax.plot(x, optimiz_metric, lw=4, marker="h", ms=10)
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
        plt.savefig(plotdir+'/HPO_{0}_{1}_parameters{2}.pdf'.format(self.BDT,metric,tag))
        plt.close()


    def storeExtendedReport(self, plotdir, tag=""):
        with open(plotdir+"/HPO_{0}_extendedReport{1}.txt".format(self.BDT,tag), "w") as reportFile:
            for j in range(len(self.extended_reports)):
                reportFile.write("Number of boosting rounds = {0}\n".format(self.min_num_trees+j))
                for i, res in enumerate(self.extended_reports[j]):
                    reportFile.write("    >>> Iteration {}: \t{}\n".format(i, res))
        reportFile.close()
