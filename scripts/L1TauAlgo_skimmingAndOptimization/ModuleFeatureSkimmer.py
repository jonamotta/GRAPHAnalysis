from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from itertools import combinations
import xgboost as xgb
import numpy as np

class FeatureSkimmer():
    
    # instantiate with hyperparameters and minimum number fo features
    def __init__(self, BDT, initial_features, hyperparams, num_trees, k_features, metric):
        self.BDT = BDT
        self.features = initial_features # this list should be ordered by importance!!
        self.num_trees = num_trees
        self.hyperparams = hyperparams
        self.k_features = k_features
        self.metric = metric
   
    # define function for sequential backward search of the 
    def SequentialBackwardSearch(self, X_train, y_train):
        self.Xtrain = X_train
        self.ytrain = y_train
        self.score_results = [self.fit(self.features)[0]]
        self.std_results = [self.fit(self.features)[1]]
        self.feat_results = [self.features]

        # iterate through all the dimensions until k_features is reached
        # this '>' on the length of features array actually works like a '>='
        while len(self.features) > self.k_features:
            print('         - testing BDT with {0} features'.format(len(self.features)-1))
            self.features = self.features[:-1]
            score, std = self.fit(self.features)

            # store the results
            self.score_results.append(score)
            self.std_results.append(std)
            self.feat_results.append(self.features)

        # reverse the outputs so that they are in rising order by number of features used
        self.score_results.reverse()
        self.std_results.reverse()
        self.feat_results.reverse()
        
        return self.score_results, self.std_results, self.feat_results

    # define fit function
    def RandomSearch(self, X_train, y_train):
        self.Xtrain = X_train
        self.ytrain = y_train
        self.score_results = [self.fit(self.features)[0]]
        self.std_results = [self.fit(self.features)[1]]
        self.feat_results = [self.features]

        # iterate through all the dimensions until k_features is reached
        # this '>' on the length of features array actually works like a '>='
        while len(self.features) > self.k_features:
            print('         - testing BDT with {0} features'.format(len(self.features)-1))
            scores = []
            stds = []
            subsets = []
            # iterate through different combinations of features, fit the model, record the score
            for comb in combinations(self.features, r=len(self.features) - 1):
                score, std = self.fit(list(comb))
                scores.append(score)
                stds.append(std)
                subsets.append(comb)

            # get the index of best score and store the results
            best_score_index = np.argmax(scores)
            self.score_results.append(scores[best_score_index])
            self.std_results.append(stds[best_score_index])
            self.feat_results.append(subsets[best_score_index])

            # update features
            self.features = subsets[best_score_index]

        # reverse the outputs so that they are in rising order by number of features used
        self.score_results.reverse()
        self.std_results.reverse()
        self.feat_results.reverse()
        
        return self.score_results, self.std_results, self.feat_results


    # train models with specific set of features and get score
    def fit(self, features):
        train = xgb.DMatrix(data=self.Xtrain[features],label=self.ytrain, feature_names=features)
        cv = xgb.cv(self.hyperparams, train, metrics=['auc'], nfold=10, num_boost_round=self.num_trees, stratified=True)
        score = 0.0
        std = 0.0
        if self.metric == "test":  score = cv['test-auc-mean'][self.num_trees-1] ; std = cv['test-auc-std'][self.num_trees-1]
        if self.metric == "train": score = cv['train-auc-mean'][self.num_trees-1] ; std = cv['train-auc-std'][self.num_trees-1]
        return score, std


    # plot the results as a function of the features names
    def plotterFctFeats(self, plotdir, tag=""):
        height = len(self.feat_results[len(self.feat_results)-1])
        fig, ax = plt.subplots(figsize=( height*3 , height ))
        x = np.linspace(2,len(self.feat_results),len(self.feat_results))
        ax.errorbar(x, self.score_results, self.std_results, lw=2, marker="h", elinewidth=2)
        ax.set_xticks(x)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(self.feat_results)):
            string = ""
            for j in self.feat_results[i]:
                string = string+j.replace("cl3d_","")+"\n"
            labels[i] = string
        ax.set_xticklabels(labels)
        plt.grid(linestyle=':')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.xlabel('Features used in training')
        plt.ylabel(r'Optimization metric')
        plt.savefig(plotdir+'/FS_{0}_{1}AUROC_features{2}.pdf'.format(self.BDT,self.metric,tag))
        plt.close()
