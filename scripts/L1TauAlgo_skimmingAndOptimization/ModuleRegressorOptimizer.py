from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np

class RegressorOptimizer():
    
    # instantiate with hyperparameters and minimum number fo features
    def __init__(self, BDT, dataset, initial_features, target, k_features):
        self.BDT = BDT
        self.features = initial_features
        self.dfTr = dataset
        self.k_features = k_features
        self.target = target


    # function to perform random search of most important variables
    def FeaturesRandomSearch(self, inputFeats):
        self.input = inputFeats
        self.score_results = [self.fit(self.features)]
        self.feat_results = [self.features]

        # iterate through all the dimensions until k_features is reached
        # this '>' on the length of features array actually works like a '>='
        while len(self.features) > self.k_features:
            print('         - testing BDT with {0} features'.format(len(self.features)-1))
            scores = []
            subsets = []
            # iterate through different combinations of features, fit the model, record the score
            for comb in combinations(self.features, r=len(self.features) - 1):
                score = self.fit(list(comb))
                scores.append(score)
                subsets.append(comb)

            # get the index of best score and store the results
            best_score_index = np.argmax(scores)
            self.score_results.append(scores[best_score_index])
            self.feat_results.append(subsets[best_score_index])

            # update features
            self.features = subsets[best_score_index]

        # reverse the outputs so that they are in rising order by number of features used
        self.score_results.reverse()
        self.feat_results.reverse()
        
        return self.score_results, self.feat_results


    def HyperparameterOptimizer(self, minBoostRounds, maxBoostRounds):
        self.hpo_results = []
        self.hpo_rounds = []

        # 200 boosting rounds is too much
        if maxBoostRounds > 200: maxBoostRounds = 200

        for br in range(minBoostRounds,maxBoostRounds+1):
            print('         - testing BDT with {0} boosting rounds'.format(br))
            score = self.fit(self.features, br)
            self.hpo_results.append(score)
            self.hpo_rounds.append(br)

            if len(self.hpo_results) <= 20: continue
            # if over the last XX boosting rounds the increase in score has been less than 1% we stop
            check1 = self.hpo_results[-10]
            check2 = self.hpo_results[-1]
            if (check2-check1)/check1 < 0.01: break

        return self.hpo_results, self.hpo_rounds

    # train models with specific set of features and get score
    def fit(self, features, boostRounds=100):
        model = GradientBoostingRegressor(n_estimators=boostRounds, learning_rate=0.1, max_depth=5, random_state=0, loss='huber').fit(self.dfTr[features], self.target)
        return model.score(self.dfTr[features], self.target)


    # plot the results as a function of the features names
    def plotterFctFeats(self, plotdir, tag=""):
        height = len(self.feat_results[len(self.feat_results)-1])
        fig, ax = plt.subplots(figsize=( height*3 , height ))
        x = np.linspace(2,len(self.feat_results),len(self.feat_results))
        ax.plot(x, self.score_results, lw=2, marker="h")
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
        plt.savefig(plotdir+'/FS_{0}_score_features{1}.pdf'.format(self.BDT,tag))
        plt.close()


    # plot the results as a function of the number of boosting rounds
    def plotterFctBR(self, plotdir, tag=""):
        fig, ax = plt.subplots(figsize=(30,10))
        ax.plot(self.hpo_rounds, self.hpo_results, lw=2, marker="h")
        ax.set_xticks(self.hpo_rounds)
        plt.grid(linestyle=':')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.xlabel('Features used in training')
        plt.ylabel(r'Optimization metric')
        plt.savefig(plotdir+'/HPO_{0}_score_boostRounds{1}.pdf'.format(self.BDT,tag))
        plt.close()