from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from itertools import combinations
import xgboost as xgb
import numpy as np


class FeatureSuperSkimmer():