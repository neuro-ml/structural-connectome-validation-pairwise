import numpy as np
import os
import sys
import pandas as pd
from scipy.spatial.distance import *
from scipy.sparse.csgraph import dijkstra, shortest_path, connected_components, laplacian

from sklearn.base import  BaseEstimator, TransformerMixin
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import networkx as nx
import igraph as ig
import scipy
import time


##########################
# Normalizer's functions #
##########################

def no_norm(matrix):
    return matrix

def max_norm(matrix):
    normed_matrix = matrix / np.max(matrix)
    return normed_matrix

def binar_norm(matrix):
    bin_matrix = matrix.copy()
    bin_matrix[bin_matrix > 0] = 1
    return bin_matrix

def mean_norm(matrix):
    normed_matrix = matrix / np.mean(matrix)
    return normed_matrix

def double_norm(function, matrix1, matrix2):
    return function(matrix1), function(matrix2)

##########################
# Featurizer's functions #
##########################

def bag_of_edges(X, SPL=None, symmetric = True, return_df = False, offset = 1):
    size = X.shape[1]
    if symmetric:
        indices = np.triu_indices(size, k = offset)
    else:
        grid = np.indices(X.shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(X.shape) == 3:
        featurized_X = X[:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        featurized_X = X[indices[0], indices[1]]
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j) for i,j in zip(indices[0], indices[1])]
        featurized_X = pd.DataFrame(featurized_X, columns=col_names)
    return featurized_X

def degrees(X, return_df = False):
    if len(X.shape) == 3:
        featurized_X = np.sum(X, axis=1)
        shape = (X.shape[0], X.shape[1])
    elif len(X.shape) == 2:
        featurized_X = np.sum(X, axis=1)
        shape = (1, X.shape[1])
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size). ')

    if return_df:
        col_names = ['degree_' + str(i) for i in range(X.shape[1])]
        featurized_X = pd.DataFrame(featurized_X.reshape(shape), columns=col_names)
    return featurized_X

def closeness_centrality(X):
    n_nodes = X.shape[0]
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
            unweighted=False)
    sum_distances_vector = np.sum(SPL, 1)
    cl_c = float(n_nodes - 1)/sum_distances_vector
    featurized_X = cl_c
    return featurized_X

def betweenness_centrality(X):
    n_nodes = X.shape[0]
    A_inv = 1./X
    G_inv = ig.Graph.Weighted_Adjacency(list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
    btw = np.array(G_inv.betweenness(weights='weight', directed=False))*2./((n_nodes-1)*(n_nodes-2))
    return btw

def eigenvector_centrality(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                attr="weight", loops=False)
    eigc = G.eigenvector_centrality(weights='weight', directed=False)
    return np.array(eigc)

def pagerank(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="DIRECTED", attr="weight", loops=False)
    return np.array(G.pagerank(weights="weight"))

def efficiency(X):
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False, unweighted=False)
    inv_SPL_with_inf = 1./SPL
    inv_SPL_with_nan = inv_SPL_with_inf.copy()
    inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)]=np.nan
    efs = np.nanmean(inv_SPL_with_nan, 1)
    return efs

def clustering_coefficient(X):
    Gnx = nx.from_numpy_matrix(X)
    clst_geommean = list(nx.clustering(Gnx, weight='weight').values())
    clst_geommean
    return np.array(clst_geommean)

def triangles(X):
    clust = clustering_coefficient(X)

    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
            attr="weight", loops=False)
    non_weighted_degrees = np.array(G.degree())
    non_weighted_deg_by_deg_minus_one = np.multiply(non_weighted_degrees,
            (non_weighted_degrees - 1))
    tr = np.multiply(np.array(clust),
            np.array(non_weighted_deg_by_deg_minus_one, dtype = float))/2.
    return tr


########################
# TRANSFORMERS CLASSES #
########################

pairs_data = pd.read_csv('../data/parkinson_pairs_data.csv', index_col = None)

def generate_even_sample(data, n = 1000, seed = 0):
    sample_of_1 = data[data.are_same == 1].sample(n=n, random_state=seed)
    sample_of_0 = data[data.are_same == 0].sample(n=n, random_state=seed)
    return pd.concat([sample_of_1, sample_of_0], axis=0)

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target, copy=True):
        self.target = target

    def fit(self, pairs_data, y=None):
        return self

    def transform(self, pairs_data, data_dir='../data/parkinson/'):
        matrices = {}
        if self.target != 'All':
            pairs_data = pairs_data[(pairs_data.subject1_target == self.target) & (pairs_data.subject2_target == self.target)]
        file_ids = np.unique(pairs_data[['subject1_id', 'subject2_id']])
        for file_id in file_ids:
            for file in os.listdir(data_dir+file_id):
                if 'FULL' in file:
                    print(data_dir + file_id + '/' + file)
                    matrix = np.loadtxt(data_dir + file_id + '/' + file)
                    matrix = np.delete(matrix, [3,38], axis = 1)
                    matrix = np.delete(matrix, [3,38], axis = 0)
                    np.fill_diagonal(matrix, 0)
                    matrices[file_id] = matrix

        pairs_data = generate_even_sample(pairs_data, n = int(pairs_data.are_same.sum()))
        return {'pairs_data': pairs_data,
                'matrices': matrices}

class MatrixNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm, copy=True):
        self.norm    = norm
        self.copy    = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = {}

        for key in X['matrices'].keys():
            X_transformed[key] = self.norm(X['matrices'][key])

        return {'pairs_data': X['pairs_data'],
                'matrices': X_transformed}

class MatrixFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, features, copy=True):
        self.features = features
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        cur_features = {}
        for key in X['matrices'].keys():
            cur_features[key] = self.features[0](X['matrices'][key])
            for feature_func in self.features[1:]:
                cur_features[key] = np.append(cur_features[key], feature_func(X['matrices'][key]))

        return {'pairs_data': X['pairs_data'],
                'features': cur_features}

def gen_dist(p): return lambda x,y: minkowski(x.reshape(-1),y.reshape(-1),p)
func_list = [chebyshev] + [gen_dist(i) for i in [1, 2]]

class VectorFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, func_list):
        self.func_list = func_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors1 = X['pairs_data'].subject1_id.apply(lambda x: X['features'][x])
        vectors2 = X['pairs_data'].subject2_id.apply(lambda x: X['features'][x])
        features = []
        for index in vectors1.index:
            feats = []
            for function in self.func_list:
                feats.append(function(vectors1[index], vectors2[index]))
            features.append(feats)

        return np.array(features)

######################
# PARAMETERS SETTING #
######################

sys.path.append(os.path.abspath('../../../Reskit/'))

from reskit.core import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

grid_cv = StratifiedKFold(n_splits=10, shuffle=True,  random_state=0)

eval_cv = StratifiedShuffleSplit(
            n_splits = 100,
            test_size = 0.2,
            random_state = 0 )

steps = [
    ('norm', [
        ('-', [
            ('object', MatrixNormalizer),
            ('params', [
                ('param', {'norm': no_norm}) ]) ]),
        ('max', [
            ('object', MatrixNormalizer),
            ('params', [
                ('param', {'norm': max_norm}) ]) ]),
        ('binar', [
            ('object', MatrixNormalizer),
            ('params', [
                ('param', {'norm': binar_norm}) ]) ]),
        ('mean', [
            ('object', MatrixNormalizer),
            ('params', [
                ('param', {'norm': mean_norm}) ]) ]) ]),
    ('base_features', [
        ('bag_of_edges', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [bag_of_edges]}) ]) ]),
        ('degrees', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [degrees]}) ]) ]),
        ('closeness_centrality', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [closeness_centrality]}) ]) ]),
        ('betweenness_centrality', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [betweenness_centrality]}) ]) ]),
        ('eigenvector_centrality', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [eigenvector_centrality]}) ]) ]),
        ('pagerank', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [pagerank]}) ]) ]),
        ('efficiency', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [efficiency]}) ]) ]),
        ('clustering_coefficient', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [clustering_coefficient]}) ]) ]),
        ('triangles', [
            ('object', MatrixFeaturizer),
            ('params', [
                ('param', {'features': [triangles]}) ]) ]) ]),
    ('pairwise_features', [
        ('l1_l2_linf', [
            ('object', VectorFeaturizer),
            ('params', [
                ('param', {'func_list': func_list}) ]) ]) ]),
    ('scaler', [
        ('standard', [
            ('object', StandardScaler),
            ('params', [
                ('None', {}) ]) ]) ]),
    ('classifier', [
        ('grid', [
            ('object', GridSearchCV),
            ('params', [
                ('LR', {
                    'estimator': LogisticRegression(),
                    'cv': grid_cv,
                    'n_jobs': -1,
                    'param_grid': {
                        'penalty': ['l1', 'l2'],
                        'C':[0.05*i for i in range(1,20)],
                        'fit_intercept': [True],
                        'max_iter': [50, 100],
                        'random_state': [0],
                        'solver': ['liblinear'],
                        'n_jobs': [1] } }),
                ('RF', {
                    'estimator': RandomForestClassifier(),
                    'cv': grid_cv,
                    'n_jobs': -1,
                    'param_grid': {
                        'n_estimators': [100],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [2, 3, 3, 5, 7],
                        'min_samples_split': [2],
                        'min_samples_leaf': [1],
                        'min_weight_fraction_leaf': [0.0],
                        'max_features': [0.5, 1],
                        'max_leaf_nodes': [None],
                        'bootstrap': [True],
                        'oob_score': [False],
                        'n_jobs':[1],
                        'random_state': [0],
                        'verbose': [0],
                        'warm_start': [False],
                        'class_weight': [None] } }),
                ('GBT', {
                    'estimator': XGBClassifier(),
                    'cv': grid_cv,
                    'n_jobs': -1,
                    'param_grid': {
                        'max_depth': [2,3,4,5],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                        'n_estimators': [100],
                        'silent': [True],
                        'objective': ['binary:logistic'],
                        'nthread': [1],
                        'gamma': [0],
                        'subsample': [1.0, 0.9, 0.8, 0.7, 0.6],
                        'colsample_bytree': [1.0],
                        'base_score': [0.5],
                        'seed': [0] } }),
                ('SGD', {
                    'estimator': SGDClassifier(),
                    'cv': grid_cv,
                    'n_jobs': -1,
                    'param_grid': {
                        'loss':['hinge', 'log', 'modified_huber'],
                        'penalty': ['elasticnet'],
                        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
                        'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        'fit_intercept': [True],
                        'n_iter': [50, 100, 200],
                        'shuffle': [True],
                        'verbose':[0],
                        'epsilon': [0.1],
                        'n_jobs': [-1],
                        'random_state':[0],
                        'learning_rate': ['optimal'],
                        'eta0': [0.0],
                        'power_t': [0.5],
                        'class_weight': [None] } }),
                ('SVC', {
                    'estimator': SVC(),
                    'cv': grid_cv,
                    'n_jobs': -1,
                    'param_grid': {
                        'C': [0.05*i for i in range(1,11)],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': list(range(2,5)),
                        'gamma': [0.01, 0.001, 0.05, 0.1, 0.2],
                        'max_iter': [50, 100],
                        'random_state': [0],
                        'shrinking': [True] } }) ]) ]) ]) ]

scoring = ['accuracy', 'roc_auc']
targets = ['All', 'PD', 'Control']
steps = Steps(steps)

cfg = Config(   steps = steps,
                eval_cv = eval_cv,
                scoring = scoring   )

output_folder = 'PPMI_pairwise_results/'

for target in targets:
    pp = Pipeliner( config = cfg )
    y = DataTransformer( target ).fit_transform( pairs_data )['pairs_data'].are_same
    X = DataTransformer( target ).fit_transform( pairs_data )

    pp.get_results( X, y.values,
                    featuring_steps = [ 'norm',
                                        'base_features',
                                        'pairwise_features' ],
                    results_file    = output_folder + target + '.csv' )
    print('\n\n', target, 'Done')
