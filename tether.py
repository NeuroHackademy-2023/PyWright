# this file will have functions to create a connectivity matrix based on
# Vazquez-Rodrıguez et al. 2019, PNAS, created for Neurohackademy 2023

import numpy as np
from scipy.stats import pearsonr
from netneurotools.network import struct_consensus
def group_consensus_mats(func_mats, sc_mats):
     
     # compute group-consensus FC matrix by averaging all individual matrices

     group_fcmat = np.mean(func_mats, axis=2)

    # compute group-consensus SC matrix using previously-established methods
    (data, distance, hemiid, weighted=False)
     group_scmat = struct_consensus(data = sc_mats,
                                    distance = dist_mat,
                                    hemiid = , # N x 1 dimensional array, 0s and 1s for LH and RH (needs to be input by user or hard-coded for each atlas)
                                    weighted = False)


def make_matrix(mat_function, struct_mat):
    pass

def get_predictor_vectors(mats, nodes):
    pass

def predict_function(predictors, functional, prediction_method):
    pass

def get_r_values(node_prediction, functional):
    pass

def euclidean_distance(parcellation):
    pass

def communicability(group_scmat, normalize=False):
    #  weighted  sum  of  all  paths  and  walks  between those  nodes
    #  takes binarized structural connectome (adjacency matrix)
    
    if not np.any(np.logical_or(adjmat == 0, adjmat == 1)):
        raise ValueError('Input matrix must be binary.')

    # normalize by largest eigenvalue (prevents extremely large values)
    if normalize:
        norm = np.linalg.eigvals(adjmat).max()
        adjmat = adjmat / norm

    # expm from scipy.linalg computes the matrix exponential using Padé approximation
    return expm(adjmat)

def shortest_path_length(matrix):
    pass


def tether(func_mats, struct_mats, matrices_functions=[], get_r2=True, prediction_method='linear'):
    # this function will take a group-consensus tructural matrix and a group-consensus fuctional matrix and create a
    # connectivity matrix based on the Vazquez-Rodrıguez et al. 2019, PNAS

    mats = []
    for subject in struct_mats:
        mats.append([make_matrix(mat_func, struct_mats) for mat_func in matrices_functions])
    # run over each node in the matrices and create vectors for each matrix for each node
    # then run the regression and get the r value
    n_nodes = np.shape(func_mats)[1]
    node_predictions = []
    if get_r2:
        r_values = []
    for node in range(n_nodes):
        predictors = get_predictor_vectors(mats, node)
        functional = func_mats[:, node]
        # run regression
        node_prediction = predict_function(predictors, functional, prediction_method)
        if get_r:
            r_values.append(get_r_values(node_prediction, functional))
        node_predictions.append(node_prediction)
    if get_r2:
        return node_predictions, r_values
    else:
        return node_predictions