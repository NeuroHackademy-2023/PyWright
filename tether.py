# this file will have functions to create a connectivity matrix based on
# Vazquez-Rodrıguez et al. 2019, PNAS, created for Neurohackademy 2023

import numpy as np
import nilearn as nl
from sklearn.linear_model import LinearRegression
import nibabel as nib
from netneurotools.network import struct_consensus
from scipy.linalg import expm
import networkx as nx

def group_consensus_mats(func_mats, sc_mats, atlas_hemiid, dist_mat):
    """
    this function will take a list of functional matrices and a list of structural matrices and return the grouped
    comcemus matrices for each ome.
    :param func_mats: inputs are a 3D NxNxs FC matrix
    :param sc_mats: a 3D NxNxs SC matrix
    :param atlas_hemiid: a vector of length N that has 0s for left hemisphere and 1s for right hemisphere
    :return: a 2D NxN group consensus FC matrix and a 2D NxN group consensus SC matrix
    """
    if not atlas_hemiid:
        raise ValueError('need `atlas_hemiid` argument, a N x 1 dimensional array with 0s and 1s for left- and right- hemisphere')
    
    # compute group-consensus FC matrix by averaging all individual matrices
     group_fcmat = np.mean(func_mats, axis=2)

    # compute group-consensus SC matrix using neurotools from Masic lab @ McGill
    group_scmat = struct_consensus(data = sc_mats,
                                    distance = dist_mat,
                                    hemiid = atlas_hemiid,
                                    weighted = False)
    return group_fcmat, group_scmat


def get_predictor_vectors(mats, nodes):
    """this function will take a list of matrices and a node number and return a vector for each matrix with the values
    for that node
    :param mats: list of structural matrices used to predict functional matrices. these have to be the same size and
    with the same order of nodes.
    :param nodes: node number to get the values for
    :return: a vector for each matrix with the values for the connection of that node to all other nodes"""
    # gets list of matrices and a node number
    # returns a vector for each matrix with the values for that node
    return [mat[:, nodes] for mat in mats]

  
def predict_function(predictors, functional, prediction_method, return_model=True):
    """this function will take a list of structural predictors and a functional matrix and return a predicted
    functional matrix
    :param predictors: list of vectors for each structural matrix with the values for the connection of that
    node to all other nodes
    :param functional: a vector of functional values for each node's connection to all other nodes
    :param prediction_method: the method to use for prediction. currently only linear regression is implemented
    :return: a predicted functional matrix"""

    # standarize predictors
    predictors = [predictor - np.mean(predictor)/np.std(predictor) for predictor in predictors]
    # fit multiple linear regression
    if prediction_method == 'linear':
        model = LinearRegression()
    else:
        raise TypeError(f'{prediction_method} not implemented for prediction method')
    model.fit(predictors, functional)
    # predict functional values
    if return_model:
        return model, model.predict(predictors)
    return model.predict(predictors)



def euclidean_distance(parcellation):
    """
    This function will take a parcellation and return a matrix of the euclidean distance between each pair of nodes
    :param parcellation: path to a parcellation file or a nibabel image object
    :return: a matrix of the euclidean distance between each pair of nodes

    """
    if isinstance(parcellation, str):
        parcellation = nib.load(parcellation)
    coords = nl.find_parcellation_cut_coords(parcellation)
    dist_mat = []
    # for each node in the parcellation, calculate the distance to each other node
    for node in coords:
        dist_mat.append([np.linalg.norm(node - other_node) for other_node in coords])
    return np.array(dist_mat)

def communicability(adjmat, normalize=False, thresh=0):
    """
    This function will take an adjacency matrix and return a communicability matrix
    the communicability matrix is the weighted  sum  of  all  paths  and  walks  between all pairs of nodes
    :param adjmat:
    :param normalize:
    :return:
    """
    
    if not np.any(np.logical_or(adjmat == 0, adjmat == 0)):
        # binarize the matrix and print a warning
        print(f'Input matrix is not binary. Converting to binary with threshold {thresh}.')
        adjmat = (adjmat > thresh).astype(int)

    # normalize by largest eigenvalue (prevents extremely large values)
    if normalize:
        norm = np.linalg.eigvals(adjmat).max()
        adjmat = adjmat / norm

    # expm from scipy.linalg computes the matrix exponential using Padé approximation
    return expm(adjmat)

def shortest_path_length(matrix, threshold=-1):
    """this function will take a matrix and return a matrix of the shortest path length between each pair of nodes
    :param matrix: a matrix of the connection strength between each pair of nodes
    :param threshold: the threshold to use for binarizing the matrix. if -1, the matrix will not be binarized
    :return: a matrix of the shortest path length between each pair of nodes"""

    if threshold == -1:
        pass
    else:
        matrix = (matrix > threshold).astype(int)

    # compute the shortest path pairs and convert them back to a matrix
    return nx.floyd_warshall_numpy(nx.from_numpy_matrix(matrix))





def tether(func_mats, struct_mats, parcellation, matrices_functions=[], get_r2=True, prediction_method='linear',
           include_eucledian=True):
    # this function will take a structural matrix and a fuctional matrix and create a
    # connectivity matrix based on the Vazquez-Rodrıguez et al. 2019, PNAS
    # method
    eucledian_matrix = euclidean_distance(parcellation)
    func_group_mat, struct_group_mat = group_consensus_mats(func_mats, struct_mats, atlas_hemiid, eucledian_matrix)

    mats = [mat_func(struct_group_mat) for mat_func in matrices_functions]
    if include_eucledian:
        mats.append(eucledian_matrix)

    # run over each node in the matrices and create vectors for each matrix for each node
    # then run the regression and get the r value
    n_nodes = np.shape(func_mats)[1]
    node_predictions = []
    if get_r2:
        r_values = []
    for node in range(n_nodes):
        predictors = get_predictor_vectors(mats, node)
        functional_truth = func_mats[:, node]
        # run regression
        node_prediction = predict_function(predictors, functional_truth, prediction_method, return_model=get_r)
        if get_r2:
            r_values.append(node_prediction[0].score(functional_truth))
            node_prediction = node_prediction[1]
        node_predictions.append(node_prediction)
    if get_r2:
        return node_predictions, r_values
    else:
        return node_predictions
