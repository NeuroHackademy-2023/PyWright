# this file will have functions to create a connectivity matrix based on
# Vazquez-Rodrıguez et al. 2019, PNAS, created for Neurohackademy 2023

import numpy as np
from scipy.stats import pearsonr
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

def communicability(matrix):
    pass

def shortest_path_length(matrix):
    
    # Binarize the matrix and convert to float
    A = (matrix != 0).astype(float)
    
    l = 1
    Lpath = A.copy()
    D = A.copy()
    
    Idx = np.ones_like(A, dtype=bool)
    
    while Idx.any():
        l += 1
        Lpath = np.dot(Lpath, A)
        Idx = (Lpath != 0) & (D == 0)
        D[Idx] = l

    # Assign infinity to disconnected nodes

    # Clear diagonal (set to 0)
    pass


def tether(func_mats, struct_mats, matrices_functions=[], get_r=True, prediction_method='linear'):
    # this function will take a structural matrix and a fuctional matrix and create a
    # connectivity matrix based on the Vazquez-Rodrıguez et al. 2019, PNAS
    # method
    mats = []
    for subject in struct_mats:
        mats.append([make_matrix(mat_func, struct_mats) for mat_func in matrices_functions])
    # run over each node in the matrices and create vectors for each matrix for each node
    # then run the regression and get the r value
    n_nodes = np.shape(func_mats)[1]
    node_predictions = []
    if get_r:
        r_values = []
    for node in range(n_nodes):
        predictors = get_predictor_vectors(mats, node)
        functional = func_mats[:, node]
        # run regression
        node_prediction = predict_function(predictors, functional, prediction_method)
        if get_r:
            r_values.append(get_r_values(node_prediction, functional))
        node_predictions.append(node_prediction)
    if get_r:
        return node_predictions, r_values
    else:
        return node_predictions
