# this file will have functions to create a connectivity matrix based on
# Vazquez-Rodrıguez et al. 2019, PNAS, created for Neurohackademy 2023

import numpy as np
import nilearn as nl
from sklearn.linear_model import LinearRegression
import nibabel as nib

def get_predictor_vectors(mats, nodes):
    # gets list of matrices and a node number
    # returns a vector for each matrix with the values for that node
    return [mat[:, nodes] for mat in mats]

def predict_function(predictors, functional, prediction_method, return_model=True):
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



def get_r_values(model, functional, score_method='r_squared'):
    # get r squared values for the model
    return model.score(functional, score_method)

def euclidean_distance(parcellation):
    """
    This function will take a parcellation and return a matrix of the euclidean distance between each node
    input: parcellation file path (string)
    output: matrix of euclidean distance between each node (numpy array)
    """
    coords = nl.find_parcellation_cut_coords(nib.load(parcellation))
    dist_mat = []
    # for each node in the parcellation, calculate the distance to each other node
    for node in coords:
        dist_mat.append([np.linalg.norm(node - other_node) for other_node in coords])
    return np.array(dist_mat)

def communicability(matrix):
    pass

def shortest_path_length(matrix):
    pass


def tether(func_mats, struct_mats, parcellation, matrices_functions=[], get_r=True, prediction_method='linear',
           include_eucledian=True):
    # this function will take a structural matrix and a fuctional matrix and create a
    # connectivity matrix based on the Vazquez-Rodrıguez et al. 2019, PNAS
    # method
    eucledian_matrix = euclidean_distance(parcellation)
    func_group_mat, struct_group_mat = group_concensus_maps(func_mats, struct_mats)

    mats = [mat_func(struct_group_mat) for mat_func in matrices_functions]
    if include_eucledian:
        mats.append(eucledian_matrix)

    # run over each node in the matrices and create vectors for each matrix for each node
    # then run the regression and get the r value
    n_nodes = np.shape(func_mats)[1]
    node_predictions = []
    if get_r:
        r_values = []
    for node in range(n_nodes):
        predictors = get_predictor_vectors(mats, node)
        functional_truth = func_mats[:, node]
        # run regression
        node_prediction = predict_function(predictors, functional_truth, prediction_method, return_model=get_r)
        if get_r:
            r_values.append(get_r_values(node_prediction[0], functional_truth))
            node_prediction = node_prediction[1]
        node_predictions.append(node_prediction)
    if get_r:
        return node_predictions, r_values
    else:
        return node_predictions