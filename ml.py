import os
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFpr
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import f_regression
import warnings

def ml(eff_conn_list, output_labels):
    '''
    eff_conn_list should be NxNxnumSubj, each effective connectivity matrices should be NxN
    output_labels should be a 1-D numeric array of size numSubj
    '''
    OAtrain_LASSO_vcts = []
    for sub in range(0,eff_conn_list.shape[2]):
        subslice = eff_conn_list[:,:,sub] # each slice of cube is a subject's matrix
        subtril = np.tril(subslice, k=0) # upper triangle is 0
        edgelist = np.reshape(subtril, len(subtril)**2) # reshape each matrix to a list with length 246*246
        edgelist = np.nan_to_num(edgelist) # change nan to 0 (so regularized model doesn't fail)
        OAtrain_LASSO_vcts.append(edgelist)
    
    pred_behav = []
    real_behav = []  # To store real IQ values

    # define data
    X = np.asarray(OAtrain_LASSO_vcts)
    y = np.asarray(output_labels)

    #Intersection network to visualize
    intersection_network = np.zeros(eff_conn_list.shape[0:2])

    # use LOO CV to test predictive power of ElasticNet model with optimal parameters
    for leftout in range(0, len(y)):
        print("Leaving out subject " + str(leftout+1) + " (" + str(leftout+1) + "/" + str(len(y)) + ")")

        # define train and test data
        X_train = np.delete(X, leftout, axis=0)
        y_train = np.delete(y, leftout)
        X_test = X[leftout,:]
        y_test = y[leftout]

        # filter model by arbitrarysignificance threshold
        threshold = 0.05
        filter_idx = SelectFpr(f_regression, alpha=threshold).fit(X_train, y_train).get_support(indices=True)
        filtered_data = X_train[:,filter_idx]

        # use CV to find optimal alpha and l1 parameters for ElasticNet model
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
        ratios = np.arange(0.1,1,0.1)
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        enetcv = ElasticNetCV(l1_ratio=ratios, cv=cv, alphas=alphas).fit(filtered_data, y_train)
        print("Selected alpha: "+str(enetcv.alpha_)+", selected L1 ratio: "+str(enetcv.l1_ratio_))

        # add current network to intersection network
        surviving_indices_elastic_net = np.where(enetcv.coef_ != 0)[0] # indices of nonzero elasticnet features
        final_coefficients_original = np.zeros(X.shape[1]) # intialize vector of 0s
        original_indices_elastic_net = filter_idx[surviving_indices_elastic_net] # indices of nonzero 
        final_coefficients_original[original_indices_elastic_net] = 1   
        current_network = np.reshape(final_coefficients_original, intersection_network.shape)
        intersection_network = np.add(intersection_network, current_network)

        edges = (len(enetcv.coef_[enetcv.coef_!=0]))
        #print("Number of edges in training data network: "+str(edges))
        y_pred = enetcv.predict(X_test[filter_idx].reshape(1, -1))
        y_pred = y_pred[0]
        pred_behav.append(y_pred)
        real_behav.append(y_test)

    # Create a DataFrame to store and display the results
    results_df = pd.DataFrame({
        'Real IQ Values': real_behav,
        'Predicted IQ Values': pred_behav
    })

    
    return results_df