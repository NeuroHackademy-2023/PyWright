import numpy as np
import nibabel as nib
import copy
import pandas as pd
from sklearn import preprocessing as p

def _preprocess_BOLD_coneff(averaged_bold_seq):
    mean_sig = np.mean(averaged_bold_seq, axis=1)
    averaged_bold_seq = (averaged_bold_seq.T - mean_sig).T
    averaged_bold_seq = np.nan_to_num(averaged_bold_seq)
    min_max_scaler = p.MinMaxScaler()
    averaged_bold_seq_demean = min_max_scaler.fit_transform(averaged_bold_seq)
    return averaged_bold_seq_demean

def _threshold_struc_mat(struc_mat, threshold):
    return struc_mat * (struc_mat > threshold)

def _compute_2ndstep_mat(A):
    #calculate implement 1 degree separation matrix
    N = A.shape[0]
    B = np.zeros((N, N), dtype=np.int64)
    
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:  # If e_ij is above 0 in A, set e_ij as 0 in B
                B[i, j] = 0
            else:
                max_strength = 0
                for k in range(N):
                    if k != i and k != j:
                        max_strength = max(max_strength, min(A[i, k], A[k, j]))
                B[i, j] = max_strength
    
    return B

def _get_binary_mat(mat):
    mat[mat > 1] = 1
    return mat

def _normlaize_mat(mat):
    max_val = np.max(mat)
    mat = mat / max_val
    return mat

def _gradient_descrent(error_th, maxiter, skip_bold, lr, use_multistep, averaged_bold_seq, M, noconn, M_2nstep = None, gd_pass = 1):
    #TODO: check if multistep, have to input M_2nstep
    # check gd_pass can only be 1 if not multistep
    _iter = 0
    etem = []
    error_des = 10000
    while error_des > error_th and _iter < maxiter:
        error_des = 0
        gradient = np.zeros((averaged_bold_seq.shape[0], averaged_bold_seq.shape[0]))
        
        t = averaged_bold_seq.shape[1]

        # initialization- initial error
        for jj in range(skip_bold, t - 1):
            if gd_pass == 1:
                error_des += np.linalg.norm(
                        np.dot(M, averaged_bold_seq[:,jj]) - averaged_bold_seq[:,jj + 1]
                )
            elif use_multistep and (gd_pass == 2):
                error_des += np.linalg.norm(
                    np.dot(M, averaged_bold_seq[:,jj])
                    + np.dot(M_2nstep, averaged_bold_seq[:,jj])
                    - averaged_bold_seq[:,jj + 1]
                )

        error_des *= 0.5
        etem.append(error_des)

        # calculate gradient
        for jj in range(skip_bold, t - 1):
            gradient += (
                    np.dot(np.dot(M, averaged_bold_seq[:,jj]).reshape([-1,1]),
                           np.transpose(averaged_bold_seq[:,jj].reshape([-1,1])))
                    - np.dot(averaged_bold_seq[:,jj + 1].reshape([-1,1]), np.transpose(averaged_bold_seq[:,jj].reshape([-1,1])))
                )
            if use_multistep:
                gradient += (
                    np.dot(np.dot(M_2nstep, averaged_bold_seq[:,jj]).reshape([-1,1]),
                           np.transpose(averaged_bold_seq[:,jj].reshape([-1,1])))
                )
            #print(gradient)
         

        if gd_pass == 1:
            #print(gradient)
            M = M - lr * gradient
            M = M * noconn
        elif gd_pass == 2:
            M_2nstep = M_2nstep - lr * gradient
            M_2nstep = M_2nstep * noconn
            
        _iter += 1
        if (_iter % 100 == 0):
            print("iterations %f completed" %_iter)
            print("current error: %f" %error_des)
        
    return M, M_2nstep, etem #effective connectivity matrix, error over iterations

def structured_G_causality(struc_mat, averaged_bold_seq, use_multistep = 1, norm_opt = 1, use_deconv = 0, structural_thres = 2,
                           lr= 10**(-4), maxiter=500, error_th=10, skip_bold=5):
    #TODO: check data structure of struc_mat and bold_seq
    
    etem_2 = np.zeros(1)

    averaged_bold_seq = _preprocess_BOLD_coneff(averaged_bold_seq)
    #print(averaged_bold_seq)
    structure_direct = _threshold_struc_mat(struc_mat, structural_thres)
    structure_2nstep = _threshold_struc_mat(_compute_2ndstep_mat(struc_mat), structural_thres)

    if norm_opt == 1:
        print("Initial matrix binarized")
        M = _get_binary_mat(structure_direct)
        M_2nstep = _get_binary_mat(structure_2nstep)
        
    elif norm_opt == 2:
        print("Initial matrix normalized according to its largest value")
        M = _normlaize_mat(structure_direct)
        M_2nstep = _normlaize_mat(structure_2nstep)
        
    noconn_M = copy.deepcopy(M)
    noconn_M[noconn_M > 1] = 1
    noconn_M2 = copy.deepcopy(M_2nstep)
    noconn_M2[noconn_M2 > 1] = 1
    
    #print(noconn_M)

    print("start gradient descent")
    M, M_2nstep ,etem = _gradient_descrent(error_th, maxiter, skip_bold, lr, use_multistep, averaged_bold_seq, M, noconn_M,
                                           M_2nstep, 1)
    if use_multistep:
        print("start second pass gradient descent")
        M, M_2nstep, etem_2 = _gradient_descrent(error_th, maxiter, skip_bold, lr, use_multistep, averaged_bold_seq, M,noconn_M2,
                                                 M_2nstep, 2)
    
    
    etem = np.array(etem)
    etem = etem[etem != 0]
    if use_multistep:
        etem_2 = np.array(etem_2)
        etem_2 = etem_2[etem_2 != 0] #do something with etem, etem2?

    return M, M_2nstep, etem, etem_2