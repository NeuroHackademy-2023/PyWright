#Inputs:
#- `matrix`: structural connectivity matrix
#- `do_norm`: 0 = no normalization, 1 = binarize, 2 = normalize
#- *optional* `bin_thresh`: default = o, otherwise = min> threshold >max

# Output:
#- norm: the normalized matrix

# Code:
#load the matrix


def normalize_matrix (matrix, do_norm = 0, bin_thresh = 0):
mini = matrix.min()
maxi = matrix.max()
    if do_norm == 0:
        print ('No normalization of structural connectivity matrix')
    elif do_norm == 1:
        print ('Binarization of structural matrix')
        if bin_thresh == 0:
	    threshold = 0
        elif mini <= bin_thresh <= maxi:
            threshold = bin_thresh
	else:
	    print('Threshold must be in values range.
	    matrix minimum value > bin_thresh > matrix maxmimum value')
    return matrix = matrix[matrix <= threshold] = 0;
    elif do_norm == 2:
        print ('Normalization of structural matrix')
    return matrix = matrix/maxi
    else:
        print('Specify normalization input. 0 = no normalization, 1 = binarize, 2 = normalize')

