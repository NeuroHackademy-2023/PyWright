from cloudpathlib import S3Path, S3Client
import nibabel as nib
from pathlib import Path
import os
from shutil import rmtree
import json
import pandas as pd
from dipy.tracking import utils
import numpy as np
from scipy.stats import pearsonr


def read_json(path):
    with open(path) as f:
        return json.load(f)

def file_load(path, sub_name, client):
    path = S3Path(str(path).replace("{SUB}", sub_name), client=client)
    ext = Path(path).suffix
    extentions_d = {
                    ".gz": lambda x: nib.load(x),
                    ".csv": lambda x: pd.read_csv(x),
                    ".trk": lambda x: nib.streamlines.load(x),
                    ".json": read_json
                    }
    return extentions_d[ext](path)

def get_HCP_sub(preprocessed_file_paths, diffusion_file_paths, nsubs=100):
    """
    This function will take a list of file paths and return a generator that yields a dictionary of the file paths
    for each subject in the HCP dataset and the files for that subject.
    the files will be downloaded to "/temp/cache" and deleted after the generator yields the data.
    :param file_paths: a list of the paths to the files to be downloaded, the path should include "{SUB}" where the
    subject number should be in the file name. include the part of the path that is the same for all subjects (that is,
    everything after the subject folder).
    :param data_type: either preprocessed or diffusion, to indicate which dataset you want to download from
    :return: subject name and dictionary of file paths and the files
    """
    cache_path = Path('/tmp/cache')
    if os.path.isdir(cache_path):
        rmtree(cache_path)
    os.mkdir(cache_path)

    # Create a client that uses our cache path and that does not try to
    # authenticate with S3.
    client = S3Client(
        local_cache_dir=cache_path,
        no_sign_request=True)

    hcp_derivs_path_preproc = S3Path("s3://hcp-openaccess/HCP_1200", client=client)
    hcp_derivs_path_diffusion = S3Path("s3://open-neurodata/rokem/hcp1200/afq",
            client=client)

    for sub in hcp_derivs_path_preproc.glob("*"):
        if os.path.isdir(cache_path):
            rmtree(cache_path)
        os.mkdir(cache_path)
        sub_name = os.path.split(sub)[-1]
        preproc_data = {file_path: file_load(sub / file_path, sub_name, client) for file_path in file_paths}

        diff_sub = hcp_derivs_path_diffusion / sub_name
        diffusion_data = {file_path: file_load(diff_sub / file_path, sub_name, client) for file_path in diffusion_file_paths}
        preproc_data.update(diffusion_data)
        if nsubs <= 0:
            break
        yield sub_name, preproc_data
        nsubs -= 1


# usage example:

def get_connectivity_matrix(tracts, parcellation):
    affine_mat = parcellation.affine
    parcellation = parcellation.get_data()
    tracts.remove_invalid_streamlines()
    tracts = tracts.streamlines
    new_labels, lookup = utils.reduce_labels(parcellation.get_fdata())
    m, grouping = utils.connectivity_matrix(tracts, affine_mat, new_labels, return_mapping=True,
                                            mapping_as_streamlines=True)
    return m

def con_eff_bold_data(data_bold, data_labels):
    result = []
    for hemi in ['R', 'L']:
        avg_h_bold_mat = []
        # unique lh label
        unique_h = np.unique(data_labels)
        # loop through unique lh labels
        for label in unique_h:  # for each label
            label_idx = np.where(data_labels == label)[1]  # get the index of the vertices that match this label
            label_timeseriesda = np.mean(data_bold[:, label_idx],
                                         axis=1)  # take the mean of the vertices within the region
            avg_h_bold_mat.append(
                label_timeseriesda)  # append the mean BOLD signal of this region at each time point (volume)
        result.append(avg_h_bold_mat)  # stack all the regions together into a matrix


    return np.vstack(*result)


def func_mats(subj_bold, avg_bold_mat):
    ### Generate N x N FC matrix for each subject
    subj_bold_data = subj_bold.get_fdata()

    num_regions = avg_bold_mat.shape[0]
    FC_mat = np.zeros((num_regions, num_regions))  # intialize empty matrix

    # loop through each region index
    for i_region in range(num_regions):  # for each region (x-axis)
        for j_region in range(num_regions):  # and for each region (y-axis)
            if i_region == j_region:
                FC_mat == 0  # the correlation between the region and itself is set to 0 by convention
            else:  # otherwise compute the pearson correlation between the time series of each region with every other region
                FC_mat[i_region, j_region] = pearsonr(subj_bold_data[:, i_region], subj_bold_data[:, j_region])[0]

    return np.arctanh(FC_mat)  # to make the distribution of the values more normal, fisher-z transform them

def struct_mats():
    cms = []
    for sub, files in get_HCP_sub(["mri/MNINonLinear/aparc+aseg.nii.gz"], ["ses-01/{SUB}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.trk"], "diffusion"):
        tract_file = files["ses-01/{SUB}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.trk"]
        parc = files["mri/MNINonLinear/aparc+aseg.nii.gz"]
        cms.append(get_connectivity_matrix(tract_file, parc))
    return cms

def funct_dat_and_mats():
    bold_sigs = []
    mats = []
    for sub, files in get_HCP_sub(['MNINonLinear/fsaverage_LR59k/115825.aparc.59k_fs_LR.dlabel.nii',
                                   'MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_Atlas_1.6mm_hp2000_clean.dtseries.nii'])
        labels = files['MNINonLinear/fsaverage_LR59k/115825.aparc.59k_fs_LR.dlabel.nii']
        bold = files['MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_Atlas_1.6mm_hp2000_clean.dtseries.nii']
        bold_sigs.append(con_eff_bold_data(bold, labels))
        mats.append(func_mats(bold, labels))
    return bold_sigs, mats











