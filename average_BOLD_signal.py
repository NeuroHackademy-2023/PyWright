

import os
import glob
import nibabel as nib 
import numpy as np


def average_BOLD_signal(subjects_directory, atlas_data, brain_par, bold_filename):
    # Search for all subject folders
    subject_folders = glob.glob(os.path.join(subjects_directory, "Subject_*"))

    # Loop through each subject's folder
    for subject_folder in subject_folders:
        bold_file_path = os.path.join(subject_folder, "fake_BOLD.nii")
        if os.path.exists(bold_file_path):
            bold_nifti = nib.load(bold_file_path)
            bold_data = bold_nifti.get_fdata()

            # Check if the bold_data has 4D shape
            if bold_data.ndim != 4:
                raise ValueError(f"Expected 4D BOLD data, but got data with shape {bold_data.shape}")

            # Create the array that will store averaged BOLD signal 
            r, c, d, t = bold_data.shape
            averaged_bold_seq = np.zeros((t, brain_par))            

            # Averaging Bold signal
            for time in range(t):
                bold_time = bold_data[:, :, :, time]
                for brain_area in range(1, brain_par + 1):
                    temp = []
                    for axial_plane in range(d):
                        # Find indices of voxels in the current axial plane that belong to the current brain area
                        indices_x, indices_y = np.where(atlas_data[:, :, axial_plane] == brain_area)
                        for index in range(len(indices_x)):
                            x_idx = indices_x[index]
                            y_idx = indices_y[index]
                            # Ensure the voxel indices are within valid range
                            if x_idx < r and y_idx < c:
                                # Append the BOLD value of the voxel to the temporary list
                                temp.append(bold_time[x_idx, y_idx, axial_plane])
                    print(f"Subject: {subject_folder}, Timeseries: {time}, Brain area: {brain_area}")
                    # Convert the temporary list to a NumPy array and remove zero values
                    temp = np.array(temp)
                    temp = temp[temp != 0]
                    # Calculate the mean of the BOLD values and store in the averaged_bold_seq array
                    averaged_bold_seq[time, brain_area - 1] = np.mean(temp)
                #Save the data for each subject
                subject_csv_filename = os.path.join(subject_folder, f"{subject_folder}_averaged_bold_data.csv")
                np.savetxt(subject_csv_filename, averaged_bold_seq, delimiter=",")
                print(f"Saved averaged_bold_seq to {subject_csv_filename}")
        else:
            print(f"No BOLD file found for {subject_folder}")
    # Calculate the mean of the averaged_bold_seq array along axis 1
    mean_sig = np.mean(averaged_bold_seq, axis=1)

    for b in range(brain_par):
        averaged_bold_seq[:, b] = averaged_bold_seq[:, b] - mean_sig

    csv_filename = "averaged_bold_data.csv"
    np.savetxt(csv_filename, averaged_bold_seq, delimiter=",")
    print(f"Saved averaged_bold_seq to {csv_filename}")

    return averaged_bold_seq
