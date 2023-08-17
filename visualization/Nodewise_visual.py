import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib


# from your_first_script_name import tether, ...

# Paths to your pial files
lh_pial_file = '/freesurfer/100307/output/surf/lh.pial'
rh_pial_file = '/freesurfer/100307/output/surf/rh.pial'

# Load the surfaces
lh_verts, _ = nib.freesurfer.read_geometry(lh_pial_file)
rh_verts, _ = nib.freesurfer.read_geometry(rh_pial_file)

# Combine coordinates for visualization
all_verts = np.vstack([lh_verts, rh_verts])

# Downsample: select every nth vertex to reduce density
n = 30
downsampled_verts = all_verts[::n, :]

# Placeholder: Load structural and functional matrices and atlas metadata
# func_mats = ...
# struct_mats = ...
# atlas_metadata = ...
# parcellation = ...

# Get predicted matrices and R^2 values from the tether function
# predicted_matrices, r2_values = tether(func_mats, struct_mats, parcellation, atlas_metadata)
# R2 = np.array(r2_values)[::n]

# For the time being, I'm commenting out the above and using the mock values as placeholder
R2 = np.random.rand(downsampled_verts.shape[0])

# Size and color of nodes based on R^2
size = 5 * (1 - R2)  # Adjust size to make nodes more visible
colors = plt.cm.viridis(1 - R2)

# Extract coordinates
x, y, z = downsampled_verts[:, 0], downsampled_verts[:, 1], downsampled_verts[:, 2]

# Visualize
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, s=size, c=colors)

# Add a colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Inverse R^2')

ax.set_title("Spatial Distribution of Structure–Function Correspondence in Brain")
plt.show()
