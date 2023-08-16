# This file creates a figure of the spatial distribution of structure-function correspondence based on
# Vazquez-Rodríguez et al. 2019, PNAS. Assistance provided by ChatGPT-4. 
# Created for Neurohackademy 2023.

import numpy as np
import matplotlib.pyplot as plt

def plot_spatial_distribution(aij, coor, r2_values):
    """
    Plot the spatial distribution of structure-function correspondence.
    
    Args:
    - aij (np.array): Adjacency matrix.
    - coor (np.array): Node spatial coordinates.
    - r2_values (np.array): R^2 values for each node.
    """
    
    # Normalize the R^2 values to the range [0, 1]
    normalized_r2 = (r2_values - np.min(r2_values)) / (np.max(r2_values) - np.min(r2_values))
    
    # Calculate sizes and colors inversely proportional to R^2 values
    sizes = (1.0 - normalized_r2) * 100  # adjust multiplier for suitable sizes
    colors = plt.cm.viridis(1.0 - normalized_r2)
    
    if coor.shape[1] == 2:
        plt.scatter(coor[:, 0], coor[:, 1], s=sizes, c=colors)
        plt.colorbar(label='$R^2$ value')
        plt.title('Spatial distribution of structure-function correspondence (2D)')
        plt.show()
        
    elif coor.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coor[:, 0], coor[:, 1], coor[:, 2], s=sizes, c=colors)
        ax.set_title('Spatial distribution of structure-function correspondence (3D)')
        plt.show()

# Example:
aij = np.array([
    # example adjacency matrix values
])

coor = np.array([
    # example 2D or 3D coordinates
])

r2_values = np.array([
    # example R^2 values for each node
])

plot_spatial_distribution(aij, coor, r2_values)
