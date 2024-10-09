import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_on_glass_data_F2023():
    # Step 1: Load the data
    glass_data = np.loadtxt('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/GlassPCA/glass_data.txt', comments="%", delimiter=' ')
    
    x = glass_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    # Normalize data
    mn = np.mean(x, axis=0)
    data = x - mn
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spread = maxs - mins
    data = data / spread

    # Assuming Sodium (Na) is the second feature in your dataset
    # Adjust the index if needed based on your dataset's structure
    sodium_index = 1  # Adjust this if Na is not the second column
    first_sodium_value = data[0, sodium_index]
    print(f"Answer: First value of Sodium (Na) after normalization: {first_sodium_value:.3f}")

    # Compute covariance matrix
    c_x = np.cov(data.T)
    print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")

    # Eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(c_x)
    v_norm = values / values.sum() * 100

    # Plot variance explained
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    # Project data onto principal components
    pc_proj = vectors.T.dot(data.T)

    # Find the maximum absolute value among the projected data
    max_proj_val = np.max(np.abs(pc_proj))
    print(f"Answer: maximum absolute projected value {max_proj_val:.3f}")

    # Variance explained by the first three PCs
    answer = v_norm[0] + v_norm[1] + v_norm[2]
    print(f"Answer: Variance explained by the first three PC: {answer:.2f}%")

if __name__ == "__main__":
    pca_on_glass_data_F2023()