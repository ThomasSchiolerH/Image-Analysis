import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1) Load the data
data = pd.read_csv('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/pistachio/pistachio_data.txt', sep=" ", comment='%', header=None)
data.columns = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ", "SOLIDITY", "CONVEX_AREA", "EXTENT", "ASPECT_RATIO", "ROUNDNESS", "COMPACTNESS"]

# 2) Subtract the mean from the data
data_centered = data - data.mean()

# 3) Compute the standard deviation of each measurement
std_dev = data_centered.std()

# Find the measurement with the smallest standard deviation
min_std_dev_measurement = std_dev.idxmin()

print(f"The measurement with the smallest standard deviation is: {min_std_dev_measurement}")

# 4) Divide each measurement by its own standard deviation
data_standardized = data_centered / std_dev

# Compute the covariance matrix and find the maximum absolute value
cov_matrix = data_standardized.cov()
max_abs_value = cov_matrix.abs().max().max()

print(f"The maximum absolute value in the covariance matrix is: {max_abs_value}")


# 5) Do the PCA
pca = PCA()
pca_result = pca.fit_transform(data_standardized)

#print(pca_result)
# Compute the sum of squared projected values for the first nut
first_nut = pca_result[0]
sum_of_squares = np.sum(first_nut**2)

print(sum_of_squares)

# Find the number of components needed to explain at least 97% of the variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
num_components = np.where(cumulative_variance >= 0.97)[0][0] + 1

print(f"The number of components needed to explain at least 97% of the variance is: {num_components}")
