import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data, ignoring the first line since it's a header with '%'
# Assuming the data uses whitespace as delimiter, since your header seems to suggest this
data = np.loadtxt('pistachio_data.txt', delimiter=' ', skiprows=1)

# Subtract the mean
data_mean = data - np.mean(data, axis=0)

# Compute standard deviation and scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_mean)

# Perform PCA
pca = PCA()
pca.fit(data_scaled)

# Access the covariance matrix and find the maximum absolute value
covariance_matrix = np.cov(data_scaled, rowvar=False)
max_abs_value = np.max(np.abs(covariance_matrix))

print("Maximum absolute value in the covariance matrix:", max_abs_value)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data with np.loadtxt, skipping the header and assuming space delimiter
data = np.loadtxt('pistachio_data.txt', delimiter=' ', skiprows=1)

# Subtract the mean
data_mean = data - np.mean(data, axis=0)

# Compute standard deviation and scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_mean)

# Perform PCA
pca = PCA()
pca.fit(data_scaled)

# Question 2: Calculate cumulative variance explained by the PCA components
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.where(cumulative_variance >= 0.97)[0][0] + 1  # plus one as index starts at 0

print("Number of components needed to explain at least 97% variance:", num_components)

# Question 3: Project the measurements of the first nut onto the PCA components
first_nut_projection = pca.transform(data_scaled[0:1])
sum_of_squares = np.sum(first_nut_projection**2)

print("Sum of squared projected values for the first nut:", sum_of_squares)

# Question 4: Identify the feature with the smallest standard deviation
feature_std = np.std(data, axis=0)
min_std_index = np.argmin(feature_std)
features = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY', 'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS']
smallest_std_feature = features[min_std_index]

print("Feature with the smallest standard deviation:", smallest_std_feature)
