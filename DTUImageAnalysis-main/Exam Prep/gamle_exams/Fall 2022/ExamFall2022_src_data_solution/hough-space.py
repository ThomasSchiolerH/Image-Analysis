import numpy as np

# Given data points
data_points = np.array([(7, 13), (9, 10), (6, 10), (6, 8), (3, 6)])
theta_deg = 151
rho_target = 0.29

# Convert theta to radians
theta_rad = np.deg2rad(theta_deg)

# Compute rho for each point
computed_rhos = np.cos(theta_rad) * data_points[:, 0] + np.sin(theta_rad) * data_points[:, 1]

# Find the points that correspond to rho closest to the target
closest_indices = np.abs(computed_rhos - rho_target).argsort()[:2]  # Get the indices of the two closest points
closest_points = data_points[closest_indices]
closest_rhos = computed_rhos[closest_indices]

closest_points, closest_rhos
print(closest_points)