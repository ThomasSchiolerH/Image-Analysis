import numpy as np

# Example landmark coordinates in (x, y) format
# Replace these with the actual coordinates from your images
standing_coords = np.array([[0, 1], [0, 5], [2, 4], [4, 4], [3, 6]])  # Replace with actual coordinates
running_coords = np.array([[3, 1], [7, 1], [3.5, 3], [5.5, 5], [4.5, 6]])   # Replace with actual coordinates

import numpy as np
from skimage.transform import EuclideanTransform

# Define landmarks as numpy arrays
src = np.array([[0, 1], [0, 5], [2, 4], [4, 4], [3, 6]]) 
dst = np.array([[3, 1], [7, 1], [3.5, 3], [5.5, 5], [4.5, 6]])

# Compute initial misalignment error
e_x = src[:, 0] - dst[:, 0]
e_y = src[:, 1] - dst[:, 1]
initial_error = np.dot(e_x, e_x) + np.dot(e_y, e_y)
print(f"Initial Landmark alignment error F: {initial_error}")

# Estimate Euclidean transformation
tform = EuclideanTransform()
tform.estimate(src, dst)

# Apply the transformation to the source landmarks
src_transformed = tform(src)

# Compute final alignment error
e_x_transformed = src_transformed[:, 0] - dst[:, 0]
e_y_transformed = src_transformed[:, 1] - dst[:, 1]
final_error = np.dot(e_x_transformed, e_x_transformed) + np.dot(e_y_transformed, e_y_transformed)
print(f"Final Landmark alignment error F after transformation: {final_error}")
