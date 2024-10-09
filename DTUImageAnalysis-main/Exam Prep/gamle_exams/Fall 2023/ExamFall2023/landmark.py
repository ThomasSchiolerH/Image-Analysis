import numpy as np
import math
from skimage.transform import SimilarityTransform, matrix_transform

src = np.array([[3, 1], [3.5, 3], [4.5, 6], [5.5, 5], [7, 1]])
dst = np.array([[1, 0], [2, 4], [3, 6], [4, 4], [5, 0]])

# Compute the differences between corresponding points
differences = src - dst

# Square the differences
squared_differences = differences ** 2

# Sum up the squared differences
sum_of_squared_distances = np.sum(squared_differences)

print(sum_of_squared_distances)

# Compute the centroids of src and dst
centroid_src = np.mean(src, axis=0)
centroid_dst = np.mean(dst, axis=0)

# Compute the translation vector
translation = centroid_dst - centroid_src

print(translation)

# Compute the centroids of src and dst
cm_1 = np.mean(src, axis=0)
cm_2 = np.mean(dst, axis=0)

# Compute the translation vector
translations = cm_2 - cm_1
print(f"Answer: translation {translations}")

# Estimate the similarity transform
tform = SimilarityTransform()
tform.estimate(src, dst)
print(f"Answer: rotation {abs(tform.rotation * 180 / np.pi):.2f} degrees")

# Apply the transformation to the source landmarks
src_transform = matrix_transform(src, tform.params)

# Compute the differences between the transformed source landmarks and the destination landmarks
e_x = src_transform[:, 0] - dst[:, 0]
e_y = src_transform[:, 1] - dst[:, 1]

# Compute the sum-of-squared-distances error after alignment
f_after = np.dot(e_x, e_x) + np.dot(e_y, e_y)
#print(f"Aligned landmark alignment error F: {f_after}")