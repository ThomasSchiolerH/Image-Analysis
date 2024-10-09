# Student number: s214968
from skimage import color, io
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_erosion, binary_dilation
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import prewitt
from skimage.filters import median
from skimage.util import img_as_float
from skimage import segmentation
from skimage import measure
import math
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd
from skimage.transform import rescale, resize
from skimage import color, data, io, morphology, measure, segmentation, img_as_ubyte, filters
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.color import label2rgb
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Load all images
image_files = glob.glob('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/flowers/flower*.jpg')
# images = [io.imread(file) for file in image_files]

# # Resize images to the size of the first image
# images = [resize(image, images[0].shape) for image in images]

# # Compute the average image
# average_image = np.mean(images, axis=0)

# # Flatten each image and stack into a 2D array
# image_array = np.array([image.flatten() for image in images])

# # Perform PCA with 5 components
# image_pca = PCA(n_components=5)
# image_pca.fit(image_array)

# # Generate synthetic images
# synth_image_plus = average_image + 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :].reshape(average_image.shape)
# synth_image_minus = average_image - 3 * np.sqrt(image_pca.explained_variance_[0]) * image_pca.components_[0, :].reshape(average_image.shape)

# # Display the average image and synthetic images
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(average_image, cmap='gray')
# plt.title('Average Image')
# plt.subplot(1, 3, 2)
# plt.imshow(synth_image_plus, cmap='gray')
# plt.title('Synthetic Image Plus')
# plt.subplot(1, 3, 3)
# plt.imshow(synth_image_minus, cmap='gray')
# plt.title('Synthetic Image Minus')
# plt.show()

# # Print the percentage of the total variation explained by the first principal component
# print(f"The first principal component explains {image_pca.explained_variance_ratio_[0]*100}% of the total variation.")

# # Transform all the images using PCA
# image_pca_transformed = image_pca.transform(image_array)

# # Find the indices of the flowers with the maximum and minimum projections onto the first principal component

images = [io.imread(file) for file in image_files]

# Resize images to the size of the first image
images = [resize(image, images[0].shape) for image in images]

# Compute the average image
average_image = np.mean(images, axis=0)

# Flatten each image and stack into a 2D array
image_array = np.array([image.flatten() for image in images])

# Perform PCA with 5 components
image_pca = PCA(n_components=5)
image_pca.fit(image_array)

# Print the percentage of the total variation explained by the first principal component
print(f"The first principal component explains {image_pca.explained_variance_ratio_[0]*100}% of the total variation.")

# Transform all the images using PCA
image_pca_transformed = image_pca.transform(image_array)

# Extract the projections onto the first principal component
projections = image_pca_transformed[:, 0]

# Define the possible answers as given
possible_answers = [
    (3, 7),
    (2, 5),
    (8, 9),
    (10, 12),
    (1, 12)
]

# Calculate the distances for each possible pair
distances = []
for (i, j) in possible_answers:
    distance = abs(projections[i - 1] - projections[j - 1])  # Subtract 1 for zero-based indexing
    distances.append((i, j, distance))

# Find the pair with the maximum distance
furthest_pair = max(distances, key=lambda x: x[2])
print(f"The two flowers that are furthest away from each other when projected onto the first principal component are Flower {furthest_pair[0]} and Flower {furthest_pair[1]}.")

# Display the images of the furthest pair
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[furthest_pair[0] - 1], cmap='gray')
plt.title(f'Flower {furthest_pair[0]}')
plt.subplot(1, 2, 2)
plt.imshow(images[furthest_pair[1] - 1], cmap='gray')
plt.title(f'Flower {furthest_pair[1]}')
plt.show()
