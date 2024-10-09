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
import pandas as pd
import seaborn as sns
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

# Load images
images = [io.imread(f'/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/ImagePCA/spoon{i}.png') for i in range(1, 7)]

# Convert to gray scale if necessary (assuming the images might not be in gray scale)
gray_images = [color.rgb2gray(img) if img.ndim == 3 else img for img in images]

# Convert each image to float for accurate averaging
gray_images = [img_as_float(img) for img in gray_images]

# Compute the average image
average_image = np.mean(gray_images, axis=0)

# Access the pixel value at position (500, 100) in a 1-based coordinate system
# Convert to 0-based by subtracting 1 from each index
pixel_value = average_image[499, 99]

print(f'The value of the pixel at (row=500, column=100) is: {pixel_value}')
pixel_value_255 = int(pixel_value * 255)

print(f'The value of the pixel at (row=500, column=100) in the 0-255 range is: {pixel_value_255}')

# Threshold the images to binary
binary_images = [img > 100/255 for img in gray_images]  # Normalized threshold since images are in [0,1]

# Stack images into a single numpy array for PCA
# Flatten each image to a 1D array because PCA expects 2D input (samples, features)
image_stack = np.stack([img.ravel() for img in binary_images])

# Perform PCA
pca = PCA(n_components=5)  # You can adjust the number of components if needed
pca.fit(image_stack)

# Percentage of variance explained by the first principal component
variance_explained = pca.explained_variance_ratio_[0] * 100  # Multiply by 100 to convert to percentage

print(f'The first principal component explains {variance_explained:.2f}% of the total variation.')

# Assuming 'image_stack' contains the flattened gray scale images already centered by subtracting the mean
Psi = np.mean(image_stack, axis=0)
A = image_stack - Psi

# Fit PCA to the centered data
pca = PCA()
pca.fit(A)

# Get the cumulative variance to determine the number of components to retain
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance >= 0.65)[0][0] + 1  # At least 65% variance explained

# Refit PCA with the desired number of components
pca = PCA(n_components=n_components)
pca.fit(A)

# Project 'spoon1.png' onto the PCA space
spoon1_projection = pca.transform(A[0].reshape(1, -1))

print(f"Number of components used: {n_components}")
print(f"Coordinates of spoon1.png in the reduced PCA space: {spoon1_projection.flatten()}")

# Define the angle in degrees and convert to radians
theta = 20
theta_rad = np.deg2rad(theta)

# Rotation matrix
R = np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad), np.cos(theta_rad)]
])

# Scaling matrix
s = 2
S = np.array([
    [s, 0],
    [0, s]
])

# Translation matrix, in homogeneous coordinates
tx, ty = 3.1, -3.3
T = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

# Combine transformations by matrix multiplication, first rotate, then scale, then translate
# Note that the point also needs to be in homogeneous coordinates
point = np.array([10, 10, 1])
transformed_point = T @ (S @ np.vstack((R @ point[:2], 1)))

# Print the resulting point
print("Resulting position of the point (10, 10) after transformations is:", transformed_point[:2])

