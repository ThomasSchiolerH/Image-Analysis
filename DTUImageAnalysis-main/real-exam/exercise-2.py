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


# Load the data
data_name = "/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/winePCA/wine-data.txt" 
x_org = np.loadtxt(data_name, comments="%")

# Separate measurements and producer information
x = x_org[:, :13]
producer = x_org[:, 13]

# Normalize the measurements
x_mean = np.mean(x, axis=0)
x_range = np.ptp(x, axis=0)  # ptp (peak-to-peak) function returns the range of values (max - min)
x_normalized = (x - x_mean) / x_range

# Compute the covariance matrix
cov_matrix = np.cov(x_normalized, rowvar=False)

# Compute the average value of the elements in the covariance matrix
average_cov_value = np.mean(cov_matrix)

print("The average value of the elements in the covariance matrix is:", average_cov_value)

# Perform PCA
pca = PCA()
pca.fit(x_normalized)

# Project the normalized measurements to the PCA space
x_pca = pca.transform(x_normalized)

# Separate the projected measurements for wines from producer 1 and producer 2
x_pca_producer1 = x_pca[producer == 1]
x_pca_producer2 = x_pca[producer == 2]

# Compute the average projected value on the first principal component for each producer
avg_pca1 = np.mean(x_pca_producer1[:, 0])
avg_pca2 = np.mean(x_pca_producer2[:, 0])

# Compute the difference between the two average values
difference = avg_pca2 - avg_pca1  # Note the swapped order

print("The difference between the average projected values on the first principal component for producer 2 and producer 1 is:", difference)

# Find the minimum and maximum projected coordinates on the first principal component
min_pca1 = np.min(x_pca[:, 0])
max_pca1 = np.max(x_pca[:, 0])

# Compute the difference between the maximum and minimum values
difference = max_pca1 - min_pca1

print("The difference between the maximum and minimum projected coordinates on the first principal component is:", difference)

# Get the alcohol level of the first wine after normalization
normalized_alcohol_level_first_wine = x_normalized[0, 0]

print("The alcohol level of the first wine after normalization is:", normalized_alcohol_level_first_wine)

# Compute the percentage of total variation explained by the first five principal components
variance_explained = np.sum(pca.explained_variance_ratio_[:5]) * 100

print("The first five principal components explain {:.2f}% of the total variation in the dataset.".format(variance_explained))
