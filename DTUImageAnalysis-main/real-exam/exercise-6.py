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

import numpy as np
import matplotlib.pyplot as plt

# Given parameters
mu1 = np.array([24, 3])
mu2 = np.array([45, 7])
cov_matrix = np.array([[2, 0], [0, 2]])
prior1 = 0.5
prior2 = 0.5

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.inv(cov_matrix)

# Calculate the coefficients for the linear discriminant function
w = np.dot(cov_inv, (mu2 - mu1))
b = -0.5 * np.dot(np.dot(mu1.T, cov_inv), mu1) + 0.5 * np.dot(np.dot(mu2.T, cov_inv), mu2) + np.log(prior1 / prior2)

# Function to compute the LDA decision boundary
def lda_decision_boundary(x1, w, b):
    return -(w[0] * x1 + b) / w[1]

# Number of samples
num_samples = 100

# Generate random samples for class 1
class1_samples = np.random.multivariate_normal(mu1, cov_matrix, num_samples)

# Generate random samples for class 2
class2_samples = np.random.multivariate_normal(mu2, cov_matrix, num_samples)

# Define x1 values for the decision boundary
x1_vals = np.linspace(15, 50, 200)

# Compute x2 values for the decision boundary without plotting it
x2_vals = lda_decision_boundary(x1_vals, w, b)

# Plot the sample measurements
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], c='blue', alpha=0.5, label='Class 1 Samples')
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], c='red', alpha=0.5, label='Class 2 Samples')
plt.xlabel('Camera 1 Measurements (x1)')
plt.ylabel('Camera 2 Measurements (x2)')
plt.title('Sample Measurements from Camera')
plt.legend()
plt.grid(True)

# Adjust the plot limits to fit the values better
plt.xlim(15, 50)
plt.ylim(-7, 20)

plt.show()

# Given sample
sample = np.array([30, 10])

# Compute the linear discriminant function value
y_x = np.dot(w, sample) + b

# Determine the class
if y_x > 0:
    sample_class = "Class 2"
else:
    sample_class = "Class 1"

print(f"Computed y(x) value: {y_x}")
print(f"The plastic piece belongs to: {sample_class}")


