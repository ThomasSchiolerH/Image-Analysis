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

# Load the RGB image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/PixelWiseOps/pixelwise.png')

# Convert to grayscale properly using skimage's color module
gray_image = color.rgb2gray(image)

# Convert the grayscale image to floating point format
gray_image = img_as_float(gray_image)

# Perform linear transformation
min_val, max_val = 0.1, 0.6
scaled_image = (max_val - min_val) * (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image)) + min_val

# Compute Otsu's threshold (make sure the image is in a single-channel format)
threshold = filters.threshold_otsu(scaled_image)

# Apply the threshold
binary_image = scaled_image > threshold

# Plotting the results
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(gray_image, cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('off')

ax[2].imshow(scaled_image, cmap='gray')
ax[2].set_title('Scaled Image')
ax[2].axis('off')

ax[3].imshow(binary_image, cmap='gray')
ax[3].set_title('Binary Image')
ax[3].axis('off')

#plt.show()

print(threshold)

# Load the rocket image
rocket_image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/Filtering/rocket.png')

# Convert to grayscale
gray_rocket_image = color.rgb2gray(rocket_image)

# Apply Prewitt filter
filtered_image = filters.prewitt(gray_rocket_image)

# Apply threshold
threshold_value = 0.06
binary_image = filtered_image > threshold_value

# Count the number of white pixels
num_white_pixels = np.sum(binary_image)

print(f"The number of white pixels in the resulting image is {num_white_pixels}")
