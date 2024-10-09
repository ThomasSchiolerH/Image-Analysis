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

# Load the original grayscale image and the binary masks
zebra_image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/zebra/Zebra.png')
white_mask = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/zebra/Zebra_whiteStripes.png')
black_mask = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/zebra/Zebra_blackStripes.png')
mask = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/zebra/Zebra_MASK.png')
# Ensure the masks are binary
white_mask = white_mask > 0
black_mask = black_mask > 0

# Extract pixel values for white and black stripes using the masks
white_pixels = zebra_image[white_mask]
black_pixels = zebra_image[black_mask]

# Calculate mean and standard deviation for the white stripes
mean_white = np.mean(white_pixels)
std_white = np.std(white_pixels)

# Print the results
print(f"Mean of white stripes: {mean_white}")
print(f"Standard deviation of white stripes: {std_white}")

# Ensure the mask is binary
mask = mask > 0

# Extract pixel values from the zebra image inside the round mask
masked_pixels = zebra_image[mask]

# Classify the pixels inside the mask
# Using the threshold derived from the white and black stripe means
threshold = (mean_white + np.mean(black_pixels)) / 2
classified_white = masked_pixels > threshold

# Count the number of pixels classified as white
num_white_pixels = np.sum(classified_white)

# Print the result
print(f"Number of pixels classified as white stripe: {num_white_pixels}")

# Calculate mean and standard deviation for the black stripes
mean_black = np.mean(black_pixels)
std_black = np.std(black_pixels)

# Ensure the class range is within valid pixel values
black_range_min = max(0, mean_black - 2 * std_black)
black_range_max = min(255, mean_black + 2 * std_black)

# Print the class range for black stripes
print(f"Class range for black stripes: [{black_range_min}, {black_range_max}]")

