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

# Load the image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/Letters/Letters.png')

# Extract the R, G, B color channels
R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]

# Create a new binary image
binary_image = np.zeros_like(R)
binary_image[(R > 100) & (G < 100) & (B < 100)] = 1

# Erode the binary image using a disk shaped structuring element with radius=3
selem = morphology.disk(3)
eroded_image = morphology.binary_erosion(binary_image, selem)

# Count the number of foreground pixels in the eroded image
num_foreground_pixels = np.sum(eroded_image)

print(f'The number of foreground pixels in the eroded image is: {num_foreground_pixels}')

# Convert the input photo from RGB to gray scale
gray_image = color.rgb2gray(image)

# Apply a median filter to the gray scale image with a square footprint of size 8
filtered_image = filters.median(gray_image, morphology.square(8))

# Print the value at the pixel at (100, 100) in the resulting image
print(f'The value at the pixel at (100, 100) in the resulting image is: {filtered_image[100, 100]}')

# Create a new binary image
binary_image = np.zeros_like(R)
binary_image[(R > 100) & (G < 100) & (B < 100)] = 1

# Erode the binary image using a disk shaped structuring element with radius=3
selem = morphology.disk(3)
eroded_image = morphology.binary_erosion(binary_image, selem)

# Compute all the BLOBs in the image
labels = measure.label(eroded_image)

# Compute the area and perimeter of all found BLOBs
regions = measure.regionprops(labels)

# Remove all BLOBs with an area<1000 or an area>4000 or a perimeter<300
for region in regions:
    if region.area < 1000 or region.area > 4000 or region.perimeter < 300:
        labels[labels == region.label] = 0

# Display the remaining BLOBs
io.imshow(labels > 0)
plt.show()