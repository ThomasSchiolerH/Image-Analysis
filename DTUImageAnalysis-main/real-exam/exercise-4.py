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

# Load the DICOM file
dicom_file = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/vertebraCT/1-353.dcm'
dicom_data = dicom.dcmread(dicom_file)
dicom_pixels = dicom_data.pixel_array

# Convert pixel values to Hounsfield units (HU)
# Assuming rescale intercept and slope are provided in the DICOM file metadata
intercept = dicom_data.RescaleIntercept
slope = dicom_data.RescaleSlope
hu_image = dicom_pixels * slope + intercept

# Apply thresholding to create a binary image
threshold = 200
binary_image = hu_image > threshold

# Perform morphological closing to remove small noise structures
selem = morphology.disk(3)
closed_image = morphology.binary_closing(binary_image, selem)

# Label connected regions (BLOBs)
label_image = measure.label(closed_image)
blobs = measure.regionprops(label_image)

# Filter BLOBs by area
min_area = 500
filtered_blobs = [blob for blob in blobs if blob.area > min_area]

# Create an empty image to store the filtered BLOBs
filtered_image = np.zeros_like(closed_image)
for blob in filtered_blobs:
    for coord in blob.coords:
        filtered_image[coord[0], coord[1]] = 1


# Plot the original DICOM image, the binary image, and the closed image
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].imshow(hu_image, cmap='gray')
# ax[0].set_title('Original DICOM Image')
# ax[0].axis('off')

# ax[1].imshow(binary_image, cmap='gray')
# ax[1].set_title('Binary Image')
# ax[1].axis('off')

# ax[2].imshow(closed_image, cmap='gray')
# ax[2].set_title('Closed Image')
# ax[2].axis('off')

# plt.show()

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(hu_image, cmap='gray')
ax[0].set_title('Original DICOM Image')
ax[0].axis('off')

ax[1].imshow(binary_image, cmap='gray')
ax[1].set_title('Binary Image')
ax[1].axis('off')

ax[2].imshow(closed_image, cmap='gray')
ax[2].set_title('Closed Image')
ax[2].axis('off')

ax[3].imshow(filtered_image, cmap='gray')
ax[3].set_title('Filtered Image')
ax[3].axis('off')

plt.show()

from sklearn.metrics import jaccard_score

# Load the ground truth mask
gt_mask = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/vertebraCT/vertebra_gt.png', as_gray=True)
gt_mask = gt_mask > 0  # Convert to binary

# Ensure the filtered_image and gt_mask have the same shape
if filtered_image.shape != gt_mask.shape:
    raise ValueError("The filtered image and ground truth mask must have the same shape.")

# Compute the DICE score
intersection = np.logical_and(filtered_image, gt_mask)
dice_score = 2. * intersection.sum() / (filtered_image.sum() + gt_mask.sum())

# Print the DICE score
print(f'DICE Score: {dice_score:.4f}')

# Sample the original pixel values in the mask found by the algorithm
sampled_hu_values = hu_image[filtered_image == 1]

# Compute the mean and standard deviation of the sampled HU values
mean_hu = np.mean(sampled_hu_values)
std_hu = np.std(sampled_hu_values)

# Print the mean and standard deviation of the HU values
print(f'Mean HU value: {mean_hu:.2f}')
print(f'Standard Deviation of HU values: {std_hu:.2f}')

# Compute the areas of all the BLOBs
blob_areas = [blob.area for blob in blobs]

# Calculate the minimum and maximum areas
min_area = np.min(blob_areas)
max_area = np.max(blob_areas)

# Print the minimum and maximum areas
print(f'Minimum BLOB area: {min_area} pixels')
print(f'Maximum BLOB area: {max_area} pixels')

# Extract the underlying pixel values from the DICOM slice using the expert mask
expert_mask_values = hu_image[gt_mask == 1]

# Plot a histogram of these values with 100 bins
plt.figure(figsize=(10, 6))
plt.hist(expert_mask_values, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Hounsfield Units in Expert Masked Area')
plt.xlabel('Hounsfield Units (HU)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

