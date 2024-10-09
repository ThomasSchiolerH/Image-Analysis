import pydicom
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from matplotlib import pyplot as plt

# Read DICOM file
dicom_file = pydicom.dcmread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/HeartCT/1-001.dcm')
pixels = dicom_file.pixel_array

# Read ROI images
myocardium_roi = plt.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/HeartCT/MyocardiumROI.png')
blood_roi = plt.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/HeartCT/BloodROI.png')

# Extract pixel values of the ROIs
myocardium_pixels = pixels[myocardium_roi == 1]
blood_pixels = pixels[blood_roi == 1]

# Compute Mu and Sigma
mu = np.mean(blood_pixels)
sigma = np.std(blood_pixels)
# Print class range
print('Class range:', (mu - 3*sigma, mu + 3*sigma))


# Create binary image
binary_image = ((pixels > mu - 3*sigma) & (pixels < mu + 3*sigma)).astype(np.uint8)

# Perform morphological operations
closed_image = ndimage.binary_closing(binary_image, structure=morphology.disk(3))
opened_image = ndimage.binary_opening(closed_image, structure=morphology.disk(5))

# Perform BLOB analysis
labels = measure.label(opened_image)
# Print number of BLOBs
num_blobs = len(np.unique(labels)) - 1  # Subtract 1 to exclude the background label
print('Number of BLOBs before filtering:', num_blobs)

props = measure.regionprops(labels)
filtered_blobs = np.zeros_like(labels)
for prop in props:
    if 2000 < prop.area < 5000:
        filtered_blobs[labels == prop.label] = 1

# Read ground truth image
ground_truth = plt.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/HeartCT/BloodGT.png')

# Compute DICE score
intersection = np.sum(filtered_blobs[ground_truth == 1])
dice_score = 2. * intersection / (np.sum(filtered_blobs) + np.sum(ground_truth))

print('DICE score:', dice_score)
# Compute means of the two classes
mu_myocardium = np.mean(myocardium_pixels)
mu_blood = np.mean(blood_pixels)

# Compute class limit
class_limit = (mu_myocardium + mu_blood) / 2

print('Class limit:', class_limit)