import pydicom as dicom
from skimage import io, measure, segmentation, morphology
import numpy as np
import math

# Load the DICOM image
dicom_image = dicom.dcmread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/Aorta/1-442.dcm').pixel_array

# Load the mask image
mask = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/Aorta/AortaROI.png')

# Use the mask to extract the grey-values of the blood in the aorta
blood_grey_values = dicom_image[mask > 0]

# Compute the average and standard deviation of the grey-values
average = np.mean(blood_grey_values)
std_dev = np.std(blood_grey_values)

print(f'Average grey-value of the blood: {average}')
print(f'Standard deviation of the grey-values: {std_dev}')

# Define the threshold value
T = 90

# Apply threshold to the DICOM image
binary_image = dicom_image > T

# Check the output after thresholding
print("Unique values in binary image:", np.unique(binary_image))

# Remove blobs connected to the image border
cleared_image = segmentation.clear_border(binary_image)

# Check the output after clearing border-connected blobs
print("Unique values in cleared image:", np.unique(cleared_image))

# Label the image with 8-connectivity
labels = measure.label(cleared_image, connectivity=2)

# Perform region properties to calculate area and perimeter of each blob
properties = measure.regionprops(labels)

# Calculate circularity and filter based on conditions
aorta_candidates = []
for prop in properties:
    area = prop.area
    perimeter = prop.perimeter
    if perimeter == 0:
        continue  # Avoid division by zero in circularity calculation
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    print(f"Blob area: {area}, perimeter: {perimeter}, circularity: {circularity}")
    if circularity > 0.95 and area > 200:
        aorta_candidates.append(prop)

# Assuming there is one main aorta candidate, calculate its physical area
if aorta_candidates:
    pixel_area = aorta_candidates[0].area
    physical_area = pixel_area * (0.75 ** 2)  # Convert pixel area to physical area (mm^2)
else:
    physical_area = 0

print("The area of the aorta in the scan is:", physical_area, "mm^2")
