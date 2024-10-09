import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import disk, closing, opening, label
from skimage.measure import regionprops

# Load the DICOM file
dicom_file = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/HeartCT/1-001.dcm'
ds = pydicom.dcmread(dicom_file)
pixel_array = ds.pixel_array

# Load the blood ROI image
roi_image = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/HeartCT/BloodROI.png'
roi = Image.open(roi_image).convert('L')
roi_array = np.array(roi)

# Mask the pixel values with the ROI
masked_pixels = pixel_array[roi_array > 0]

# Compute mean and standard deviation
mu = np.mean(masked_pixels)
sigma = np.std(masked_pixels)

# Calculate the class range
lower_bound = mu - 3 * sigma
upper_bound = mu + 3 * sigma
print(f"Class range: [{lower_bound:.0f}, {upper_bound:.0f}]")

# Threshold the image based on class range
thresholded_image = (pixel_array > lower_bound) & (pixel_array < upper_bound)

# Apply morphological operations
selem = disk(3)
closed_image = closing(thresholded_image, selem)
selem = disk(5)
opened_image = opening(closed_image, selem)

# Perform blob analysis
label_image = label(opened_image)
props = regionprops(label_image)
filtered_labels = [prop.label for prop in props if 2000 < prop.area < 5000]

# Generate filtered segmentation based on blob area
filtered_segmentation = np.isin(label_image, filtered_labels)

# Save the result for Dice score calculation
auto_seg_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/HeartCT/automated_segmentation.png'
Image.fromarray((filtered_segmentation * 255).astype(np.uint8)).save(auto_seg_path)

# Load the manual and automated segmentations
manual_seg_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/HeartCT/BloodGT.png'
manual_seg = Image.open(manual_seg_path).convert('L')
manual_seg_array = np.array(manual_seg) > 0

auto_seg = Image.open(auto_seg_path).convert('L')
auto_seg_array = np.array(auto_seg) > 0

# Calculate the intersection and union for the Dice score
intersection = np.logical_and(manual_seg_array, auto_seg_array)
dice_score = 2 * intersection.sum() / (manual_seg_array.sum() + auto_seg_array.sum())

print(f"Dice score: {dice_score:.2f}")

# Perform blob analysis
label_image = label(opened_image)
props = regionprops(label_image)

# Count and print the number of BLOBs before filtering
initial_blob_count = len(props)
print(f"Number of BLOBs found before filtering: {initial_blob_count}")

# Apply area filtering
filtered_labels = [prop.label for prop in props if 2000 < prop.area < 5000]

# Generate filtered segmentation based on blob area
filtered_segmentation = np.isin(label_image, filtered_labels)

# Load the myocardium ROI image
myo_roi_image = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/HeartCT/MyocardiumROI.png'
myo_roi = Image.open(myo_roi_image).convert('L')
myo_roi_array = np.array(myo_roi)

# Mask the pixel values with the myocardium ROI
myo_masked_pixels = pixel_array[myo_roi_array > 0]

# Compute mean for myocardium
mu_myo = np.mean(myo_masked_pixels)

# Already computed mean for blood
mu_blood = mu

# Calculate the class limit as the midpoint between the myocardium and blood means
class_limit = (mu_myo + mu_blood) / 2
print(f"Class limit between myocardium and blood: {class_limit:.0f}")

# Update to include these steps in your existing script
