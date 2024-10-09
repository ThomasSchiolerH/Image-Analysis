import pydicom
import numpy as np
from skimage import io
from skimage.morphology import disk, dilation, erosion, binary_opening, binary_dilation, binary_erosion
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import measurements
import numpy as np
import pydicom
from skimage import io, measure, color
from skimage.morphology import disk, dilation, erosion
from scipy.spatial import distance

# Step 1: Read the DICOM file and the expert annotations
dicom_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/Abdominal/1-166.dcm'
liver_roi_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/Abdominal/LiverROI.png'
kidney_l_roi_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/Abdominal/KidneyRoi_l.png'
kidney_r_roi_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/Abdominal/KidneyRoi_r.png'

dicom_image = pydicom.dcmread(dicom_path).pixel_array
liver_mask = io.imread(liver_roi_path) > 0
kidney_l_mask = io.imread(kidney_l_roi_path) > 0
kidney_r_mask = io.imread(kidney_r_roi_path) > 0

# Step 2: Extract the pixel values using the masks
liver_pixels = dicom_image[liver_mask]
kidney_l_pixels = dicom_image[kidney_l_mask]
kidney_r_pixels = dicom_image[kidney_r_mask]

# Step 3: Compute the average Hounsfield units for the kidneys
avg_hu_kidney_l = np.mean(kidney_l_pixels)
avg_hu_kidney_r = np.mean(kidney_r_pixels)

print("Average HU for Left Kidney:", avg_hu_kidney_l)
print("Average HU for Right Kidney:", avg_hu_kidney_r)

# Step 4: Compute the average and the standard deviation of the Hounsfield units in the liver
avg_hu_liver = np.mean(liver_pixels)
std_hu_liver = np.std(liver_pixels)

print("Average HU for Liver:", avg_hu_liver)
print("Standard Deviation HU for Liver:", std_hu_liver)

# Step 5: Compute the threshold t_1 (average - std deviation)
t_1 = avg_hu_liver - std_hu_liver

# Step 6: Compute the threshold t_2 (average + std deviation)
t_2 = avg_hu_liver + std_hu_liver

print("Threshold t_1 for Liver Segmentation:", t_1)
print("Threshold t_2 for Liver Segmentation:", t_2)



def dice_score(im1, im2):
    return 1 - distance.dice(im1.ravel(), im2.ravel())

def image_analysis_pipeline():
    # Paths and file reading
    in_dir = "./data/abdominal"
    ct = pydicom.dcmread(dicom_path)
    img = ct.pixel_array

    liver_roi = io.imread(liver_roi_path)
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]

    # Fit and calculate statistics for liver
    liver_mean = np.mean(liver_values)
    liver_std = np.std(liver_values)
    min_hu = liver_mean - liver_std
    max_hu = liver_mean + liver_std

    # Binary image creation
    bin_img = (img > min_hu) & (img < max_hu)

    # Morphological operations
    footprint = disk(3)
    dilated = dilation(bin_img, footprint)
    footprint = disk(10)
    eroded = erosion(dilated, footprint)
    dilated_final = dilation(eroded, footprint)

    # Labeling and filtering based on region properties
    label_img = measure.label(dilated_final)
    props = measure.regionprops(label_img)
    filtered_img = np.zeros_like(label_img, dtype=bool)
    for prop in props:
        if 1500 <= prop.area <= 7000 and prop.perimeter > 300:
            filtered_img[label_img == prop.label] = True

    # DICE score calculation
    dice_score_result = dice_score(filtered_img, liver_mask)
    print(f"Final DICE Score: {dice_score_result:.3f}")

# Run the analysis pipeline
image_analysis_pipeline()