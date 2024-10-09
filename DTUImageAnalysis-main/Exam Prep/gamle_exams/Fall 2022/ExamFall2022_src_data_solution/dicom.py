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

# Step 1: Load the DICOM file
dicom_image = dicom.dcmread("/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/dicom/1-162.dcm")
image_data = dicom_image.pixel_array

# Step 2: Load ROI masks and extract pixel values for liver, kidney, and aorta
liver_roi = io.imread("/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/dicom/LiverROI.png").astype(bool)
kidney_roi = io.imread("/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/dicom/KidneyROI.png").astype(bool)
aorta_roi = io.imread("/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/dicom/AortaROI.png").astype(bool)

# Extract pixel values where masks are true
liver_pixels = image_data[liver_roi]
kidney_pixels = image_data[kidney_roi]
aorta_pixels = image_data[aorta_roi]

# Step 3: Compute the mean values and determine thresholds
liver_mean = np.mean(liver_pixels) if liver_pixels.size > 0 else 0
kidney_mean = np.mean(kidney_pixels) if kidney_pixels.size > 0 else 0
aorta_mean = np.mean(aorta_pixels) if aorta_pixels.size > 0 else 0

t1 = (liver_mean + kidney_mean) / 2
t2 = (kidney_mean + aorta_mean) / 2

# Step 4: Segment the image based on thresholds
segmented_image = np.logical_and(image_data > t1, image_data < t2).astype(int)

# Step 5: Compute DICE score
intersection = np.logical_and(segmented_image, kidney_roi)
dice_score = 2 * intersection.sum() / (segmented_image.sum() + kidney_roi.sum()) if kidney_roi.sum() > 0 else 0

print("Threshold t1:", t1)
print("Threshold t2:", t2)
print("DICE Score:", dice_score)