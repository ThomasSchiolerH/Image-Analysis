from skimage import io, color
from skimage.transform import rotate
from skimage.filters import threshold_otsu
import numpy as np

# Load the image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/GeomTrans/lights.png')

# Rotate the image 11 degrees around the center (40, 40)
image_rotated = rotate(image, angle=11, center=(40, 40))

# Convert the image to grayscale
image_gray = color.rgb2gray(image_rotated)

# Compute the threshold using Otsu's method
thresh = threshold_otsu(image_gray)

# Apply the threshold to create a binary image
binary = image_gray > thresh

# Compute the percentage of foreground pixels
percentage_foreground = np.sum(binary) / binary.size * 100

print(f"The percentage of foreground pixels is {percentage_foreground}%")
print(f"The threshold {thresh}")
