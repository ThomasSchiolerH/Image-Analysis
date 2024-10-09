import numpy as np
from skimage import transform, io, util

# Define source and destination landmarks
src = np.array([(220, 55), (105, 675), (315, 675)])
dst = np.array([(100, 165), (200, 605), (379, 525)])

# Define function to compute Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Compute initial alignment error F
F_initial = np.sum([euclidean_distance(a, b) for a, b in zip(src, dst)])
# Compute initial alignment error F
F_initial = np.sum([euclidean_distance(a, b)**2 for a, b in zip(src, dst)])
print(F_initial)

# Compute Euclidean transformation
tform = transform.EuclideanTransform()
tform.estimate(src, dst)

# Compute transformed source landmarks
src_transformed = tform(src)

# Compute alignment error F again
F_final = np.sum([euclidean_distance(a, b)**2 for a, b in zip(src_transformed, dst)])
print(F_final)

# Compute alignment error F again
F_final = np.sum([euclidean_distance(a, b) for a, b in zip(src_transformed, dst)])

# Load rocket image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/GeomTrans/rocket.png')

# Apply transformation to image
warped_image = transform.warp(image, tform.inverse)

# Convert warped image to bytes
warped_image_bytes = util.img_as_ubyte(warped_image)

# Return warped image
warped_image_bytes

# Get pixel value at (row=150, column=150)
pixel_value = warped_image_bytes[150, 150]
#print(pixel_value)

from skimage import color, filters

# Convert image to grayscale
warped_image_gray = color.rgb2gray(warped_image)

# Apply Gaussian filter
filtered_image = filters.gaussian(warped_image_gray, sigma=3)

# Convert back to byte image
filtered_image_bytes = util.img_as_ubyte(filtered_image)

# Get pixel value at (row=100, column=100)
pixel_value2 = filtered_image_bytes[100, 100]
print(f"The pixel value at (row=100, column=100) in the resulting image is: {pixel_value2}")

from skimage import transform, io, util

# Load the image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/GeomTrans/CPHSun.png')

# Rotate the image
rotated_image = transform.rotate(image, angle=16, center=(20, 20))

# Convert back to byte image
rotated_image_bytes = util.img_as_ubyte(rotated_image)

# Get pixel value at (row=200, column=200)
pixel_value = rotated_image_bytes[200, 200]
print(f"The pixel value at (row=200, column=200) in the resulting image is: {pixel_value}")
