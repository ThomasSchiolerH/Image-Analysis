from skimage import io
from skimage.transform import SimilarityTransform, warp, resize
import numpy as np
from skimage import img_as_ubyte

# Load the images
shoe_1 = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/LMRegistration/shoe_1.png')
shoe_2 = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/LMRegistration/shoe_2.png')

# Manually placed landmarks
src = np.array([[40, 320], [425, 120], [740, 330]])
dst = np.array([[80, 320], [380, 155], [670, 300]])

# Compute the similarity transform
tform = SimilarityTransform()
tform.estimate(src, dst)

# Warp shoe_1 to shoe_2 using the estimated transformation
shoe_1_warped = warp(shoe_1, tform.inverse)

# Extract the scale of the transform
scale = tform.scale
print(f'Scale of the transform: {scale}')

# Compute the alignment error before and after the registration
error_before = np.sum((src - dst)**2)
error_after = np.sum((tform(src) - dst)**2)

print(f'Alignment error before registration: {error_before}')
print(f'Alignment error after registration: {error_after}')

# Resize shoe_2 to match the dimensions of shoe_1_warped
shoe_2_resized = resize(shoe_2, shoe_1_warped.shape)

# Compare the blue component of the color values of the aligned images
blue_component_diff = np.abs(shoe_1_warped[..., 2] - shoe_2_resized[..., 2])

# Convert both images to bytes
shoe_1_warped_bytes = img_as_ubyte(shoe_1_warped)
shoe_2_resized_bytes = img_as_ubyte(shoe_2_resized)

# Extract the blue component at position (200, 200)
blue_component_shoe_1 = shoe_1_warped_bytes[200, 200, 2]
blue_component_shoe_2 = shoe_2_resized_bytes[200, 200, 2]
# Compute the absolute difference
blue_component_diff = np.abs(blue_component_shoe_1 - blue_component_shoe_2)

print(f'Absolute difference in blue component at position (200, 200): {blue_component_diff}')
print(f'Blue component difference: {np.mean(blue_component_diff)}')