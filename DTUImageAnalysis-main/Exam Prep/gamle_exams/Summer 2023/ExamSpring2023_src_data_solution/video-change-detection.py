from skimage import color, io
import numpy as np

# Load the images
background = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/ChangeDetection/background.png')
new_frame = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/ChangeDetection/new_frame.png')

# Convert the images to grayscale
background_gray = color.rgb2gray(background)
new_frame_gray = color.rgb2gray(new_frame)

# Update the background image
alpha = 0.90
new_background = alpha * background_gray + (1 - alpha) * new_frame_gray

# Compute the absolute difference image
diff_image = np.abs(new_frame_gray - new_background)

# Compute the number of changed pixels
changed_pixels = np.sum(diff_image > 0.1)

print(f"The number of changed pixels is {changed_pixels}")

# Calculate the average value in the region [150:200, 150:200]
average_value = np.mean(new_background[150:200, 150:200])

print(f"The average value of the estimated new background image in the pixel region [150:200, 150:200] is {average_value}")