from skimage import io, color
import numpy as np

# Load the images
image1 = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/ChangeDetection/change1.png')
image2 = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/ChangeDetection/change2.png')

# Convert the images to grayscale
image1_gray = color.rgb2gray(image1)
image2_gray = color.rgb2gray(image2)

# Compute the absolute difference image
diff_image = np.abs(image1_gray - image2_gray)

# Compute the number of changed pixels
changed_pixels = np.sum(diff_image > 0.3)

# Compute the percentage of changed pixels
total_pixels = image1_gray.size
percentage_changed = (changed_pixels / total_pixels) * 100

print(f"The percentage of changed pixels is {percentage_changed}%")