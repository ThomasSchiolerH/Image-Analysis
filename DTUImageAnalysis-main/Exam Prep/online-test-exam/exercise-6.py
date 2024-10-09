import cv2
import numpy as np
from skimage.measure import label, regionprops

# Load the images
frame_1 = cv2.imread('ChangeDetection/frame_1.jpg')
frame_2 = cv2.imread('ChangeDetection/frame_2.jpg')

# Convert the images to HSV color space
hsv_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)
hsv_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV)

# Extract the S channel and scale it
s_1 = hsv_1[:,:,1] * 255
s_2 = hsv_2[:,:,1] * 255

# Compute the absolute difference image
diff = cv2.absdiff(s_1, s_2)

# Compute the average value and the standard deviation
avg = np.mean(diff)
std_dev = np.std(diff)

# Compute the threshold
threshold = avg + 2 * std_dev

# Compute the binary change image
binary_change = np.where(diff > threshold, 1, 0)

# Compute the number of changed pixels
num_changed_pixels = np.sum(binary_change)

# Perform a BLOB analysis on the binary change image
label_img = label(binary_change)
regions = regionprops(label_img)

# Find the BLOB with the largest area
max_area = max(region.area for region in regions)

print(f'Threshold: {threshold}')
print(f'Number of changed pixels: {num_changed_pixels}')
print(f'Max are BLOB: {max_area}')