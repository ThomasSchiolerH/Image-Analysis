from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage import io, color, morphology
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import disk


# Load the image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/CarData/road.png')

# Convert from RGB to HSV
hsv_image = color.rgb2hsv(image)

# Create a binary image: pixels with V > 0.9 are set to foreground
binary_image = hsv_image[:, :, 2] > 0.9

# Label connected components with 8-connectivity
labeled_image = label(binary_image, connectivity=2)

# Calculate the area of each component
regions = regionprops(labeled_image)

# Find the minimum area that results in only two BLOBs remaining
areas = [region.area for region in regions]
sorted_areas = sorted(areas, reverse=True)
min_area = sorted_areas[1]  # Assuming we need to keep the largest two areas

# Remove small objects smaller than the second largest
final_image = remove_small_objects(labeled_image, min_size=min_area)

# Display the result
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(binary_image, cmap='gray')
ax[1].set_title('Binary Image')
ax[2].imshow(final_image, cmap='gray')
ax[2].set_title('Filtered Image')
plt.show()

# Return the minimum area
min_area
print(min_area)

# Load the car image (you need to provide the path to car.png)
car_image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/CarData/car.png')

# Convert from RGB to HSV
hsv_car_image = color.rgb2hsv(car_image)

# Create a binary image: pixels with S > 0.7 are set to foreground
binary_car_image = hsv_car_image[:, :, 1] > 0.7

# Apply morphological erosion with a disk-shaped structuring element, radius=6
eroded_image = morphology.binary_erosion(binary_car_image, footprint=disk(6))

# Apply morphological dilation with a disk-shaped structuring element, radius=4
dilated_image = morphology.binary_dilation(eroded_image, footprint=disk(4))

# Calculate the number of foreground pixels in the final image
foreground_pixels = np.sum(dilated_image)

# Display the result
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(car_image)
ax[0].set_title('Original Car Image')
ax[1].imshow(binary_car_image, cmap='gray')
ax[1].set_title('Binary Image')
ax[2].imshow(eroded_image, cmap='gray')
ax[2].set_title('Eroded Image')
ax[3].imshow(dilated_image, cmap='gray')
ax[3].set_title('Dilated Image')
plt.show()

# Output the number of foreground pixels
print(f'Number of foreground pixels in the final image: {foreground_pixels}')