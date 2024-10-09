import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import prewitt
from skimage.color import rgb2gray
import imageio

# Load the image using imageio
image_rgb = imageio.imread('ardeche_river.jpg')

# Convert image to grayscale
image_gray = rgb2gray(image_rgb)

# Normalize image to range [0, 1] if not already
if image_gray.max() > 1:
    image_gray = image_gray / 255.0

# Apply linear histogram stretching
min_val, max_val = 0.2, 0.8
stretched_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min())
stretched_gray = stretched_gray * (max_val - min_val) + min_val

# Computing the average value of the histogram stretched image
average_value = np.mean(stretched_gray)

# Using the Prewitt filter to extract edges
edges = prewitt(stretched_gray)

# Computing the maximum absolute value of the Prewitt filtered image
max_abs_value = np.max(np.abs(edges))

# Creating a binary image with a threshold at the average value
binary_image = (stretched_gray >= average_value).astype(int)

# Computing the number of foreground pixels in the binary image
foreground_pixels = np.sum(binary_image)

print(f"Average value of the image: {average_value}")
print(f"Maximum absolute value of the Prewitt filtered image: {max_abs_value}")
print(f"Number of foreground pixels: {foreground_pixels}")
