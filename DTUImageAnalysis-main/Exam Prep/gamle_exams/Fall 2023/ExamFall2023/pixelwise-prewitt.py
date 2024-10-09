import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.filters import prewitt_h

def process_image(image_path):
    # Load the image as an RGB image
    image_rgb = io.imread(image_path)

    # Convert the image to gray scale and normalize to [0, 1]
    image_gray = color.rgb2gray(image_rgb) / 255

    # Linear gray scale histogram stretch to [0.2, 0.8]
    image_stretched = 0.6 * (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray)) + 0.2

    # Computing the average value of the histogram stretched image
    average_value = np.mean(image_stretched)

    # Use the Prewitt_h filter to extract edges in the image
    edges_prewitt = prewitt_h(image_stretched)

    # Computing the maximum absolute value of the Prewitt filtered image
    max_abs_prewitt = np.max(np.abs(edges_prewitt))

    # Creating a binary image from the histogram stretched image using a threshold equal to the average value
    binary_image = image_stretched > average_value

    # Computing the number of foreground pixels in the binary image
    foreground_pixels = np.sum(binary_image)

    return average_value, max_abs_prewitt, foreground_pixels

# Example usage
image_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/Pixelwise/ardeche_river.jpg'
results = process_image(image_path)
print("Average Value:", results[0])
print("Max Absolute Value of Prewitt Filtered Image:", results[1])
print("Number of Foreground Pixels:", results[2])
