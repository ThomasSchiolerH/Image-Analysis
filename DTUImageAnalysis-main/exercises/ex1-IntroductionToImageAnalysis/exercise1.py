from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Exercise 1

# Directory containing data and images
#in_dir = "data/"

# X-ray image
#im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
#im_org = io.imread(in_dir + im_name)

# Exercise 2

# Image dimensions
#print(im_org.shape)

# Exercise 3

# Pixel type
#print(im_org.dtype)

# Exercise 4

# Grey image
# io.imshow(im_org)
# plt.title('Metacarpal image')
# io.show()

# Exercise 5 & 6

# Image with colour
# io.imshow(im_org, cmap="hot")
# plt.title('Metacarpal image (with colormap)')
# io.show()

# Exercise 7

# Image with grey scale range
# Below values of 20 = black
# Above values 170 = white
# io.imshow(im_org, vmin=20, vmax=170)
# plt.title('Metacarpal image (with gray level scaling)')
# io.show()

# Highest value pixel is white and lowest is black
# Because the imshow function scales the image's colors 
# based on the minimum and maximum values of the data provided if vmin and vmax are not specified
# io.imshow(im_org)
# plt.title('Metacarpal image (with automatic gray level scaling)')
# io.show()

# Exercise 8

# Histogram
# plt.hist(im_org.ravel(), bins=256)
# plt.title('Image histogram')
# io.show()

# Store bin values of histogram
#h = plt.hist(im_org.ravel(), bins=256)

# Find value of given bin
#bin_no = 100
##count = h[0][bin_no]
#print(f"There are {count} pixel values in bin {bin_no}")

# Tuple with first element (bin count) and second element (bin edge)
# So the bin edges can be found
#bin_left = h[1][bin_no]
#bin_right = h[1][bin_no + 1]
#print(f"Bin edges: {bin_left} to {bin_right}")

#Alternative way of calling histogram function
#y, x, _ = plt.hist(im_org.ravel(), bins=256)
#io.show()

# Exercise 9

# Find most common intensity
#print(f"Most common intensity: {max(y)} in bin {np.argmax(y)}")

# Exercise 10

# (row, col) with (0,0) in top left corner
#r = 110
#c = 90
#im_val = im_org[r, c]
#print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# Exericse 11

# What does this operation do:
# im_org[:30] = 0
# io.imshow(im_org)
# io.show()
# All pixels in row 0 to 30 are set to black (0)

# Exercise 12

# Binary image with the same size as the original
#mask = im_org > 150
# io.imshow(mask)
# io.show()
# Values 1 is white and value 0 is black

# Exericse 13

# What does this code do:
#im_org[mask] = 255
#io.imshow(im_org)
#io.show()

# Exercise 14

# Directory containing data and images
in_dir = "data/"
# X-ray image
im_name_2 = "ardeche.jpg"
# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org_2 = io.imread(in_dir + im_name_2)
print(im_org_2.shape)
print(im_org_2.dtype)
#(600, 800, 3)
#uint8
# 600 rows, 800 cols
#io.imshow(im_org_2)
#io.show()

# Exercise 15

# RGB values at 110,90?
im_val_2 = im_org_2[110, 90]
print(f"The pixel value at (r,c) = ({110}, {90}) is: {im_val_2}")
# [119 178 238]

# Exercise 16

# Colour top half of image green
rowsToColor = im_org_2.shape[0] // 2
im_org_2[:rowsToColor] = [0,255,0]
io.imshow(im_org_2)
io.show()
