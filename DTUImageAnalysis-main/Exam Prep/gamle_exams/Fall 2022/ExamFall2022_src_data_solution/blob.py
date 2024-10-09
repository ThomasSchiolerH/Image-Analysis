from skimage import io, color, filters, segmentation, measure

# Load the image
image = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/BLOBs/figures.png')

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Compute the threshold using Otsu's method
threshold = filters.threshold_otsu(gray_image)

# Compute a binary image
binary_image = gray_image <= threshold

# Remove blobs connected to the image border
cleared_image = segmentation.clear_border(binary_image)

# Label each blob in the image
label_image = measure.label(cleared_image)

# Compute the properties of each blob
properties = measure.regionprops(label_image)

# Count the number of blobs with an area larger than 13000 pixels
large_blobs = [prop for prop in properties if prop.area > 13000]

print(f"There are {len(large_blobs)} BLOBs with an area larger than 13000 pixels.")

# Initialize the largest blob area and perimeter
largest_blob_area = 0
largest_blob_perimeter = 0

# Go through each blob
for prop in properties:
    # If this blob's area is larger than the current largest, update the largest area and perimeter
    if prop.area > largest_blob_area:
        largest_blob_area = prop.area
        largest_blob_perimeter = prop.perimeter

print(f"The largest BLOB has an area of {largest_blob_area} pixels and a perimeter of {largest_blob_perimeter} pixels.")
