import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.measure import label, regionprops

def process_change_detection(image_path1, image_path2):
    try:
        # Read the images
        image1_rgb = io.imread(image_path1)
        image2_rgb = io.imread(image_path2)
        
        # Convert images to HSV color space
        image1_hsv = color.rgb2hsv(image1_rgb)
        image2_hsv = color.rgb2hsv(image2_rgb)
        
        # Extract the S channel and scale it
        s1 = image1_hsv[:, :, 1] * 255
        s2 = image2_hsv[:, :, 1] * 255
        
        # Compute the absolute difference image between the two S images
        difference_image = np.abs(s1 - s2)
        
        # Compute the average value and the standard deviation of the difference image
        average_value = np.mean(difference_image)
        std_deviation = np.std(difference_image)
        
        # Compute a threshold as the average value plus two times the standard deviation
        threshold = average_value + 2 * std_deviation
        
        # Print the threshold used for binary conversion
        print("Threshold for binary image conversion:", threshold)
        
        # Compute a binary change image
        change_image = difference_image > threshold
        
        # Compute the number of changed pixels
        num_changed_pixels = np.sum(change_image)
        
        # Perform a BLOB analysis on the binary change image
        labeled_image = label(change_image)
        regions = regionprops(labeled_image)
        
        # Find the BLOB with the largest area
        largest_blob_area = 0
        for region in regions:
            if region.area > largest_blob_area:
                largest_blob_area = region.area
        
        return num_changed_pixels, largest_blob_area
    except Exception as e:
        print("Failed to process the images:", e)
        return None

# Example usage
image_path1 = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/ChangeDetection/frame_1.jpg'
image_path2 = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/ChangeDetection/frame_2.jpg'
results = process_change_detection(image_path1, image_path2)
if results:
    print("Number of Changed Pixels:", results[0])
    print("Largest BLOB Area:", results[1])
else:
    print("Change detection processing failed.")
