# To solve this problem, we first need to calculate how many bytes one image comprises.
# Each pixel in an RGB image has 3 channels (Red, Green, Blue), with each channel being 1 byte.
# The size of each image is 2400 x 1200 pixels.

# Image size in pixels
image_width = 2400
image_height = 1200

# Bytes per pixel (3 channels, 1 byte each)
bytes_per_pixel = 3

# Total bytes per image
bytes_per_image = image_width * image_height * bytes_per_pixel

# The connection can transfer 35 megabytes per second.
# Convert megabytes to bytes for consistency (1 megabyte = 1,000,000 bytes).
transfer_rate = 35 * 1_000_000  # in bytes per second

# Now calculate how many images can be transferred per second.
images_per_second_transfer = transfer_rate / bytes_per_image

# The image analysis algorithm takes 130 milliseconds to analyze one image.
# Convert milliseconds to seconds for consistency.
analysis_time_per_image = 130 / 1000  # in seconds

# Calculate the number of images that can be analyzed per second.
images_per_second_analysis = 1 / analysis_time_per_image

# The system frame rate is determined by the lesser of these two rates
# (the bottleneck being either transfer or processing).
system_frame_rate = min(images_per_second_transfer, images_per_second_analysis)

bytes_per_image, images_per_second_transfer, images_per_second_analysis, system_frame_rate
print(f"System framerate {system_frame_rate:.1f}")