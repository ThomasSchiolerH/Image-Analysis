# Constants
image_width = 1600
image_height = 800
bits_per_channel = 8
channels = 3 # RGB
images_per_second = 6.25
time_to_process_one_frame = 230 / 1000 # convert ms to s

# Calculate overall system framerate
system_framerate = 1 / time_to_process_one_frame
print(f"The overall system framerate is {system_framerate} frames per second.")

# Calculate data transfer rate
image_size = image_width * image_height * bits_per_channel * channels / 8 # in bytes
data_transfer_rate = images_per_second * image_size # in bytes per second
print(f"The USB-2 connection can transfer {data_transfer_rate} bytes per second from the camera to the computer.")
