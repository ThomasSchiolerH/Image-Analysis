import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Check current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir('.'))

# Paths to the images
fish_images = ["Fish/discus.jpg", "Fish/guppy.jpg", "Fish/kribensis.jpg", "Fish/neon.jpg", "Fish/oscar.jpg", 
               "Fish/platy.jpg", "Fish/rummy.jpg", "Fish/scalare.jpg", "Fish/tiger.jpg", "Fish/zebra.jpg"]


# Load the images and convert each to grayscale
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize((100, 100))         # Resize to 100x100
        images.append(np.array(img))
    return images

images = load_images(fish_images)

# Compute the average fish image
average_image = np.mean(images, axis=0)

# Display the average fish image
plt.imshow(average_image, cmap='gray')
plt.title('Average Fish Image')
plt.show()

# Flatten the images to prepare for PCA
flattened_images = [img.flatten() for img in images]

# Perform PCA
pca = PCA(n_components=6)
principal_components = pca.fit_transform(flattened_images)

# Display the variance explained by each principal component
print("Variance explained by each component:", pca.explained_variance_ratio_)
# Calculate the percentage of total variance explained by the first two components
variance_explained_first_two = sum(pca.explained_variance_ratio_[:2]) * 100  # Multiply by 100 to get percentage

# Print the result
print(f"Variance explained by the first two components: {variance_explained_first_two:.2f}%")
# Assuming the images have been loaded and are named appropriately in the images list
# and that 'neon.jpg' and 'guppy.jpg' are at the correct indices in fish_images
neon_index = fish_images.index("Fish/neon.jpg")
guppy_index = fish_images.index("Fish/guppy.jpg")

# Access the specific images
neon_image = images[neon_index]
guppy_image = images[guppy_index]

# Calculate the pixelwise sum of squared differences
squared_differences = np.sum((neon_image - guppy_image) ** 2)

# Print the result
print(f"Pixelwise sum of squared differences: {squared_differences}")

