from skimage import io
import numpy as np
from sklearn.decomposition import PCA

in_dir = "/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2023/ExamFall2023/data/Fish/"
all_images = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
              "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]

# Load and flatten images
images = [io.imread(in_dir + f).flatten() for f in all_images]

# Compute average image
average_image = np.mean(images, axis=0)

# Center data by subtracting average image
centered_images = images - average_image

# Compute PCA
pca = PCA(n_components=6)
pca.fit(centered_images)

# Compute explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print the total variance explained by the first two components
print(f"Total variance explained by the first two components: {sum(explained_variance_ratio[:2]) * 100:.2f}%")

# Load the images
neon_image = io.imread(in_dir + "neon.jpg").flatten()
guppy_image = io.imread(in_dir + "guppy.jpg").flatten()

# Compute the pixelwise sum of squared differences
sum_of_squared_differences = np.sum((neon_image - guppy_image) ** 2)

print(f"Pixelwise sum of squared differences: {sum_of_squared_differences}")

# Project all images onto the principal components
projections = pca.transform(centered_images)

# Find the projection of the neon fish
neon_index = all_images.index("neon.jpg")
neon_projection = projections[neon_index]

# Compute the Euclidean distance between the neon fish and all other fish
distances = np.sqrt(np.sum((projections - neon_projection) ** 2, axis=1))

# Find the fish with the maximum distance
max_distance_index = np.argmax(distances)
max_distance_fish = all_images[max_distance_index]

print(f"The fish that is furthest away from the neon fish in PCA space is: {max_distance_fish}")

