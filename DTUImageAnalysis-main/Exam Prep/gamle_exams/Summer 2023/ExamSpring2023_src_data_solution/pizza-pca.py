from skimage import io
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load the images
image_list = []
for filename in glob.glob('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/PizzaPCA/training/*.png'): # assuming the images are .jpg
    im = io.imread(filename, as_gray=True)
    image_list.append(im)

# Convert the list to numpy array
image_array = np.array([im.ravel() for im in image_list])

# Compute the average pizza
average_pizza = np.mean(image_array, axis=0)

# Reshape the average pizza to the original shape
average_pizza_image = np.reshape(average_pizza, image_list[0].shape)

# Display the average pizza
plt.imshow(average_pizza_image, cmap='gray')
plt.title('Average Pizza')
plt.show()

# Perform PCA
pca = PCA(n_components=5)
pca.fit(image_array)

# The principal components
principal_components = pca.components_

# Display the principal components
# for i, component in enumerate(principal_components):
#     component_image = component.reshape(image_list[0].shape)
#     plt.imshow(component_image, cmap='gray')
#     plt.title(f'Principal Component {i+1}')
#     plt.show()
# Compute the sum of squared differences between each pizza and the average pizza
ssd = np.sum((image_array - average_pizza)**2, axis=1)

# Find the index of the pizza with the largest sum of squared differences
max_ssd_index = np.argmax(ssd)

# Display the pizza that is visually as far away from the average pizza as possible
plt.imshow(image_list[max_ssd_index], cmap='gray')
plt.title('Pizza for Experimental Eater')
plt.show()

# Compute the proportion of the total variation explained by the first principal component
explained_variance_ratio = pca.explained_variance_ratio_
first_component_explained_variance = explained_variance_ratio[0]

print(f"The first principal component explains {first_component_explained_variance * 100:.2f}% of the total variation.")

# Project the pizzas onto the first principal component
projections = np.dot(image_array - average_pizza, principal_components[0])

# Find the indices of the pizzas with the maximum and minimum projections
max_projection_index = np.argmax(projections)
min_projection_index = np.argmin(projections)

# Display the pizzas that are the furthest away on the first principal axes
plt.imshow(image_list[max_projection_index], cmap='gray')
plt.title('Signature Pizza 1')
plt.show()

plt.imshow(image_list[min_projection_index], cmap='gray')
plt.title('Signature Pizza 2')
plt.show()

# Load the photo of the wanted pizza
wanted_pizza = io.imread('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2023/ExamSpring2023_src_data_solution/data/PizzaPCA/super_pizza.png', as_gray=True)

# Flatten the wanted pizza and subtract the average pizza
wanted_pizza_flattened = wanted_pizza.ravel() - average_pizza

# Project the wanted pizza onto the PCA space
wanted_pizza_projection = np.dot(wanted_pizza_flattened, principal_components.T)

# Project all pizzas onto the PCA space
all_pizzas_projection = np.dot(image_array - average_pizza, principal_components.T)

# Compute the Euclidean distance between the projection of the wanted pizza and the projections of the pizzas on the menu
distances = np.sqrt(np.sum((all_pizzas_projection - wanted_pizza_projection)**2, axis=1))

# Find the index of the pizza with the smallest distance
closest_pizza_index = np.argmin(distances)

# Display the pizza that looks most similar to the wanted pizza
plt.imshow(image_list[closest_pizza_index], cmap='gray')
plt.title('Pizza Most Similar to Wanted Pizza')
plt.show()


