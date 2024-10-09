# Student number: s214968
from skimage import color, io
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_erosion, binary_dilation
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import prewitt
from skimage.filters import median
from skimage.util import img_as_float
from skimage import segmentation
from skimage import measure
import math
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd
import seaborn as sns
from skimage.transform import rescale, resize
from skimage import color, data, io, morphology, measure, segmentation, img_as_ubyte, filters
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.color import label2rgb
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.morphology import ball, binary_closing, binary_erosion, closing


# Load the 3D MRI image
image_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
# sitk_image = sitk.ReadImage(image_path)

# # Convert to numpy array
# image_data = sitk.GetArrayFromImage(sitk_image)

# # Define the rigid transformation (yaw and pitch in radians)
# yaw = 10  # degrees
# pitch = -30  # degrees
# yaw_rad = np.deg2rad(yaw)
# pitch_rad = np.deg2rad(pitch)

# # Create the transformation object
# transform = sitk.Euler3DTransform()
# transform.SetRotation(yaw_rad, pitch_rad, 0)

# # Apply the rigid transformation to the image
# resampler = sitk.ResampleImageFilter()
# resampler.SetReferenceImage(sitk_image)
# resampler.SetInterpolator(sitk.sitkLinear)
# resampler.SetTransform(transform)
# moving_image = resampler.Execute(sitk_image)

# # Convert back to numpy array
# moving_image_data = sitk.GetArrayFromImage(moving_image)

# # Generate the mask using Otsu thresholding
# threshold = threshold_otsu(image_data)
# binary_mask = image_data > threshold

# # Apply morphological closing with ball structuring element (radius 5)
# closed_mask = binary_closing(binary_mask, ball(5))

# # Apply erosion with ball structuring element (radius 3)
# final_mask = binary_erosion(closed_mask, ball(3))

# # Apply the mask to both images
# masked_template_image = image_data * final_mask
# masked_moving_image = moving_image_data * final_mask

# # Compute the mean intensities of the masked regions
# mean_template_masked = np.mean(masked_template_image[final_mask])
# mean_moving_masked = np.mean(masked_moving_image[final_mask])

# # Compute the NCC for the binary mask
# numerator_masked = np.sum((masked_template_image[final_mask] - mean_template_masked) * (masked_moving_image[final_mask] - mean_moving_masked))
# denominator_masked = np.sqrt(np.sum((masked_template_image[final_mask] - mean_template_masked)**2) * np.sum((masked_moving_image[final_mask] - mean_moving_masked)**2))

# ncc_masked = numerator_masked / denominator_masked

# print("Normalized Correlation Coefficient (NCC) with binary mask:", ncc_masked)









# # Load the 3D MRI image
# file_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/real-exam/02502_exam_spring_2024_data/brain/T1_brain_template.nii.gz'
# image = sitk.ReadImage(file_path)
# template_array = sitk.GetArrayFromImage(image)

# # Step 2: Apply rigid registration with the specified rotation angles
# def apply_rigid_transform(image, yaw, pitch):
#     # Rotate around the z-axis (yaw) -> in the x-y plane
#     image = rotate(image, angle=yaw, axes=(1, 2), reshape=False)
#     # Rotate around the x-axis (pitch) -> in the y-z plane
#     image = rotate(image, angle=pitch, axes=(0, 2), reshape=False)
#     return image

# moving_array = apply_rigid_transform(template_array, 10, -30)

# # Step 3: Generate the mask using Otsu thresholding and morphological operations
# threshold_value = threshold_otsu(template_array)
# binary_mask = template_array > threshold_value

# # Apply morphological closing with a ball structuring element with radius 5
# structuring_element = ball(5)
# binary_mask = binary_closing(binary_mask, structuring_element)

# # Apply erosion with a ball with radius 3
# structuring_element = ball(3)
# binary_mask = binary_erosion(binary_mask, structuring_element)

# # Step 4: Apply the mask to both the moving and template images
# template_masked = template_array * binary_mask
# moving_masked = moving_array * binary_mask

# # Step 5: Compute the intensity-based normalized correlation coefficient
# def compute_ncc(template, moving, mask):
#     template_mean = np.mean(template[mask])
#     moving_mean = np.mean(moving[mask])

#     numerator = np.sum((template[mask] - template_mean) * (moving[mask] - moving_mean))
#     denominator = np.sqrt(np.sum((template[mask] - template_mean) ** 2) * np.sum((moving[mask] - moving_mean) ** 2))
    
#     return numerator / denominator

# ncc_value = compute_ncc(template_masked, moving_masked, binary_mask)
# print(f'Normalized Correlation Coefficient (NCC): {ncc_value}')








fixedImage = sitk.ReadImage(image_path)

# Convert the image to a numpy array
fixedImageArray = sitk.GetArrayFromImage(fixedImage)

# Define the yaw and pitch in radians
yaw = np.deg2rad(10)  # degrees
pitch = np.deg2rad(-30)  # degrees

# Function to create a rotation matrix for affine transform
def rotation_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cr, -cy*sr, sy],
        [sp*sy*cr + cp*sr, cp*cr - sp*sy*sr, -sp*cy],
        [-cp*sy*cr + sp*sr, sp*cy + cp*sy*sr, cp*cy]
    ])

# Create the Affine transform and set the rotation
transform = sitk.AffineTransform(3)
rot_matrix = rotation_matrix(yaw, pitch, 0)[:3, :3]
transform.SetMatrix(rot_matrix.T.flatten())

# Compute the center of the image
image_center = np.array(fixedImage.GetSize()) / 2.0
image_center = fixedImage.TransformContinuousIndexToPhysicalPoint(image_center)

# Set the center of rotation
transform.SetCenter(image_center)

# Apply the transformation to the image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixedImage)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkLinear)
movingImage = resampler.Execute(fixedImage)
movingImageArray = sitk.GetArrayFromImage(movingImage)

# Function to display orthogonal views of a 3D volume
def imshow_orthogonal_view(image, origin=None, title=None):
    data = sitk.GetArrayViewFromImage(image)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = data / np.max(data)
    axes[0].imshow(data[origin[0]], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[:, origin[1]], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[:, :, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.show()

# Display orthogonal views of the transformed moving image
imshow_orthogonal_view(movingImage, title="Transformed Moving Image")

# Apply Otsu thresholding
threshold = threshold_otsu(fixedImageArray)
binary_mask = fixedImageArray > threshold

# Apply morphological closing with a ball structuring element
closed_mask = binary_closing(binary_mask, ball(5))

# Apply erosion with a ball structuring element
eroded_mask = binary_erosion(closed_mask, ball(3))

# Apply the mask to both the moving and template images
masked_template = fixedImageArray * eroded_mask
masked_moving = movingImageArray * eroded_mask

# Debugging: Verify the masked images and intermediate values
mean_template_masked = np.mean(masked_template[eroded_mask])
mean_moving_masked = np.mean(masked_moving[eroded_mask])

print("Mean intensity of template image (masked):", mean_template_masked)
print("Mean intensity of moving image (masked):", mean_moving_masked)

numerator_masked = np.sum((masked_template[eroded_mask] - mean_template_masked) * (masked_moving[eroded_mask] - mean_moving_masked))
denominator_masked = np.sqrt(np.sum((masked_template[eroded_mask] - mean_template_masked)**2) * np.sum((masked_moving[eroded_mask] - mean_moving_masked)**2))

print("Numerator (masked):", numerator_masked)
print("Denominator (masked):", denominator_masked)

ncc = numerator_masked / denominator_masked
print(f"Normalized Correlation Coefficient (NCC): {ncc}")

# Visualize the final masks and their application
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(eroded_mask[fixedImageArray.shape[0] // 2], cmap='gray')
axes[0].set_title('Final Mask (Axial)')

axes[1].imshow(masked_template[fixedImageArray.shape[0] // 2], cmap='gray')
axes[1].set_title('Masked Template (Axial)')

axes[2].imshow(masked_moving[fixedImageArray.shape[0] // 2], cmap='gray')
axes[2].set_title('Masked Moving (Axial)')

plt.tight_layout()
plt.show()






def imshow_orthogonal_view(sitkImage, origin=None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize data for display
    axes[0].imshow(data[origin[0], :, :], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[:, origin[1], :], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[:, :, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.show()

def display_mask(mask):
    """
    Display the orthogonal views of a 3D binary mask from the middle of the volume.

    Parameters
    ----------
    mask : numpy array
        Binary mask to display.
    """
    origin = np.array(mask.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask[origin[0], :, :], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(mask[:, origin[1], :], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(mask[:, :, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    fig.suptitle("Binary Mask", fontsize=16)
    plt.show()

# Load the 3D MRI image

image = sitk.ReadImage(image_path)
image_array = sitk.GetArrayFromImage(image)

# Define the rigid transformation (yaw = 10 degrees, pitch = -30 degrees)
transform = sitk.Euler3DTransform()
transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0))
transform.SetRotation(np.deg2rad(10), np.deg2rad(-30), 0)  # Yaw, pitch

# Resample the image using the rigid transformation
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(image)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(transform)
moving_image = resampler.Execute(image)
moving_image_array = sitk.GetArrayFromImage(moving_image)

# Generate a mask using Otsu thresholding
otsu_threshold = threshold_otsu(image_array)
mask = image_array > otsu_threshold

# Apply morphological closing with a ball radius 5, then erosion with a ball radius 3
mask = closing(mask, ball(5))
mask = erosion(mask, ball(3))

# Apply the mask to both the moving and template images
template_image_masked = image_array * mask
moving_image_masked = moving_image_array * mask

# Calculate the NCC
def normalized_cross_correlation(template, moving):
    template_mean = np.mean(template)
    moving_mean = np.mean(moving)

    numerator = np.sum((template - template_mean) * (moving - moving_mean))
    denominator = np.sqrt(np.sum((template - template_mean) ** 2) * np.sum((moving - moving_mean) ** 2))

    return numerator / denominator

ncc_value = normalized_cross_correlation(template_image_masked, moving_image_masked)
print(f"Normalized Correlation Coefficient (NCC): {ncc_value}")

# Display orthogonal views
imshow_orthogonal_view(image, title="Original Image")
imshow_orthogonal_view(moving_image, title="Transformed Image")
display_mask(mask)
