import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# Function to display the orthogonal views of a 3D volume
def imshow_orthogonal_view(sitkImage, origin=None, title=None):
    data = sitk.GetArrayFromImage(sitkImage)
    if origin is None:
        origin = np.array(data.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    data = img_as_ubyte(data / np.max(data)) if np.max(data) > 0 else data
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

# Load MRI images
imgT1_v1 = sitk.ReadImage('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/ImgT1_v1.nii.gz')
imgT1_v2 = sitk.ReadImage('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/online-test-exam/ImgT1_v2.nii.gz')

# Affine transformation matrix
transform = sitk.AffineTransform(3)
transform.SetMatrix([0.98, -0.16, 0.17, 0, 0.97, 0, -0.17, 0.04, 0.98])

# Set minimal translation and center of rotation
transform.SetTranslation([0, 0, 0])
size = imgT1_v2.GetSize()
center = (size[0] / 2, size[1] / 2, size[2] / 2)
transform.SetCenter(center)

# Apply the transformation
transformed_imgT1_v2 = sitk.Resample(imgT1_v2, imgT1_v2, transform)

# Visualize the transformed image
imshow_orthogonal_view(transformed_imgT1_v2)
