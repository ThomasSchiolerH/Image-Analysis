from skimage import color, io
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_erosion, binary_dilation
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import prewitt
from skimage.filters import median
from skimage import segmentation
from skimage import measure
import math
from scipy.stats import norm
import pandas as pd
import seaborn as sns
from skimage.transform import rescale, resize
from skimage import color, data, io, morphology, measure, segmentation, img_as_ubyte
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

def rotation_matrix(roll, yaw, deg=True):
    """
    Return the rotation matrix associated with the Euler angles roll (around z-axis) and yaw (around y-axis).

    Parameters
    ----------
    roll : float
        The rotation angle around the z-axis.
    yaw : float
        The rotation angle around the y-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given in radians.
    """
    if deg:
        roll = np.deg2rad(roll)
        yaw = np.deg2rad(yaw)

    # Roll (rotation around Z-axis)
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0, 0],
                    [np.sin(roll), np.cos(roll), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Yaw (rotation around Y-axis)
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw), 0],
                    [0, 1, 0, 0],
                    [-np.sin(yaw), 0, np.cos(yaw), 0],
                    [0, 0, 0, 1]])

    # Combine rotations
    R = np.dot(R_y, R_z)

    return R

def create_affine_transform(roll, yaw, translation_x):
    # Get rotation matrix for roll and yaw
    R = rotation_matrix(roll, yaw)
    
    # Create affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R[:3, :3]  # set rotation part
    affine_matrix[:3, 3] = [translation_x, 0, 0]  # set translation part

    return affine_matrix

# Calculate the affine transformation matrix
affine_transform_matrix = create_affine_transform(30, 10, 10)
affine_transform_matrix
