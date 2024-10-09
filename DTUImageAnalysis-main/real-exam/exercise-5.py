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

# Initial values
x1, x2 = 4, 3
alpha = 0.07

# Gradient descent
for _ in range(5):
    dx1 = 2*x1 - x2 + 3*x1**2
    dx2 = -x1 + 6*x2
    x1 = x1 - alpha * dx1
    x2 = x2 - alpha * dx2

print(x1)

# Reset initial values
x1, x2 = 4, 3

# Gradient descent
iteration = 0
while True:
    # Calculate cost
    cost = x1**2 - x1*x2 + 3*x2**2 + x1**3
    if cost < 0.20:
        break

    # Calculate gradients
    dx1 = 2*x1 - x2 + 3*x1**2
    dx2 = -x1 + 6*x2

    # Update parameters
    x1 = x1 - alpha * dx1
    x2 = x2 - alpha * dx2

    iteration += 1

print(iteration)
