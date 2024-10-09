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

# Define the mean and standard deviation for each grain quality
mu1 = 52  # medium quality grain
sigma1 = 2
mu2 = 150  # high quality grain
sigma2 = 30

# Calculate the coefficients of the quadratic equation
a = 1/sigma1**2 - 1/sigma2**2
b = -2*(mu1/sigma1**2 - mu2/sigma2**2)
c = mu1**2/sigma1**2 - mu2**2/sigma2**2

# Solve the quadratic equation
x1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
x2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

print(f"The pixel values that separate the medium quality grain from the high quality grain are approximately {x1} and {x2}")