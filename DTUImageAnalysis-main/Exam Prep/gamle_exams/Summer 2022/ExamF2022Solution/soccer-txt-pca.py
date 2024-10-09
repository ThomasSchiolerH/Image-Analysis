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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the data
data_path = '/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Summer 2022/ExamF2022Solution/PCAData/soccer_data.txt'
data = np.genfromtxt(data_path, comments='%', delimiter=' ', skip_header=1, filling_values=np.nan)

# Handle NaN values by replacing them with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imputer.fit_transform(data)

# Perform PCA
pca = PCA(n_components=6)
pca_data = pca.fit_transform(data_imputed)

# Find the maximum absolute value
max_value = np.max(np.abs(pca_data))
print("The maximum absolute value of all the projected player values is:", max_value)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Create the Scree Plot
plt.figure(figsize=(6, 4))
plt.bar(range(6), explained_variance_ratio, alpha=0.5, align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

