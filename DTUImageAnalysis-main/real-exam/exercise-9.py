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

