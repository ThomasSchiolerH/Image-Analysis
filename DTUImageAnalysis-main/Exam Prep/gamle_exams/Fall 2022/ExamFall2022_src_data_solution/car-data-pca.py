import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd

# Define column names
column_names = ["wheel-base", "length", "width", "height", "curb-weight", "engine-size", "horsepower", "highway-mpg"]

# Load the data
df = pd.read_csv('/Users/thomasschioler/Documents/DTU/Semester 6/Image Analysis/Exam Prep/gamle_exams/Fall 2022/ExamFall2022_src_data_solution/data/CarPCA/car_data.txt', sep=" ", skiprows=1, names=column_names)

# Normalize the data
scaler = StandardScaler()
normalized_df = scaler.fit_transform(df)

# Perform PCA
pca = PCA()
principalComponents = pca.fit_transform(normalized_df)

# Convert the principal components into a DataFrame
principalDf = pd.DataFrame(data = principalComponents)

# Print the explained variance ratio
print(pca.explained_variance_ratio_)
print(normalized_df[0, 0])
print(sum(pca.explained_variance_ratio_[:2]))
print(abs(principalComponents[0, 0]))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming pca and normalized_df are defined and pca is fitted on normalized_df

# Transform the entire dataset
principalComponents = pca.transform(normalized_df)

# Project the original data onto the PCA space
projected_data = np.dot(normalized_df, pca.components_.T)

# Create a DataFrame with the first three projected measurements
projected_df = pd.DataFrame(data = projected_data[:, :3], columns = ['PC1', 'PC2', 'PC3'])

# Create a pairplot of the first three projected measurements
sns.pairplot(projected_df)
plt.show()