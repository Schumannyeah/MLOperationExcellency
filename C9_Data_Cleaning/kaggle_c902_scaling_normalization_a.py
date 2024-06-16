# in scaling, you're changing the range of your data, while
# in normalization, you're changing the shape of the distribution of your data.


# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)



# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
# It creates a figure with a single row and two columns using subplots, specifying the overall figure size as 15 inches wide and 3 inches tall.
# For each subplot:
# It plots a histogram (sns.histplot) of the respective data series (original_data in the first subplot, scaled_data in the second).
# Enables kernel density estimation (kde=True) to overlay a smooth line representing the distribution's density.
# Disables the legend to keep the plot clean.
# Sets a title for each subplot to differentiate between the original and scaled data.

# 1, 2: These two numbers represent the layout of the subplots in terms of rows and columns.
# In this case, 1 indicates that there is 1 row, and 2 indicates there are 2 columns.
# So, the figure will have a total of 2 subplots laid out horizontally.
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()




# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()