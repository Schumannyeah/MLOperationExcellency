# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../dataset/NFL Play by Play 2009-2016 (v3).csv")

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

# # set seed for reproducibility
# # if there are any missing values, which will be reprsented with NaN or None.
# print(np.random.seed(0))
#
# print(nfl_data.head())
#
# # get the number of missing data points per column
# missing_values_count = nfl_data.isnull().sum()
#
# # look at the # of missing points in the first ten columns
# print(missing_values_count[0:10])
#
# # how many total missing values do we have?
# total_cells = np.product(nfl_data.shape)
# total_missing = missing_values_count.sum()
#
# # percent of data that is missing
# percent_missing = (total_missing/total_cells) * 100
# print(total_cells, total_missing, percent_missing)
#
# # remove all the rows that contain a missing value
# # this might drop all the rows if each row happens to have a null
# nfl_data.dropna()
#
# # remove all columns with at least one missing value
# columns_with_na_dropped = nfl_data.dropna(axis=1)
# print(columns_with_na_dropped.head())
# print(columns_with_na_dropped.shape)
# print(nfl_data.shape)


# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
print(subset_nfl_data)

# replace all NA's with 0
# subset_nfl_data = subset_nfl_data.fillna(0)
# print(subset_nfl_data)

# replace all NA's the value that comes directly after it in the same column,
# then replace all the remaining na's with 0
subset_nfl_data =subset_nfl_data.bfill(axis=0).fillna(0)
print(subset_nfl_data)




