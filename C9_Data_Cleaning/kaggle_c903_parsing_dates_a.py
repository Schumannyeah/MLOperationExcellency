# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# plotting modules
import matplotlib.pyplot as plt


# read in our data
landslides = pd.read_csv("../dataset/catalog.csv")

# set seed for reproducibility
np.random.seed(0)

print(landslides['date'].head())

# check the data type of our date column
landslides['date'].dtype

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# print the first few rows
print(landslides['date_parsed'].head())


# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())


# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
# Histogram Bins: A histogram represents the distribution of numerical data by dividing the entire range of values into intervals (or bins),
# and counting how many values fall into each interval. These intervals are the "bins."
#
# Purpose: By specifying bins=31 in sns.distplot(day_of_month_landslides, kde=False, bins=31),
# you're asking Seaborn to create a histogram with 31 equal-width bins to categorize the data points.
# This is particularly useful when you expect your data to have a natural grouping or when you want a fine-grained view of the distribution.
sns.distplot(day_of_month_landslides, kde=False, bins=31)

plt.show()