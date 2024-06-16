# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
# plotting modules
import matplotlib.pyplot as plt
# read in our data
earthquakes = pd.read_csv("../dataset/database.csv")

# set seed for reproducibility
np.random.seed(0)

print(earthquakes['Date'].head())

date_lengths = earthquakes.Date.str.len()
print(date_lengths.value_counts())


indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
print(earthquakes.loc[indices])

earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

# try to get the day of the month from the date column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day

# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.displot(day_of_month_earthquakes, kde=False, bins=31)

plt.show()