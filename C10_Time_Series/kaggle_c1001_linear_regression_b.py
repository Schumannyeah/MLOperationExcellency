# Tunnel Traffic is a time series describing the number of vehicles
# traveling through the Baregg Tunnel in Switzerland each day from November 2003 to November 2005.
# In this example, we'll get some practice applying linear regression to time-step features and lag features.


from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
# %config InlineBackend.figure_format = 'retina'


# Load Tunnel Traffic dataset
data_dir = Path("../dataset")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

# Create a time series in Pandas by setting the index to a date
# column. We parsed "Day" as a date type by using `parse_dates` when
# loading the data.
tunnel = tunnel.set_index("Day")
print(tunnel.head())
print(tunnel.dtypes)

print("-"*30)

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.
tunnel = tunnel.to_period()

print(tunnel.head())
# # Check the type of the index
# print(tunnel.index)
print(tunnel.dtypes)

df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))

print("="*30)
print(df.head())



# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
# The predict method is used on the trained model with X to generate predicted values of y.
# These predictions are then wrapped in a Pandas Series object with the same index as X (and y),
# ensuring they're aligned properly for comparison.
y_pred = pd.Series(model.predict(X), index=X.index)

# This plots the actual 'NumVehicles' values (y) using the previously defined
# plotting parameters (plot_params).
ax = y.plot(**plot_params)
# On the same axes (ax), the predicted values (y_pred) are plotted with a thicker line width for distinction.
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');
# plt.show()




# Pandas provides us a simple method to lag a series, the shift method.
df['Lag_1'] = df['NumVehicles'].shift(1)
print("*"*30)
print(df.head())

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
# align() is used to ensure y and X have matching indices after removing NaNs from X.
# Rows in both X and y that don't have corresponding entries are dropped.
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
# A scatter plot (ax.plot(X['Lag_1'], y, '.', color='0.25')) is created to visualize the relationship
# between the lagged feature Lag_1 and the target NumVehicles. Each dot represents an observation.
ax.plot(X['Lag_1'], y, '.', color='0.25')
# The predicted values (y_pred) are plotted over the scatter plot of actual data points,
# likely as a line to show the regression line fitted by the model.
ax.plot(X['Lag_1'], y_pred)
# set_aspect('equal') ensures the aspect ratio of the plot is equal,
# making it easier to visually interpret relationships.
# The other options like auto,scaled, image, box, a numeric figure
# A numeric value: You can also directly specify the aspect ratio as a ratio of y-unit to x-unit.
# For example, set_aspect(2) would make the y-axis twice as tall as the x-axis for each unit of data.
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');

plt.show()