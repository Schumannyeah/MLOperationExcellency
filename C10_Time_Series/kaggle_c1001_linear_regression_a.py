# Linear Regression with Time Series
# For the first part of this course, we'll use the linear regression algorithm to construct forecasting models.
# Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks.
#
# The linear regression algorithm learns how to make a weighted sum from its input features.
# target = weight_1 * feature_1 + weight_2 * feature_2 + bias

# During training, the regression algorithm learns values for the parameters weight_1, weight_2, and bias that best fit the target.
# (This algorithm is often called ordinary least squares since it chooses values that minimize the squared error between the target and the predictions.)
# The weights are also called regression coefficients and the bias is also called the intercept
# because it tells you where the graph of this function crosses the y-axis.

# details refer to C99_Statistics/ordinary_least_square.py
#
# Time-step features
# There are two kinds of features unique to time series: time-step features and lag features.
#
# Time-step features are features we can derive directly from the time index.
# The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(
    "../dataset/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

print(df.head())






plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
# %config InlineBackend.figure_format = 'retina'

df['Time'] = np.arange(len(df.index))
# df.head()
# print(df['Time'])

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');



df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

print(df.head())

# Linear regression with a lag feature produces the model:
# target = weight * lag + bias
# So lag features let us fit curves to lag plots where each observation in a series is plotted against the previous observation.

fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');

# You can see from the lag plot that sales on one day (Hardcover) are correlated with sales from the previous day (Lag_1).
# When you see a relationship like this, you know a lag feature will be useful.
# More generally, lag features let you model serial dependence.
# A time series has serial dependence when an observation can be predicted from previous observations.

plt.show()