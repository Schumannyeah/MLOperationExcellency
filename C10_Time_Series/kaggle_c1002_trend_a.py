# Trend
# Model long-term changes with moving averages and the time dummy.

# Moving Average Plots
# To see what kind of trend a time series might have, we can use a moving average plot.
# To compute a moving average of a time series, we compute the average of the values within a sliding window of some defined width.
# Each point on the graph represents the average of all the values in the series that fall within the window on either side.
# The idea is to smooth out any short-term fluctuations in the series so that only long-term changes remain.

# Engineering Trend
# Once we've identified the shape of the trend, we can attempt to model it using a time-step feature.
# We've already seen how using the time dummy itself will model a linear trend:
#
# target = a * time + b
# We can fit many other kinds of trend through transformations of the time dummy.
# If the trend appears to be quadratic (a parabola), we just need to add the square of the time dummy to the feature set, giving us:
#
# target = a * time ** 2 + b * time + c
# Linear regression will learn the coefficients a, b, and c.




from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
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


# Load Tunnel Traffic dataset
data_dir = Path("../dataset")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()

# To create a moving average, first use the rolling method to begin a windowed computation.
# Follow this by the mean method to compute the average over the window.
# As we can see, the trend of Tunnel Traffic appears to be about linear.

moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
);

plt.show()


# In kaggle_c1001_linear_regression_b.py, we engineered our time dummy in Pandas directly.
# From now on, however, we'll use a function from the statsmodels library called DeterministicProcess.
# Using this function will help us avoid some tricky failure cases that can arise with time series and linear regression.
# The order argument refers to polynomial order: 1 for linear, 2 for quadratic, 3 for cubic, and so on.

# (A deterministic process, by the way, is a technical term for a time series that is non-random or completely determined,
# like the const and trend series are. Features derived from the time index will generally be deterministic.)
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

print(X.head())



from sklearn.linear_model import LinearRegression

y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

# The trend discovered by our LinearRegression model is almost identical to the moving average plot,
# which suggests that a linear trend was the right decision in this case.

ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

plt.show()


# To make a forecast, we apply our model to "out of sample" features.
# "Out of sample" refers to times outside of the observation period of the training data.
# Here's how we could make a 30-day forecast:

X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X), index=X.index)

print(y_fore.head())

ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()

plt.show()

