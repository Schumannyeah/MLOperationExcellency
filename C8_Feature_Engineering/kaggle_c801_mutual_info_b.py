import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
# plt.rc: This is a function to set the rc parameters (short for runtime configuration).
# Rc parameters are essentially default settings for various properties in matplotlib.
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


# Load data
df = pd.read_csv("../dataset/ames.csv")

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

# print(df.head())
# check all the unique values of a column
unique_values = df['BldgType'].unique()
values_count = df['BldgType'].value_counts()
print(unique_values)
print(values_count)


# print(df.shape)
# print(df.dtypes)
# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

# features = ["YearBuilt", "MoSold", "ScreenPorch"]
# sns.relplot(
#     x="value", y="SalePrice", col="variable", data=df.melt(id_vars="SalePrice", value_vars=features), facet_kws=dict(sharex=False),
# );
# plt.show()




# X = df.copy()
# y = X.pop('SalePrice')
#
# mi_scores = make_mi_scores(X, y)
#
# print(mi_scores.head(20))
# # print(mi_scores.tail(20))  # uncomment to see bottom 20
#
# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores.head(20))
# # plot_mi_scores(mi_scores.tail(20))  # uncomment to see bottom 20
#
# plt.show()





# sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen");
# plt.show()



# GrLivArea  # Above ground living area
# MoSold     # Month sold


# The trend lines being significantly different from one category to the next indicates an interaction effect.
# # YOUR CODE HERE:
# feature = "GrLivArea"
#
# sns.lmplot(
#     x=feature, y="SalePrice", hue="BldgType", col="BldgType",
#     data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
# );
# plt.show()




# YOUR CODE HERE:
feature = "MoSold"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
plt.show()