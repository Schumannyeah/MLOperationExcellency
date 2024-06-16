import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-v0_8-whitegrid")
# print(plt.style.available)

df = pd.read_csv("../dataset/autos.csv")

# # check the schema or dtypes of a dataframe
# print(df.dtypes)
#
# # check the rows count
# print("the rows count is ", len(df))
#
# # check how many columns from shape (rows, columns)
# print("the columns count is ", df.shape[1])
#
# # check all the unique values of a column
# unique_values_aspiration = df['fuel_system'].unique()
# values_count_aspiration = df['fuel_system'].value_counts()
# print(unique_values_aspiration)
# print(values_count_aspiration)

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

# print(df.head())


# The scikit-learn algorithm for MI treats discrete features differently from continuous features.
# Consequently, you need to tell it which are which.
# As a rule of thumb, anything that must have a float dtype is not discrete.
# Categoricals (object or categorial dtype) can be treated as discrete by giving them a label encoding.

X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
# X.select_dtypes("object"):
# Selects columns with the data type object, which typically includes categorical variables.
# X[colname].factorize():
# Converts categorical values into numerical labels. The factorize function assigns a unique integer to each unique category.
# The underscore (_) is used to ignore the second output of factorize, which contains the original unique values.
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()


# print(X.head())


# All discrete features should now have integer dtypes (double-check this before using MI!)
# X.dtypes produces a Series with the data types of each column in the DataFrame X.
# X.dtypes == int results in a boolean Series and then be assigned to discrete_features
discrete_features = X.dtypes == int
# discrete_features: A boolean array-like object (same length as the number of columns in X)
# indicating which features are discrete (True) and which are continuous (False).
# in our case it is all False, meaning continuous after factorize

# print(discrete_features)

# Scikit-learn has two mutual information metrics in its feature_selection module:
# one for real-valued targets (mutual_info_regression) and
# one for categorical targets (mutual_info_classif).
def make_mi_scores(X, y, discrete_features):
    # Here, mutual_info_regression from the sklearn.feature_selection module is used to compute the mutual information
    # between each feature in X and the target y. The discrete_features parameter specifies whether each feature
    # should be treated as discrete or continuous. Higher mutual information scores indicate a stronger statistical
    # dependence between the feature and the target.
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    # The raw mutual information scores, which are in a simple array form, are converted into a pandas Series.
    # Each element in the Series corresponds to a feature in X, with the index of the Series set to the column names of X.
    # The name of the Series is set to "MI Scores" for clarity.
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
# print(mi_scores[::3])  # show a few features with their MI scores

# now a bar plot to make comparisions easier:
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    # width: An array of indices (0 to n-1, where n is the number of features) created using numpy.arange,
    # which will be used to position the bars horizontally.
    # ticks: A list of the index values (feature names) extracted from scores, to be used as y-axis tick labels.

    width = np.arange(len(scores))
    ticks = list(scores.index)

    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

# This sets up the figure with a resolution of 100 DPI and a size of 8 inches in width and 5 inches in height.
plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)

# As we might expect, the high-scoring curb_weight feature exhibits a strong relationship
# with price, the target.
# sns.relplot(x="curb_weight", y="price", data=df);


# The fuel_type feature has a fairly low MI score, but as we can see from the figure,
# it clearly separates two price populations with different trends within the horsepower
# feature. This indicates that fuel_type contributes an interaction effect and might
# not be unimportant after all.
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);
plt.show()





