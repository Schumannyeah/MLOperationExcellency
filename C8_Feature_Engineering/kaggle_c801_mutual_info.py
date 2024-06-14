import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-v0_8-whitegrid")
# print(plt.style.available)

df = pd.read_csv("../dataset/autos.csv")
print(df.head())

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

# All discrete features should now have integer dtypes (double-check this before using MI!)
# X.dtypes produces a Series with the data types of each column in the DataFrame X.
# X.dtypes == int results in a boolean Series and then be assigned to discrete_features
discrete_features = X.dtypes == int

# Scikit-learn has two mutual information metrics in its feature_selection module:
# one for real-valued targets (mutual_info_regression) and
# one for categorical targets (mutual_info_classif).
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

# now a bar plot to make comparisions easier:
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()