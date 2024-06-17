# principal component analysis (PCA).
# 主成分分析：一种常用的统计分析方法，用于降低数据维度并发现数据中的主要变化模式。它通过线性变换将原始数据转换为一组互相不相关的主成分，以便更好地理解和解释数据。
# Just like clustering is a partitioning of the dataset based on proximity, you could think of PCA as a partitioning of the variation in the data.
# PCA is a great tool to help you discover important relationships in the data and can also be used to create more informative features.

# Technical note:
# PCA is typically applied to standardized data. With standardized data "variation" means "correlation".
# With unstandardized data "variation" means "covariance". All data in this course will be standardized before applying PCA.

# How to use PCA for Feature Engineering
# The first way is to use it as a descriptive technique.
# The second way is to use the components themselves as if features:
# Because the components expose the variational structure of the data directly,
# they can often be more informative than the original features. Here are some use-cases:
# Dimensionality reduction:
#       When your features are highly redundant (multicollinear, specifically),
#       PCA will partition out the redundancy into one or more near-zero variance components,
#       which you can then drop since they will contain little or no information.
# Anomaly detection:
#       Unusual variation, not apparent from the original features, will often show up in the low-variance components.
#       These components could be highly informative in an anomaly or outlier detection task.
# Noise reduction:
#       A collection of sensor readings will often share some common background noise.
#       PCA can sometimes collect the (informative) signal into a smaller number of features while leaving the noise alone,
#       thus boosting the signal-to-noise ratio.
# Decorrelation:
#       Some ML algorithms struggle with highly-correlated features.
#       PCA transforms correlated features into uncorrelated components, which could be easier for your algorithm to work with.


# PCA Best Practices
# PCA only works with numeric features, like continuous quantities or counts.
# PCA is sensitive to scale. It's good practice to standardize your data before applying PCA, unless you know you have good reason not to.
# Consider removing or constraining outliers, since they can have an undue influence on the results.



# example to use it as a descriptive technique.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    # fig: the figure object that will contain the plots
    # axs: an array of two subplots, arranged in a 1*2 grid
    fig, axs = plt.subplots(1, 2)
    # n is the number of principal components in the PCA object
    # grid: an array representing the components, starting from 1 to n
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    # evr: the explained variance ratio for each principal component
    evr = pca.explained_variance_ratio_
    # axs[0]: the first plot, which displays a bar chart of the explained variance ratio
    # the y-axis limits are set from 0.0 to 1.0
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    # cv: cumulative sum of the explained variance ratios
    cv = np.cumsum(evr)
    # np.r_[0, grid], np.r_[0, cv] prepends a 0 to the beginning of the 'grid' and cv" arrays, so the plot starts at the origin (0,0)s
    # "o": it specifies the marker style. Here represents circular markers at each data point
    # "-": it specifies the line style. The "-" indicates a solid line connecting the data points
    # This format string is part of Matplotlib's extensive system for customizing plot appearance.
    # Other common marker styles include "s" for squares, "^" for triangles, "*" for stars,
    # and other line styles include "--" for dashed lines, "-." for dash-dot lines, and ":" for dotted lines.
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


df = pd.read_csv("../dataset/autos.csv")


# We've selected four features that cover a range of properties.
# Each of these features also has a high MI score with the target, price.
# We'll standardize the data since these features aren't naturally on the same scale.
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = df.copy()
y = X.pop('price')
X = X.loc[:, features]
# print(X)

# Standardize
# We've selected four features that cover a range of properties.
# Each of these features also has a high MI score with the target, price.
# We'll standardize the data since these features aren't naturally on the same scale.

# (X - X.mean(axis=0)): Subtracts the mean of each feature (column) from the respective values
# / X.std(axis=0): Divides by the standard deviation of each feature (column).
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)


# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# print(X_pca)

# Convert to dataframe
# Now we can fit scikit-learn's PCA estimator and create the principal components.
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

print(pca.components_)
# print(X_pca.head())




# After fitting, the PCA instance contains the loadings in its components_ attribute.
# (Terminology for PCA is inconsistent, unfortunately. We're following the convention
# that calls the transformed columns in X_pca the components, which otherwise don't have a name.)
# We'll wrap the loadings up in a dataframe.
# In PCA, pca.components_ attribute stores the principal axes in feature space, representing directions of maximum variance in the data.
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
# print(X.columns)
print(loadings)



# Recall that the signs and magnitudes of a component's loadings tell us what kind of variation it's captured.
# The first component (PC1) shows a contrast between large, powerful vehicles with poor gas milage, and smaller, more economical vehicles with good gas milage.
# We might call this the "Luxury/Economy" axis. The next figure shows that our four chosen features mostly vary along the Luxury/Economy axis.
plot_variance(pca);
plt.show()


# Let's also look at the MI scores of the components.
# Not surprisingly, PC1 is highly informative, though the remaining components, despite their small variance, still have a significant relationship with price.
# Examining those components could be worthwhile to find relationships not captured by the main Luxury/Economy axis.
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
print(mi_scores)



# The third component shows a contrast between horsepower and curb_weight -- sports cars vs. wagons, it seems.
# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
print(df.loc[idx, cols])

df["sports_or_wagon"] = X.curb_weight / X.horsepower
# plots a regression plot to explore the relationship between sports_or_wagon and price.
# The order=2 parameter specifies a second-degree polynomial fit.
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);
plt.show()