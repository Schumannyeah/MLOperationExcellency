# Target Encoding
# Boost any categorical feature with this powerful technique.

# It's a method of encoding categories as numbers, like one-hot or label encoding,
# with the difference that it also uses the target to create the encoding.
# This makes it what we call a supervised feature engineering technique.

# A target encoding is any kind of encoding that replaces a feature's categories with some number derived from the target.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import pandas as pd

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

autos = pd.read_csv("../dataset/autos.csv")
# print(autos.head())

autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
# print(autos[["make", "price", "make_encoded"]].head(10))





# Use Cases for Target Encoding
# Target encoding is great for:
# High-cardinality features:
#   A feature with a large number of categories can be troublesome to encode:
#   a one-hot encoding would generate too many features and alternatives, like a label encoding, might not be appropriate for that feature.
#   A target encoding derives numbers for the categories using the feature's most important property: its relationship with the target.
# Domain-motivated features:
#   From prior experience, you might suspect that a categorical feature should be important even if it scored poorly with a feature metric.
#   A target encoding can help reveal a feature's true informativeness


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
warnings.filterwarnings('ignore')


df = pd.read_csv("../dataset/movielens1m.csv")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))

X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]

from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)

plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating']);
plt.show()