# Prerequisites:
# To run pip install scikit-learn first
# Categorical Variables
# 1. Ordinal Encoding:
# This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).
# This assumption makes sense in this example, because there is an indisputable ranking to the categories.
# Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables.
# For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables.
# 2. One-Hot Encoding
# In contrast to ordinal encoding, one-hot encoding does not assume an ordering of the categories.
# Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data
# (e.g., "Red" is neither more nor less than "Yellow"). We refer to categorical variables without an intrinsic ranking as nominal variables.
# One-hot encoding generally does not perform well if the categorical variable takes on a large number of values
# (i.e., you generally won't use it for variables taking more than 15 different values).

# In general, one-hot encoding (Approach 3) will typically perform best,
# and dropping the categorical columns (Approach 1) typically performs worst, but it varies on a case-by-case basis.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

# Read the data
X = pd.read_csv('../dataset/train.csv', index_col='Id')
X_test = pd.read_csv('../dataset/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
# ====================================================================================
# Function learning
#  Q & A Section Part A

# s = (X_train.dtypes == 'object')
# object_cols = list(s[s].index)
#
# print("Categorical variables:")
# print(object_cols)

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())


# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
# set(X_valid[col]): This converts the values of the column col in the validation data (X_valid) into a set.
# This set will contain unique values present in the column of the validation data.
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



#  Q & A Section Part B
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# Get number of unique entries in each column with categorical data
# map(lambda col: X_train[col].nunique(), object_cols): this applies the nunique() method to each column "col"
# in the list 'object_cols' within the training dataset 'X_train'
# The 'nunique()' method returns the number of unique entries in each column.
# 'list': This converts the result of the 'map' function, which is an iterable, into a list

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
# the zip function pairs each column name with its corresponding number of unique entries then
# is converted into a dictionary by "dict"
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
# sorted: this sorts the dictionary by its values (number of unique entries)
# items() method returns a view object that displays a list of a dictionary's key-value tuple pairs.
# sorting is done based on the 2nd element x[1] of each tuple, which is the number of unique entries

sorted(d.items(), key=lambda x: x[1])

# ------------------------------------------------------------------------------------

