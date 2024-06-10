# In general, the larger the validation set, the less randomness (aka "noise") there is in our measure of model quality,
# and the more reliable it will be. Unfortunately, we can only get a large validation set by removing rows from our training data,
# and smaller training datasets mean worse models!

# sklearn has been deprecated and using scikit-learn instead
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Read the data
train_data = pd.read_csv('../dataset/train.csv', index_col='Id')
test_data = pd.read_csv('../dataset/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

# print(X.head())

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

# print("Average MAE score:", scores.mean())


# Begin by writing a function get_score() that reports the average (over three cross-validation folds) MAE of a machine learning pipeline that uses:
# the data in X and y to create folds,
# SimpleImputer() (with all parameters left as default) to replace missing values, and
# RandomForestRegressor() (with random_state=0) to fit a random forest model.
# The n_estimators parameter supplied to get_score() is used when setting the number of trees in the random forest model.
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

print(results)

import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))
plt.show()