# Pipelines are a simple way to keep your data preprocessing and modeling code organized.
# Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
#
# Many data scientists hack together models without pipelines, but pipelines have some important benefits. Those include:
#
# Cleaner Code: Accounting for data at each step of preprocessing can get messy. With a pipeline,
# you won't need to manually keep track of your training and validation data at each step.
# Fewer Bugs: There are fewer opportunities to misapply a step or forget a preprocessing step.
# Easier to Productionize: It can be surprisingly hard to transition a model from a prototype to something deployable at scale.
# We won't go into the many related concerns here, but pipelines can help.
# More Options for Model Validation: You will see an example in the next tutorial, which covers cross-validation.

# sklearn has been deprecated and using scikit-learn instead
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
X_full = pd.read_csv('../dataset/train.csv', index_col='Id')
X_test_full = pd.read_csv('../dataset/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# print(X_train.head())

# The next code cell uses code from the tutorial to preprocess the data and train a model
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


# Preprocessing of test data, fit model
preds_test = clf.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('../dataset/submission.csv', index=False)