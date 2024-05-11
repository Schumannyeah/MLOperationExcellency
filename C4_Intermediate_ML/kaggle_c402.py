# Prerequisites:
# To run pip install scikit-learn first

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
# Reads the training data from a CSV file named "train.csv" located in the "../dataset" directory.
# It sets the "Id" column as the index of the dataframe.
X_full = pd.read_csv('../dataset/train.csv', index_col='Id')
X_test_full = pd.read_csv('../dataset/test.csv', index_col='Id')

# ====================================================================================
# Function learning

# # Retrieve all column names (index) within the DataFrame
# column_index = X_full.columns
# # Print the column names
# print("Column index:")
# print(column_index)

# # To return all the null dataframe and not null dataframe separately
# null_sales_price = X_full[X_full['SalePrice'].isnull()]
# not_null_sales_price = X_full[~X_full['SalePrice'].isnull()]
# print(not_null_sales_price)
#
# # Count of null values in the 'SalesPrice' column
# null_count = X_full['SalePrice'].isnull().sum()
# print("Number of null values in the 'SalesPrice' column:", null_count)
#
# # Count of rows with non-null 'SalesPrice'
# # .shape: This attribute of a DataFrame returns a tuple representing the dimensions of the DataFrame.
# # The first element of the tuple represents the number of rows, and the second element represents the number of columns.
# not_null_count = not_null_sales_price.shape[0]
# print(not_null_count)
#
# # Check for null values in each column
# # This is a boolean series where each element represents whether the corresponding column in the DataFrame contains any null values
# columns_with_null = X_full.isnull().any()
# # Display columns with null values
# print("Columns with null values:")
# # columns_with_null[columns_with_null]:
# # This expression filters the boolean series columns_with_null to only include the columns where the value is True.
# # This effectively gives us a subset of columns_with_null containing only the columns with null values.
# print(columns_with_null[columns_with_null].index)

# ------------------------------------------------------------------------------------

# Remove rows with missing target, separate target from predictors
# Removes rows from the training data (X_full) where the "SalePrice" (target variable) is missing.
# The inplace=True argument modifies the dataframe itself.
# .dropna(): This method is used to remove rows or columns with null (missing) values from a DataFrame.
# axis=0: This parameter specifies that rows will be considered for removal. Setting axis=0 means that the method will operate along the row axis.
# subset=['SalePrice']: This parameter specifies the subset of columns to consider for null values.
# In this case, it specifies that null values in the 'SalePrice' column should be considered for removal.
# inplace=True: This parameter specifies whether to modify the DataFrame in place or return a new DataFrame.
# Setting inplace=True modifies the DataFrame in place and returns None.
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
# X: This represents the features (predictors) DataFrame.
# y: This represents the target (response) Series or array.
# train_size=0.8: This parameter specifies the proportion of the dataset to include in the training split.
# Here, it's set to 80%, meaning 80% of the data will be used for training.
# test_size=0.2: This parameter specifies the proportion of the dataset to include in the validation split.
# Here, it's set to 20%, meaning 20% of the data will be used for validation.
# random_state=0: This parameter sets the random seed for reproducibility.
# It ensures that each time you run the code, you get the same split. Setting it to a specific value (e.g., 0) ensures reproducibility.

# X_train: Features for training.
# X_valid: Features for validation.
# y_train: Target values for training.
# y_valid: Target values for validation.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# print(X_train.head())
# print(X_train.shape)
#
# # Number of missing values in each column of training data
# missing_val_count_by_column = (X_train.isnull().sum())
# print(missing_val_count_by_column)
# print(missing_val_count_by_column[missing_val_count_by_column > 0])



# ====================================================================================
#  Q & A Section Part A

# # Fill in the line below: How many rows are in the training data?
# num_rows = X_train.shape[0]
# print(num_rows)
#
# # Fill in the line below: How many columns in the training data
# # have missing values?
# # num_cols_with_missing is return a list of columns showing with True/False if containing null
# # num_cols_with_missing = X_train.isnull().any()
# # Then X_train.isnull().any().sum() shows how many null columns
# print(X_train.isnull().any().sum())
#
# # Fill in the line below: How many missing entries are contained in
# # all of the training data?
# tot_missing = X_train.isnull().sum()
# print(tot_missing[tot_missing > 0])


#  Q & A Section Part B
# Since there are relatively few missing entries in the data (the column with the greatest percentage of missing values is missing less than 20% of its entries),
# we can expect that dropping columns is unlikely to yield good results.
# This is because we'd be throwing away a lot of valuable data, and so imputation will likely perform better.

# To compare different approaches to dealing with missing values, you'll use the same score_dataset() function from the tutorial.
# This function reports the mean absolute error (MAE) from a random forest model.

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


#  Q & A Section Part C
# Use the next code cell to impute missing values with the mean value along each column.
# Set the preprocessed DataFrames to imputed_X_train and imputed_X_valid. Make sure that the column names match those in X_train and X_valid.

# The fit_transform method fits the imputer on the training data and replaces missing values with the learned statistics.
# Here, missing values in X_train are replaced with the mean, median, or most frequent value of each column,
# depending on the strategy specified (which defaults to 'mean').
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))



#  Q & A Section Part D
# Compare the MAE from each approach. Does anything surprise you about the results? Why do you think one approach performed better than the other?
# Given that thre are so few missing values in the dataset, we'd expect imputation to perform better than dropping columns entirely.
# However, we see that dropping columns performs slightly better! While this can probably partially be attributed to noise in the dataset,
# another potential explanation is that the imputation method is not a great match to this dataset.
# That is, maybe instead of filling in the mean value, it makes more sense to set every missing value to a value of 0,
# to fill in the most frequently encountered value, or to use some other method.
# For instance, consider the GarageYrBlt column (which indicates the year that the garage was built).
# It's likely that in some cases, a missing value could indicate a house that does not have a garage.
# Does it make more sense to fill in the median value along each column in this case?
# Or could we get better results by filling in the minimum value along each column?
# It's not quite clear what's best in this case, but perhaps we can rule out some options immediately - for instance,
# setting missing values in this column to 0 is likely to yield horrible results!


# In this final step, you'll use any approach of your choosing to deal with missing values.
# Once you've preprocessed the training and validation features, you'll train and evaluate a random forest model.
# Then, you'll preprocess the test data before generating predictions that can be submitted to the competition!

# you need only ensure:
# the preprocessed DataFrames have the same number of columns,
# the preprocessed DataFrames have no missing values,
# final_X_train and y_train have the same number of rows, and
# final_X_valid and y_valid have the same number of rows.

# Imputation
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))


#  Q & A Section Part E
# Preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))

# Get test predictions
preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission402.csv', index=False)

# ------------------------------------------------------------------------------------
