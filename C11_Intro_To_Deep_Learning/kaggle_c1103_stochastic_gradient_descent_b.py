# Setup plotting
import matplotlib.pyplot as plt
from learntools.deep_learning_intro.dltools import animate_sgd


plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

fuel = pd.read_csv('../dataset/fuel.csv')

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

X = fuel.copy()
# Remove target
y = X.pop('FE')

# make_column_transformer: This function creates a transformer that applies a list of transformers to subsets of columns in a DataFrame.
# Tuple 1 - Handling Numerical Columns:
#
# StandardScaler: This transformer will standardize the features by removing the mean and scaling to unit variance.
# It's ideal for numerical columns where the scale matters for many machine learning algorithms.
# make_column_selector(dtype_include=np.number): This function selects columns from the DataFrame that include numeric data types.
# These selected columns will then be passed to StandardScaler.
#
# Tuple 2 - Handling Categorical Columns:
#
# OneHotEncoder(sparse_output=False): The OneHotEncoder is used to convert categorical variables into a format that can be provided to ML algorithms.
# By setting sparse_output=False, the encoded categorical variables are returned as a dense array instead of a sparse matrix,
# which can be more memory-intensive but easier to work with in some contexts.
# make_column_selector(dtype_include=object): This selector picks columns of object data type,
# which are typically used to represent string-based categorical data in pandas DataFrames. These columns are then one-hot encoded.
#
# StandardScaler with example
#
# dense array with example
#
# sparse matrix with example
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

# The reason why there are 50 columns in the input_shape is because the OneHotEncoder has transfromed
# many object columns into more columns
input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

print(fuel.head())
print(pd.DataFrame(X[:10,:]).head())



from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)



import pandas as pd

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();

plt.show()

# With the learning rate and the batch size, you have some control over:
#
# How long it takes to train a model
# How noisy the learning curves are
# How small the loss becomes
#
# You probably saw that smaller batch sizes gave noisier weight updates and loss curves.
# This is because each batch is a small sample of data and smaller samples tend to give noisier estimates.
# Smaller batches can have an "averaging" effect though which can be beneficial.
#
# Smaller learning rates make the updates smaller and the training takes longer to converge.
# Large learning rates can speed up training, but don't "settle in" to a minimum as well.
# When the learning rate is too large, the training can fail completely. (Try setting the learning rate to a large value like 0.99 to see this.)



