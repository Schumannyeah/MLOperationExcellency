import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')



import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

spotify = pd.read_csv('../dataset/spotify.csv')

X = spotify.copy().dropna()
# y = X.pop('track_popularity') separates the target variable,
# which is assumed to be the track's popularity, from the feature set X.
# The pop method removes the specified column from the DataFrame X and returns it, which is assigned to y.
# This modifies X to no longer include the 'track_popularity' column.
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.

# The function group_split you've shared is designed to perform a grouped data split,
# which is particularly useful when you want to ensure that all observations related to a certain group (in this case, an artist)
# stay within either the training set or the testing set.
# This prevents data leakage, which can occur if information about an artist leaks from the training data into the testing data,
# making evaluation less reliable.
def group_split(X, y, group, train_size=0.75):
    # It generates indices to split data into training and test sets while respecting the grouping defined by the groups parameter.
    splitter = GroupShuffleSplit(train_size=train_size)
    # next(splitter.split(X, y, groups=group)): This line actually performs the split, returning indices
    # for the training and testing sets.It uses the X, y, and group parameters to ensure that data points (songs)
    # belonging to the same group (artist) are kept together in either the training or testing subset.
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))




model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));





# Start the plot at epoch 10
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));




# Now let's add some capacity to our network. We'll add three hidden layers with 128 units each.
# Run the next cell to train the network and see the learning curves.

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

# Now the validation loss begins to rise very early (yellow), while the training loss continues to decrease (green).
# This indicates that the network has begun to overfit. At this point, we would need to try something to prevent it,
# either by reducing the number of units or through a method like early stopping. (We'll see another in the next lesson!)





from tensorflow.keras import callbacks

# YOUR CODE HERE: define an early stopping callback
# If you like, try experimenting with patience and min_delta to see what difference it might make.
early_stopping = callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));


plt.show()