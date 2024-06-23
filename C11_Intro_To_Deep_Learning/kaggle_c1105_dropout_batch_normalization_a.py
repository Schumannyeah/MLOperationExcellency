# In this lesson, we'll learn about a two kinds of special layers, not containing any neurons themselves,
# but that add some functionality that can sometimes benefit a model in various ways. Both are commonly used in modern architectures.

# Dropout
# The first of these is the "dropout layer", which can help correct overfitting.
#
# In the last lesson we talked about how overfitting is caused by the network learning spurious patterns in the training data.
# To recognize these spurious patterns a network will often rely on very a specific combinations of weight, a kind of "conspiracy" of weights.
# Being so specific, they tend to be fragile: remove one and the conspiracy falls apart.
#
# This is the idea behind dropout. To break up these conspiracies, we randomly drop out some fraction of a layer's input units every step of training,
# making it much harder for the network to learn those spurious patterns in the training data.
# Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust.

# Batch Normalization
# The next special layer we'll look at performs "batch normalization" (or "batchnorm"), which can help correct training that is slow or unstable.

# Now, if it's good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better!
# In fact, we have a special kind of layer that can do this, the batch normalization layer.
# A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation,
# and then also putting the data on a new scale with two trainable rescaling parameters.
# Batchnorm, in effect, performs a kind of coordinated rescaling of its inputs.


# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


import pandas as pd
red_wine = pd.read_csv('../dataset/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']


# When adding dropout, you may need to increase the number of units in your Dense layers.
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

# There's nothing to change this time in how we set up the training.
model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    verbose=0,
)


# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();

plt.show()










