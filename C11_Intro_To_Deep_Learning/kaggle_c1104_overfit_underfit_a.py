# Underfitting :
# the training set is when the loss is not as low as it could be because the model hasn't learned enough signal.
# Overfitting :
# the training set is when the loss is not as low as it could be because the model learned too much noise.
# The trick to training deep learning models is finding the best balance between the two.

# Capacity
# A model's capacity refers to the size and complexity of the patterns it is able to learn.
# For neural networks, this will largely be determined by how many neurons it has and how they are connected together.
# If it appears that your network is underfitting the data, you should try increasing its capacity.
#
# You can increase the capacity of a network either by making it wider (more units to existing layers) or
# by making it deeper (adding more layers).
# Wider networks have an easier time learning more linear relationships,
# while deeper networks prefer more nonlinear ones. Which is better just depends on the dataset.

# Early Stopping
# We mentioned that when a model is too eagerly learning noise, the validation loss may start to increase during training.
# To prevent this, we can simply stop the training whenever it seems the validation loss isn't decreasing anymore.
# Interrupting the training this way is called early stopping.
# Just set your training epochs to some large number (more than you'll need), and early stopping will take care of the rest.
# In Keras, we include early stopping in our training through a callback.


import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('../dataset/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']



from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

plt.show()
















