# In addition to the training data, we need two more things:
#
# An "optimizer" that can tell the network how to change its weights.
# A "loss function" that measures how good the network's predictions are.

# The loss function measures the disparity between the the target's true value and the value the model predicts.
# A common loss function for regression problems is the mean absolute error or MAE. For each prediction y_pred,
# MAE measures the disparity from the true target y_true by an absolute difference abs(y_true - y_pred).
# The total MAE loss on a dataset is the mean of all these absolute differences.

# Besides MAE, other loss functions you might see for regression problems are the mean-squared error (MSE)
# or the Huber loss (both available in Keras).

# During training, the model will use the loss function as a guide for finding the correct values of its weights (lower loss is better).
# In other words, the loss function tells the network its objective.

# The Optimizer - Stochastic Gradient Descent
# We've described the problem we want the network to solve, but now we need to say how to solve it.
# This is the job of the optimizer. The optimizer is an algorithm that adjusts the weights to minimize the loss.

# Virtually all of the optimization algorithms used in deep learning belong to a family called stochastic gradient descent.
# They are iterative algorithms that train a network in steps. One step of training goes like this:
#
# Sample some training data and run it through the network to make predictions.
# Measure the loss between the predictions and the true values.
# Finally, adjust the weights in a direction that makes the loss smaller.
# Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)

# Each iteration's sample of training data is called a minibatch (or often just "batch"),
# while a complete round of the training data is called an epoch. The number of epochs you train
# for is how many times the network will see each training example.
#
# The animation shows the linear model from Lesson 1 being trained with SGD. (Stochastic Gradient Descent)
# The pale red dots depict the entire training set, while the solid red dots are the minibatches.
# Every time SGD sees a new minibatch, it will shift the weights (w the slope and b the y-intercept)
# toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit.
# You can see that the loss gets smaller as the weights get closer to their true values.

# Learning Rate and Batch Size
# Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way).
# The size of these shifts is determined by the learning rate.
# A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.
#
# The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds.
# Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)
#
# Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results.
# Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning
# (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.

# What's In a Name?
# The gradient is a vector that tells us in what direction the weights need to go.
# More precisely, it tells us how to change the weights to make the loss change fastest.
# We call our process gradient descent because it uses the gradient to descend the loss curve towards a minimum.
# Stochastic means "determined by chance." Our training is stochastic because the minibatches are random samples from the dataset.
# And that's why it's called SGD!

import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('../dataset/red-wine.csv')

# Create training and validation splits
# random_state=0: The random_state parameter is a seed value for the random number generator.
# By setting it to a fixed value (like 0), you ensure that the sampling is reproducible;
# running the code multiple times will give you the same sampled rows. This is particularly useful for debugging, reproducing experiments,
# or comparing results consistently across runs.
df_train = red_wine.sample(frac=0.7, random_state=0)
# By passing df_train.index to the drop method,
# you're instructing pandas to remove all rows from red_wine DataFrame that are present in the training set,
# effectively leaving you with the remaining rows that constitute the validation set.
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
# Here, df_train.max() is used to find the maximum value in the DataFrame.
# The axis=0 argument specifies that the operation should be performed column-wise.
# This means that for each column, the function finds the maximum value across all rows.
# The result, stored in the variable max_, is a Series where each element corresponds to the maximum value found in the respective column of df_train.
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

print(X_train.shape)

# Eleven columns means eleven inputs.
# We've chosen a three-layer network with over 1500 neurons. This network should be capable of learning fairly complex relationships in the data.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# Deciding the architecture of your model should be part of a process.
# Start simple and use the validation loss as your guide. You'll learn more about model development in the exercises.
#
# After defining the model, we compile in the optimizer and loss function.

model.compile(
    optimizer='adam',
    loss='mae',
)

# Now we're ready to start the training! We've told Keras to feed the optimizer 256 rows of the training data at a time
# (the batch_size) and to do that 10 times all the way through the dataset (the epochs).

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# You can see that Keras will keep you updated on the loss as the model trains.
#
# Often, a better way to view the loss though is to plot it.
# The fit method in fact keeps a record of the loss produced during training in a History object.
# We'll convert the data to a Pandas dataframe, which makes the plotting easy.

import pandas as pd
import matplotlib.pyplot as plt

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();

plt.show()

