# Data leakage (or leakage)
# happens when your training data contains information about the target,
# but similar data will not be available when the model is used for prediction.
# This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

# Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
# It is important to think about target leakage in terms of the timing or chronological order that data becomes available,
# not merely whether a feature helps make good predictions.

# To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.


