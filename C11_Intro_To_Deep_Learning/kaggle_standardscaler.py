# The StandardScaler in scikit-learn is a transformer that standardizes features by removing the mean and scaling to unit variance.
# This process is also known as standardization, and it's particularly useful for many machine learning algorithms
# that assume the input features have a Gaussian distribution or when the algorithm is sensitive to the scale of the features.
#
# "Removing the mean and scaling to unit variance" refers to the process of standardization,
# which transforms each feature in your dataset to have a mean (average) of 0 and a standard deviation (measure of spread) of 1.
# This is done to ensure that all features are on the same scale and to mitigate the effects of units and ranges across different features.
# Let's illustrate this using the example features 'Feature1' and 'Feature2' from the previous dataset:
#
# Original Features
# Feature1: [10, 20, 30, 40, 50]
# Feature2: [2, 4, 6, 8, 10]
# Step 1: Removing the Mean (Centering)
# For each feature, we calculate its mean and subtract that mean from every value in the feature.
# This shifts the data so that the feature has a mean (average) of 0.
#
# Mean of Feature1: (10 + 20 + 30 + 40 + 50) / 5 = 30
# Mean of Feature2: (2 + 4 + 6 + 8 + 10) / 5 = 6
# After centering:
#
# Feature1_centered: [-20, -10, 0, 10, 20]
# Feature2_centered: [-4, -2, 0, 2, 4]
# Step 2: Scaling to Unit Variance
# Next, we divide each centered value by the standard deviation of its respective feature.
# This scales the data so that each feature has a standard deviation of 1.
# The standard deviation measures how much the values deviate from the mean.
#
# Standard Deviation of Feature1: sqrt(((10-30)^2 + (20-30)^2 + ... + (50-30)^2) / 5) = sqrt(200) ≈ 14.142
# Standard Deviation of Feature2: sqrt(((2-6)^2 + (4-6)^2 + ... + (10-6)^2) / 5) = sqrt(40) ≈ 2.828
# After scaling:
#
# Feature1_scaled: [-20/14.14, -10/14.14, 0/14.14, 10/14.14, 20/14.14] ≈ [-1.414, -0.707, 0, 0.707, 1.414]
# Feature2_scaled: [-4/2.828, -2/2.828, 0/2.828, 2/2.828, 4/2.828] ≈ [-1.414, -0.707, 0, 0.707, 1.414]
# The final scaled features now have a mean of approximately 0 and a standard deviation of approximately 1,
# putting them on the same scale and making them comparable in magnitude for machine learning algorithms.
# This entire process is what StandardScaler automates.

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Create a sample dataset
data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(df)

# Convert back to DataFrame for better readability (optional)
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

print("Original DataFrame:")
print(df)
print("\nScaled DataFrame:")
print(df_scaled)