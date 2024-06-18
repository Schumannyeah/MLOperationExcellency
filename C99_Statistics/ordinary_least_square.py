# Example Calculation of OLS
# Let's consider a simple linear regression example with one independent variable.
#
# Data
# Suppose we have the following dataset:
#
# X (Independent Variable): [1, 2, 3, 4, 5]
# Y (Dependent Variable): [2, 3, 5, 7, 11]
# We want to fit a linear model
# 𝑌=𝛽0+𝛽1𝑋 to this data.
#
# 1. Calculate Means of X and Y:
# 𝑋ˉ=(1+2+3+4+5)/5=3
# 𝑌ˉ=(2+3+5+7+11)/5=5.6
#
# 2. Calculate the Slope(𝛽1):
#     𝛽1=∑(𝑋𝑖−𝑋ˉ)(𝑌𝑖−𝑌ˉ)/∑(𝑋𝑖−𝑋ˉ)2
#
# 3. Compute the components:
#     ∑(𝑋𝑖−𝑋ˉ)(𝑌𝑖−𝑌ˉ) = 22
#     ∑(𝑋𝑖−𝑋ˉ)2 = 10
#     𝛽1=2.2
#
# Calculate the Intercept (𝛽0):
#     β0=Yˉ−β1Xˉ
#     𝛽0=5.6−2.2×3=5.6−6.6=−1
#
# Summary
# Using OLS, we have estimated the linear relationship between 𝑋 and 𝑌 as:
#     𝑌=−1+2.2𝑋
#

import numpy as np
import statsmodels.api as sm

# Data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 3, 5, 7, 11])

# Add constant to X (for intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(Y, X).fit()

# Print the coefficients
print(f"Intercept (beta_0): {model.params[0]}")
print(f"Slope (beta_1): {model.params[1]}")

# Print the summary
print(model.summary())
