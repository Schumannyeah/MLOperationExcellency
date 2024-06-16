import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

accidents = pd.read_csv("../dataset/accidents.csv")
autos = pd.read_csv("../dataset/autos.csv")
concrete = pd.read_csv("../dataset/concrete.csv")
customer = pd.read_csv("../dataset/customer.csv")

# Mathematical Transforms
# Relationships among numerical features are often expressed through mathematical formulas,
# which you'll frequently come across as part of your domain research

autos["stroke_ratio"] = autos.stroke / autos.bore
# print(autos[["stroke", "bore", "stroke_ratio"]].head())


autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1]);

# plt.show()



# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)

# Set the maximum width of each column to prevent line-wrapping
pd.set_option('display.width', None)

# In Traffic Accidents are several features indicating whether some roadway object was near the accident.
# This will create a count of the total number of roadway features nearby using the sum method
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

# print(accidents[roadway_features + ["RoadwayFeatures"]].head(10))




components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

# print(concrete[components + ["Components"]].head(10))






# Building-Up and Breaking-Down Features
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

# print(customer[["Policy", "Type", "Level"]].head(10))



autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
# print(autos[["make", "body_style", "make_and_style"]].head())




# Group Transforms
# The mean function is a built-in dataframe method, which means we can pass it as a string to transform.
# Other handy methods include max, min, median, var, std, and count. Here's how you could calculate the frequency
# with which each state occurs in the dataset:
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

# print(customer[["State", "Income", "AverageIncome"]].head(10))


customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

# print(customer[["State", "StateFreq"]].head(10))




# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

print(df_valid[["Coverage", "AverageClaim"]].head(10))



# Tips on Creating Features
# It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:
# Linear models learn sums and differences naturally, but can't learn anything more complex.
# Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
# Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
# Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
# Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.
