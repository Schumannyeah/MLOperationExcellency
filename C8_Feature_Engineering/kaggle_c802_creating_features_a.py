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

plt.show()