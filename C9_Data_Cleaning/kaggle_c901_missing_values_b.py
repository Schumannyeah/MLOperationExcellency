# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
sf_permits = pd.read_csv("../dataset/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0)