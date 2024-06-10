import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
insurance_filepath = "../dataset/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)

print(insurance_data.head())

# Bar chart showing average score for racing games by platform
plt.figure(figsize=(16,8))

# sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
# sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
plt.show()


# Path of the file to read
candy_filepath = "../dataset/candy.csv"

# Read the file into a variable insurance_data
candy_data = pd.read_csv(candy_filepath, index_col="id")