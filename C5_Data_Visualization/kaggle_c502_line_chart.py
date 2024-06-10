import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
data_filepath = "../dataset/museum_visitors.csv"

# Read the file into a variable fifa_data
data = pd.read_csv(data_filepath, index_col="Date", parse_dates=True)

# to list all the available columns
print(list(data.columns))

# to check the head or tail
print(data.head())
print(data.tail())

# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Add title
plt.title("Museum Visitors")

# Line chart showing all series
sns.lineplot(data=data)

# to show only one serie
# sns.lineplot(data=data['Firehouse Museum'], label="Firehouse Museum Vistor By Date")

# Add label for horizontal axis
plt.xlabel("Date")

plt.show()