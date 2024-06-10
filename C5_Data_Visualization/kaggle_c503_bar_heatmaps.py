import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
data_filepath = "../dataset/flight_delays.csv"

# Read the file into a variable fifa_data
data = pd.read_csv(data_filepath, index_col="Month")

print(data)

# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=data.index, y=data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")

plt.show()

# Generate heatmaps
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")

plt.show()



# Path of the file to read
ign_filepath = "../dataset/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

average_scores = ign_data.drop(columns='Platform').mean(axis=1)
print(average_scores)