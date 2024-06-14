# Unfortunately, BigQuery doesn't directly download datasets for local use.
# BigQuery is a serverless data warehouse designed for analyzing large datasets in the cloud.
# Therefore, consider only using pandas and other sql api to run the test instead of using BigQuery

from google.cloud import bigquery

# Create a client object (assuming you have authentication set up)
client = bigquery.Client()

# Define your query to fetch specific data
query = """
SELECT * FROM `bigquery-public-data.chicago_crime.crime`
LIMIT 100;  -- Limit to 100 rows for example
"""

# Run the query
query_job = client.query(query)

# Get the results (rows) as a list of dictionaries
results = list(query_job.result())

# You can now process this data or potentially use pandas to write it to a CSV file

# Example using pandas (optional)
import pandas as pd

df = pd.DataFrame(results)  # Create DataFrame from results
df.to_csv("crime_data.csv", index=False)  # Save to CSV file (optional)
