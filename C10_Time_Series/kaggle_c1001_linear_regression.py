import pandas as pd

df = pd.read_csv(
    "../dataset/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

print(df.head())