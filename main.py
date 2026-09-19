import pandas as pd
df = pd.read_csv("data/Gold Futures Historical Data (6).csv")
print(df.columns.tolist())
print(df.head())