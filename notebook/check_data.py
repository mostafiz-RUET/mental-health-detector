import pandas as pd

df = pd.read_csv("data/mental_health.csv")

print(df.head())
print("\nTotal rows:", len(df))
print("\nLabel counts:")
print(df['label'].value_counts())
