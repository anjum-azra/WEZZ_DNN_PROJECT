import pandas as pd

df = pd.read_csv("data/wez_dataset.csv")

print("=" * 60)

print("DATASET QUALITY")

print("=" * 60)

print("Rows :", len(df))

print()

print(df.describe())

print()

print("Minimum Rmax :", df["Rmax"].min())
print("Maximum Rmax :", df["Rmax"].max())
print("Mean Rmax    :", df["Rmax"].mean())
print("Std Dev      :", df["Rmax"].std())