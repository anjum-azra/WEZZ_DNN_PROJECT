"""
analysis/validate_dataset.py

Validate WEZ Dataset

Features
--------
✓ Dataset Summary
✓ Missing Value Check
✓ Duplicate Check
✓ Correlation Matrix
✓ Histograms
✓ Scatter Plots
✓ Boxplots

Author: Anjum Azra
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


DATASET = "data/wez_dataset.csv"


def save_histogram(df, column):

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column],
        bins=30,
    )

    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/{column}_histogram.png",
        dpi=300,
    )

    plt.close()


def save_boxplot(df, column):

    plt.figure(figsize=(6, 5))

    plt.boxplot(df[column])

    plt.title(f"{column} Boxplot")

    plt.tight_layout()

    plt.savefig(
        f"results/{column}_boxplot.png",
        dpi=300,
    )

    plt.close()


def save_scatter(df, x):

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df[x],
        df["Rmax"],
        alpha=0.6,
    )

    plt.xlabel(x)
    plt.ylabel("Rmax")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/{x}_vs_Rmax.png",
        dpi=300,
    )

    plt.close()


def main():

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET)

    print()

    print("Shape")

    print(df.shape)

    print()

    print("=" * 60)
    print("FIRST FIVE ROWS")
    print("=" * 60)

    print(df.head())

    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(df.describe())

    print()

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isnull().sum())

    print()

    print("=" * 60)
    print("DUPLICATES")
    print("=" * 60)

    print(df.duplicated().sum())

    print()

    print("=" * 60)
    print("CORRELATION")
    print("=" * 60)

    correlation = df.corr(numeric_only=True)

    print(correlation)

    correlation.to_csv(
        "results/correlation_matrix.csv"
    )

    columns = [

        "ShooterAltitude",
        "ShooterVelocity",
        "ShooterPitch",
        "TargetAltitude",
        "TargetVelocity",
        "TargetHeading",
        "TargetOffBoresight",
        "Rmax",

    ]

    print()

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    for column in columns:

        save_histogram(df, column)

        save_boxplot(df, column)

    for column in columns[:-1]:

        save_scatter(df, column)

    print()

    print("Files Generated")

    print("---------------------------")

    print("✓ correlation_matrix.csv")

    for column in columns:

        print(f"✓ {column}_histogram.png")
        print(f"✓ {column}_boxplot.png")

    for column in columns[:-1]:

        print(f"✓ {column}_vs_Rmax.png")

    print()

    print("=" * 60)
    print("DATASET VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()