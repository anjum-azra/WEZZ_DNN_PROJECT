"""
dataset/generator.py

High Performance WEZ Dataset Generator

Features
--------
✓ Latin Hypercube Sampling
✓ Multiprocessing
✓ tqdm Progress Bar
✓ Resume Interrupted Generation
✓ Automatic Checkpoint Saving
✓ Logging

Author: Anjum Azra
"""

import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from config.config import (
    SHOOTER_ALTITUDE,
    SHOOTER_SPEED,
    SHOOTER_PITCH,
    TARGET_ALTITUDE,
    TARGET_SPEED,
    TARGET_HEADING,
    TARGET_OFFBORESIGHT,
)

from dataset.lhs import generate_lhs_samples
from dataset.worker import process_sample
from utils.logger import logger


class DatasetGenerator:

    def __init__(self, number_of_samples=100):

        self.number_of_samples = number_of_samples

        self.output_file = "data/wez_dataset.csv"

        self.columns = [

            "ShooterAltitude",
            "ShooterVelocity",
            "ShooterPitch",

            "TargetAltitude",
            "TargetVelocity",
            "TargetHeading",
            "TargetOffBoresight",

            "Rmax",

        ]

        self.bounds = [

            SHOOTER_ALTITUDE,
            SHOOTER_SPEED,
            SHOOTER_PITCH,

            TARGET_ALTITUDE,
            TARGET_SPEED,
            TARGET_HEADING,
            TARGET_OFFBORESIGHT,

        ]

    # ----------------------------------------------------------

    def save_checkpoint(self, dataset):

        df = pd.DataFrame(
            dataset,
            columns=self.columns,
        )

        df.to_csv(
            self.output_file,
            index=False,
        )

    # ----------------------------------------------------------

    def print_statistics(self, df, rejected):

        print()
        print("=" * 60)
        print("DATASET GENERATED SUCCESSFULLY")
        print("=" * 60)

        print(f"Requested Samples : {self.number_of_samples}")
        print(f"Valid Samples     : {len(df)}")
        print(f"Rejected Samples  : {rejected}")

        if len(df) == 0:
            return

        print()

        print("Rmax Statistics")

        print(f"Minimum : {df['Rmax'].min():.2f}")
        print(f"Maximum : {df['Rmax'].max():.2f}")
        print(f"Mean    : {df['Rmax'].mean():.2f}")
        print(f"Median  : {df['Rmax'].median():.2f}")
        print(f"Std Dev : {df['Rmax'].std():.2f}")

        print()

        print("Top 10 Most Frequent Rmax Values")

        print(
            df["Rmax"]
            .round()
            .value_counts()
            .head(10)
        )

        print()

        print("First Five Rows")

        print(df.head())

        print()

        print(f"Dataset saved to : {self.output_file}")

    # ----------------------------------------------------------

    def generate(self):

        logger.info("Dataset generation started.")

        print("=" * 60)
        print("GENERATING WEZ DATASET")
        print("=" * 60)

        os.makedirs("data", exist_ok=True)

        samples = generate_lhs_samples(
            self.bounds,
            self.number_of_samples,
        )

        dataset = []

        rejected = 0

        start_index = 0

        # --------------------------------------------------
        # Resume Existing Dataset
        # --------------------------------------------------

        if os.path.exists(self.output_file):

            existing = pd.read_csv(self.output_file)

            dataset = existing.values.tolist()

            start_index = len(dataset)

            print()

            print("Existing dataset found.")

            print(f"Already generated : {start_index}")

            logger.info(
                "Resuming dataset from sample %d",
                start_index,
            )

        if start_index >= self.number_of_samples:

            print()

            print("Dataset already completed.")

            logger.info("Dataset already completed.")

            return pd.DataFrame(
                dataset,
                columns=self.columns,
            )

        remaining_samples = samples[start_index:]

        print(f"Remaining samples : {len(remaining_samples)}")

        # --------------------------------------------------
        # Multiprocessing
        # --------------------------------------------------

        with ProcessPoolExecutor() as executor:

            results = executor.map(

                process_sample,

                remaining_samples,

                chunksize=10,

            )

            for result in tqdm(

                results,

                total=len(remaining_samples),

                desc="Generating",

                unit="scenario",

                colour="green",

            ):

                if result is None:

                    rejected += 1

                    continue

                dataset.append(result)

                logger.info(

                    "Generated Sample %d",

                    len(dataset),

                )

                # Save every 100 valid samples

                if len(dataset) % 100 == 0:

                    self.save_checkpoint(dataset)

        # --------------------------------------------------
        # Final Save
        # --------------------------------------------------

        self.save_checkpoint(dataset)

        df = pd.DataFrame(
            dataset,
            columns=self.columns,
        )

        self.print_statistics(
            df,
            rejected,
        )

        logger.info(

            "Dataset generation completed. Valid=%d Rejected=%d",

            len(df),

            rejected,

        )

        return df