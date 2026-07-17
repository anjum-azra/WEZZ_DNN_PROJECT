"""
main.py

WEZ Deep Neural Network Project

Main Entry Point

Author: Anjum Azra
"""

from dataset.generator import DatasetGenerator

from dnn.train import main as train_model
from dnn.evaluate import main as evaluate_model
from dnn.predict import main as predict_model

from analysis.validate_dataset import main as validate_dataset


def print_menu():

    print("\n" + "=" * 60)
    print("        WEAPON ENGAGEMENT ZONE DNN PROJECT")
    print("=" * 60)

    print("1. Generate Dataset")
    print("2. Validate Dataset")
    print("3. Train DNN")
    print("4. Evaluate DNN")
    print("5. Predict Rmax")
    print("6. Exit")

    print("=" * 60)


def generate_dataset():

    while True:

        try:

            samples = int(
                input("\nNumber of samples to generate: ")
            )

            if samples <= 0:

                print("Enter a positive number.")

                continue

            break

        except ValueError:

            print("Please enter a valid integer.")

    generator = DatasetGenerator(samples)

    generator.generate()


def main():

    while True:

        print_menu()

        choice = input("Enter your choice: ").strip()

        if choice == "1":

            generate_dataset()

        elif choice == "2":

            validate_dataset()

        elif choice == "3":

            train_model()

        elif choice == "4":

            evaluate_model()

        elif choice == "5":

            predict_model()

        elif choice == "6":

            print("\nThank you for using WEZ DNN Project.")
            break

        else:

            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":

    main()