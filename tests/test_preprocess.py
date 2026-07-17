from dnn.preprocess import DataPreprocessor

processor = DataPreprocessor(
    "data/wez_dataset.csv"
)

X_train, X_valid, X_test, y_train, y_valid, y_test = processor.process()

print()

print("Training :", X_train.shape)

print("Validation :", X_valid.shape)

print("Testing :", X_test.shape)