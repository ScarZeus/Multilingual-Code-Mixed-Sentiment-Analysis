import pandas as pd

def preprocess_data(location: str):
    train_csv = pd.read_csv(f"{location}/raw_train.tsv", sep="\t")
    test_csv = pd.read_csv(f"{location}/raw_test.tsv", sep="\t")
    validation_csv = pd.read_csv(f"{location}/raw_validation.tsv", sep="\t")
    sentiment_labels = pd.read_csv(f"{location}/sentiment_label.txt", sep="\t")

    print("Train Data:")
    print(train_csv.head(), "\n")
    print("Test Data:")
    print(test_csv.head(), "\n")
    print("Validation Data:")
    print(validation_csv.head(), "\n")
    print("Sentiment Labels:")
    print(sentiment_labels.head(), "\n")

    return train_csv, test_csv, validation_csv, sentiment_labels


# usage
train, test, val, labels = preprocess_data("data/raw")
