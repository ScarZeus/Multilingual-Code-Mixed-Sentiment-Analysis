import pandas as pd 

def preprocess_data(df: pd.DataFrame,location:str):
    train_csv = pd.read_csv("data/raw/raw_train.tsv" ,sep = "\t")
    test_csv = pd.read_csv("data/raw/raw_test.tsv" ,sep = "\t")
    validation_csv = pd.read_csv("data/raw/raw_validation.tsv" ,sep = "\t")
    sentiment_labels = pd.read_csv("data/raw/sentiment_labels.txt" ,sep = ",")
    print(train_csv.head())
    print(test_csv.head())
    print(validation_csv.head())
    print(sentiment_labels.head())
