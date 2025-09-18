import pandas as pd
import torch
from torch.utils.data import Dataset


def load_data(train_path, val_path, test_path, sentiment_path):
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["uid", "text", "label"])
    val_df = pd.read_csv(val_path, sep="\t", header=None, names=["uid", "text", "label"])
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["uid", "text"])

    # Merge test labels
    sentiment_df = pd.read_csv(sentiment_path)
    test_df = test_df.merge(sentiment_df, left_on="uid", right_on="Uid", how="left")
    test_df.drop("Uid", axis=1, inplace=True)
    test_df.rename(columns={"Sentiment": "label"}, inplace=True)

    return train_df, val_df, test_df


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item
