from src.preprocessing import load_data, EmotionDataset
from src.model import get_model_and_tokenizer
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Load dataset
    train_df, val_df, test_df = load_data("raw_train.tsv", "raw_validation.tsv", "raw_test.tsv", "sentiment_label.txt")

    # Model + Tokenizer
    tokenizer, model = get_model_and_tokenizer(model_type="mbert", num_labels=3)  # change to "xlmr"

    # Prepare datasets
    train_dataset = EmotionDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = EmotionDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_dataset = EmotionDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    # Train
    trainer = train_model(model, train_dataset, val_dataset, output_dir="./results")

    # Evaluate
    report = evaluate_model(trainer, test_dataset, label_names=["negative", "neutral", "positive"])
    print(report)


if __name__ == "__main__":
    main()
