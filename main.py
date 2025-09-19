from src.preprocessing import load_data, EmotionDataset
from src.model import get_model_and_tokenizer
from src.train import train_model
from src.evaluate import evaluate_model
from src.plot_metrics import plot_training_curves


def main():
    # Load dataset
    train_df, val_df, test_df = load_data(
        "data/raw/raw_train.tsv", "data/raw/raw_validation.tsv","data/raw/raw_test.tsv","data/raw/sentiment_label.txt"
    )

    candidates = ["mbert", "xlmr"]
    results = {}

    for model_type in candidates:
        print(f"\n=== Training {model_type.upper()} ===")

        tokenizer, model = get_model_and_tokenizer(model_type=model_type, num_labels=3)


        train_dataset = EmotionDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
        val_dataset = EmotionDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
        test_dataset = EmotionDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

        trainer = train_model(model, train_dataset, val_dataset, model_name=model_type)

        plot_training_curves(f"./outputs/{model_type}/training_log.csv", model_type)


        report = evaluate_model(trainer, test_dataset, label_names=["negative", "neutral", "positive"])
        print(f"\n=== {model_type.upper()} Test Report ===")
        print(report)


        metrics = trainer.evaluate(val_dataset)
        results[model_type] = metrics["eval_f1"]

    best_model_type = max(results, key=results.get)
    print("\n==============================")
    print(f" Best Model: {best_model_type.upper()} with F1 = {results[best_model_type]:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
