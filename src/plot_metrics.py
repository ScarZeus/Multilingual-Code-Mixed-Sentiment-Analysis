import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(log_file, model_name, save_png=True):
    df = pd.read_csv(log_file)
    df = df.dropna(subset=["epoch"])

    # Loss curve
    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["loss"], label="Train Loss")
    if "eval_loss" in df.columns:
        plt.plot(df["epoch"], df["eval_loss"], label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} - Loss Curve")
    plt.legend()
    if save_png:
        plt.savefig(f"./results/{model_name}/loss_curve.png")
    plt.show()

    # Accuracy curve
    if "eval_accuracy" in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["eval_accuracy"], marker="o", label="Eval Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name.upper()} - Accuracy Curve")
        plt.legend()
        if save_png:
            plt.savefig(f"./results/{model_name}/accuracy_curve.png")
        plt.show()

    # F1 curve
    if "eval_f1" in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["eval_f1"], marker="s", color="green", label="Eval F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title(f"{model_name.upper()} - F1 Score Curve")
        plt.legend()
        if save_png:
            plt.savefig(f"./results/{model_name}/f1_curve.png")
        plt.show()
