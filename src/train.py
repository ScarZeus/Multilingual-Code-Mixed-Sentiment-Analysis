from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def train_model(model, train_dataset, val_dataset, model_name, output_dir="./results", epochs=3, batch_size=16):
    save_dir = f"{output_dir}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{save_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"   # disable external loggers like wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    trainer.save_model(f"{save_dir}/best_model")

    # Save training logs
    history = trainer.state.log_history
    pd.DataFrame(history).to_csv(f"{save_dir}/training_log.csv", index=False)

    return trainer
