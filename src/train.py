from transformers import __version__ as transformers_version
from packaging import version
import os


def train_model(model, train_dataset, val_dataset, model_name, output_dir="./outputs", epochs=3, batch_size=16):
    save_dir = f"{output_dir}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # check transformers version
    if version.parse(transformers_version) >= version.parse("4.2.0"):
        training_args = TrainingArguments(
            output_dir=save_dir,
            evaluation_strategy="epoch",   # works in new versions
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
            report_to="none"  
        )
    else:
        training_args = TrainingArguments(
            output_dir=save_dir,
            evaluate_during_training=True,  # fallback for old versions
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"{save_dir}/logs",
            logging_steps=50
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

    history = trainer.state.log_history
    pd.DataFrame(history).to_csv(f"{save_dir}/training_log.csv", index=False)

    return trainer
