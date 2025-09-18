from sklearn.metrics import classification_report


def evaluate_model(trainer, dataset, label_names=None):
    preds = trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)

    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    return report
