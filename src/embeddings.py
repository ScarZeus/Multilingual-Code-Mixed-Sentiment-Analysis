import torch


def extract_embeddings(model, tokenizer, text, max_len=64):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    with torch.no_grad():
        outputs = model.bert(**inputs) if hasattr(model, "bert") else model.roberta(**inputs)
    return outputs.last_hidden_state
