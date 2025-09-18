from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


def get_model_and_tokenizer(model_type="mbert", num_labels=3):
    if model_type == "mbert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
    elif model_type == "xlmr":
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
    else:
        raise ValueError("Choose model_type as 'mbert' or 'xlmr'")
    return tokenizer, model
