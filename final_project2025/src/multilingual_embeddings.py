import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel


def get_multilingual_embeddings(texts):
    """Get multilingual embeddings using XLM-RoBERTa."""
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Return the last hidden state (embeddings)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence-level embedding
