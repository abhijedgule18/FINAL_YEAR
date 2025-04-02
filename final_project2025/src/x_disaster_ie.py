import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer

class XDisasterIE(nn.Module):
    def __init__(self, num_labels, roberta_model_name='xlm-roberta-base'):
        super(XDisasterIE, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Extract the sequence output
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Example Usage (Illustrative):
if __name__ == '__main__':
    # Load pre-trained model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XDisasterIE(num_labels=3)  # Example: 3 labels (Disaster Type, Location, Severity)

    # Example input text
    text = "Severe flooding in Jakarta causes widespread damage."

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    outputs = model(**inputs)

    # Print output shape
    print("Output shape:", outputs.shape) # Should be something like: [1 (batch size), sequence length, num_labels]
