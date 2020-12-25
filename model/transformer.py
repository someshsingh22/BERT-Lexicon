import numpy as np
from torch import nn
from transformers import BertForSequenceClassification


class BERT(nn.Module):
    def __init__(self, name="bert-base-uncased"):
        super(BERT, self).__init__()
        self.name = name
        self.encoder = BertForSequenceClassification.from_pretrained(name)

    def forward(self, text, label):
        """
        Convert Labels
        """
        label = label - 1
        loss, logits = self.encoder(text, labels=label)[:2]
        return loss, logits

    def convertLogits(self, logits):
        """
        Convert Logits To Labels
        """
        logits = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return logits + 1
