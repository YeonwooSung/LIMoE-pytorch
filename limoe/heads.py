import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from transformers.models.bert.configuration_bert import BertConfig


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained("bert-large-uncased")
        self.transform = BertPredictionHeadTransform(self.bert_config)

        if self.bert_config.vocab_size != config.vocab_size:
            self.has_projection = True
            self.projection = nn.Linear(self.bert_config.vocab_size, config.vocab_size)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        if self.has_projection:
            x = self.projection(x)
        x = self.decoder(x) + self.bias
        return x


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x



def compute_mlm(mlm_logits, mlm_labels, vocab_size):
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, vocab_size),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
    }

    return ret


def compute_itm(itm_logits, itm_labels):
    itm_loss = F.cross_entropy(
        itm_logits.view(-1, 2),
        itm_labels.view(-1),
    )

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    return ret
