import math
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from accelerate.utils import send_to_device
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoConfig

from hdsc.modules import ClassificationHead, SentenceLevelEncoder

PAD_ID = 0


class PHQTotalMulticlassAttentionModelBERT(nn.Module):
    """Hierarchical Attention Classification model with BERT in the word level.

    See Also
    --------
    PHQTotalMulticlassAttentionModel
    BertWordLevelEncoder

    References
    ----------

    This model was inspired by [Zichao_2016]_. Refer to their paper for more details.

    .. [Zichao_2016] Yang, Zichao, et al. "Hierarchical attention networks for document classification."
       Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics:
       human language technologies. 2016.

    """

    def __init__(
        self,
        bert_model: str,
        encoder_hidden_dim: int,
        encoder_num_layers: int,
        dropout: float,
        num_classes: int,
        attention_type: str = "hierarchical",
        pooling: str = "mean",
        binary_only: bool = True,
        bidirectional: bool = True,
        multilabel: bool = False,
        regression: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.bert_model = bert_model
        self.bidirectional = bidirectional
        self.multilabel = multilabel
        self.regression = regression
        self.num_directions = 2 if bidirectional else 1
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.attention_type = attention_type
        self.pooling = pooling
        self.binary_only = binary_only
        self.dropout = dropout
        self.device = device

        self.encoder_config = AutoConfig.from_pretrained(self.bert_model)
        self.encoder = AutoModel.from_config(self.encoder_config)
        self.word_output_dim = self.encoder.config.hidden_size

        self.sent_encoder = SentenceLevelEncoder(
            self.word_output_dim,
            encoder_hidden_dim,
            encoder_num_layers,
            dropout,
            attention_type,
            self.binary_only,
            bidirectional,
        )
        self.linear_in_size = self.num_directions * encoder_hidden_dim

        self.clf_binary = ClassificationHead(self.linear_in_size, dropout, num_classes)

        if not self.binary_only:
            self.clf_regression = ClassificationHead(self.linear_in_size, dropout, 1)

        self.softmax = nn.LogSoftmax(dim=1)

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def initialize_encoder_weights(self):
        self.encoder = AutoModel.from_pretrained(self.bert_model)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ) -> torch.Tensor:
        model_output = self.encoder(inputs["input_ids"], inputs["attention_mask"])
        sentence_embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        word_outputs = torch.split(sentence_embeddings, inputs["text_lens"].tolist())

        output = pad_sequence(word_outputs, batch_first=True, padding_value=PAD_ID)

        batch_size = inputs["text_lens"].size(0)
        sent_h0, sent_c0 = self.init_hidden(batch_size, device=self.device)
        (
            final_hidden_binary,
            final_hidden_regression,
            attn_binary,
            attn_regression,
            sent_conicity,
        ) = self.sent_encoder(
            output,
            send_to_device(inputs["text_lens"], torch.device("cpu")),
            sent_h0,
            sent_c0,
            self.device,
            pooling=self.pooling,
        )

        pred_binary = self.clf_binary(final_hidden_binary)
        if self.multilabel or self.regression:
            # pred_binary_final = torch.sigmoid(pred_binary)
            pred_binary_final = pred_binary
        else:
            pred_binary = self.softmax(pred_binary)
        pred_regression = None

        if not self.binary_only:
            pred_regression = self.clf_regression(final_hidden_regression)
            pred_regression = pred_regression.squeeze(1)

        if not self.multilabel and not self.regression:
            _, topi = pred_binary.topk(k=1, dim=1)
            pred_binary_final = topi.squeeze(-1)

        return pred_binary

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(
            self.encoder_num_layers * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            requires_grad=False,
        )
        c0 = torch.zeros(
            self.encoder_num_layers * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            requires_grad=False,
        )
        h0 = h0.to(device)
        c0 = c0.to(device)
        return h0, c0
