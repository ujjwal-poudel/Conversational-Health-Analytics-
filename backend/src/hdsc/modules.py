import math
from math import inf
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

PAD_ID = 0


class SentenceLevelEncoder(nn.Module):
    """BiLSTM attention encoder for sentence-level sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        attention_type: str = "hierarchical",
        binary_only: bool = True,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the network.

        Parameters
        ----------
        input_size : int
            Input size for the LSTM.
        hidden_size : int
            Hidden size for the LSTM.
        num_layers : int
            Number of hidden layers for the LSTM.
        dropout : float
            Dropout after each LSTM layer.
            If there is only one hidden layer, no LSTM dropout is applied.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.binary_only = binary_only
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if self.num_layers > 1 else 0,
        )
        # self.attention = SoftDotAttention(2 * self.hidden_size, "general")
        if attention_type == "hierarchical":
            self.attention_binary = HierarchicalAttention(self.num_directions * self.hidden_size)
            if not self.binary_only:
                self.attention_regression = HierarchicalAttention(self.num_directions * self.hidden_size)
        elif attention_type == "multihead":
            self.attention_binary = nn.MultiheadAttention(self.num_directions * self.hidden_size, 4)
            if not self.binary_only:
                self.attention_regression = nn.MultiheadAttention(self.num_directions * self.hidden_size, 4)

    def forward(
        self,
        word_encodings: torch.Tensor,
        text_lens: List[int],
        h0: torch.Tensor,
        c0: torch.Tensor,
        device: torch.device,
        pooling: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model.

        Parameters
        ----------
        word_encodings : torch.Tensor
            [description]
        text_lens : List[int]
            [description]
        h0, c0 : torch.Tensor
            [description]
        device : torch.device
            [description]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [description]
        """
        mask = torch.zeros(word_encodings.size(0), word_encodings.size(1), dtype=torch.bool).to(device)
        for i in range(mask.size(0)):
            mask[i, text_lens[i] :] = True

        packed_inp = pack_padded_sequence(word_encodings, text_lens, batch_first=True, enforce_sorted=False)
        packed_hidden, (hn, cn) = self.lstm(packed_inp, (h0, c0))
        hidden, output_lens = pad_packed_sequence(packed_hidden, batch_first=True, padding_value=PAD_ID)

        if self.num_directions == 2:
            hn = torch.cat((hn[-1], hn[-2]), 1)
            cn = torch.cat((cn[-1], cn[-2]), 1)
        else:
            hn = hn.squeeze(0)

        if self.attention_type == "hierarchical":
            final_hidden_binary, attn_binary = self.attention_binary(hidden, mask=mask)
            if not self.binary_only:
                final_hidden_regression, attn_regression = self.attention_regression(hidden, mask=mask)
            else:
                final_hidden_regression, attn_regression = None, None
        elif self.attention_type == "multihead":
            word_encodings = hidden.transpose(0, 1)
            final_hidden, attn = self.attention_binary(
                word_encodings, word_encodings, word_encodings, key_padding_mask=mask
            )
            final_hidden = final_hidden.transpose(0, 1)
            input_mask_expanded = (mask == False).unsqueeze(-1).expand(final_hidden.size()).float()
            hidden_sum = torch.sum(final_hidden * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            final_hidden = hidden_sum / sum_mask
        elif self.attention_type == "none":
            final_hidden_binary = hn
            final_hidden_regression = hn
            attn_binary = None
            attn_regression = None

        # if pooling == "mean":
        #     final_hidden = weighted_inputs
        #     input_mask_expanded = (
        #         (mask == False).unsqueeze(-1).expand(final_hidden.size()).float()
        #     )
        #     hidden_sum = torch.sum(final_hidden * input_mask_expanded, dim=1)
        #     sum_mask = input_mask_expanded.sum(1)
        #     sum_mask = torch.clamp(sum_mask, min=1e-9)
        #     final_hidden = hidden_sum / sum_mask
        # elif pooling == "last":
        #     final_hidden = hn

        reverse_mask = mask == False
        input_mask_expanded = reverse_mask.unsqueeze(-1).expand(hidden.size()).float()
        hidden_sum = torch.sum(hidden * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        hidden_mean = hidden_sum / sum_mask
        hidden_mean = hidden_mean.unsqueeze(1)
        cosine_sim = torch.abs(F.cosine_similarity(hidden, hidden_mean, dim=2)) * reverse_mask
        conicity = cosine_sim.sum(1) / reverse_mask.sum(1)

        return (
            final_hidden_binary,
            final_hidden_regression,
            attn_binary,
            attn_regression,
            conicity,
        )


class BertWordLevelEncoder(nn.Module):
    def __init__(self, bert_model: str, freeze_weights: bool = True):
        super().__init__()
        self.freeze_weights = freeze_weights
        self.bert = AutoModel.from_pretrained(bert_model)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, inputs):
        if self.freeze_weights:
            with torch.no_grad():
                bert_output = self.bert(**inputs)
        else:
            bert_output = self.bert(**inputs)

        sent_embeddings = self.mean_pooling(bert_output, inputs["attention_mask"])
        return sent_embeddings


class SoftDotAttention(nn.Module):
    r"""Soft (Global) Dot Attention.

    Given the target hidden state :math:`h_t` and the source-side
    context vector :math:`c_t`, an attentional hidden state :math:`\tilde{h}_t`
    is calculated as follows:

    .. math::
        \tilde{h}_t = \text{tanh}(W_c[c_t;h_t])

    At each timestep :math:`t`, the model infers a *variable-length* alignment weight vector
    :math:`\alpha_t` based on the current target state :math:`h_t` and all source states :math:`\bar{h}_s`.
    A global contextvector :math:`c_t` is then computed as the weighted average, according to
    :math:`\alpha_t`, over all the source states.

    .. math::
        \begin{align} \\
            \alpha_t(s) &= \text{align}(h_t, \bar{h}_s) \\
            &= \frac{\exp(\text{score}(h_t, \bar{h}_s))}
                {\sum_{s'}\exp(\text{score}(h_t, \bar{h}_{s'}))}
        \end{align}

    where :math:`\text{score}` is referred as a *content-based* function
    which is defined in three ways:

    .. math::
        \begin{equation} \\
            \text{score}(h_t, \bar{h}_s) = \begin{cases}
                h_t^\top \bar{h}_s & \text{dot} \\
                h_t^\top W_\alpha \bar{h}_s & \text{general} \\
                v_\alpha^\top \text{tanh}(W_\alpha[h_t;\bar{h}_s]) & \text{concat}\\
            \end{cases} \\
        \end{equation}

    Refer to [Luong_2015]_ for more information.

    .. [Luong_2015] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning.
        "Effective approaches to attention-based neural machine translation."
        arXiv preprint arXiv:1508.04025 (2015).

    Args:
        dim (int): The input dimension of the word annotation.

    Inputs:
        - **target_hidden_state** of shape `(batch, dim)`: tensor containing
          the hidden state of the target element.
        - **source_hidden_states** of shape `(batch, seq_len, dim)`: tensor
          containing the hidden states for the whole source sequence.
        - **mask** (optional) of shape `(batch, seq_len)`: boolean tensor
          containing the positions that should not be assigned any
          attention (e.g. padding symbols).
        - **external_weights** (optional) of shape `(batch, seq_len)`: tensor
          containing external weights to combine with the attention scores.

    Outputs:
        - **h_tilde** of shape `(batch, dim)`: tensor containing a weighted 
          average of all source hidden states.
        - **attn** of shape `(batch, seq_len)`: tensor containing attention scores 
          for each item in the source sequence.

    """

    def __init__(self, dim: int, score: str = "dot"):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.LeakyReLU()
        self.mask = None
        self.score = score
        assert score in [
            "dot",
            "general",
            "concat",
        ], "Incorrect score type! Choose between ['dot', 'general', 'concat']!"
        nn.init.xavier_uniform_(
            self.linear_in.weight,
            gain=nn.init.calculate_gain("leaky_relu", 0.2),
        )
        nn.init.xavier_uniform_(
            self.linear_out.weight,
            gain=nn.init.calculate_gain("leaky_relu", 0.2),
        )

    def forward(
        self,
        target_hidden_state: torch.Tensor,
        source_hidden_states: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        external_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate target_hidden_state through the network."""
        if self.score == "general":
            target = self.linear_in(target_hidden_state).unsqueeze(2)  # batch x dim x 1
        elif self.score == "dot":
            target = target_hidden_state.unsqueeze(2)

        # Get attention
        attn = torch.bmm(source_hidden_states, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -inf)

        if external_weights is not None:
            assert external_weights.size() == attn.size(), "External weights size must match the attention size!"
            attn = attn * external_weights

        attn = self.softmax(torch.nan_to_num(attn, neginf=-inf, nan=-inf))

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, source_hidden_states).squeeze(1)  # batch x dim
        # h_tilde = torch.cat((weighted_context, target_hidden_state), 1)

        # h_tilde = self.relu(self.linear_out(h_tilde))
        h_tilde = self.relu(weighted_context)
        # print("h_tilde:", h_tilde)
        # print("weighted_context:", weighted_context)

        return h_tilde, attn


class HierarchicalAttention(nn.Module):
    """Attention as in Hierarchical Attention Networks for Document Classification.

    URL: https://www.cc.gatech.edu/~dyang888/docs/naacl16.pdf

    Args:
        dim (int): The input dimension of the word annotation.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, 1, bias=False)
        # self.external = nn.Sequential(
        #     nn.Linear(2, 50), nn.LeakyReLU(), nn.Linear(50, 1), nn.LeakyReLU()
        # )
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        external_weights: Optional[torch.Tensor] = None,
        average_outputs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tanh(self.linear_in(source_hidden_states))
        similarity = self.context(hidden).squeeze(2)

        # if external_weights is not None:
        #     similarity = torch.cat(
        #         (similarity.unsqueeze(2), external_weights.unsqueeze(2)), dim=2
        #     )
        #     similarity = self.external(similarity).squeeze(2)

        if mask is not None:
            similarity.masked_fill_(mask, -inf)

        importance = self.softmax(similarity)

        if average_outputs:
            outputs = torch.bmm(importance.unsqueeze(1), source_hidden_states).squeeze(1)
        else:
            outputs = source_hidden_states * importance.unsqueeze(-1)

        return outputs, importance


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float, num_classes: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.relu = nn.LeakyReLU()
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def init_weights(self):
        nn.init.xavier_uniform_(
            self.dense.weight,
            nn.init.calculate_gain("leaky_relu", 0.2),
        )
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(inputs)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output
