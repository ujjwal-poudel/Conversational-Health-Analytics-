from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from matplotlib.lines import Line2D
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding

from hdsc.model import PHQTotalMulticlassAttentionModelBERT as PHQTotalMulticlassAttentionModelBERTDist

PAD_ID = 0


def plot_grad_flow(named_parameters, save_name=None):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)


def classification_accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Returns the average accuracy

    Args:
        pred (torch.Tensor): prediction of the model (without softmax)
        gold (torch.Tensor): true labels

    Returns:
        float: average accuracy
    """

    if pred.dim() == 1:
        if pred.type() == "torch.FloatTensor":
            pred = torch.round(pred)
    else:
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        gold = gold.long()
    return torch.sum(pred == gold).item() / len(pred)


def save_model(model: torch.nn.Module, path: str, accelerator: Accelerator) -> None:
    path = Path(path)
    model_to_save = accelerator.unwrap_model(model)
    save_dict = {
        "name": "bert",
        "model": model_to_save.state_dict(),
        "kwargs": {
            "bert_model": model_to_save.bert_model,
            "encoder_hidden_dim": model_to_save.encoder_hidden_dim,
            "encoder_num_layers": model_to_save.encoder_num_layers,
            "dropout": model_to_save.dropout,
            "num_classes": model_to_save.num_classes,
            "attention_type": model_to_save.attention_type,
            "pooling": model_to_save.pooling,
            "binary_only": model_to_save.binary_only,
            "bidirectional": model_to_save.bidirectional,
            "multilabel": model_to_save.multilabel,
        },
    }
    accelerator.save(save_dict, path)


def load_model(model_path: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    loaded_dict = torch.load(model_path)
    model_kwargs = loaded_dict["kwargs"]
    model_state_dict = loaded_dict["model"]
    model: nn.Module = PHQTotalMulticlassAttentionModelBERT(device=device, **model_kwargs)
    model.load_state_dict(model_state_dict)

    return model
