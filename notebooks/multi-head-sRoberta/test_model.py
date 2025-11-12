import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import torch
import torch.nn as nn
from datasets import load_dataset
from torchmetrics.functional import f1_score, mean_absolute_error
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

from hdsc.model import PHQTotalMulticlassAttentionModelBERT

parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Name of the model to test", type=str)
parser.add_argument(
    "chunking",
    help="Type of dialog chunking",
    type=str,
    choices=["lines", "lines_inverse", "pairs", "semantic", "word_window", "participant_only"],
)
parser.add_argument("--multilabel", help="The model gives multilabel prediction.", action="store_true")
parser.add_argument("--regression", help="The model gives regression prediction.", action="store_true")
parser.add_argument("--five_classes", help="The model gives five classes prediction.", action="store_true")
parser.add_argument(
    "--store_gradient",
    help="Store the gradient for each prediction with the respect to the input.",
    action="store_true",
)
parser.add_argument("--predict_on_train", help="Also predict on the training set.", action="store_true")
args = parser.parse_args()

MODEL_NAME = args.model_name
CHUNKING = args.chunking
MULTILABEL = args.multilabel
REGRESSION = args.regression
FIVE_CLASS = args.five_classes
STORE_GRADIENT = args.store_gradient

print(f"Running evaluation for model {MODEL_NAME} with {CHUNKING} chunking...")


def collate_fn(batch):
    labels = torch.vstack([x["labels"] for x in batch])
    input_ids = torch.vstack([x["input_ids"] for x in batch])
    attention_mask = torch.vstack([x["attention_mask"] for x in batch])
    text_lens = torch.tensor([len(x["input_ids"]) for x in batch])
    return {
        "labels": labels,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_lens": text_lens,
    }


def load_model(model_path: str, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    loaded_dict = torch.load(model_path, weights_only=True)
    model_kwargs = loaded_dict["kwargs"]
    model_state_dict = loaded_dict["model"]
    model = PHQTotalMulticlassAttentionModelBERT(
        device=device,
        **model_kwargs,
    )
    model = model.to(device)
    model.load_state_dict(model_state_dict)

    return model


def read_labels(labels_path, test=False):
    labels = {}
    labels_start = 2 if test else 4
    with open(Path(labels_path), encoding="utf-8") as f:
        labels_reader = csv.reader(f)
        next(labels_reader, None)
        for row in labels_reader:
            if not row or len(row) <= labels_start:
                continue  # skip empty or malformed rows
            if row[0] not in ["451", "458"]:  # broken interviews
                try:
                    labels[row[0]] = [int(x) for x in row[labels_start:]]
                except ValueError:
                    logging.warning("[Could not read the labels for the transcript %s!]", row[0])
    return labels


def set_dropout_to_eval(m):
    if type(m) == nn.Dropout:
        m.eval()


@torch.no_grad()
def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> List[torch.Tensor]:
    preds: List[torch.Tensor] = []

    model.eval()

    for i, batch in enumerate(tqdm(dataloader)):
        pred_binary = model(batch)
        preds.append(pred_binary.cpu())
    return preds


@torch.enable_grad()
def predict_and_save_grad(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    preds: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []

    model.train()
    model.apply(set_dropout_to_eval)
    for param in model.encoder.parameters():
        param.requires_grad = False

    for i, batch in enumerate(tqdm(dataloader)):
        model.zero_grad()
        for j in range(8):
            texts: List[Dict[str, torch.Tensor]] = []
            for i in range(batch["text_lens"].size(0)):
                texts.append(
                    {
                        "input_ids": batch["input_ids"][i].to(device),
                        "attention_mask": batch["attention_mask"][i].to(device),
                    }
                )
            outputs = model(texts, batch["text_lens"])
            outputs.pred_binary_final[0, j].backward()
            if j == 0:
                preds.append(outputs.pred_binary_final.cpu())
            grads.append(outputs.chunk_hidden_states.grad.data)
    return preds, grads


device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
dataset = load_dataset("daic_woz.py", CHUNKING, download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD, trust_remote_code=True)

encoded_dataset = dataset.map(
    lambda examples: tokenizer(examples["turns"], padding="max_length", truncation=True),
    load_from_cache_file=False,
)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], device=device)
train_mean = torch.tensor([0.7358, 0.8019, 1.0189, 1.0377, 0.9151, 0.8585, 0.6792, 0.3113])
splits = ["test"]

for _split in splits:
    dataset_split = "validation" if _split == "dev" else _split
    dataloader = torch.utils.data.DataLoader(
        encoded_dataset[dataset_split],
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )

    label_split = _split == "test"

    # Add an if-statement to handle the test split's unique name
    if _split == "test":
        # Use the test file name
        labels_path = Path("D:/volumes/MACBACKUP/Regression/full_test_split.csv/")
    else:
        # This is the old code, which works for "dev" or "train"
        labels_path = Path(f"D:/volumes/MACBACKUP/Regression/{label_split}_split_Depression_AVEC2017.csv")

    test = _split == "test"
    labels = read_labels(labels_path, test=test)

    # -- Modification made to ensure, the model evaluate the final head for test set --
    # labels = torch.tensor([label for label in labels.values()])

    # sum_labels = torch.sum(labels, dim=1)

    # -- Uncomment the top code to change for dev --

    labels_list = list(labels.values())
    labels_tensor = torch.tensor(labels_list)

    # # Handle case: test set has only one regression value per participant (total score)
    # if labels_tensor.ndim == 1:
    #     # only total PHQ score available
    #     sum_labels = labels_tensor
    # else:
    #     # per-head PHQ items available
    #     sum_labels = torch.sum(labels_tensor, dim=1)

    # Detect whether labels are per-item (8D) or total scores (1D)
    if labels_tensor.ndim == 1:
        print(f"Detected 1D labels for {_split} set — using total PHQ scores only.")
        sum_labels = labels_tensor
        per_item_labels = False
    else:
        print(f"Detected per-item PHQ labels for {_split} set — computing totals.")
        sum_labels = torch.sum(labels_tensor, dim=1)
        per_item_labels = True

    labels = labels_tensor

    # -- End of modification --

    pos_ids = sum_labels > 9
    neg_ids = sum_labels <= 9
    one_ids = sum_labels <= 4
    two_ids = (sum_labels > 4) & (sum_labels <= 9)
    three_ids = (sum_labels > 9) & (sum_labels <= 14)
    four_ids = (sum_labels > 14) & (sum_labels <= 19)
    five_ids = sum_labels > 19
    five_bin_ids = [one_ids, two_ids, three_ids, four_ids, five_ids]
    two_bin_ids = [pos_ids, neg_ids]

    model_dir = Path("saved_models") / MODEL_NAME
    model_preds = []
    for model_path in model_dir.iterdir():
        if model_path.suffix == ".pt" and model_path.name.startswith("model"):
            model: PHQTotalMulticlassAttentionModelBERT = load_model(model_path, device)
            if STORE_GRADIENT:
                preds, grads = predict_and_save_grad(model, dataloader)
                print("Saving gradients to", model_dir / f"grads_{model_path.name}_{_split}.pt")
                torch.save(grads, model_dir / f"grads_{model_path.name}_{_split}.pt")
            else:
                preds = predict(model, dataloader)
            preds = torch.vstack(preds) if MULTILABEL or FIVE_CLASS else torch.hstack(preds)
            model_preds.append(preds)

    torch.save(model_preds, model_dir / f"preds_{_split}.pt")
    torch.save(labels, model_dir / f"labels_{_split}.pt")

    if not per_item_labels:
        print(f"Skipping per-item (8-head) evaluation for {_split} — only total PHQ scores available.")
        continue

    if MULTILABEL:
        all_mae_micro = torch.zeros(len(model_preds))
        all_mae_macro = torch.zeros(len(model_preds))
        all_mae_micro_sum = torch.zeros(len(model_preds))
        all_mae_macro_sum = torch.zeros(len(model_preds))
        all_mae_five = torch.zeros(len(model_preds))
        all_mae_sep = torch.zeros(len(model_preds), 8)
        all_rrmse_sep = torch.zeros(len(model_preds), 8)
        for i, preds in enumerate(model_preds):
            try:
                mae_micro = mean_absolute_error(preds, labels)
            except RuntimeError as e:
                print("Preds:", preds)
                print("Labels:", labels)
                raise e
            mae_macro = torch.mean(torch.tensor([mean_absolute_error(preds[ids], labels[ids]) for ids in two_bin_ids]))
            try:
                mae_micro_sum = mean_absolute_error(torch.sum(preds, dim=1), sum_labels)
            except RuntimeError as e:
                print("Preds:", preds)
                print("Labels:", labels)
                raise e
            mae_macro_sum = torch.mean(
                torch.tensor(
                    [mean_absolute_error(torch.sum(preds, dim=1)[ids], sum_labels[ids]) for ids in two_bin_ids]
                )
            )
            mae_five = torch.mean(torch.tensor([mean_absolute_error(preds[ids], labels[ids]) for ids in five_bin_ids]))
            mae_sep = torch.mean(torch.abs(preds - labels), dim=0)

            rmse_sep_pred = torch.sqrt(torch.mean(torch.pow((preds - labels), 2), dim=0))
            rmse_sep_train = torch.sqrt(torch.mean(torch.pow((labels - train_mean.expand_as(labels)), 2), dim=0))
            rrmse = rmse_sep_pred / rmse_sep_train

            all_mae_micro[i] = mae_micro
            all_mae_macro[i] = mae_macro
            all_mae_micro_sum[i] = mae_micro_sum
            all_mae_macro_sum[i] = mae_macro_sum
            all_mae_five[i] = mae_five
            all_mae_sep[i] = mae_sep
            all_rrmse_sep[i] = rrmse

        all_mae_sep_mean = all_mae_sep.mean(dim=0)
        all_mae_sep_std = all_mae_sep.std(dim=0, unbiased=False)
        all_mae_sep_str = " & ".join([str(round(all_mae_sep_mean[i].item(), 3)) for i in range(all_mae_sep.size(1))])
        all_mae_sep_std_str = " & ".join(
            [
                str(round(all_mae_sep_mean[i].item(), 3)) + r" $\pm$ " + str(round(all_mae_sep_std[i].item(), 3))
                for i in range(all_mae_sep.size(1))
            ]
        )

        all_rrmse_sep_mean = all_rrmse_sep.mean(dim=0)
        all_rrmse_sep_std = all_rrmse_sep.std(dim=0, unbiased=False)
        all_rrmse_sep_str = " & ".join(
            [str(round(all_rrmse_sep_mean[i].item(), 3)) for i in range(all_rrmse_sep.size(1))]
        )
        all_rrmse_sep_std_str = " & ".join(
            [
                str(round(all_rrmse_sep_mean[i].item(), 3)) + r" $\pm$ " + str(round(all_rrmse_sep_std[i].item(), 3))
                for i in range(all_rrmse_sep.size(1))
            ]
        )

        print(
            f"{_split} MAE micro: {all_mae_micro.mean().item():.4f} ± {all_mae_micro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} MAE macro: {all_mae_macro.mean().item():.4f} ± {all_mae_macro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} MAE sum micro: {all_mae_micro_sum.mean().item():.4f} "
            + f"± {all_mae_micro_sum.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} MAE sum macro: {all_mae_macro_sum.mean().item():.4f} "
            f"± {all_mae_macro_sum.std(unbiased=False).item():.4f}"
        )
        print(f"{_split} MAE five: {all_mae_five.mean().item():.4f} ± {all_mae_five.std(unbiased=False).item():.4f}")
        print(f"{_split} MAE separate: {all_mae_sep_str}")
        print(f"{_split} MAE separate with STDs: {all_mae_sep_std_str}")
        print(f"{_split} RRMSE separate: {all_rrmse_sep_str}")
        print(f"{_split} RRMSE separate with STDs: {all_rrmse_sep_std_str}")
        print(f"{_split} aRRMSE: {all_rrmse_sep.mean().item():.4f} ± {all_rrmse_sep.std(unbiased=False).item():.4f}")

        all_f1_micro = torch.zeros(len(model_preds))
        all_f1_macro = torch.zeros(len(model_preds))
        all_f1_five_micro = torch.zeros(len(model_preds))
        all_f1_five_macro = torch.zeros(len(model_preds))
        all_mae_sep = torch.zeros(len(model_preds), 8)

        for idx, preds in enumerate(model_preds):
            sum_preds = torch.sum(preds, dim=1)
            pred_one_ids = sum_preds <= 4
            pred_two_ids = (sum_preds > 4) & (sum_preds <= 9)
            pred_three_ids = (sum_preds > 9) & (sum_preds <= 14)
            pred_four_ids = (sum_preds > 14) & (sum_preds <= 19)
            pred_five_ids = sum_preds > 19
            pred_five_bin_ids = [
                pred_one_ids,
                pred_two_ids,
                pred_three_ids,
                pred_four_ids,
                pred_five_ids,
            ]

            preds_bin = sum_preds > 9
            labels_bin = sum_labels > 9

            labels_five = torch.clone(sum_labels)
            preds_five = torch.floor(torch.clone(sum_preds)).to(torch.int64)
            for i, ids in enumerate(five_bin_ids):
                labels_five = torch.where(ids, i, labels_five)
            for i, ids in enumerate(pred_five_bin_ids):
                preds_five = torch.where(ids, i, preds_five)

            f1_micro = f1_score(preds_bin, labels_bin, task="binary", average="micro")
            f1_macro = f1_score(preds_bin, labels_bin, task="binary", average="macro")
            f1_five_micro = f1_score(preds_five, labels_five, task="multiclass", average="micro", num_classes=5)
            f1_five_macro = f1_score(preds_five, labels_five, task="multiclass", average="macro", num_classes=5)


            all_f1_micro[idx] = f1_micro
            all_f1_macro[idx] = f1_macro
            all_f1_five_micro[idx] = f1_five_micro
            all_f1_five_macro[idx] = f1_five_macro

        print(f"{_split} F1 micro: {all_f1_micro.mean().item():.4f} ± {all_f1_micro.std(unbiased=False).item():.4f}")
        print(f"{_split} F1 macro: {all_f1_macro.mean().item():.4f} ± {all_f1_macro.std(unbiased=False).item():.4f}")
        print(
            f"{_split} F1 five micro: {all_f1_five_micro.mean().item():.4f} "
            + f"± {all_f1_five_micro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} F1 five macro: {all_f1_five_macro.mean().item():.4f} "
            + f"± {all_f1_five_macro.std(unbiased=False).item():.4f}"
        )
    elif REGRESSION:
        all_mae_micro = torch.zeros(len(model_preds))
        all_mae_macro = torch.zeros(len(model_preds))
        for i, preds in enumerate(model_preds):
            preds = preds.squeeze()
            print(preds, sum_labels)
            try:
                mae_micro = mean_absolute_error(preds, sum_labels)
            except RuntimeError as e:
                print("Preds:", preds)
                print("Labels:", labels)
                raise e
            mae_macro = torch.mean(
                torch.tensor([mean_absolute_error(preds[ids], sum_labels[ids]) for ids in two_bin_ids])
            )

            all_mae_micro[i] = mae_micro
            all_mae_macro[i] = mae_macro

        print(
            f"{_split} MAE micro: {all_mae_micro.mean().item():.4f} ± {all_mae_micro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} MAE macro: {all_mae_macro.mean().item():.4f} ± {all_mae_macro.std(unbiased=False).item():.4f}"
        )

        all_f1_micro = torch.zeros(len(model_preds))
        all_f1_macro = torch.zeros(len(model_preds))
        all_f1_five_micro = torch.zeros(len(model_preds))
        all_f1_five_macro = torch.zeros(len(model_preds))

        for idx, preds in enumerate(model_preds):
            preds = preds.squeeze()

            preds_bin = preds > 9
            labels_bin = sum_labels > 9

            preds_five = torch.div(preds, 5, rounding_mode="floor").to(torch.long)
            labels_five = torch.div(sum_labels, 5, rounding_mode="floor").to(torch.long)

            f1_micro = f1_score(preds_bin, labels_bin, task="binary", average="micro")
            f1_macro = f1_score(preds_bin, labels_bin, task="binary", average="macro")
            f1_five_micro = f1_score(preds_five, labels_five, task="multiclass", average="micro", num_classes=5)
            f1_five_macro = f1_score(preds_five, labels_five, task="multiclass", average="macro", num_classes=5)


            all_f1_micro[idx] = f1_micro
            all_f1_macro[idx] = f1_macro
            all_f1_five_micro[idx] = f1_five_micro
            all_f1_five_macro[idx] = f1_five_macro

        print(f"{_split} F1 micro: {all_f1_micro.mean().item():.4f} ± {all_f1_micro.std(unbiased=False).item():.4f}")
        print(f"{_split} F1 macro: {all_f1_macro.mean().item():.4f} ± {all_f1_macro.std(unbiased=False).item():.4f}")
        print(
            f"{_split} F1 five micro: {all_f1_five_micro.mean().item():.4f} "
            + f"± {all_f1_five_micro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} F1 five macro: {all_f1_five_macro.mean().item():.4f} "
            + f"± {all_f1_five_macro.std(unbiased=False).item():.4f}"
        )
    elif FIVE_CLASS:
        all_f1_micro = torch.zeros(len(model_preds))
        all_f1_macro = torch.zeros(len(model_preds))
        all_f1_five_micro = torch.zeros(len(model_preds))
        all_f1_five_macro = torch.zeros(len(model_preds))
        all_mae_sep = torch.zeros(len(model_preds), 8)

        for idx, preds in enumerate(model_preds):
            preds_five = preds.topk(k=1, dim=1)[1].squeeze(-1).to(torch.long)
            labels_five = torch.div(sum_labels, 5, rounding_mode="floor").to(torch.long)

            preds_bin = preds_five > 1
            labels_bin = sum_labels > 9

            print(preds_five)
            print(labels_five)

            f1_micro = f1_score(preds_bin, labels_bin, task="binary", average="micro")
            f1_macro = f1_score(preds_bin, labels_bin, task="binary", average="macro")
            f1_five_micro = f1_score(preds_five, labels_five, task="multiclass", average="micro", num_classes=5)
            f1_five_macro = f1_score(preds_five, labels_five, task="multiclass", average="macro", num_classes=5)


            all_f1_micro[idx] = f1_micro
            all_f1_macro[idx] = f1_macro
            all_f1_five_micro[idx] = f1_five_micro
            all_f1_five_macro[idx] = f1_five_macro

        print(f"{_split} F1 micro: {all_f1_micro.mean().item():.4f} ± {all_f1_micro.std(unbiased=False).item():.4f}")
        print(f"{_split} F1 macro: {all_f1_macro.mean().item():.4f} ± {all_f1_macro.std(unbiased=False).item():.4f}")
        print(
            f"{_split} F1 five micro: {all_f1_five_micro.mean().item():.4f} "
            + f"± {all_f1_five_micro.std(unbiased=False).item():.4f}"
        )
        print(
            f"{_split} F1 five macro: {all_f1_five_macro.mean().item():.4f} "
            + f"± {all_f1_five_macro.std(unbiased=False).item():.4f}"
        )
    else:
        all_f1_micro = torch.zeros(len(model_preds))
        all_f1_macro = torch.zeros(len(model_preds))
        for idx, preds in enumerate(model_preds):
            preds_bin = preds.squeeze(-1)
            labels_bin = sum_labels > 9

            f1_micro = f1_score(preds_bin, labels_bin, task="binary", average="micro")
            f1_macro = f1_score(preds_bin, labels_bin, task="binary", average="macro")


            all_f1_micro[idx] = f1_micro
            all_f1_macro[idx] = f1_macro

        print(f"{_split} F1 micro: {all_f1_micro.mean().item():.4f} ± {all_f1_micro.std(unbiased=False).item():.4f}")
        print(f"{_split} F1 macro: {all_f1_macro.mean().item():.4f} ± {all_f1_macro.std(unbiased=False).item():.4f}")
