import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import tomli
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torchinfo import summary
from torchmetrics.functional import accuracy, f1_score, mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from hdsc.losses import MultiLabelLoss
from hdsc.model import PHQTotalMulticlassAttentionModelBERT
from hdsc.utils import save_model

logger = get_logger(__name__)
logger.setLevel("INFO")


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


def training_step(
    config: Dict,
    model: nn.Module,
    inputs: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    scheduler: Optional[GradScaler] = None,
) -> Dict[str, torch.Tensor]:
    """Perform one training step on a batch

    Args:
        model (nn.Module): Model to train.
        inputs (DataItem): Inputs for the model.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        rank (torch.device): GPU device used for DDP training.
        world_size (int): Number of GPUs for DDP training.
        scaler (Optional[GradScaler], optional): Scaler use for mixed precision training. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: Following outputs:
            - loss: overall loss on the training batch;
            - loss_1: loss on the multilabel task;
            - loss_2: regularization term. L2 distance of sum of all labels and total PHQ score;
            - preds: all the predictions;
            - labels: all the multilabel labels;
            - reg_labels: all the regression labels.
    """
    model.train()
    optimizer.zero_grad()

    phq_score = torch.sum(inputs["labels"], dim=1)

    if config["model"]["multilabel"]:
        labels = inputs["labels"]
    elif config["model"]["regression"]:
        labels = phq_score
    elif config["five_classes"]:
        labels = torch.div(phq_score, 5, rounding_mode="floor")
        labels = labels.to(torch.long)
    else:
        labels = phq_score >= 10

    with accelerator.autocast():
        pred_binary = model(inputs)

        if config["model"]["multilabel"]:
            loss, loss_1, loss_2 = loss_fn(
                pred_binary,
                labels,
                phq_score,
            )
        else:
            loss = loss_fn(pred_binary, labels)

    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    # Gather the outputs from all GPUs
    cat_fn = torch.vstack if config["model"]["multilabel"] else torch.hstack
    gathered_preds = accelerator.gather(pred_binary).detach()
    gathered_labels = accelerator.gather(labels).detach()
    gathered_reg_labels = accelerator.gather(phq_score).detach()

    # Recalculate the loss on all inputs
    if config["model"]["multilabel"]:
        loss, loss_1, loss_2 = loss_fn(gathered_preds, gathered_labels, gathered_reg_labels)
    else:
        loss = loss_fn(gathered_preds, gathered_labels)
        loss_1 = torch.zeros(1)
        loss_2 = torch.zeros(1)

    update_outputs = {
        "loss": loss,
        "loss_1": loss_1,
        "loss_2": loss_2,
        "preds": gathered_preds,
        "labels": gathered_labels,
        "reg_labels": gathered_reg_labels,
    }

    return update_outputs


def evaluate(
    config: Dict,
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accelerator: Accelerator,
) -> Dict[str, torch.Tensor]:
    """Evaluates the model on the validation dataset.

    Args:
        model (nn.Module): Model to evaluate.
        val_dataloader (DataLoader): Validation dataloader.
        loss_fn (nn.Module): Loss function.
        rank (torch.device): GPU device used for DDP training.
        world_size (int): Number of GPUs for DDP training.

    Returns:
        Dict[str, torch.Tensor]: Following outputs:
            - loss: overall loss on the training batch;
            - loss_1: loss on the multilabel task;
            - loss_2: regularization term. L2 distance of sum of all labels and total PHQ score;
            - preds: all the predictions;
            - labels: all the multilabel labels;
            - reg_labels: all the regression labels.
    """
    running_loss: float = 0.0
    running_loss_1: float = 0.0
    running_loss_2: float = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_reg_labels: List[torch.Tensor] = []

    n_steps = len(val_dataloader)
    cat_fn = torch.vstack if config["model"]["multilabel"] else torch.hstack

    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            phq_score = torch.sum(batch["labels"], dim=1)
            if config["model"]["multilabel"]:
                labels = batch["labels"]
            elif config["model"]["regression"]:
                labels = phq_score
            elif config["five_classes"]:
                labels = torch.div(phq_score, 5, rounding_mode="floor")
                labels = labels.to(torch.long)
            else:
                labels = phq_score >= 10
            pred_binary = model(batch)

            # Gather the outputs from all GPUs
            gathered_preds = accelerator.gather(pred_binary).detach()
            gathered_labels = accelerator.gather(labels).detach()
            gathered_reg_labels = accelerator.gather(phq_score).detach()

            # Recalculate the loss on all inputs
            if config["model"]["multilabel"]:
                loss, loss_1, loss_2 = loss_fn(gathered_preds, gathered_labels, gathered_reg_labels)
            else:
                loss = loss_fn(gathered_preds, gathered_labels)
                loss_1 = torch.zeros(1)
                loss_2 = torch.zeros(1)

            running_loss += loss
            running_loss_1 += loss_1
            running_loss_2 += loss_2
            all_preds.append(gathered_preds)
            all_labels.append(gathered_labels)
            all_reg_labels.append(gathered_reg_labels)

    avg_loss = running_loss.item() / n_steps
    avg_loss_1 = running_loss_1.item() / n_steps
    avg_loss_2 = running_loss_2.item() / n_steps

    all_preds = torch.vstack(all_preds)
    if not config["model"]["multilabel"] and not config["model"]["regression"]:
        all_preds = all_preds.topk(k=1, dim=1)[1].squeeze(-1)
    all_labels = cat_fn(all_labels)
    all_reg_labels = torch.hstack(all_reg_labels)

    eval_outputs = {
        "loss": avg_loss,
        "loss_1": avg_loss_1,
        "loss_2": avg_loss_2,
        "preds": all_preds,
        "labels": all_labels,
        "reg_labels": all_reg_labels,
    }

    return eval_outputs


def main():
    accelerator = Accelerator()
    device = accelerator.device

    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    model_name = "_".join(
        [
            "lstm" if config["model"]["bert_model"] == "lstm" else "robert",
            "multilabel" if config["model"]["multilabel"] else "binary",
            "regression" if config["model"]["regression"] else "no-regression",
            "five_classes" if config["five_classes"] else "",
        ]
    )
    save_dir = Path(config["save_dir"]) / model_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
    dataset = load_dataset("daic_woz.py", "lines")
    encoded_dataset = dataset.map(
        lambda examples: tokenizer(examples["turns"], padding="max_length", truncation=True),
        load_from_cache_file=False,
    )
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    cat_fn = torch.vstack if config["model"]["multilabel"] else torch.hstack

    train_dataloader = torch.utils.data.DataLoader(
        encoded_dataset["train"],
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        encoded_dataset["validation"],
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        shuffle=False,
    )

    model = PHQTotalMulticlassAttentionModelBERT(device=device, **config["model"])
    model.initialize_encoder_weights()

    if config["model"]["multilabel"]:
        loss_fn = MultiLabelLoss(
            regularization=config["regularization_loss"],
            l=config["loss_l"],
        )
    elif config["model"]["regression"]:
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.NLLLoss()

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    total_steps = len(train_dataloader) * config["num_iters"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

    model = accelerator.prepare(model)
    optimizer, train_dataloader, scheduler = accelerator.prepare(optimizer, train_dataloader, scheduler)
    validation_dataloader = accelerator.prepare(validation_dataloader)

    if config["encoder_layers_to_freeze"] is not None:
        if config["encoder_layers_to_freeze"] == "all":
            model.encoder.requires_grad_(False)
        else:
            layers_to_freeze = [
                str(x) for x in list(itertools.chain.from_iterable(config["encoder_layers_to_freeze"]))
            ]
            for name, param in model.encoder.named_parameters():
                if any([True for layer in layers_to_freeze if layer in name]):
                    param.requires_grad_(False)

    accelerator.print(config)
    model_stats = summary(model, depth=4, verbose=0)
    accelerator.print(model_stats)

    best_loss = float("inf")
    best_f1 = 0.0
    running_loss = 0.0
    running_loss_bin = 0.0
    running_loss_reg = 0.0
    preds = []
    dev_preds = []
    labels = []
    labels_reg = []
    dev_labels = []
    dev_labels_reg = []
    global_step = 0
    eval_every = len(train_dataloader)

    with open(save_dir / f"log_{model_name}_{config['seed']}.tsv", "w", encoding="utf-8") as f:
        f.write(
            "epoch\ttrain_loss\ttrain_loss_bin\ttrain_loss_reg\tdev_loss\tdev_loss_bin\tdev_loss_reg\t"
            + "train_acc\tdev_acc\tf1_micro\tf1_macro\tdev_mae\tdev_mse\n"
        )
        for epoch in range(config["num_iters"]):
            running_loss = 0.0
            preds = []
            for batch in train_dataloader:
                outputs = training_step(config, model, batch, loss_fn, optimizer, accelerator, scheduler)

                running_loss += outputs["loss"]
                running_loss_bin += outputs["loss_1"]
                running_loss_reg += outputs["loss_2"]
                preds.append(outputs["preds"])
                labels.append(outputs["labels"])
                labels_reg.append(outputs["reg_labels"])

                global_step += 1

                if global_step % eval_every == 0:
                    eval_outputs = evaluate(config, model, validation_dataloader, loss_fn, accelerator)

                    average_train_loss = running_loss.item() / eval_every
                    average_train_loss_bin = running_loss_bin.item() / eval_every
                    average_train_loss_reg = running_loss_reg.item() / eval_every
                    average_dev_loss = eval_outputs["loss"]
                    average_dev_loss_bin = eval_outputs["loss_1"]
                    average_dev_loss_reg = eval_outputs["loss_2"]

                    preds = torch.vstack(preds)
                    if not config["model"]["multilabel"] and not config["model"]["regression"]:
                        preds = preds.topk(k=1, dim=1)[1].squeeze(-1)
                    dev_preds = eval_outputs["preds"]
                    preds = preds.float()
                    dev_preds = dev_preds.float()
                    labels = cat_fn(labels)
                    dev_labels = eval_outputs["labels"]
                    dev_labels_reg = eval_outputs["reg_labels"]

                    if config["model"]["multilabel"]:
                        average_train_acc = mean_absolute_error(preds, labels).item()
                        average_dev_acc = mean_absolute_error(dev_preds, dev_labels).item()

                        dev_labels_bin = (torch.sum(dev_labels, dim=1) >= 10).to(torch.long)
                        dev_preds_bin = (torch.sum(dev_preds, dim=1) >= 10).to(torch.long)
                        f1_micro = f1_score(dev_preds_bin, dev_labels_bin, average="micro").item()
                        f1_macro = f1_score(
                            dev_preds_bin,
                            dev_labels_bin,
                            average="macro",
                            num_classes=2,
                        ).item()
                        f1_samples = f1_score(
                            dev_labels >= 1.5,
                            dev_preds >= 1.5,
                            task="multilabel",
                            average="micro",
                            num_labels=config["model"]["num_classes"],
                        ).item()

                        mae = mean_absolute_error(
                            torch.sum(dev_preds, dim=1),
                            dev_labels_reg,
                        ).item()
                        mse = mean_squared_error(
                            torch.sum(dev_preds, dim=1),
                            dev_labels_reg,
                        ).item()
                        f1_mae = f1_micro / mae
                    elif config["model"]["regression"]:
                        preds = preds.squeeze(-1)
                        dev_preds = dev_preds.squeeze(-1)
                        average_train_acc = mean_absolute_error(preds, labels).item()
                        average_dev_acc = mean_absolute_error(dev_preds, dev_labels).item()

                        dev_labels_bin = (dev_labels >= 10).to(torch.long)
                        dev_preds_bin = (dev_preds >= 10).to(torch.long)
                        f1_micro = f1_score(dev_preds_bin, dev_labels_bin, average="micro").item()
                        f1_macro = f1_score(
                            dev_preds_bin,
                            dev_labels_bin,
                            average="macro",
                            num_classes=2,
                        ).item()
                        f1_samples = 0.0
                        mae = mean_absolute_error(
                            dev_preds,
                            dev_labels_reg,
                        ).item()
                        mse = mean_squared_error(
                            dev_preds,
                            dev_labels_reg,
                        ).item()
                        f1_mae = f1_micro / mae
                    elif config["five_classes"]:
                        average_train_acc = accuracy(preds.to(torch.long), labels).item()
                        average_dev_acc = accuracy(dev_preds.to(torch.long), dev_labels).item()

                        f1_micro = f1_score(dev_preds.to(torch.long), dev_labels, average="micro").item()
                        f1_macro = f1_score(
                            dev_preds.to(torch.long),
                            dev_labels,
                            average="macro",
                            multiclass=True,
                            num_classes=5,
                        ).item()

                        mae = 0.0
                        mse = 0.0
                        f1_mae = 0.0
                        f1_samples = 0.0
                    else:
                        average_train_acc = accuracy(preds, labels).item()
                        average_dev_acc = accuracy(dev_preds, dev_labels).item()

                        f1_micro = f1_score(dev_preds, dev_labels, average="micro").item()
                        f1_macro = f1_score(
                            dev_preds,
                            dev_labels,
                            average="macro",
                            multiclass=True,
                            num_classes=2,
                        ).item()

                        mae = 0.0
                        mse = 0.0
                        f1_mae = 0.0
                        f1_samples = 0.0

                    preds_save_path = save_dir / f"preds_{config['seed']}_{epoch}.json"
                    preds_to_json = {
                        "pred": dev_preds.tolist(),
                        "true": dev_labels.tolist(),
                    }

                    if not save_dir.exists():
                        save_dir.mkdir()
                    json.dump(preds_to_json, open(preds_save_path, "w", encoding="utf-8"))

                    running_loss = 0.0
                    running_loss_bin = 0.0
                    running_loss_reg = 0.0
                    preds = []
                    dev_preds = []
                    labels = []
                    labels_reg = []
                    dev_labels = []
                    dev_labels_reg = []

                    stats_message = (
                        f"Epoch [{epoch+1}/{config['num_iters']}], Step [{global_step}/{config['num_iters']*len(train_dataloader)}], "
                        + f"Train Loss: {average_train_loss:.4f}, Dev Loss: {average_dev_loss:.4f}, "
                        + f"Train Loss Binary: {average_train_loss_bin:.4f}, Dev Loss Binary: {average_dev_loss_bin:.4f}, "
                        + f"Train Loss Regression: {average_train_loss_reg:.4f}, Dev Loss Regression: {average_dev_loss_reg:.4f}, "
                        + f"Train Acc/MAE: {average_train_acc:.2f}, Dev Acc/MAE: {average_dev_acc:.2f}, "
                        + f"F1 Micro: {f1_micro*100:.2f}, F1 Macro: {f1_macro*100:.2f}, F1 Samples: {f1_samples*100:.2f}, "
                        + f"LR: {scheduler.get_last_lr()}"
                    )
                    stats_message += f", Dev MAE: {mae:.3f}, Dev MSE: {mse:.3f}, F1 Macro / MAE: {f1_mae:.3f}"
                    accelerator.print(stats_message)

                    if accelerator.is_main_process:
                        f.write(
                            "\t".join(
                                [
                                    str(epoch + 1),
                                    str(average_train_loss),
                                    str(average_train_loss_bin),
                                    str(average_train_loss_reg),
                                    str(average_dev_loss),
                                    str(average_dev_loss_bin),
                                    str(average_dev_loss_reg),
                                    str(average_train_acc),
                                    str(average_dev_acc),
                                    str(f1_micro),
                                    str(f1_macro),
                                    str(mae),
                                    str(mse),
                                ]
                            )
                        )
                        f.write("\n")

                    accelerator.wait_for_everyone()
                    if config["save_every_epoch"]:
                        save_path = save_dir / f"model_{config['seed']}_{epoch}.pt"
                        save_model(model, save_path, accelerator)
                    else:
                        if config["model"]["multilabel"]:
                            save_path = save_dir / f"model_best_loss_{config['seed']}.pt"
                            if average_dev_loss < best_loss:
                                accelerator.print(
                                    f"Dev loss decreased ({best_loss} -> {average_dev_loss}). Saving the model to {save_path}..."
                                )
                                save_model(model, save_path, accelerator)
                                best_loss = average_dev_loss
                        elif config["model"]["regression"]:
                            save_path = save_dir / f"model_best_mae_{config['seed']}.pt"
                            if mae < best_loss:
                                accelerator.print(
                                    f"Dev MAE decreased ({best_loss} -> {mae}). Saving the model to {save_path}..."
                                )
                                save_model(model, save_path, accelerator)
                                best_loss = mae
                        else:
                            save_path = save_dir / f"model_best_f1_{config['seed']}.pt"
                            if f1_macro > best_f1:
                                accelerator.print(
                                    f"Dev F1 increased ({best_f1} -> {f1_macro}). Saving the model to {save_path}..."
                                )
                                save_model(model, save_path, accelerator)
                                best_f1 = f1_macro
    accelerator.end_training()


if __name__ == "__main__":
    main()
