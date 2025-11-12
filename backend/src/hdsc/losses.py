from typing import List, Optional

import torch
import torch.nn as nn


class MultiLabelLoss(nn.Module):
    def __init__(
        self,
        regularization: bool = True,
        l: float = 0.1,
        intervals: Optional[List[int]] = None,
    ):
        super().__init__()
        reduction = "none" if intervals else "mean"
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        self.l = l
        self.reg = regularization
        self.intervals = intervals

    def forward(self, pred, true_1, true_2):
        loss_1 = self.loss_fn(pred, true_1)
        if self.intervals:
            losses = []
            true_sum = torch.sum(true_1, dim=1)
            if len(self.intervals) == 1:
                losses.append(torch.mean(loss_1[true_sum < self.intervals[0]]))
                losses.append(torch.mean(loss_1[true_sum >= self.intervals[0]]))
            else:
                losses.append(torch.mean(loss_1[true_sum <= self.intervals[0]]))
                for i in range(len(self.intervals) - 1):
                    losses.append(
                        torch.mean(loss_1[(true_sum > self.intervals[i]) & (true_sum <= self.intervals[i + 1])])
                    )
                losses.append(torch.mean(loss_1[true_sum > self.intervals[-1]]))
            loss_1 = torch.mean(torch.hstack([loss for loss in losses if not loss.isnan()]))
        loss_2 = torch.dist(torch.sum(pred, dim=1), true_2, p=2)
        if self.reg:
            loss = loss_1 + self.l * loss_2
        else:
            loss = loss_1
        return loss, loss_1, loss_2
