import torch
from torch.nn import Module


class NegativeSamplingInBatchLoss(Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, y_pred: torch.Tensor, y_true: [torch.Tensor]) -> torch.Tensor:
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        index1 = ((y_true == 1).nonzero(as_tuple=True)[0])
        numerator = y_pred[index1]
        denominator = torch.sum(y_pred)

        l = numerator / denominator
        l = -1 * torch.log(l)
        return l
