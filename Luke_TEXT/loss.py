import torch
from torch.nn import Module

class NegativeSamplingInBatchLoss(Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, y_pred):
        shape = y_pred.shape[0]
        y_true = torch.eye(shape).to(self.device)
        y_pred = y_pred.to(self.device)

        surat = torch.mul(y_true, y_pred)
        surat = torch.sum(surat, dim=0)
        makhrag = torch.sum(y_pred, dim=0)
        kasr = torch.div(surat, makhrag)

        log = torch.log(kasr)
        sum = torch.sum(log)
        loss = sum / shape

        if torch.is_nonzero(loss):
            loss = -1 * loss
        else:
            loss = torch.tensor(2e-10)

        return loss
