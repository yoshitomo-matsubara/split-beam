from torch import nn
from torchdistill.losses.single import register_single_loss


@register_single_loss
class NormalizedReconstructionLoss(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__(reduction='none')

    def forward(self, preds, targets):
        errors = super().forward(preds, targets)
        normalized_errors = errors.sum(dim=1) / targets.norm(dim=1)
        return normalized_errors.mean()
