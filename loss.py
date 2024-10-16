import torch
import torch.nn as nn

from utils import euler_angles_to_matrix

class GeodesicLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        input = input.reshape(-1, 3, 3)
        target = target.reshape(-1, 3, 3)
        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()


class TransformationLoss(nn.Module):
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(TransformationLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = GeodesicLoss()

    def forward(self, prediction, target):
        # Separate translation and rotation components
        translation_prediction = prediction[:, :3]
        translation_target = target[:, :3]
        rotation_prediction = euler_angles_to_matrix(prediction[:, 3:], 'XYZ')
        rotation_target = euler_angles_to_matrix(target[:, 3:], 'XYZ')

        # Calculate translation and rotation losses
        translation_loss = self.translation_loss(translation_prediction, translation_target)
        rotation_loss = self.rotation_loss(rotation_prediction, rotation_target)

        total_loss = (self.translation_weight * translation_loss + self.rotation_weight * rotation_loss).to(torch.float)

        # Weighted sum of translation and rotation losses
        return total_loss