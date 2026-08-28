import torch.nn as nn
from pytorch_lightning import LightningModule

from downstream_tasks.classificationHead import ClassificationHead


class ClassificationHeadMLP(ClassificationHead):
    def __init__(self, args, encoders: {str: LightningModule}, loss_weights=None):
        super().__init__(args, encoders, loss_weights)

        # attention dimension
        if self.fusion_strategy == "mean":
            self.linear_dim = int(self.hparams.projection_dim)
        else:
            self.linear_dim = int(len(self.modalities) * self.hparams.projection_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.linear_dim, self.linear_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.linear_dim, self.hparams.output_dim, bias=False),
        )

    def forward(self, x, y, flag=False):
        fused_vector = super().get_fused_vector(x)

        # return predictions
        preds = self.linear(fused_vector.cuda())
        y = y.float() if self.task_type_regression else y

        return preds, y
