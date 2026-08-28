import torch.nn as nn
from pytorch_lightning import LightningModule

from downstream_tasks.classificationHead import ClassificationHead


class ClassificationHeadLinear(ClassificationHead):
    def __init__(self, args, encoders: {str: LightningModule}):
        super().__init__(args, encoders)

        # attention dimension
        if self.fusion_strategy == "mean":
            self.linear_dim = int(self.hparams.projection_dim)
        else:
            self.linear_dim = int(len(self.modalities) * self.hparams.projection_dim)

        self.linear = nn.Linear(self.linear_dim, self.output_dim).to('cuda')

    def forward(self, x, y, flag=False):
        fused_vector = super().get_fused_vector(x)

        # return predictions
        preds = self.linear(fused_vector)
        y = y.float() if self.task_type_regression else y.long()

        return preds, y
