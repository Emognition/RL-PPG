import torch
from pytorch_lightning import LightningModule
from torch import nn

from downstream_tasks.classificationHead import ClassificationHead

"""
Inspiration:
https://github.com/klean2050/tiles_ecg_model/blob/master/src/trainers/supervised_learning.py
"""


class ClassificationHeadAttention(ClassificationHead):
    def __init__(self, args, encoders: {str: LightningModule}):
        super().__init__(args, encoders)

        # attention dimension
        if self.fusion_strategy == "mean":
            self.att_dim = int(self.hparams.projection_dim)
        else:
            self.att_dim = int(len(self.modalities) * self.hparams.projection_dim)

        # attention layer
        self.query = nn.Linear(self.att_dim, self.att_dim).to('cuda')
        self.key = nn.Linear(self.att_dim, self.att_dim).to('cuda')
        self.value = nn.Linear(self.att_dim, self.att_dim).to('cuda')

        # task projector
        self.projector = nn.Linear(self.att_dim, self.output_dim).to('cuda')

    def forward(self, x, y, flag=False):
        fused_vector = super().get_fused_vector(x)

        # attention mechanism
        q = self.query(fused_vector)
        k = self.key(fused_vector)
        v = self.value(fused_vector)

        weights = torch.matmul(q, k.T)
        weights = (weights / q.shape[-1] ** 0.5).softmax(0)
        fused_vector = torch.matmul(weights, v)

        # return predictions
        preds = self.projector(fused_vector)
        y = y.float() if self.task_type_regression else y.long()

        return preds, y
