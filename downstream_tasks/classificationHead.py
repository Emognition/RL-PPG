import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error



class ClassificationHead(LightningModule):
    def __init__(self, args, encoders: {str: LightningModule}, loss_weights = None):
        super().__init__()
        self.save_hyperparameters(args)
        self.accuracy = accuracy_score
        self.batch_size = args.batch_size
        self.modalities = args.modalities
        self.task_type_regression = args.task_type_regression
        self.output_dim = args.output_dim
        self.fusion_strategy = args.fusion_strategy
        self.args = args
        self.loss_weights = torch.tensor(loss_weights)

        # sanity checks
        if self.task_type_regression:
            assert self.output_dim == 1


        if encoders is not None and len(encoders) > 0:
            self.encoders = encoders
            for encoder in self.encoders.values():
                if self.args.backbone_fine_tuning_strategy == "freeze":
                    encoder.eval()
                elif self.args.backbone_fine_tuning_strategy == "finetune":
                    encoder.train()
                encoder.to('cuda')
        else:
            self.encoders = None

        # helper modules
        self.validation_true, self.validation_pred = [], []
        self.test_true, self.test_pred = [], []

    def get_fused_vector(self, x):
        if self.encoders is None:
            all_vectors = {m: x[m] for m in self.modalities}
        else:
            # extract embeddings
            all_vectors = {}
            for modality in self.modalities:
                if 'larfield' in self.args.dataset.lower()  :
                    if modality == 'ECG':
                        agregated_vectors = [
                            self.encoders[modality](x[modality][:, None, element])
                            for element in range(x[modality].shape[1])
                        ]
                    elif modality == 'PPG':
                        agregated_vectors = [
                            self.encoders[modality](x[modality][:, element])
                            for element in range(x[modality].shape[1])
                        ]
                    elif modality == 'ACC':
                        agregated_vectors = [
                            self.encoders[modality](x[modality][:, element, :])
                            for element in range(x[modality].shape[1])
                        ]
                    else:
                        raise NotImplementedError(f"Not handled modality: {modality}")
                    agregated_vectors = [torch.mean(x[1], -1) for x in agregated_vectors]
                    all_vectors[modality] = torch.mean(torch.stack(agregated_vectors, dim=0), dim=0)
                elif self.args.dataset == 'prosi':
                    if modality == 'ECG':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :, :])[1], -1)
                    elif modality == 'PPG':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :, :])[1], -1)
                    elif modality == 'ACC':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :3, :])[1], -1)
                    else:
                        raise NotImplementedError(f"Not handled modality: {modality}")
                elif self.args.dataset == 'wesad':
                    if modality == 'ECG':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :, :])[1], -1)
                    elif modality == 'ACC':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :3, :])[1], -1)
                    else:
                        raise NotImplementedError(f"Not handled modality: {modality}")
                elif self.args.dataset == 'wisdm':
                    if modality == 'ACC':
                        all_vectors[modality] = torch.mean(self.encoders[modality](x[modality][:, :3, :])[1], -1)
                    else:
                        raise NotImplementedError(f"Not handled modality: {modality}")

        if len(self.modalities) > 1:
            # multimodal fusion
            if self.fusion_strategy == "mean":
                fused_vector = torch.stack(list(all_vectors.values()))
                fused_vector = torch.mean(fused_vector, dim=0)
            elif self.fusion_strategy == "WildECG":
                # from WildECG
                fused_vector = torch.cat(all_vectors, dim=-1)
                fused_vector = fused_vector.half() if flag else fused_vector.float()
                fused_vector = fused_vector / fused_vector.norm(dim=1, keepdim=True)
            else:
                fused_vector = torch.cat(list(all_vectors.values()), dim=-1)
        else:
            fused_vector = all_vectors[self.modalities[0]]
        return fused_vector

    def training_step(self, batch, _):
        data, y = batch
        x = {key: data[key] for key in self.modalities}
        preds, y = self.forward(x, y, True)
        loss = self.compute_loss(preds, y)
        self.log("Train/loss", loss.mean().item(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        data, y = batch
        x = {key: data[key] for key in self.modalities}
        preds, y = self.forward(x, y, True)
        loss = self.compute_loss(preds, y, val=True)
        val_true = y.cpu()
        val_pred = preds.cpu()


        preds = val_pred.argmax(dim=1)
        f1 = f1_score(val_true, preds, average="macro", zero_division=0)
        self.log("Valid/f1", f1, sync_dist=True, batch_size=self.batch_size, prog_bar=True)

        acc = accuracy_score(val_true, preds)
        self.log("Valid/acc", acc, sync_dist=True, batch_size=self.batch_size, prog_bar=True)

        # todo: not sure about it; when self.task_type_regression==True
        self.log("Valid/loss", loss.mean().item(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)


    def test_step(self, batch, _):
        data, y = batch
        x = {key: data[key] for key in self.modalities}
        test_pred, y = self.forward(x, y, True)
        test_true = y.cpu()
        test_pred = test_pred.cpu()


        test_pred = test_pred.argmax(dim=1)
        f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
        self.log("Test/f1", f1, sync_dist=True, batch_size=self.batch_size)

        acc = accuracy_score(test_true, test_pred)
        self.log("Test/acc", acc, sync_dist=True, batch_size=self.batch_size)

        pred_path = f"{self.logger.save_dir}_test_pred/{self.logger.name}_test/{self.logger.version}"
        os.makedirs(pred_path, exist_ok=True)
        np.save(f"{pred_path}/predicted_labels_{str(datetime.now()).split(':')[-1]}.npy",
                test_pred.detach().numpy())
        np.save(f"{pred_path}/true_labels_{str(datetime.now()).split(':')[-1]}.npy",
                test_true.detach().numpy())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay),
            # betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
        )
        return {"optimizer": optimizer}

    def compute_loss(self, preds, y, val=False):

        loss = nn.CrossEntropyLoss(weight=self.loss_weights.to('cuda'))

        return loss(preds, y)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
