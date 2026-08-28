import gc

import pandas as pd
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import numpy as np
from .augmentations import DataTransform
import random

class Larfield(Dataset):
    def __init__(self, dataset_path, config, training_mode):
        super(Larfield, self).__init__()
        self.training_mode = training_mode
        self.config = config
        self.path = dataset_path
        self.samples = []
        self.samples_acc = []
        self.timestamps = []
        self.train_or_test = []

        # added temporary loading both acc and ppg data to make sure we exclude the same samples in pretraining

        data_ppg = np.load(dataset_path + f"/ppg/0.npy")
        data_acc = np.load(dataset_path + f"/acc/0.npy")
        data_acc = np.transpose(data_acc,(1, 0, 2))


        # load ECG data of participant, the last value is timestamp, so we skip it
        if isinstance(data_ppg, np.ndarray):
            self.samples = torch.from_numpy(data_ppg)[:, :-1].float()
            self.timestamps = torch.from_numpy(data_ppg)[:, -1].float()

        else:
            for s in data_ppg:
                self.samples.append(torch.tensor(s[:-1].copy()).float())
                self.timestamps.append(torch.tensor(s[-1].copy()).float())
        if isinstance(data_acc, np.ndarray):
            self.samples_acc = torch.from_numpy(data_acc)[:, config.acc_axis, :-1].float()
        else:
            for s in data_acc:
                self.samples_acc.append(torch.tensor(s[config.acc_axis, :-1].copy()).float())

        # Dimensionality check ECG
        if len(self.samples.shape) < 3:
            self.samples = self.samples.unsqueeze(2)
        if self.samples.shape.index(min(self.samples.shape)) != 1:  # make sure the Channels in second dim
            self.samples = self.samples.permute(0, 2, 1)

        # Dimensionality check ACC
        if len(self.samples_acc.shape) < 3:
            self.samples_acc = self.samples_acc.unsqueeze(2)
        if self.samples_acc.shape.index(min(self.samples_acc.shape)) != 1:  # make sure the Channels in second dim
            self.samples_acc = self.samples_acc.permute(0, 2, 1)

        dates = pd.to_datetime( self.timestamps, unit="s", utc=True).tz_convert("Europe/Warsaw").date
        dates = [str(d) for d in dates]
        dates_unique = sorted(np.unique(dates))

        for d in dates:
            if dates_unique.index(d) <= int(0.5 * len(dates_unique)):
                if dates_unique.index(d) > int(0.3 * len(dates_unique)):
                    self.train_or_test.append("valid")
                else:
                    self.train_or_test.append("train")
            else:
                self.train_or_test.append("test")

        # Check and remove if there are nans - temporary, will fix this in data processing
        mask = np.ones(len(self.samples_acc), dtype=bool)
        wrong_idx = np.argwhere(np.isnan(self.samples_acc).any(axis=-1))
        if wrong_idx.shape[-1] > 0:
            for wi in wrong_idx[0]:
                mask[wi] = False
            self.samples_acc = self.samples_acc[mask, ...]
            self.samples = self.samples[mask, ...]
            self.timestamps = self.timestamps[mask, ...]
            self.train_or_test = self.train_or_test[mask, ...]

        # not used later
        del self.samples_acc

    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
        x_data = self.samples[idx]
        if "self_supervised" in self.training_mode:
            aug1, aug2 = DataTransform(x_data[None, :, :], self.config)
            return x_data, torch.Tensor([-1]).long(), aug1.squeeze(axis=0), aug2.squeeze(axis=0)
        else:
            return x_data, torch.Tensor([-1]).long(), x_data, x_data


class Larfield_multimodal(Dataset):
    def __init__(self, dataset_path, config, training_mode):
        super(Larfield_multimodal, self).__init__()
        self.training_mode = training_mode
        self.config = config
        self.path = dataset_path
        self.samples_ppg = []
        self.samples_acc = []
        self.features_ppg = []
        self.features_acc = []
        data_ppg = np.load(dataset_path + f"/ppg/0.npy")
        data_acc = np.load(dataset_path + f"/acc/0.npy")
        data_acc = np.transpose(data_acc,(1, 0, 2))


        # load ECG data of participant, the last value is timestamp, so we skip it
        if isinstance(data_ppg, np.ndarray):
            if "feature" in self.training_mode:
                self.samples_ppg = torch.from_numpy(data_ppg)[:, :-6].float()
                self.features_ppg = torch.from_numpy(np.nan_to_num(data_ppg,  nan=-10, posinf=-10, neginf=-10))[:, -4:-3].float()
            else:
                self.samples_ppg = torch.from_numpy(data_ppg)[:, :-1].float()

        else:

            if "feature" in self.training_mode:
                for s in data_ppg:
                    self.samples_ppg.append(torch.tensor(s[:-6].copy()).float())
                    self.features_ppg.append(torch.tensor(np.nan_to_num(s[-5:].copy(), nan=-10, posinf=-10, neginf=-10))[:, -4:-3].float())
            else:
                for s in data_ppg:
                    self.samples_ppg.append(torch.tensor(s[:-1].copy()).float())

        del(data_ppg)
        gc.collect()
        if isinstance(data_acc, np.ndarray):
            if "feature" in self.training_mode:
                self.samples_acc = torch.from_numpy(data_acc)[:, config.acc_axis, :-6].float()
                self.features_acc = torch.from_numpy(np.nan_to_num(data_acc, nan=-10, posinf=-10, neginf=-10))[:, 0, -4:-3].float() # features are the same for all axis
            else:
                self.samples_acc = torch.from_numpy(data_acc)[:, config.acc_axis, :-1].float()
        else:
            if "feature" in self.training_mode:
                for s in data_acc:
                    self.samples_acc.append(torch.tensor(s[config.acc_axis, :-6].copy()).float())
                    self.features_acc.append(torch.tensor(np.nan_to_num(s[0,-5:].copy(), nan=-10, posinf=-10, neginf=-10))[:, 0, -4:-3].float())
            else:
                for s in data_acc:
                    self.samples_acc.append(torch.tensor(s[config.acc_axis, :-1].copy()).float())
        del(data_acc)
        gc.collect()
        # Dimensionality check ECG
        if len(self.samples_ppg.shape) < 3:
            self.samples_ppg = self.samples_ppg.unsqueeze(2)
        if self.samples_ppg.shape.index(min(self.samples_ppg.shape)) != 1:  # make sure the Channels in second dim
            self.samples_ppg = self.samples_ppg.permute(0, 2, 1)

        # Dimensionality check ACC
        if len(self.samples_acc.shape) < 3:
            self.samples_acc = self.samples_acc.unsqueeze(2)
        if self.samples_acc.shape.index(min(self.samples_acc.shape)) != 1:  # make sure the Channels in second dim
            self.samples_acc = self.samples_acc.permute(0, 2, 1)

        # Check and remove if there are nans - temporary, will fix this in data processing
        mask = np.ones(len(self.samples_acc), dtype=bool)
        wrong_idx = np.argwhere(np.isnan(self.samples_acc).any(axis=-1))
        if wrong_idx.shape[-1] > 0:
            for wi in wrong_idx[0]:
                mask[wi] = False
            self.samples_acc = self.samples_acc[mask, ...]
            self.samples_ppg = self.samples_ppg[mask, ...]


    def __len__(self):
        return len(self.samples_ppg)

    def __getitem__(self, idx):
        ppg_data = self.samples_ppg[idx]
        acc_data = self.samples_acc[idx]

        if self.training_mode == "self_supervised_multimodal":
            return ppg_data, acc_data, ppg_data, acc_data
        elif "feature" in self.training_mode:
            ppg_features = self.features_ppg[idx]
            acc_features = self.features_acc[idx]
            return ppg_data, acc_data, ppg_features, acc_features
        else:
            return ppg_data, acc_data, ppg_data, acc_data


class Larfield_acc(Dataset):
    def __init__(self, dataset_path, config, training_mode):
        super(Larfield_acc, self).__init__()
        self.training_mode = training_mode
        self.config = config
        self.path = dataset_path
        self.samples = []

        # added temporary loading both acc and ppg data to make sure we exclude the same samples in pretraining

        data_acc = np.load(dataset_path + f"/acc/0.npy")
        data_acc = np.transpose(data_acc,(1, 0, 2))


        if isinstance(data_acc, np.ndarray):
            self.samples = torch.from_numpy(data_acc)[:, config.acc_axis, :-1].float()
        else:
            for s in data_acc:
                self.samples.append(torch.tensor(s[config.axis, :-1].copy()).float())
        # Dimensionality check ACC
        if len(self.samples.shape) < 3:
            self.samples_acc = self.samples.unsqueeze(2)
        if self.samples.shape.index(min(self.samples.shape)) != 1:  # make sure the Channels in second dim
            self.samples = self.samples_acc.permute(0, 2, 1)

        # Check and remove if there are nans - temporary, will fix this in data processing
        mask = np.ones(len(self.samples), dtype=bool)
        wrong_idx = np.argwhere(np.isnan(self.samples).any(axis=-1))
        if wrong_idx.shape[-1] > 0:
            for wi in wrong_idx[0]:
                mask[wi] = False
            self.samples = self.samples[mask, ...]


    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
        x_data = self.samples[idx]
        if "self_supervised" in self.training_mode:
            aug1, aug2 = DataTransform(x_data[None, :], self.config)
            return x_data, torch.Tensor([-1]).long(), aug1.squeeze(axis=0), aug2.squeeze(axis=0)
        else:
            return x_data, torch.Tensor([-1]).long(), x_data, x_data

def data_generator_larfield(data_path, configs, training_mode, train_size_ablation=None):
    if "feature" in training_mode:
        # TODO change it
        train_sp = [data_path + "train_features/" + p for p in os.listdir(data_path + "train_features")]
        valid_sp = [data_path + "valid_features/" + p for p in os.listdir(data_path + "valid_features")]
    elif "personal" in training_mode:
        # personal models for people from test set, trained only on 30% od their initial days
        prcp_id = training_mode.split("personal_")[1]
        train_sp = [data_path + "test_ppg/" + prcp_id]
        valid_sp = [data_path + "test_ppg/" + prcp_id]

    else:
        train_sp = [data_path + "train_ppg/" + p for p in os.listdir(data_path + "train_ppg")]
        valid_sp = [data_path + "valid_ppg/" + p for p in os.listdir(data_path + "valid_ppg")]


    if train_size_ablation is not None:
        # Set the random seed for reproducibility
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        n = int(np.floor(len(train_sp) * (train_size_ablation / 100)))
        train_sp = random.sample(train_sp, n)
    all_datasets_train = []
    all_datasets_valid = []
    if "multimodal" in training_mode:
        for train_p in train_sp:
            all_datasets_train.append(Larfield_multimodal(train_p, configs, training_mode))
        for valid_p in valid_sp:
            all_datasets_valid.append(Larfield_multimodal(valid_p, configs, training_mode))
    elif "acc" in training_mode:
        for train_p in train_sp:
            all_datasets_train.append(Larfield_acc(train_p, configs, training_mode))
        for valid_p in valid_sp:
            all_datasets_valid.append(Larfield_acc(valid_p, configs, training_mode))
    else:
        for train_p in train_sp:
            all_datasets_train.append(
                Larfield(train_p, configs, training_mode))
        for valid_p in valid_sp:
            all_datasets_valid.append(
                Larfield(valid_p, configs, training_mode))

    final_train_dataset = ConcatDataset(all_datasets_train)
    final_valid_dataset = ConcatDataset(all_datasets_valid)
    if "personal" in training_mode:
        train_idx = [idx for idx, x in enumerate(all_datasets_train[0].train_or_test) if x == "train"]
        final_train_dataset = Subset(final_train_dataset, train_idx)
        valid_idx = [idx for idx, x in enumerate(all_datasets_valid[0].train_or_test) if x == "valid"]
        final_valid_dataset = Subset(final_valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=final_train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=final_valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = None
    return train_loader, valid_loader, test_loader


