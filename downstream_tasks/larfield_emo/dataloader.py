import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from random import sample, seed
from sklearn.preprocessing import OneHotEncoder
import neurokit2 as nk

TASK_LIST = ["intense_emotion"]

LABELS_MULTICLASS_DICT = {0: "no_emotion", 1: "intense_emotion"}


class LarFieldEmo(Dataset):
    def __init__(self, dataset_path, modalities=["PPG"]):
        super(LarFieldEmo, self).__init__()
        self.modalities = modalities
        self.samples = []
        self.timestamps = []
        self.participant_list = []
        self.train_or_test = []
        lab_emo, val, aro, names = [], [], [], []

        for prcp in os.listdir(dataset_path):
            if ".DS_Store" not in prcp:
                ppg_temp = np.load(dataset_path + prcp + "/ppg" + "/0.npy")
                ppg = ppg_temp[:, :, :-4]
                prcp_timestamps = []

                for i in range(len(ppg)):
                    data_sample = {
                        'PPG': torch.from_numpy(ppg[i])[None, :, :].float(),
                    }
                    self.samples.append(data_sample)
                    lab_emo.append(ppg_temp[i][0][-4])
                    val.append(ppg_temp[i][0][-3])
                    aro.append(ppg_temp[i][0][-2])
                    prcp_timestamps.append(ppg_temp[i][0][-1])
                    self.timestamps.append(ppg_temp[i][0][-1])
                    names.append(prcp)

            dates = pd.to_datetime(prcp_timestamps, unit="s", utc=True).tz_convert("Europe/Warsaw").date
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

        self.participant_list = np.hstack(names)

        y_train_bin = np.hstack(lab_emo)
        y_train_val = np.hstack(val)
        y_train_aro = np.hstack(aro)

        print(f"y_train.shape{y_train_bin.shape}")
        print(f"y_train_bin: {y_train_bin}")
        print(f"y_train_val: {y_train_val}")
        print(f"y_train_aro: {y_train_aro}")

        labels_bin = torch.Tensor(y_train_bin).float()
        labels_val = torch.Tensor(y_train_val).float()
        labels_aro = torch.Tensor(y_train_aro).float()

        dates = pd.to_datetime(self.timestamps, unit="s", utc=True).tz_convert("Europe/Warsaw").date
        self.labels = {
            "intense_emotion": labels_bin, "valence": labels_val,
            "arousal": labels_aro,
        }

        self.set_task("intense_emotion")

    def __getitem__(self, index):
        return self.samples[index], torch.Tensor(self.labels[self.task][index]).long()

    def __len__(self):
        return len(self.samples)

    def get_sample_dim(self):

        return self.samples[0]['PPG'].shape[1]

    def get_num_classes(self):
        return self.num_classes

    def set_task(self, task: str) -> None:
        assert task in TASK_LIST, f"Task {task} is not recognized :("
        self.task = task
        self.num_classes = len(set(self.labels[self.task].tolist()))

    def get_participant_list(self) -> [str]:

        return sorted(list(set(self.participant_list)))

    def get_participants_train_test_split(self, test_participants: [str],
                                          validation_size: int = 2, train_size_ablation: int = None, LOSO=True,
                                          across_time=False, cold_start=True) -> ([int], [int], [int]):

        participant_list_copy = self.participant_list[:].tolist()
        p_list = self.participant_list
        t_t_v_list = self.train_or_test

        for participant in test_participants:
            participant_list_copy.remove(participant)
        seed(123)
        val_participants = sample(participant_list_copy, validation_size)
        if LOSO and not across_time:
            test_idxes = [idx for idx, x in enumerate(p_list) if x in test_participants]
            val_idxes = [idx for idx, x in enumerate(p_list) if x in val_participants]
            train_idxes = [idx for idx in range(0, len(p_list)) if
                           idx not in test_idxes and idx not in val_idxes]
        elif LOSO and across_time and not cold_start:
            test_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if
                          x in test_participants and ttv == "test"]
            val_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if
                         (x in val_participants or x in test_participants) and ttv == "valid"]
            train_idxes = [idx for idx in range(0, len(p_list)) if t_t_v_list[idx] == "train"]
        elif LOSO and across_time and cold_start:
            test_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if
                          x in test_participants and ttv == "test"]
            val_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if
                         x in val_participants and ttv == "valid"]
            train_idxes = [idx for idx in range(0, len(p_list)) if
                           p_list[idx] not in test_participants and p_list[idx] not in val_participants and t_t_v_list[
                               idx] == "train"]
        elif not LOSO and across_time and not cold_start:
            test_idxes = [idx for idx, x in enumerate(t_t_v_list) if x == "test" and p_list[idx] in test_participants]
            val_idxes = [idx for idx, x in enumerate(t_t_v_list) if x == "valid" and p_list[idx] in test_participants]
            train_idxes = [idx for idx, x in enumerate(t_t_v_list) if
                           x == "train" and p_list[idx] in test_participants]

        if train_size_ablation is not None:
            n = int(np.floor(len(train_idxes) * (train_size_ablation / 100)))
            train_idxes = sample(train_idxes, n)

        return train_idxes, test_idxes, val_idxes


def data_generator_larfield_emo(full_dataset, configs, train_idx, valid_idx):
    train_dataset = Subset(full_dataset, train_idx)
    print("len train", len(train_dataset))
    valid_dataset = Subset(full_dataset, valid_idx)
    print("len valid", len(valid_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = None
    return train_loader, valid_loader, test_loader
