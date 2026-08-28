import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
import os
import numpy as np
from random import sample, seed
from sklearn.preprocessing import OneHotEncoder
import neurokit2 as nk
from decimal import Decimal, ROUND_HALF_UP

TASK_LIST = [
    "binary", "multiclass",
    "4-sitting", "3-running",
    "3-sitting",
    "binary_without_pb", "multiclass_without_pb",
    "binary_without_s", "multiclass_without_s",
    "3-sitting_without_s",
    "HR_binary", "HR_3",
    "HR_regression"
]

LABELS_MULTICLASS_DICT = {0: "sitting", 1: "paced breathing", 2: "sitting moving arm", 3: "sitting tapping",
                          4: "low pace", 5: "medium pace", 6: "high pace"}


class ProSi(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset_path, modalities=["ECG"]):
        super(ProSi, self).__init__()
        self.modalities = modalities
        self.samples = {
            "all": [],
            "without_pb": [],  # removed class: 1: "paced breathing"
            "without_s": [],  # removed class: 0: "sitting"
            "4-sitting": [], "3-running": [],
            "3-sitting": [],  # removed class: 1: "paced breathing"
            "3-sitting_without_s": []
        }
        self.participant_list = {
            'all': [],
            'without_pb': [],  # removed class: 1: "paced breathing"
            "without_s": [],  # removed class: 0: "sitting"
            "4-sitting": [], "3-running": [],
            "3-sitting_without_s": []
        }

        lab_all, lab_bin, names = [], [], []
        (labels_only_sitting, names_only_sitting, labels_only_running, names_only_running,
         labels_only_3_sitting, names_only_3_sitting) = [], [], [], [], [], []
        lab_without_pb, lab_bin_without_pb, names_without_pb = [], [], []
        lab_without_s, lab_bin_without_s, names_without_s = [], [], []
        lab_without_s_3_sitting, names_without_s_3_sitting = [], []

        for prcp in os.listdir(dataset_path):
            if ".DS_Store" not in prcp:
                ppg_temp = np.load(dataset_path + prcp + "/ppg" + "/0.npy")
                ppg = ppg_temp[:, :-3]
                acc_temp = np.load(dataset_path + prcp + "/acc" + "/0.npy")
                acc_temp = np.transpose(acc_temp, (1, 0, 2))

                acc = acc_temp[:, :, :-3]

                for i in range(len(ppg)):
                    data_sample = {
                        'PPG': torch.from_numpy(ppg[i])[None, :].float(),
                        'ACC': torch.from_numpy(np.vstack(acc[i])).float()
                    }
                    self.samples['all'].append(data_sample)
                    lab_bin.append(ppg_temp[i][-2])
                    i_multiclass_label = ppg_temp[i][-3]
                    lab_all.append(i_multiclass_label)
                    names.append(prcp)

                    if i_multiclass_label != 1:
                        self.samples['without_pb'].append(data_sample)
                        lab_bin_without_pb.append(ppg_temp[i][-2])
                        lab_without_pb.append(i_multiclass_label)
                        names_without_pb.append(prcp)

                    if i_multiclass_label != 0:
                        self.samples['without_s'].append(data_sample)
                        lab_bin_without_s.append(ppg_temp[i][-2])
                        lab_without_s.append(i_multiclass_label)
                        names_without_s.append(prcp)

                    if i_multiclass_label in [0, 1, 2, 3]:
                        self.samples["4-sitting"].append(data_sample)
                        labels_only_sitting.append(i_multiclass_label)
                        names_only_sitting.append(prcp)
                        if i_multiclass_label != 1:
                            self.samples["3-sitting"].append(data_sample)
                            labels_only_3_sitting.append(i_multiclass_label)
                            names_only_3_sitting.append(prcp)
                        if i_multiclass_label != 0:
                            self.samples["3-sitting_without_s"].append(data_sample)
                            lab_without_s_3_sitting.append(i_multiclass_label)
                            names_without_s_3_sitting.append(prcp)

                    elif i_multiclass_label in [4, 5, 6]:
                        i_multiclass_label = i_multiclass_label - 4
                        self.samples["3-running"].append(data_sample)
                        labels_only_running.append(i_multiclass_label)
                        names_only_running.append(prcp)

        self.participant_list['all'] = np.hstack(names)
        self.participant_list['4-sitting'] = np.hstack(names_only_sitting)
        self.participant_list['3-sitting'] = np.hstack(names_only_3_sitting)
        self.participant_list['3-running'] = np.hstack(names_only_running)
        self.participant_list['without_pb'] = np.hstack(names_without_pb)
        self.participant_list['without_s'] = np.hstack(names_without_s)
        self.participant_list['3-sitting_without_s'] = np.hstack(names_without_s_3_sitting)

        y_train_bin = np.hstack(lab_bin)
        y_train_mult = np.hstack(lab_all)

        print(f"y_train.shape{y_train_bin.shape}")
        print(f"y_train_bin: {y_train_bin}")
        print(f"y_train_all: {y_train_mult}")

        labels_bin = torch.Tensor(y_train_bin).float()
        labels_mult = torch.Tensor(y_train_mult).float()

        labels_only_sitting = torch.Tensor(np.hstack(labels_only_sitting)).float()
        labels_only_running = torch.Tensor(np.hstack(labels_only_running)).float()
        labels_only_3_sitting = [x - 1 if x > 1 else x for x in labels_only_3_sitting]
        labels_only_3_sitting = torch.Tensor(np.hstack(labels_only_3_sitting)).float()

        lab_bin_without_pb = torch.Tensor(np.hstack(lab_bin_without_pb)).float()
        lab_without_pb = [x - 1 if x > 1 else x for x in lab_without_pb]
        lab_without_pb = torch.Tensor(np.hstack(lab_without_pb)).float()

        lab_bin_without_s = torch.Tensor(np.hstack(lab_bin_without_s)).float()
        lab_without_s = [x - 1 for x in lab_without_s]
        lab_without_s = torch.Tensor(np.hstack(lab_without_s)).float()

        lab_without_s_3_sitting = [x - 1 for x in lab_without_s_3_sitting]
        lab_without_s_3_sitting = torch.Tensor(np.hstack(lab_without_s_3_sitting)).float()



        self.labels = {
            "binary": labels_bin, "multiclass": labels_mult,
            "binary_without_pb": lab_bin_without_pb, "multiclass_without_pb": lab_without_pb,
            "binary_without_s": lab_bin_without_s, "multiclass_without_s": lab_without_s,
            "4-sitting": labels_only_sitting, "3-running": labels_only_running,
            "3-sitting": labels_only_3_sitting, "3-sitting_without_s": lab_without_s_3_sitting,
        }

        self.set_task("binary")

    def __getitem__(self, index):
        if self.task in ["binary", 'multiclass', 'HR_binary', 'HR_3', 'HR_regression']:
            return self.samples['all'][index], torch.Tensor(self.labels[self.task][index]).long()
        if self.task in ["binary_without_pb", 'multiclass_without_pb']:
            return self.samples['without_pb'][index], torch.Tensor(self.labels[self.task][index]).long()
        if self.task in ["binary_without_s", 'multiclass_without_s']:
            return self.samples['without_s'][index], torch.Tensor(self.labels[self.task][index]).long()
        return self.samples[self.task][index], torch.Tensor(self.labels[self.task][index]).long()

    def __len__(self):
        if self.task in ["binary", 'multiclass', 'HR_binary',  'HR_3', 'HR_regression']:
            return len(self.samples['all'])
        if self.task in ["binary_without_pb", 'multiclass_without_pb']:
            return len(self.samples['without_pb'])
        if self.task in ["binary_without_s", 'multiclass_without_s']:
            return len(self.samples['without_s'])
        return len(self.samples[self.task])

    def get_sample_dim(self):
        if self.task in ["binary", 'multiclass', 'HR_binary', 'HR_3', 'HR_regression']:
            return self.samples['all'][0]['ECG'].shape[1], self.samples['all'][0]['ACC'].shape[1]
        if self.task in ["binary_without_pb", 'multiclass_without_pb']:
            return self.samples['without_pb'][0]['ECG'].shape[1], self.samples['without_pb'][0]['ACC'].shape[1]
        if self.task in ["binary_without_s", 'multiclass_without_s']:
            return self.samples['without_s'][0]['ECG'].shape[1], self.samples['without_s'][0]['ACC'].shape[1]
        return self.samples[self.task][0]['ECG'].shape[1], self.samples[self.task][0]['ACC'].shape[1]

    def get_num_classes(self):
        return self.num_classes

    def set_task(self, task: str) -> None:
        assert task in TASK_LIST, f"Task {task} is not recognized :("
        self.task = task
        self.num_classes = len(set(self.labels[self.task].tolist()))

    def get_participant_list(self) -> [str]:
        if self.task in ["binary", 'multiclass', 'HR_binary', 'HR_3', 'HR_regression']:
            return sorted(list(set(self.participant_list['all'])))
        if self.task in ["binary_without_pb", 'multiclass_without_pb']:
            return sorted(list(set(self.participant_list['without_pb'])))
        return sorted(list(set(self.participant_list[self.task])))

    def get_participants_train_test_split(self, test_participants: [str],
                                          validation_size: int = 2, train_size_ablation: int = None) -> ([int], [int], [int]):
        # sample 2 participants for validation set
        if self.task in ["binary", 'multiclass', 'HR_binary', 'HR_3', 'HR_regression']:
            participant_list_copy = self.participant_list['all'][:].tolist()
            p_list = self.participant_list['all']
        elif self.task in ["binary_without_pb", 'multiclass_without_pb']:
            participant_list_copy = self.participant_list['without_pb'][:].tolist()
            p_list = self.participant_list['without_pb']
        elif self.task in ["binary_without_s", 'multiclass_without_s']:
            participant_list_copy = self.participant_list['without_s'][:].tolist()
            p_list = self.participant_list['without_s']
        else:
            participant_list_copy = self.participant_list[self.task][:].tolist()
            p_list = self.participant_list[self.task]

        for participant in test_participants:
            participant_list_copy.remove(participant)
        seed(123)
        val_participants = sample(participant_list_copy, validation_size)

        test_idxes = [idx for idx, x in enumerate(p_list) if x in test_participants]
        val_idxes = [idx for idx, x in enumerate(p_list) if x in val_participants]
        train_idxes = [idx for idx in range(0, len(p_list)) if
                       idx not in test_idxes and idx not in val_idxes]

        if train_size_ablation is not None:
            n = int(np.floor(len(train_idxes) * (train_size_ablation / 100)))
            train_idxes =sample(train_idxes, n)

        return train_idxes, test_idxes, val_idxes


def data_generator_prosi(full_dataset, configs, train_idx, valid_idx):
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
