import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
import pandas as pd
from random import sample, seed
import argparse
from sklearn.preprocessing import OneHotEncoder
import neurokit2 as nk
from argparse import Namespace
import numpy as np, pandas as pd
import os
import yaml
from sklearn.mixture import GaussianMixture

TASK_LIST = ["binary"]

LABELS_MULTICLASS_DICT = {0: "stationary", 1: "non-stationary"}


def get_personal_labels(log_variances):
    """
    Automatically finds the threshold between stationary and active
    windows for a single user using a Gaussian Mixture Model.
    """
    # Reshape data for sklearn (requires 2D array)
    X = log_variances.reshape(-1, 1)

    # Fit a 3-component GMM
    # We use 'tied' covariance to assume the clusters have similar widths,
    # which helps prevent the middle cluster from swallowing everything.
    gmm = GaussianMixture(n_components=3, covariance_type='tied', random_state=42)
    gmm_labels = gmm.fit_predict(X)

    # Get the cluster means to identify which cluster is which
    means = gmm.means_.flatten()

    # Sort the indices of the means from lowest to highest variance
    # This tells us which GMM label corresponds to which activity level
    sorted_indices = np.argsort(means)

    stationary_cluster_idx = sorted_indices[0]  # Lowest mean
    middle_cluster_idx = sorted_indices[1]  # Middle mean
    active_cluster_idx = sorted_indices[2]  # Highest mean

    # Create an array to hold our final labels (-1, 0, 1)
    final_labels = np.zeros_like(gmm_labels)

    # Map the GMM labels to our target labels
    final_labels[gmm_labels == stationary_cluster_idx] = 0  # Confident Stationary
    final_labels[gmm_labels == middle_cluster_idx] = -1  # The Muddy Middle (Unlabeled)
    final_labels[gmm_labels == active_cluster_idx] = 1  # Confident Active

    return final_labels

class LarFieldActivity(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset_path, modalities=["PPG", "ACC"]):
        super(LarFieldActivity, self).__init__()
        self.modalities = modalities
        self.samples = []
        self.timestamps = []
        self.participant_list = []
        self.train_or_test = []
        lab_activity, lab_activity_gmm, names = [], [], []


        for prcp in os.listdir(dataset_path):
            if ".DS_Store" not in prcp:
                ppg_temp = np.load(dataset_path + prcp + "/ppg" + "/0.npy")
                ppg = ppg_temp[:, :, :-2]
                acc_temp = np.load(dataset_path + prcp + "/acc" + "/0.npy")


                prcp_timestamps = []

                for i in range(len(ppg)):
                    data_sample = {
                        'PPG': torch.from_numpy(ppg[i])[None,:, :].float(),
                    }
                    self.samples.append(data_sample)

                    lab_activity.append(acc_temp[i][0][-1]) #using the last
                    prcp_timestamps.append(ppg_temp[i][0][-1])
                    self.timestamps.append(ppg_temp[i][0][-1])
                    names.append(prcp)

            dates = pd.to_datetime(prcp_timestamps, unit="s", utc=True).tz_convert("Europe/Warsaw").date
            dates = [str(d) for d in dates]
            dates_unique = sorted(np.unique(dates))

            for d in dates:
                if dates_unique.index(d) <=  int(0.5*len(dates_unique)):
                    if dates_unique.index(d) > int(0.3 * len(dates_unique)):
                        self.train_or_test.append("valid")
                    else:
                        self.train_or_test.append("train")
                else:
                    self.train_or_test.append("test")

        self.participant_list = np.hstack(names)

        y_train_bin = np.hstack(lab_activity)

        print(f"y_train.shape{y_train_bin.shape}")
        print(f"y_train_bin: {y_train_bin}")

        # Pair them up, check if label is NOT -1, and keep the sample
        self.samples = [s for s, l in zip(self.samples, y_train_bin) if l != -1]
        self.timestamps =  np.array([s for s, l in zip(self.timestamps, y_train_bin) if l != -1])
        self.participant_list = np.array([s for s, l in zip(self.participant_list, y_train_bin) if l != -1])

        # Do the same for labels (or just filter the labels list directly)
        filtered_labels = [l for l in y_train_bin if l != -1]
        filtered_labels = torch.Tensor(filtered_labels).float()

        dates = pd.to_datetime(self.timestamps, unit="s",utc=True).tz_convert("Europe/Warsaw").date
        self.labels = {
            "binary": filtered_labels,
        }

        self.set_task("binary")

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
                                          validation_size: int = 2, train_size_ablation: int = None, LOSO=True, across_time=False, cold_start=True) -> ([int], [int], [int]):

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
            test_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if x in test_participants and ttv == "test"]
            val_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if (x in val_participants or x in test_participants) and ttv == "valid"]
            train_idxes = [idx for idx in range(0, len(p_list)) if t_t_v_list[idx] == "train" ]
        elif LOSO and across_time and cold_start:
            test_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if x in test_participants and ttv == "test"]
            val_idxes = [idx for idx, (x, ttv) in enumerate(zip(p_list, t_t_v_list)) if  x in val_participants and ttv == "valid"]
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
            train_idxes =sample(train_idxes, n)

        return train_idxes, test_idxes, val_idxes


def data_generator_larfield_activity(full_dataset, configs, train_idx, valid_idx):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised learning task")
    parser.add_argument("--config", default="", type=str)
    args = parser.parse_args()

    config_file = f"/home/emognition/Desktop/ppg-ssl/downstream_tasks/config/config_larfieldactivity.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    file_config = parser.parse_args()

    config = Namespace(**config)
    config.output_dim = 2
    config.modalities = ["PPG"]

    config.dataset_dir = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/final_dataset/"
    dataset = LarFieldActivity(dataset_path=config.dataset_dir, modalities=config.modalities)
