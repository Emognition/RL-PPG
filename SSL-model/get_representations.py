import pytorch_lightning as pl
import pytorch_lightning as pl
import torch
from torch import nn, squeeze, mean
from pytorch_lightning import Trainer
import os, argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import neurokit2 as nk


def get_representations(prcp, modality):
    if ".DS_Store" not in prcp:

        final_samples = []

        # load data of a participant

        try:
            data = np.load(os.path.join(path, prcp, modality.lower()))
        except:
            data = np.load(os.path.join(path, prcp, modality.lower()) + "/0.npy",
                           allow_pickle=True)

        with torch.no_grad():
            for sample in tqdm(data, position=0):

                if dataset_name == "ProSi":
                    # for ProSi dataset the last 3 samples are 2 labels and timestamp
                    if modality == "PPG":
                        input = sample[:-3]
                elif dataset_name == "LarFieldEmo":
                    # for LarFieldEmo dataset the last 4 samples are 3 labels (valence, arousal, intense emotion) and timestamp
                    if modality == "PPG":
                        input = sample[:, :-4]
                else:
                    input = sample

                # input shape: [batch_size, channels_num, num_samples]
                if modality == "PPG":
                    if dataset_name == "LarFieldEmo":
                        x_in = torch.from_numpy(input).float()[None, :]

                    else:
                        x_in = torch.from_numpy(input).float()[None, None, :]

                logits, encoding = encoders[modality](x_in)

                encoding_mean = mean(encoding, -1)

                final_samples.append(encoding_mean.cpu())

        return np.vstack(final_samples)
    else:
        return None


if __name__ == "__main__":

    # create args parser and link to Lightning trainer:
    parser = argparse.ArgumentParser(description="demo")
    parser = Trainer.add_argparse_args(parser)

    pl.seed_everything(123, workers=True)

    # TODO: setup model name, run name, dataset name and decide whether to count PPG features.
    model_name = "larfield_ppg_10_sec_full_3e_3_26"
    run_name = "run_1"
    dataset_name = "ProSi"  # "LarFieldEmo"
    # Decide whether to use the last saved checkpoint or the one with the best validation loss
    best_or_last = "best"

    # TODO: set up the datasets paths
    out_paths = {
        "ProSi": f"representations/{model_name}_{run_name}_{best_or_last}/ProSi/",
        "LarFieldEmo": f"representations/{model_name}_{run_name}_{best_or_last}/LarFieldEmo/",

    }
    in_paths = {
        "ProSi": "/home/emognition/Desktop/ProSi/dataset_multimodal_ppg_10_s/",
        "LarFieldEmo": "/home/emognition/Desktop/LarField/test_dataset_emo_ppg/final_dataset/",

    }

    is_multimodal = False
    multimodal_string = "_ppg"
    modalities_to_encode = ["PPG"]

    # backbone path
    load_from = f"/home/emognition/Desktop/RL-PPG/SSL-model/experiments_logs/{model_name}/{run_name}/self_supervised{multimodal_string}_seed_123/saved_models/ckp_{best_or_last}.pt"

    device = torch.device("cpu")  # ("cuda")

    path = in_paths[
        dataset_name]
    out_path = out_paths[
        dataset_name]
    os.makedirs(out_path, exist_ok=True)

    # import files for specific run (the files are copied during training)

    exec(
        f'from experiments_logs.{model_name}.{run_name}.model_files.LarField_Configs import Config as Configs')
    exec(
        f'from experiments_logs.{model_name}.{run_name}.model_files.model_multimodal import base_Model_ppg, base_Model_acc')
    exec(
        f'from experiments_logs.{model_name}.{run_name}.model_files.TC import TC')

    configs = Configs()

    encoders = {}

    if not is_multimodal and modalities_to_encode == ["PPG"]:
        # Load Model
        model_ppg = base_Model_ppg(configs).to(device)

        chkpoint = torch.load(load_from, map_location=device)

        # load weights to PPG encoder
        try:
            pretrained_dict_ppg = chkpoint["model_ecg_state_dict"]  # wrong parameter name, but correct encoder
        except:
            pretrained_dict_ppg = chkpoint["model_state_dict"]
        model_ppg_dict = model_ppg.state_dict()
        model_ppg_dict.update(pretrained_dict_ppg)
        model_ppg.load_state_dict(model_ppg_dict)
        model_ppg.eval()

        encoders["PPG"] = model_ecg

    all_samples = 0

    if dataset_name == "ProSi":
        # Prosi
        for modality in modalities_to_encode:
            for prcp in tqdm(os.listdir(path), position=0):
                print(prcp)
                if f"{prcp.split('.')[0]}.npy" in os.listdir(
                        out_path) or "README" in prcp or "_ts.npy" in prcp:
                    continue
                if f"{prcp.split('.')[0]}.npz" in os.listdir(out_path):
                    continue
                samples = get_representations(prcp, modality)
                if samples is not None:
                    np.savez_compressed(
                        out_path + prcp.split('.')[0] + "_" + modality,
                        representations=samples)
    if dataset_name == "LarFieldEmo":
        # LarFieldEmo
        for modality in modalities_to_encode:
            for prcp in tqdm(os.listdir(path), position=0):
                print(prcp)
                if f"{prcp.split('.')[0]}.npy" in os.listdir(
                        out_path) or "README" in prcp or "_ts.npy" in prcp:
                    continue
                if f"{prcp.split('.')[0]}.npz" in os.listdir(out_path):
                    continue
                samples = get_representations(prcp, modality)
                if samples is not None:
                    np.savez_compressed(
                        out_path + prcp.split('.')[0] + "_" + modality,
                        representations=samples)
