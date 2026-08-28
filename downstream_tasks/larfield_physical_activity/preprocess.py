import gc
import argparse
from argparse import Namespace
import numpy as np, pandas as pd
import os, neurokit2 as nk
import yaml
from sklearn.mixture import GaussianMixture
import dask.array as da
import torch
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
import time
import h5py
from helpers import load_dataset
import torch
from multiprocessing import Process, Manager

# Check for NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")
# Fallback to CPU
else:
    device = torch.device("cpu")


def which_days(prcp):
    metadata = prcp.replace("ecg", "metadata")
    metadata = pd.read_csv(metadata)
    result = metadata.loc[metadata["rrCoverageRatio"] > 0.9, "record_id"]
    return result.values


def process_prcp_df(emo_dict):
    if emo_dict["assessment"]['response']['intenseEmotion'] == "NO":
        emotion = 0
    elif emo_dict["assessment"]['response']['intenseEmotion'] == "YES":
        emotion = 1
    else:
        emotion = -1
        return [], [], [], []

    valence = emo_dict["assessment"]['response']['valence']
    arousal = emo_dict["assessment"]['response']['arousal']
    emo_timestamp = emo_dict["assessment"]['emotionTimestamp'].replace("T", " ")

    try:
        df = emo_dict["sensors_data"]["HRM_RAW"].reset_index()
        df_acc = emo_dict["sensors_data"]["ACC"].reset_index()
    except:
        print(f"No physio data!")
        return [], [], [], []
    if "ts" in df.columns:
        df["Timestamp"] = df["ts"]
        df = df.drop(columns=["ts"])
    if "light_intensity" in df.columns:
        df["ppg"] = df["light_intensity"]
        df = df.drop(columns=["light_intensity"])
        # change column name

    if "ts" in df_acc.columns:
        df_acc["Timestamp"] = df_acc["ts"]
        df_acc = df_acc.drop(columns=["ts"])
    # sort based on Timestamp and get ECG
    df = df.set_index("Timestamp").sort_index()["ppg"]
    df_acc = df_acc.set_index("Timestamp").sort_index()[["x", "y", "z"]]

    df.index = pd.to_datetime(df.index, utc=True, format='%Y-%b-%d %H:%M:%S.%f')
    t0 = pd.Timestamp(emo_timestamp, tz="UTC")  # if t0 is a string

    df_acc.index = pd.to_datetime(df_acc.index, utc=True, format='%Y-%b-%d %H:%M:%S.%f')

    sample = df.loc[
             t0 - pd.Timedelta(seconds=5):
             t0 + pd.Timedelta(seconds=5)
             ]
    sample_acc = df_acc.loc[
                 t0 - pd.Timedelta(seconds=5):
                 t0 + pd.Timedelta(seconds=5)
                 ]

    final_samples = []
    final_timestamps = []
    final_samples_acc = []
    final_timestamps_acc = []

    i = 0
    del df
    gc.collect()

    try:
        signal = sample.to_numpy()
        signal_acc = sample_acc.to_numpy()

        if len(signal) >= 10 * 23 and len(signal) <= 10 * 27:
            # downsample to 100 Hz

            # print("resample")
            # clean via bandpass filtering
            # sample_ppg = nk.ppg_clean(signal_ppg, sampling_rate=100)
            # downsample PPG to 50 Hz
            # resample ACC to 50 Hz
            sample_acc = np.zeros((10 * 50 + 2, 3))
            for dim in range(signal_acc.shape[1]):
                sample_acc[:, dim][:500] = resample(signal_acc[:, dim][:500], 10 * 50)
            # sample_acc = resample(signal_acc, 10 * 50)
            activity_label = 0
            for dim in range(signal_acc.shape[1]):
                sample_acc[:, dim] = np.append(sample_acc[:, dim][:500], [activity_label, 0])  # binary activity label
                sample_acc[:, dim] = np.append(sample_acc[:, dim][:501], float(t0.value / 10 ** 9))  # timestamp

            # save timestamps as float

            sample_ppg = nk.ppg_process(signal, sampling_rate=len(signal) // 10)[0]["PPG_Clean"].values

            # downsample PPG to 25 Hz
            sample_ppg = resample(sample_ppg, 10 * 25)
            sample_ppg = np.append(sample_ppg, activity_label)  # physical activity label
            sample_ppg = np.append(sample_ppg, float(t0.value / 10 ** 9))  # timestamp
            final_samples.append(sample_ppg)

            final_samples_acc.append(sample_acc)

            final_timestamps.append([float(t0.value / 10 ** 9)])
            final_timestamps_acc.append([float(t0.value / 10 ** 9)])


    except Exception as e:
        print(e, sample[:1])
    i += 1

    if len(final_samples) > 0:
        final_samples = np.vstack(final_samples)
        final_samples_acc = np.vstack(final_samples_acc)
    return final_samples, final_timestamps, final_samples_acc, final_timestamps_acc


def process_prcp(prcp, wanted_experiments, return_dict, path, out_path_sessions, out_path_timestamps, config):
    print("Participant", prcp)
    print("Loading...", prcp)
    bad_quality_spp = 0
    bad_quality_hr = 0
    df = pd.DataFrame()
    # load ECG data of participant
    samples = []
    timestamps = []
    samples_acc = []
    timestamps_acc = []
    if not (f"{prcp}_" in os.listdir(out_path_sessions) or "README" in prcp):
        try:
            emo_list = load_dataset(path + prcp + "/assessments/emotion.pkl")
            for emo_dict in emo_list:
                sample_temp, timestamp_temp, sample_acc_temp, timestamp_acc_temp = process_prcp_df(emo_dict)
                if len(sample_temp) > 0:
                    samples.append(sample_temp)
                    timestamps.append(timestamp_temp)
                    samples_acc.append(sample_acc_temp)
                    timestamps_acc.append(timestamp_acc_temp)

            try:
                # print("Processing...", prcp)

                # samples = process_prcp(df_temp)
                samples = np.array(samples)
                timestamps = np.array(timestamps)
                samples_acc = np.array(samples_acc)
                timestamps_acc = np.array(timestamps_acc)

                try:
                    da.to_npy_stack(
                        out_path_sessions + prcp + "_" + "ppg_activity",
                        da.from_array(samples,
                                      chunks=len(samples)))

                    da.to_npy_stack(
                        out_path_timestamps + prcp + "_ppg_activity_ts",
                        da.from_array(timestamps, chunks=len(
                            timestamps)))  # , mode="a", header=False)

                    for dim in range(samples_acc.shape[2]):
                        da.to_npy_stack(
                            out_path_sessions_acc + prcp + f"_{dim}_acc_activity",
                            da.from_array(samples_acc[:, :, dim], chunks=len(samples_acc[:, :, dim])))

                    da.to_npy_stack(
                        out_path_timestamps_acc + prcp + "_acc_activity_ts",
                        da.from_array(timestamps_acc, chunks=len(
                            timestamps)))  # , mode="a", header=False)

                except Exception as e:
                    print(e, prcp)

            except Exception as e:
                print(
                    f"error '{e}' in processing {path + prcp + '/'}")
        except Exception as e:
            print(f"{prcp}: Empty session, {e}")

    return_dict[prcp] = {"bad_quality_spp": bad_quality_spp, "bad_quality_hr": bad_quality_hr}


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


if __name__ == "__main__":
    path = "/home/emognition/Desktop/LarField/test_ppg/"
    out_path_sessions = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/dataset_sessions_ppg/"
    out_path_timestamps = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/dataset_timestamps_ppg/"
    out_path_sessions_acc = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/dataset_sessions_acc/"
    out_path_timestamps_acc = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/dataset_timestamps_acc/"
    out_path = "/home/emognition/Desktop/LarField/test_ppg_acc_physical_activity/final_dataset/"
    wanted_experiments = {"no_emotion": (0), "intense_emotion": (1)}
    print("Start")
    # TODO: set the paths in the config file
    # config parser
    parser = argparse.ArgumentParser(description="Supervised learning task")
    parser.add_argument("--config", default="", type=str)
    args = parser.parse_args()

    config_file = f"/home/emognition/Desktop/ppg-ssl/downstream_tasks/config/config_wisdm.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    file_config = parser.parse_args()

    start = time.time()
    prcp_list = os.listdir(path)

    os.makedirs(out_path_sessions, exist_ok=True)
    os.makedirs(out_path_timestamps, exist_ok=True)
    os.makedirs(out_path_sessions_acc, exist_ok=True)
    os.makedirs(out_path_timestamps_acc, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    return_dict = {}
    config = Namespace(**config)
    config.output_dim = 2

    # """

    manager = Manager()
    return_dict = manager.dict()
    processess = [Process(target=process_prcp, args=(
    prcp, wanted_experiments, return_dict, path, out_path_sessions, out_path_timestamps, config)) for prcp in prcp_list]
    for process in processess:
        process.start()
    for process in processess:
        process.join()

    print('Multiprocessing done', flush=True)
    end_multiprocessing = time.time()
    for prcp in prcp_list:  # top_participants:
        print(prcp)
        try:
            dask_arrays = []
            ts_arrays = []
            dask_arrays_acc = {
                "0": [],
                "1": [],
                "2": [],

            }
            ts_arrays_acc = []
            df_acc = {
                "0": [],
                "1": [],
                "2": [],

            }
            for file in tqdm(os.listdir(out_path_sessions)):
                if prcp + "_" in file:
                    arr = da.from_npy_stack(out_path_sessions + file)
                    dask_arrays.append(arr)
                    arr_ts = da.from_npy_stack(out_path_timestamps + file + "_ts")
                    ts_arrays.append(arr_ts)

            for file in tqdm(os.listdir(out_path_sessions_acc)):
                for dim in ["0", "1", "2"]:
                    if prcp + "_" + dim in file:
                        dask_arrays_acc[dim].append(
                            da.from_npy_stack(out_path_sessions_acc + file))

            df = da.concatenate(dask_arrays, axis=0)
            ecg_mean = da.mean(df[:, :, -2])
            ecg_std = da.std(df[:, :, :-2])
            df[:, :, :-2] = (df[:, :, :-2] - ecg_mean) / ecg_std

            df = df.rechunk(df.shape)

            for key, value in dask_arrays_acc.items():
                df_acc[key] = da.concatenate(dask_arrays_acc[key], axis=0)
                if key != "processed":
                    # scaler = daskStandardScaler().fit(df_acc[key])
                    # df_acc[key] = scaler.transform(df_acc[key])
                    acc_one_axis_mean = da.mean(df_acc[key][:, :-2])
                    acc_one_axis_std = da.std(df_acc[key][:, :-2])
                    df_acc[key][:, :-2] = (df_acc[key][:, :-2] - acc_one_axis_mean) / acc_one_axis_std

                # df_acc[key] = da.hstack((df_acc[key], df_ts_acc))
                df_acc[key] = df_acc[key].rechunk(df_acc[key].shape)

            df_acc_stacked = da.stack((df_acc["0"], df_acc["1"], df_acc["2"]))
            df_acc_stacked = df_acc_stacked.rechunk(df_acc_stacked.shape)
            df_acc_stacked = df_acc_stacked.transpose(1, 0, 2)

            acc_temp = df_acc_stacked.compute()
            acc_temp = acc_temp[:, :, :-2]

            x = acc_temp[:, 0, :]
            y = acc_temp[:, 1, :]
            z = acc_temp[:, 2, :]

            # Calculate magnitude. The new shape will be (num_windows, window_length)
            magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            # ==========================================
            # Step 2: Calculate the variance of the magnitude for each window
            # ==========================================
            # Calculate variance along the time-step axis (axis=1)
            variances = np.var(magnitude, axis=1)
            # 1. Add a tiny constant to avoid log(0)
            safe_variances = variances + 1e-6

            # 2. Calculate the log of the variances
            log_variances = np.log(safe_variances)

            labels = get_personal_labels(log_variances)
            # Reshape labels from (N,) to (N, 1, 1)
            labels_reshaped = labels[:, None, None]

            # Broadcast the reshaped labels to (N, D, 1) so it aligns with the main array
            labels_broadcast = da.broadcast_to(labels_reshaped, (len(labels), 3, 1))

            # Concatenate along the last axis (axis=2)
            # Resulting shape will be (N, D, X + 1)
            result = da.concatenate([df_acc_stacked, labels_broadcast], axis=2)
            os.makedirs(out_path + prcp, exist_ok=True)

            da.to_npy_stack(out_path + prcp + "/ppg/", df)
            da.to_npy_stack(out_path + prcp + "/acc/", result)
            del (df)
            del (df_acc)
            del (arr)
            del (arr_ts)
        except Exception as e:
            print(e, prcp)
    end_rechunking = time.time()
    print("Multiprocessing", end_multiprocessing - start)
    print("Rechunking", end_rechunking - end_multiprocessing)
    print(return_dict)
