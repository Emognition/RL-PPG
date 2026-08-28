import gc
import numpy as np, pandas as pd
import os, neurokit2 as nk
import dask.array as da
import torch
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import signal_processing_python as spp
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
from multiprocessing import Process, Manager
import time
import h5py

from helpers import load_dataset


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
        return [], []

    valence = emo_dict["assessment"]['response']['valence']
    arousal = emo_dict["assessment"]['response']['arousal']
    emo_timestamp = emo_dict["assessment"]['emotionTimestamp'].replace("T"," ")

    try:
        df = emo_dict["sensors_data"]["HRM_RAW"].reset_index()
    except:
        print(f"No physio data!")
        return [], []
    if "ts" in df.columns:
        df["Timestamp"] = df["ts"]
        df = df.drop(columns=["ts"])
    if "light_intensity" in df.columns:
        df["ppg"] = df["light_intensity"]
        df = df.drop(columns=["light_intensity"])

    # sort based on Timestamp and get ECG
    df = df.set_index("Timestamp").sort_index()["ppg"]

    df.index = pd.to_datetime(df.index, utc=True, format='%Y-%b-%d %H:%M:%S.%f')
    t0 = pd.Timestamp(emo_timestamp, tz="UTC")  # if t0 is a string


    sample = df.loc[
             t0 - pd.Timedelta(seconds=5):
             t0 + pd.Timedelta(seconds=5)
             ]

    final_samples = []
    final_timestamps = []



    i = 0
    del df
    gc.collect()

    try:
        signal = sample.to_numpy()

        if len(signal) >= 10 * 23 and len(signal) <= 10 * 27:
            # clean via bandpass filtering
            sample_ppg = nk.ppg_process(signal, sampling_rate=len(signal) // 10)[0]["PPG_Clean"].values

            # resample PPG to 25 Hz
            sample_ppg = resample(sample_ppg, 10 * 25)
            sample_ppg = np.append(sample_ppg, emotion)  # intense emotion label
            sample_ppg = np.append(sample_ppg, valence)  # valence
            sample_ppg = np.append(sample_ppg, arousal)  # arousal
            sample_ppg = np.append(sample_ppg, float(t0.value / 10 ** 9))  # timestamp
            final_samples.append(sample_ppg)

            # save timestamps as float
            final_timestamps.append([float(t0.value / 10 ** 9)])


    except Exception as e:
        print(e, sample[:1])
    i += 1

    if len(final_samples) > 0:
        final_samples = np.vstack(final_samples)
    return final_samples, final_timestamps


def process_prcp(prcp, wanted_experiments, return_dict, path, out_path_sessions,out_path_timestamps):
    print("Participant", prcp)
    print("Loading...", prcp)
    bad_quality_spp = 0
    bad_quality_hr = 0
    df = pd.DataFrame()
    # load ECG data of participant
    samples = []
    timestamps = []
    if not (f"{prcp}_" in os.listdir(out_path_sessions) or "README" in prcp):
                try:
                    emo_list = load_dataset(path+prcp+"/assessments/emotion.pkl")
                    for emo_dict in emo_list:
                        sample_temp, timestamp_temp = process_prcp_df(emo_dict)
                        if len(sample_temp) > 0:
                            samples.append(sample_temp)
                            timestamps.append(timestamp_temp)

                    try:
                        samples = np.array(samples)
                        timestamps = np.array(timestamps)

                        try:
                            da.to_npy_stack(
                                out_path_sessions + prcp + "_" + "emotions",
                                da.from_array(samples,
                                              chunks=len(samples)))


                            da.to_npy_stack(
                                out_path_timestamps + prcp + "_emotions_ts",
                                da.from_array(timestamps, chunks=len(
                                    timestamps)))  # , mode="a", header=False)

                        except Exception as e:
                            print(e, prcp)

                    except Exception as e:
                        print(
                            f"error '{e}' in processing {path + prcp + '/'}")
                except:
                    print(f"Empty session")



    return_dict[prcp] = {"bad_quality_spp": bad_quality_spp, "bad_quality_hr": bad_quality_hr}


if __name__ == "__main__":
    path = "/home/emognition/Desktop/LarField/test_ppg/"
    out_path_sessions = "/home/emognition/Desktop/LarField/test_dataset_emo_ppg/dataset_sessions_ppg/"
    out_path_timestamps = "/home/emognition/Desktop/LarField/test_dataset_emo_ppg/dataset_timestamps_ppg/"
    out_path = "/home/emognition/Desktop/LarField/test_dataset_emo_ppg/final_dataset/"
    wanted_experiments = {"no_emotion":(0), "intense_emotion":(1)}
    print("Start")


    start = time.time()
    prcp_list = os.listdir(path)

    os.makedirs(out_path_sessions, exist_ok=True)
    os.makedirs(out_path_timestamps, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    return_dict = {}
    #"""
    manager = Manager()
    return_dict = manager.dict()
    processess = [Process(target=process_prcp, args=(prcp, wanted_experiments, return_dict, path, out_path_sessions,out_path_timestamps)) for prcp in prcp_list]
    for process in processess:
        process.start()
    for process in processess:
        process.join()
    print('Multiprocessing done', flush=True)
    end_multiprocessing = time.time()
    for prcp in  prcp_list:
        print(prcp)
        try:
            dask_arrays = []
            ts_arrays = []

            for file in tqdm(os.listdir(out_path_sessions)):
                if prcp+"_" in file:
                    arr = da.from_npy_stack(out_path_sessions + file)
                    dask_arrays.append(arr)
                    arr_ts = da.from_npy_stack(out_path_timestamps + file + "_ts")
                    ts_arrays.append(arr_ts)


            df = da.concatenate(dask_arrays, axis=0)
            df_ts = da.concatenate(ts_arrays, axis=0)
            ecg_mean = da.mean(df[:,:,-4])
            ecg_std = da.std(df[:,:,:-4])
            df[:,:,:-4] = (df[:,:,:-4] - ecg_mean) / ecg_std
            df = da.hstack((df, df_ts))
            df = df.rechunk(df.shape)


            os.makedirs(out_path + prcp, exist_ok=True)

            da.to_npy_stack(out_path + prcp + "/ppg/", df)

        except Exception as e:
            print(e, prcp)
    end_rechunking = time.time()
    print("Multiprocessing", end_multiprocessing-start)
    print("Rechunking", end_rechunking - end_multiprocessing)
    print(return_dict)

