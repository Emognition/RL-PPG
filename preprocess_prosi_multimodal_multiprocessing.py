from zipfile import ZipFile
import gc
import numpy as np, pandas as pd
import os, neurokit2 as nk
import dask.dataframe as dd
import dask.array as da
import torch
import scipy.io
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import signal_processing_python as spp
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
import datetime
import shutil
from multiprocessing import Process, Manager
import time
import h5py

from scipy.signal import butter, lfilter, medfilt

def butter_bandpass(lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def median_and_butter_bandpass_filter(data, lowcut, fs, order=3):
    data_med = medfilt(data)
    b, a = butter_bandpass(lowcut, fs, order=order)
    y = lfilter(b, a, data_med)
    return y


def which_days(prcp):
    metadata = prcp.replace("ecg", "metadata")
    metadata = pd.read_csv(metadata)
    result = metadata.loc[metadata["rrCoverageRatio"] > 0.9, "record_id"]
    return result.values


def process_prcp_df(df, df_acc, experiment, wanted_experiments, out_path_sessions):
    # load ECG of participant
    # df = pd.read_csv(os.path.join(path, prcp))

    # get days with good coverage
    # days = which_days(os.path.join(path, prcp))
    # if not len(days):
    #    return []

    # filter out days with bad coverage
    # df = df.loc[df["record_id"].isin(days)]

    # correct column notation
    # print("Remove ts")
    bad_quality_spp = 0
    bad_quality_hr = 0
    if "ts" in df.columns:
        df["Timestamp"] = df["ts"]
        df = df.drop(columns=["ts"])
    if "value" in df.columns:
        df["ecg"] = df["value"]
        df = df.drop(columns=["value"])
        # change column name
    if "ts" in df_acc.columns:
        df_acc["Timestamp"] = df_acc["ts"]
        df_acc = df_acc.drop(columns=["ts"])
    # sort based on Timestamp and get ECG
    # print("Sort")
    df = df.set_index("Timestamp").sort_index()["light_intensity"]
    # print("To datetime")

    df.index = pd.to_datetime(df.index, utc=True, format='%d-%b-%Y %H:%M:%S.%f')


    # sort based on Timestamp and get ACC axes
    df_acc = df_acc.set_index("Timestamp").sort_index()[["x", "y", "z"]].astype(
        'float32')

    # set timestamp as index
    df_acc.index = pd.to_datetime(df_acc.index, utc=True,
                                  format='%d-%b-%Y %H:%M:%S.%f')

    # calculate the magnitude of ACC
    df_acc["magnitude"] = np.sqrt(
        df_acc["x"] ** 2 + df_acc["y"] ** 2 + df_acc["z"] ** 2)
    df_acc["processed"] = spp.acc_processing(df_acc[["x", "y", "z"]].to_numpy(),
                                             "polar")
    final_samples = []
    final_timestamps = []
    final_samples_acc = {
        "x": [],
        "y": [],
        "z": [],
        "magnitude": [],
        "processed": [],
        "x_filtered": [],
        "y_filtered": [],
        "z_filtered": [],


    }
    final_timestamps_acc = []

    # print("group_by")
    df_groupby = df.groupby(pd.Grouper(freq='10s',origin=df.index[0]))

    #df_groupby = df.groupby([(df.index - df.index[0]).astype("timedelta64[10s]")])

    # print("prcp_samples")
    prcp_samples = [
        g for _, g in df_groupby
    ]

    # print("prcp_timestamps")
    prcp_timestamps = [
        g.index.values[0] for _, g in df_groupby
    ]

    prcp_samples_acc = []
    prcp_timestamps_acc = []

    # Getting the ACC data based on ECG timestamps
    for p_s in prcp_samples:
        acc_window = df_acc[
            (df_acc.index >= p_s.index[0]) & (df_acc.index <= p_s.index[-1])]
        prcp_samples_acc.append(acc_window)
        if len(acc_window) > 0:
            prcp_timestamps_acc.append(acc_window.index.values[0])
        else:
            prcp_timestamps_acc.append(None)

    i = 0
    # print("del df")
    del df
    gc.collect()
    # print("zaczynamy fora")

    for sample in range(len(prcp_samples)):#tqdm(prcp_samples):
        # if i % 10 == 0:
        try:
            # print("ecg to numpy")
            signal = prcp_samples[sample].to_numpy()
            if sample < len(prcp_samples_acc):
                try:
                    signal_acc = prcp_samples_acc[sample].to_numpy()

                    signals_acc = {
                        "x": prcp_samples_acc[sample]["x"].to_numpy(),
                        "y": prcp_samples_acc[sample]["y"].to_numpy(),
                        "z": prcp_samples_acc[sample]["z"].to_numpy(),
                        "magnitude": prcp_samples_acc[sample]["magnitude"].to_numpy(),
                        "processed": prcp_samples_acc[sample][
                            "processed"].to_numpy(),


                    }
                except Exception as e:
                    print("Error ACC conversion to numpy", e, i, prcp, prcp_samples_acc[sample][:3])

            if len(signal) >= 10 * 23 and len(signal) <= 10 * 27 and len(signal_acc) >= 10 * 45 and len(
                    signal_acc) <= 10 * 55:
                # downsample to 100 Hz
                # print("good_quality_ecg")
                #is_good_quality = spp.good_quality_ecg(signal, 1100.0, 1.0, 15.0)
                is_good_quality = 1
                if is_good_quality == 1:
                    # print("resample")

                    # clean via bandpass filtering
                    # print("clean")
                    sample_ppg = nk.ppg_process(signal, sampling_rate=len(signal) // 10)[0]["PPG_Clean"].values

                    # downsample PPG to 25 Hz
                    sample_ppg = resample(sample_ppg, 10 * 25)

                    # resample ACC to 50 Hz
                    for key, value in signals_acc.items():
                        signals_acc[key] = resample(value, 10 * 50)

                    if hr_based_exclusion:
                        pass

                    else:
                        sample_ppg = np.append(sample_ppg, wanted_experiments[experiment][0])  # multi-class label
                        sample_ppg = np.append(sample_ppg, wanted_experiments[experiment][1])  # binary label
                        final_samples.append(sample_ppg)
                        for key, value in signals_acc.items():
                            value_temp = np.append(value, wanted_experiments[experiment][0])  # multi-class label
                            value_temp = np.append(value_temp, wanted_experiments[experiment][1])  # binary label

                            final_samples_acc[key].append(value_temp)
                        # save timestamps as float
                        final_timestamps.append([prcp_timestamps[sample].astype("float") / 10 ** 9])
                        final_timestamps_acc.append([prcp_timestamps_acc[sample].astype("float") / 10 ** 9])
                else:
                    bad_quality_spp += 1
            else:
                bad_quality_spp += 1
                # print("Bad quality sample (signal processing algorithm)")
        except Exception as e:
            print(e, sample)
        i += 1

    if len(final_samples) > 0:
        final_samples = np.vstack(final_samples)
    return final_samples, final_samples_acc, final_timestamps, final_timestamps_acc, bad_quality_spp, bad_quality_hr


def process_prcp(prcp, wanted_experiments, return_dict, path, out_path_sessions, out_path_sessions_acc,out_path_timestamps, out_path_timestamps_acc):
    print("Participant", prcp)
    print("Loading...", prcp)
    bad_quality_spp = 0
    bad_quality_hr = 0
    df = pd.DataFrame()
    # load ECG data of participant

    if not (f"{prcp}_" in os.listdir(out_path_sessions) or "README" in prcp):
        for experiment in tqdm(os.listdir(path),desc=prcp, leave=True):
            if experiment in wanted_experiments.keys():

                for file in tqdm(os.listdir(path + experiment+"/"),desc=prcp+" "+experiment, leave=True):
                    if "sam_ppg" in file.lower() and ".csv" in file.lower() and prcp + "_" in file.lower():
                        try:
                            df_temp = pd.read_csv(path + experiment+"/"+file, sep=',')
                            df_temp_acc = pd.read_csv(path + experiment+"/"+file.replace("ppg", "acc"), sep=',')
                            # df = pd.concat([df, df_temp])
                            try:
                                # print("Processing...", prcp)

                                # samples = process_prcp(df_temp)
                                samples, samples_acc, timestamps, timestamps_acc, bad_quality_spp_temp, bad_quality_hr_temp = process_prcp_df(
                                    df_temp, df_temp_acc, experiment, wanted_experiments, out_path_sessions)
                                bad_quality_spp += bad_quality_spp_temp
                                bad_quality_hr += bad_quality_hr_temp
                                # print("Saving...", prcp)
                                try:
                                    da.to_npy_stack(
                                        out_path_sessions + prcp + "_" + experiment,
                                        da.from_array(samples,
                                                      chunks=len(samples)))
                                    for key, value in samples_acc.items():
                                        da.to_npy_stack(
                                            out_path_sessions_acc +
                                            prcp + "_" + experiment + f"_acc_{key}",
                                            da.from_array(value,
                                                          chunks=len(value)))
                                    dct = {"Timestamp": timestamps}
                                    timestamps_df = pd.DataFrame(dct)
                                    da.to_npy_stack(
                                        out_path_timestamps + prcp + "_" + experiment + "_ts",
                                        da.from_array(timestamps, chunks=len(
                                            timestamps)))  # , mode="a", header=False)
                                    da.to_npy_stack(
                                        out_path_timestamps_acc +
                                        prcp + "_" + experiment + "_ts",
                                        da.from_array(timestamps_acc,
                                                      chunks=len(
                                                          timestamps_acc)))
                                except Exception as e:
                                    print(e, prcp, file)

                            except Exception as e:
                                print(
                                    f"error '{e}' in processing {path + prcp + '/' + file}")
                        except:
                            print(f"Empty session {experiment} {file}")



    return_dict[prcp] = {"bad_quality_spp": bad_quality_spp, "bad_quality_hr": bad_quality_hr}


if __name__ == "__main__":
    path = "/home/emognition/Desktop/ProSi/"
    out_path_sessions = "/home/emognition/Desktop/ProSi/dataset_sessions_ppg_10_s_acc_fix/"
    out_path_sessions_acc = "/home/emognition/Desktop/ProSi/dataset_sessions_acc_10_s_acc_fix/"
    out_path_timestamps = "/home/emognition/Desktop/ProSi/dataset_timestamps_ppg_10_s_acc_fix/"
    out_path_timestamps_acc = "/home/emognition/Desktop/ProSi/dataset_timestamps_acc_10_s_acc_fix/"
    out_path = "/home/emognition/Desktop/ProSi/dataset_multimodal_ppg_10_s_acc_fix/"
    wanted_experiments = {"Breath":(0,0), "NoMove":(1,0), "Arm":(2,0), "Tap":(3,0), "RunLow":(4,1), "RunMedium":(5,1), "RunHigh":(6,1)}
    hr_based_exclusion = False
    print("Start")


    start = time.time()
    prcp_list = ["p"+str(i) for i in range(1,12)]
    bad_quality_spp_all = 0
    bad_quality_hr_all = 0
    #"""
    os.makedirs(out_path_sessions, exist_ok=True)
    os.makedirs(out_path_timestamps, exist_ok=True)
    os.makedirs(out_path_sessions_acc, exist_ok=True)
    os.makedirs(out_path_timestamps_acc, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    return_dict = {}
    #"""
    manager = Manager()
    return_dict = manager.dict()
    processess = [Process(target=process_prcp, args=(prcp, wanted_experiments, return_dict, path, out_path_sessions, out_path_sessions_acc,out_path_timestamps, out_path_timestamps_acc)) for prcp in prcp_list]
    for process in processess:
        process.start()
    for process in processess:
        process.join()
   # """
    #process_prcp(prcp_list[0], wanted_experiments, return_dict, path, out_path_sessions,  out_path_sessions_acc,out_path_timestamps, out_path_timestamps_acc)
    print('Multiprocessing done', flush=True)
    end_multiprocessing = time.time()
    #"""
    for prcp in  prcp_list:  # top_participants:
        print(prcp)
        try:
            dask_arrays = []
            ts_arrays = []
            dask_arrays_acc = {
                "x": [],
                "y": [],
                "z": [],
                "magnitude": [],
                "processed": [],

            }
            ts_arrays_acc = []
            df_acc = {
                "x": [],
                "y": [],
                "z": [],
                "magnitude": [],
                "processed": [],

            }
            for file in tqdm(os.listdir(out_path_sessions)):
                if prcp+"_" in file:
                    arr = da.from_npy_stack(out_path_sessions + file)
                    dask_arrays.append(arr)
                    arr_ts = da.from_npy_stack(out_path_timestamps + file + "_ts")
                    ts_arrays.append(arr_ts)

                    for key, value in dask_arrays_acc.items():
                        arr_acc_temp = da.from_npy_stack(
                            out_path_sessions_acc + file + f"_acc_{key}")
                        value.append(arr_acc_temp)

                    arr_ts_acc = da.from_npy_stack(
                        out_path_timestamps_acc + file + "_ts")
                    ts_arrays_acc.append(arr_ts_acc)

            df = da.concatenate(dask_arrays, axis=0)
            df_ts = da.concatenate(ts_arrays, axis=0)

            df = da.concatenate(dask_arrays, axis=0)
            df_ts = da.concatenate(ts_arrays, axis=0)
            ppg_mean = da.mean(df[:,:-2])
            ppg_std = da.std(df[:,:-2])
            df[:,:-2] = (df[:,:-2] - ppg_mean) / ppg_std
            # scaler = daskStandardScaler().fit(df_ppg)
            # df_ppg = scaler.transform(df_ppg)
            df = da.hstack((df, df_ts))
            df = df.rechunk(df.shape)

            df_ts_acc = da.concatenate(ts_arrays_acc, axis=0)

            # normalize each axis of ACC within one subject and add timestamp as last value of each axis

            for key, value in dask_arrays_acc.items():
                df_acc[key] = da.concatenate(dask_arrays_acc[key], axis=0)
                if key != "processed":
                    # scaler = daskStandardScaler().fit(df_acc[key])
                    # df_acc[key] = scaler.transform(df_acc[key])
                    acc_one_axis_mean = da.mean(df_acc[key][:,:-2])
                    acc_one_axis_std = da.std(df_acc[key][:,:-2])
                    df_acc[key][:,:-2] = (df_acc[key][:,:-2] - acc_one_axis_mean) / acc_one_axis_std

                df_acc[key] = da.hstack((df_acc[key], df_ts_acc))
                df_acc[key] = df_acc[key].rechunk(df_acc[key].shape)

            df_acc_stacked = da.stack((df_acc["x"], df_acc["y"], df_acc["z"], df_acc["magnitude"], df_acc["processed"]))
            df_acc_stacked = df_acc_stacked.rechunk(df_acc_stacked.shape)



            #"""
            os.makedirs(out_path + prcp, exist_ok=True)
            
            da.to_npy_stack(out_path + prcp + "/ppg/", df)
            da.to_npy_stack(out_path + prcp + "/acc/",
                            df_acc_stacked)
            #"""
        except Exception as e:
            print(e, prcp)
    end_rechunking = time.time()
    """
    print("Multiprocessing", end_multiprocessing-start)
    print("Rechunking", end_rechunking - end_multiprocessing)
    print(return_dict)
    """

