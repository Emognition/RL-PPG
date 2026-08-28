from zipfile import ZipFile
import gc
import numpy as np, pandas as pd
import os, neurokit2 as nk
import dask.array as da
from scipy.signal import resample
from tqdm import tqdm
import signal_processing_python as spp
from dask_ml.preprocessing import StandardScaler as daskStandardScaler
import multiprocessing
import time


def process_prcp_df(df_ppg, df_acc, df_hrm, prcp):
    bad_quality_spp = 0
    bad_quality_hr = 0
    length_out_of_range = 0
    total_time_rejected_samples = 0
    total_time_accepted_samples = 0

    # change column name
    if "ts" in df_ppg.columns:
        df_ppg["Timestamp"] = df_ppg["ts"]
        df_ppg = df_ppg.drop(columns=["ts"])

    # sort based on Timestamp and get ECG
    if df_ppg.index.name != "Timestamp":
        df_ppg = df_ppg.set_index("Timestamp")
    df_ppg = df_ppg.sort_index()["light_intensity"]

    # set timestamp as index
    df_ppg.index = pd.to_datetime(df_ppg.index, utc=True, format='%Y-%m-%dT%H:%M:%S:%f')

    # change column name
    if "ts" in df_hrm.columns:
        df_hrm["Timestamp"] = df_hrm["ts"]
        df_hrm = df_hrm.drop(columns=["ts"])

    # sort based on Timestamp and get ECG
    if df_hrm.index.name != "Timestamp":
        df_hrm = df_hrm.set_index("Timestamp")
    df_hrm = df_hrm.sort_index()["v1"]

    # set timestamp as index
    df_hrm.index = pd.to_datetime(df_hrm.index, utc=True, format='%Y-%m-%dT%H:%M:%S:%f')


    # change column name
    if "ts" in df_acc.columns:
        df_acc["Timestamp"] = df_acc["ts"]
        df_acc = df_acc.drop(columns=["ts"])

    # sort based on Timestamp and get ACC axes
    if df_acc.index.name != "Timestamp":
        df_acc = df_acc.set_index("Timestamp")
    df_acc = df_acc.sort_index()[["x", "y", "z"]].astype('float32')


    # set timestamp as index
    df_acc.index = pd.to_datetime(df_acc.index, utc=True, format='%Y-%m-%dT%H:%M:%S:%f')

    # calculate the magnitude of ACC
    df_acc["magnitude"] = np.sqrt(df_acc["x"] ** 2 + df_acc["y"] ** 2 + df_acc["z"] ** 2)
    df_acc["processed"] = spp.acc_processing(df_acc[["x", "y", "z"]].to_numpy(),
                                             "samsung")

    final_samples = []
    final_timestamps = []
    final_samples_acc = {
        "x": [],
        "y": [],
        "z": [],
        "magnitude": [],
        "processed": []
    }
    final_timestamps_acc = []

    # group data into 10 seconds long windows
    df_groupby = df_ppg.groupby(pd.Grouper(freq="10s"))

    #df_groupby = df_ppg.groupby([(df_ppg.index - df_ppg.index[0]).astype("timedelta64[10s]")])

    prcp_samples_ppg = [
        g for _, g in df_groupby
    ]
    prcp_timestamps_ppg = [
        g.index.values[0] for _, g in df_groupby
    ]

    prcp_samples_acc = []
    prcp_timestamps_acc = []
    prcp_samples_hrm = []
    prcp_timestamps_hrm = []
    # Getting the ACC data based on ECG timestamps
    for p_s in prcp_samples_ppg:
        acc_window = df_acc[(df_acc.index >= p_s.index[0]) & (df_acc.index <= p_s.index[-1])]
        hrm_window = df_hrm[(df_hrm.index >= p_s.index[0]) & (df_hrm.index <= p_s.index[-1])]
        prcp_samples_acc.append(acc_window)
        prcp_samples_hrm.append(hrm_window)
        if len(acc_window) > 0:
            prcp_timestamps_acc.append(acc_window.index.values[0])
        else:
            prcp_timestamps_acc.append(None)
        if len(hrm_window) > 0:
            prcp_timestamps_hrm.append(hrm_window.index.values[0])
        else:
            prcp_timestamps_hrm.append(None)

    i = 0

    # delete unused variables to free up some space
    del df_ppg
    del df_acc
    del df_hrm
    gc.collect()

    for sample in range(len(prcp_samples_ppg)):
        try:
            # convert signals to numpy ndarray
            try:
                signal_ppg = prcp_samples_ppg[sample].to_numpy()
            except Exception as e:
                print("Error PPG conversion to numpy", e, i, prcp, prcp_samples_ppg[sample][:3])
            if sample < len(prcp_samples_hrm):
                try:
                    signal_hrm = prcp_samples_hrm[sample].to_numpy()
                except Exception as e:
                    print("Error HRM conversion to numpy", e, i, prcp, prcp_samples_hrm[sample][:3])

            if sample < len(prcp_samples_acc):
                try:
                    signal_acc = prcp_samples_acc[sample].to_numpy()
                    if np.isnan(signal_acc.T).any() or np.isnan(signal_ppg).any():
                        length_out_of_range += 1
                        total_time_rejected_samples += len(signal_ppg) / 130
                        continue
                    signals_acc = {
                        "x": prcp_samples_acc[sample]["x"].to_numpy(),
                        "y": prcp_samples_acc[sample]["y"].to_numpy(),
                        "z": prcp_samples_acc[sample]["z"].to_numpy(),
                        "magnitude": prcp_samples_acc[sample]["magnitude"].to_numpy(),
                        "processed": prcp_samples_acc[sample]["processed"].to_numpy(),

                    }
                except Exception as e:
                    print("Error ACC conversion to numpy", e, i, prcp, prcp_samples_acc[sample][:3])

            # reject signals with messed sampling rate
            if len(signal_ppg) >= 10 * 23 and len(signal_ppg) <= 10 * 27 and len(signal_acc) >= 10 * 45 and len(
                    signal_acc) <= 10 * 55:
                # clean via bandpass filtering
                sample_ppg = nk.ppg_process(signal_ppg, sampling_rate=len(signal_ppg) // 10)[0]["PPG_Clean"].values

                # downsample PPG to 25 Hz
                sample_ppg = resample(sample_ppg, 10 * 25)

                # resample ACC to 25 Hz
                for key, value in signals_acc.items():
                    signals_acc[key] = resample(value, 10 * 25)

                final_samples.append(sample_ppg)
                for key, value in signals_acc.items():
                    final_samples_acc[key].append(value)
                # save timestamps as float
                final_timestamps.append([prcp_timestamps_ppg[sample].astype("float") / 10 ** 9])
                final_timestamps_acc.append([prcp_timestamps_acc[sample].astype("float") / 10 ** 9])

            else:
                length_out_of_range += 1
                total_time_rejected_samples += len(signal_ppg) / 25

                if len(signal_acc) > 500:
                    print(prcp, "ECG len:", len(signal_ppg), "ACC len:", len(signal_acc))
        except Exception as e:
            print(e, i, prcp, prcp_samples_ppg[sample][:3])
        i += 1

    if len(final_samples) > 0:
        return np.vstack(
            final_samples), final_samples_acc, final_timestamps, final_timestamps_acc, bad_quality_spp, bad_quality_hr, length_out_of_range, total_time_accepted_samples, total_time_rejected_samples
    else:
        return final_samples, final_samples_acc, final_timestamps, final_timestamps_acc, bad_quality_spp, bad_quality_hr, length_out_of_range, total_time_accepted_samples, total_time_rejected_samples


def process_prcp(prcp):
    print("Participant", prcp)
    print("Loading...", prcp)
    bad_quality_spp = 0
    bad_quality_hr = 0
    good_quality_samples = 0
    length_out_of_range = 0
    total_time_accepted_samples = 0
    total_time_rejected_samples = 0
    # load PPG and ACC data of participant
    if not (f"{prcp.split('.')[0]}" in os.listdir(out_path) or "README" in prcp):

        for file in tqdm(os.listdir(path + prcp), desc=prcp, leave=True):
            if "SAMSUNG" in file and ".zip" in file and os.path.getsize(path + prcp + "/" + file) > 2000:
                try:
                    with ZipFile(path + prcp + "/" + file) as zf:
                        with zf.open(file.split(".")[0] + "/HRM_RAW.csv", 'r') as ppg_file, zf.open(
                                file.split(".")[0] + "/ACC.csv", 'r') as acc_file, zf.open(file.split(".")[0] + "/HRM.csv", 'r') as hrm_file:
                            if prcp + "_" + file.split(".")[0] not in os.listdir(out_path_sessions_ppg) and prcp + "_" + \
                                    file.split(".")[0] not in os.listdir(out_path_sessions_acc):
                                try:
                                    df_temp_ppg = pd.read_csv(ppg_file, sep='\t', on_bad_lines='warn',
                                                              dtype={"light_intensity": np.float64})
                                    df_temp_hrm = pd.read_csv(hrm_file, sep='\t', on_bad_lines='warn',
                                                              dtype={"v1": np.float64,"v3": np.float64})
                                    df_temp_acc = pd.read_csv(acc_file, sep='\t', on_bad_lines='warn',
                                                              dtype={"x": np.float64, "y": np.float64, "z": np.float64})
                                except Exception as e:
                                    print(e)
                                    print(f"Empty session {file}")
                                    continue

                                try:
                                    samples_ppg, samples_acc, timestamps, timestamps_acc, bad_quality_spp_temp, bad_quality_hr_temp, length_out_of_range_temp, total_time_accepted_samples_temp, total_time_rejected_samples_temp = process_prcp_df(
                                        df_temp_ppg, df_temp_acc, df_temp_hrm, prcp)
                                    bad_quality_spp += bad_quality_spp_temp
                                    bad_quality_hr += bad_quality_hr_temp
                                    length_out_of_range += length_out_of_range_temp
                                    total_time_accepted_samples += total_time_accepted_samples_temp
                                    total_time_rejected_samples += total_time_rejected_samples_temp
                                    good_quality_samples += len(samples_ppg)
                                    if len(samples_ppg) > 0:
                                        # save data to npy stack
                                        try:
                                            da.to_npy_stack(
                                                out_path_sessions_ppg + prcp.split(".")[0] + "_" + file.split(".")[0],
                                                da.from_array(samples_ppg, chunks=len(samples_ppg)))
                                            for key, value in samples_acc.items():
                                                da.to_npy_stack(
                                                    out_path_sessions_acc + prcp.split(".")[0] + "_" + file.split(".")[
                                                        0] + f"_{key}",
                                                    da.from_array(value, chunks=len(value)))

                                            da.to_npy_stack(
                                                out_path_timestamps_ppg + prcp.split(".")[0] + "_" + file.split(".")[
                                                    0] + "_ts",
                                                da.from_array(timestamps, chunks=len(timestamps)))

                                            da.to_npy_stack(
                                                out_path_timestamps_acc + prcp.split(".")[0] + "_" + file.split(".")[
                                                    0] + "_ts",
                                                da.from_array(timestamps_acc,
                                                              chunks=len(timestamps_acc)))
                                        except Exception as e:
                                            print(e, prcp, file)
                                except Exception as e:
                                    print(f"Error '{e}' in processing {path + prcp + '/' + file}")


                except:
                    print(f"Bad zip file session {path + prcp + '/' + file}")
    try:
        percentage_of_good_quality_samples = (good_quality_samples / (
                good_quality_samples + bad_quality_spp + bad_quality_hr + length_out_of_range)) * 100
    except:
        percentage_of_good_quality_samples = 0
    return_dict = {}
    return_dict[prcp] = {"bad_quality_spp": bad_quality_spp, "bad_quality_hr": bad_quality_hr,
                         "length_out_of_range": length_out_of_range, "good_quality_samples": good_quality_samples,
                         "percentage_of_good_quality_samples": percentage_of_good_quality_samples,
                         "total_time_rejected_samples": total_time_rejected_samples,
                         "total_time_accepted_samples": total_time_accepted_samples}
    return return_dict

def set_up_pool():
    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 4 )#// 4)

    prcps = os.listdir(path)

    multiple_results = [pool.apply_async(process_prcp, args=(prcp,)) for prcp in prcps]
    return [res.get() for res in multiple_results]

if __name__ == "__main__":
    iterations = ["iteration_03"]#[ "iteration_01", "iteration_02", "iteration_03", "iteration_04", "iteration_05", "iteration_06", "iteration_07"]
    root = "/home/emognition/Desktop/LarField"
    hr_based_exclusion = False
    for iteration in iterations:
        print("Iteration:", iteration)

        # paths setup
        path = f"{root}/{iteration}/zip_participants_merged/"
        out_path_sessions_ppg = f"{root}/{iteration}_ppg_acc/dataset_sessions_ppg_fix/"
        out_path_sessions_acc = f"{root}/{iteration}_ppg_acc/dataset_sessions_acc_fix/"
        out_path_timestamps_ppg = f"{root}/{iteration}_ppg_acc/dataset_timestamps_ppg_fix/"
        out_path_timestamps_acc = f"{root}/{iteration}_ppg_acc/dataset_timestamps_acc_fix/"
        out_path = f"{root}/{iteration}_ppg_acc/dataset_fix/"

        start = time.time()

        # directories creation
        os.makedirs(out_path_sessions_ppg, exist_ok=True)
        os.makedirs(out_path_timestamps_ppg, exist_ok=True)
        os.makedirs(out_path_sessions_acc, exist_ok=True)
        os.makedirs(out_path_timestamps_acc, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)
        # multiprocessing particiants processing
        #"""
        return_dicts_list = set_up_pool()
        print(return_dicts_list)
        return_dict = {}
        for rd in return_dicts_list:
            for key, value in rd.items():
                return_dict[key] = value
        with open(f'/home/emognition/Desktop/ppg-ssl/processing_stats/{iteration}_stats.txt', 'w') as f:
            print(return_dict, file=f)

        print('Multiprocessing done', flush=True)

        end_multiprocessing = time.time()
        #"""
        for prcp in os.listdir(path):
            # merge
            try:
                dask_arrays_ppg = []
                ts_arrays_ppg = []
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
                for file in tqdm(os.listdir(out_path_sessions_ppg)):
                    if prcp.split(".")[0] in file:
                        arr_ppg = da.from_npy_stack(out_path_sessions_ppg + file)
                        dask_arrays_ppg.append(arr_ppg)
                        arr_ts_ppg = da.from_npy_stack(out_path_timestamps_ppg + file + "_ts")
                        ts_arrays_ppg.append(arr_ts_ppg)

                        for key, value in dask_arrays_acc.items():
                            arr_acc_temp = da.from_npy_stack(out_path_sessions_acc + file + f"_{key}")
                            value.append(arr_acc_temp)

                        arr_ts_acc = da.from_npy_stack(out_path_timestamps_acc + file + "_ts")
                        ts_arrays_acc.append(arr_ts_acc)

                # normalize PPG within one subject and add timestamp as last value
                df_ppg = da.concatenate(dask_arrays_ppg, axis=0)
                df_ts_ppg = da.concatenate(ts_arrays_ppg, axis=0)
                ppg_mean = da.mean(df_ppg)
                ppg_std = da.std(df_ppg)
                df_ppg  = (df_ppg - ppg_mean) / ppg_std
                df_ppg = da.hstack((df_ppg, df_ts_ppg))
                df_ppg = df_ppg.rechunk(df_ppg.shape)

                df_ts_acc = da.concatenate(ts_arrays_acc, axis=0)

                # normalize each axis of ACC within one subject and add timestamp as last value of each axis

                for key, value in dask_arrays_acc.items():
                    df_acc[key] = da.concatenate(dask_arrays_acc[key], axis=0)
                    if key != "processed":
                        acc_one_axis_mean = da.mean(df_acc[key])
                        acc_one_axis_std = da.std(df_acc[key])
                        df_acc[key] = (df_acc[key] - acc_one_axis_mean) / acc_one_axis_std

                    df_acc[key] = da.hstack((df_acc[key], df_ts_acc))
                    df_acc[key] = df_acc[key].rechunk(df_acc[key].shape)

                df_acc_stacked = da.stack((df_acc["x"], df_acc["y"], df_acc["z"], df_acc["magnitude"], df_acc["processed"]))
                df_acc_stacked = df_acc_stacked.rechunk(df_acc_stacked.shape)

                os.makedirs(out_path + prcp.split(".")[0], exist_ok=True)

                da.to_npy_stack(out_path + prcp.split(".")[0] + "/ppg/", df_ppg)
                del(df_ppg)
                del(df_acc)
                del(arr_ppg)
                del(arr_acc_temp)
                del (arr_ts_ppg)
                del (arr_ts_acc)
                gc.collect()
                da.to_npy_stack(out_path + prcp.split(".")[0] + "/acc/", df_acc_stacked)

            except Exception as e:
                print(e, prcp)

        end_rechunking = time.time()
        gc.collect()
