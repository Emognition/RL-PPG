import os
import shutil
import zipfile
import pickle

from datetime import datetime, timedelta

import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from constants import VERSIONS, TIMESTAMP_FORMAT, TIMESTAMP_FORMAT_ASSESSMENTS, MODEL_HALF_WINDOW
from helpers import load_json_file, save_dataset, load_dataset


def get_physio_features(assessment: {}, physio_data: {}) -> {}:
    result = {}

    return result


# todo handle skipping polar sessions when not declared (only watch physio should be a default option)
def window_sessions(participant: str, merged_sessions_path: str, sessions_list: [str], target_path: str,
                    overwrite: bool = False,
                    only_filled_assessments: bool = True, window_length_seconds: int = 300,
                    from_zipped_sessions: bool = True):
    # todo the sessions_list is not used
    if sessions_list is None or not sessions_list:
        sessions_list = [x.replace('.zip', '') for x in os.listdir(merged_sessions_path)]
        for definitely_not_session in ["assessments", "database"]:
            if definitely_not_session in sessions_list:
                sessions_list.remove(definitely_not_session)
    for session in os.listdir(merged_sessions_path):
       if "POLAR" in session:
           sessions_list.pop(sessions_list.index(session.replace('.zip', '')))

    # todo unzip assessments in target path so there is no confusion with .zip an unzipped files
    if from_zipped_sessions:
        zipped_assessments_path = f"{merged_sessions_path}/assessments.zip"
        if not os.path.isfile(zipped_assessments_path):
            print(f"No assessment.zip in: {merged_sessions_path}")
            return None
        try:
            with zipfile.ZipFile(f"{merged_sessions_path}/assessments.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{merged_sessions_path}")
        except zipfile.BadZipFile:
            print(f"Bad zip file: {merged_sessions_path}/assessments.zip")
            return None

    assessments_path = f"{merged_sessions_path}/assessments/emotion"

    if not os.path.isdir(assessments_path) or len(os.listdir(assessments_path)) == 0:
        print(f"No emotion assessment in: {assessments_path}")
        return None

    assessments_dict = {x: load_json_file(f"{assessments_path}/{x}")
                        for x in os.listdir(f"{assessments_path}")}
    if only_filled_assessments:
        assessments_dict = {assessment_file_name: assessment_data for assessment_file_name, assessment_data in
                            assessments_dict.items()
                            if assessment_data['status'] == 'FILLED' or assessment_data['status'] == 'FILED'}

    assessments_grouped_by_session = {
        session_name: [] for session_name in set(x['sessionName'] for x in assessments_dict.values())
        if session_name in sessions_list  # todo to jest słabe - podmień później warunek z góry żeby był tu robiony
    }
    # todo kontynuuj tutaj - potrzebujesz info o pliku żeby skopiować ankietę zamiast zapisywania jsona do nowego pliku
    for assessment_file_name, assessment_data in assessments_dict.items():
        assessment_data['filename'] = assessment_file_name
        if assessment_data['sessionName'] in assessments_grouped_by_session.keys():
            assessments_grouped_by_session[assessment_data['sessionName']].append(assessment_data)

    half_window_timedelta = timedelta(seconds=int(round(window_length_seconds / 2, 0)))

    print()
    print(f"Num of {'filled ' if only_filled_assessments else 'all '}assessments: {len(assessments_dict.keys())} "
          f"assessments in {assessments_path}")

    # TODO temporary for checking data
    print(f"All assessments: {len(assessments_dict)}")

    samples_list = []
    for assessment_session, assessments_list in tqdm(assessments_grouped_by_session.items()):
        session_data = None

        # todo here: binary model shift and cutting based on time delta not seconds in int
        for assessment in assessments_list:
            # assessment_dir_name = assessment['createdTimestamp'].replace(":", '-').replace(".", "-")
            # handle case when there is dir but no assessment
            target_dir_name = assessment['filename'].split('.')[0]
            target_dir_path = f"{target_path}/{target_dir_name}"
            if overwrite and os.path.isdir(f"{target_dir_path}"):
                shutil.rmtree(f"{target_dir_path}")

            if os.path.isdir(target_dir_path) and not os.path.isfile(f"{target_dir_path}/data.pkl"):
                shutil.rmtree(f"{target_dir_path}")

            if os.path.isdir(target_dir_path) and len(os.listdir(target_dir_path)) == 0:
                shutil.rmtree(f"{target_dir_path}")

            if not overwrite and os.path.isdir(f"{target_dir_path}"):
                print(f"Skipping sample: {target_dir_path}")
                continue

            if session_data is None:
                session_data = read_session_data(f"{merged_sessions_path}/{assessment_session}",
                                                 from_zipped_sessions=from_zipped_sessions)

            #os.makedirs(f"{target_dir_path}", exist_ok=True)
            if assessment['emotionTimestamp'] is not None:
                assessment_timestamp = assessment['emotionTimestamp']
            else:
                assessment_timestamp = assessment['startedTimestamp']

            try:
                assessment_timestamp = datetime.strptime(assessment_timestamp,
                                                         TIMESTAMP_FORMAT_ASSESSMENTS)
            except ValueError:
                print(f"ValueError for: {assessment_timestamp} in assessment: {assessment}")
                # '2023-11-28T10:49:36' does not match format '%Y-%m-%dT%H:%M:%S.%f'
                assessment_timestamp = datetime.strptime(assessment_timestamp,
                                                         '%Y-%m-%dT%H:%M:%S')
                # continue

            # todo discuss if this is needed; do not update assessment sample prop - maybe add new?
            do_offset = True
            if do_offset and assessment['source'] == 'ML_MODEL':
                try:
                    assessment_timestamp += MODEL_HALF_WINDOW[
                        assessment["binaryModelOutput"]["modelIdentifier"]]
                except:
                    assessment_timestamp += timedelta(seconds=30)

            # TODO extract this
            sample_dict = get_sliced_sample(session_data, assessment_timestamp, half_window_timedelta)

            # saving data to .csv
            # for signal, df_signal in sample_dict.items():
            #     df_signal.to_csv(f"{target_path}/{dir_name}/{signal}.csv", sep='\t', index=True,
            #                      date_format=TIMESTAMP_FORMAT_ASSESSMENTS)

            """
            # Temporal comment
            with open(f"{target_dir_path}/data.pkl", 'wb') as file:
               pickle.dump(sample_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

            #  todo copy2 ankietę; niech zostanie .json
            shutil.copy2(src=f"{merged_sessions_path}/assessments/emotion/{assessment['filename']}",
                         dst=f"{target_dir_path}")
            """
            samples_list.append({
                'participant': participant,
                'emotionTimestamp': assessment['emotionTimestamp'],
                'device': assessment['sessionName'].split('_')[0],
                'assessment': assessment,
                'sensors_data': sample_dict,
            })

    return samples_list


# assessment_timestamp -> middle_timestamp; it's middle of the sample
def get_sliced_sample(session_data, middle_timestamp, half_window_timedelta):
    time_start = (middle_timestamp - half_window_timedelta).strftime('%Y-%m-%dT%H:%M:%S')
    time_end = (middle_timestamp + half_window_timedelta).strftime('%Y-%m-%dT%H:%M:%S')
    sample_dict = {}
    for signal, df_signal in session_data.items():
        if df_signal is None:
            sample_dict[signal] = None
            continue
        # sliced_data = df_signal.loc[time_start:time_end]
        idx = df_signal.index
        sliced_data = df_signal.loc[(time_start <= idx) & (idx <= time_end)]

        if sliced_data.empty or len(sliced_data.index) == 0:
            continue
        sample_dict[signal] = sliced_data
    return sample_dict


def read_session_data(session_path: str, from_zipped_sessions: bool = True) -> dict:
    def format_datetime(dt_series):
        def get_split_date(strdt):
            split_date = strdt.split()
            str_date = split_date[1] + ' ' + split_date[2] + ' ' + split_date[5] + ' ' + split_date[3]
            return str_date

        dt_series = pd.to_datetime(dt_series.apply(lambda x: get_split_date(x)), format='%b %d %Y %H:%M:%S')

        return dt_series

    session_name, session_data = session_path.split("/")[-1][:-4], {}

    if session_name.startswith("POLAR"):
        signals = ["ACC", "ECG"]  # ["ACC", "ECG", "HR"]
    else:
        signals = VERSIONS["_v6_"]['signals']

    if from_zipped_sessions:
        session_item = zipfile.ZipFile(session_path if '.zip' in session_path else f"{session_path}.zip")

    for signal in signals:
        if from_zipped_sessions:
            signal_item = session_item.open(f'{session_path.split("/")[-1]}/{signal}.csv')
        else:
            signal_item = f"{session_path}/{signal}.csv"

        if not from_zipped_sessions and not os.path.exists(signal_item):
            print(f"No file for {signal} in {session_path}")
            session_data[signal] = None  # todo empty df would be better?
            continue
        try:
            df_signal = pd.read_csv(signal_item, sep='\t', on_bad_lines='warn', )
            # parse_dates=['ts'], )  # TODO here: use built in parser for data during reading

            # todo check if it could be optimized somehow
            df_signal['ts'] = pd.to_datetime(df_signal['ts'], format=TIMESTAMP_FORMAT)

            # todo check is it optimal? isn't it better to set index and then sort it?
            df_signal = df_signal.sort_values(by=['ts'], ascending=True)
            df_signal = df_signal.set_index("ts")
        except EmptyDataError:
            session_data[signal] = None
            print("EmptyDataError")
            continue

        session_data[signal] = df_signal
        if from_zipped_sessions:
            signal_item.close()
    if from_zipped_sessions:
        session_item.close()
    return session_data


def get_samples_for_assessments(assessments_list, half_window_timedelta, sessions_path):
    # session_name -> session_data
    loaded_sessions, assessment_data_tuples = {}, []

    for assessment in assessments_list:
        session_name = assessment['sessionName']

        if session_name in loaded_sessions.keys():
            session_data = loaded_sessions[session_name]
        else:
            # session_data = read_session_data(f"{sessions_path}/{session_name}", from_zipped_sessions=True)
            session_data = None  # TODO just for testing
            loaded_sessions[session_name] = session_data

        if assessment['emotionTimestamp'] is not None:
            assessment_timestamp = assessment['emotionTimestamp']
        else:
            assessment_timestamp = assessment['startedTimestamp']

        try:
            assessment_timestamp = datetime.strptime(assessment_timestamp,
                                                     TIMESTAMP_FORMAT_ASSESSMENTS)
        except ValueError:
            print(f"ValueError for: {assessment_timestamp} in assessment: {assessment}")
            # '2023-11-28T10:49:36' does not match format '%Y-%m-%dT%H:%M:%S.%f'
            assessment_timestamp = datetime.strptime(assessment_timestamp,
                                                     '%Y-%m-%dT%H:%M:%S')
            # continue

        # todo discuss if this is needed; do not update assessment sample prop - maybe add new?
        do_offset = True
        if do_offset and assessment['source'] == 'ML_MODEL':
            assessment_timestamp += MODEL_HALF_WINDOW[
                assessment["binaryModelOutput"]["modelIdentifier"]]

        time_start = (assessment_timestamp - half_window_timedelta).strftime('%Y-%m-%dT%H:%M:%S')
        time_end = (assessment_timestamp + half_window_timedelta).strftime('%Y-%m-%dT%H:%M:%S')

        sample_dict = {}
        # for signal, df_signal in session_data.items():
        #     if df_signal is None:
        #         sample_dict[signal] = None
        #         continue
        #     sliced_data = df_signal.loc[time_start:time_end]
        #     if sliced_data.empty or len(sliced_data.index) == 0:
        #         continue
        #     sample_dict[signal] = sliced_data
        print("DATA WINDOWING CODE IS COMMENTED FOR TESTING!!!")

        assessment_data_tuples.append((assessment, sample_dict))

    return assessment_data_tuples

if __name__=="__main__":

    test_prcps = ["/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/RydY89HJlyLzeNOrmcjn1ub6j6g1",
"/home/emognition/Desktop/LarField/iteration_05_ppg_acc/dataset_fix/6t6TTSYMybZDt2zbxLCMQvyLFdm2",
"/home/emognition/Desktop/LarField/iteration_07_ppg_acc/dataset_fix/4uF8jaUO6nVOs6RPovkIwGw53qD2",
"/home/emognition/Desktop/LarField/iteration_05_ppg_acc/dataset_fix/6SbXNCMDl3SyB97ZG0eQEnH5nI82",
"/home/emognition/Desktop/LarField/iteration_05_ppg_acc/dataset_fix/MpUE7D05ybS4objMMB3mwPa2kPC3",
"/home/emognition/Desktop/LarField/iteration_04_ppg_acc/dataset_fix/s3IpVEvxi8U17uJZXQHDGQV4SiF2",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/LWcSrXZFVZOInPb8jlIjEZSCkdv1",
"/home/emognition/Desktop/LarField/iteration_04_ppg_acc/dataset_fix/2VaupA1XhdeElSqKXSxzGQljhT02",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/NlwnRZ1tSTXnPa0tBhbgmeq51TR2",
"/home/emognition/Desktop/LarField/iteration_07_ppg_acc/dataset_fix/lxDA1xS7vOe0BpLCtFzDIYiGGZ52",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/bymVumJaGZaiLvw8xolKfE6L6Cp1",
"/home/emognition/Desktop/LarField/iteration_02_ppg_acc/dataset_fix/pmsBIM9OSvRNH9NEn4eSoOEx4b73",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/ibWYuw58sJQjJ7D1MkfRwla4sri1",
"/home/emognition/Desktop/LarField/iteration_03_ppg_acc/dataset_fix/XfuetknJVnQ1hZcBGNlrfeGB1823",
"/home/emognition/Desktop/LarField/iteration_01_ppg_acc/dataset_fix/1x6a3F7pIzd7B7BdZePh7Z0WC4A3",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/WyWnCG4vTKeh2YF0UEtBpzGUcDi1",
"/home/emognition/Desktop/LarField/iteration_02_ppg_acc/dataset_fix/9lsiyjS1FrXNnl3tC9H96zRDQUp1",
"/home/emognition/Desktop/LarField/iteration_03_ppg_acc/dataset_fix/UxG1iktUAoPonFULajPCuDfXkpQ2",
"/home/emognition/Desktop/LarField/iteration_06_ppg_acc/dataset_fix/ou9yFAPtAzV0mtnuquyflipf5iQ2",
"/home/emognition/Desktop/LarField/iteration_03_ppg_acc/dataset_fix/D19yFwUGNJhbSOEW2duM0ui6YII3",
"/home/emognition/Desktop/LarField/iteration_04_ppg_acc/dataset_fix/grVJgrgv4IX672aWnJr9a4ACfFr2",
"/home/emognition/Desktop/LarField/iteration_03_ppg_acc/dataset_fix/rPBwrd04gBQbKtYRDrhJYLlRFe33",
"/home/emognition/Desktop/LarField/iteration_07_ppg_acc/dataset_fix/CrnkUjRAp4VIZfrmcjSATt8RuKl1",
"/home/emognition/Desktop/LarField/iteration_03_ppg_acc/dataset_fix/nyhUXXi0IZXNxznwoDbabCwvHF53"]




    for prcp_path in test_prcps:

        iteration = prcp_path.split("iteration_0")[1].split("_ppg_acc")[0]
        prcp = prcp_path.split("/")[-1]
        iteration_path = f"/home/emognition/Desktop/LarField/iteration_0{iteration}/zip_participants_merged/"
        target_path = f"/home/emognition/Desktop/LarField/test_ppg/{prcp}/assessments/"

        os.makedirs(target_path, exist_ok=True)

        merged_sessions_path = f"{iteration_path}{prcp}"
        sessions_list = None
        samples = window_sessions(prcp, merged_sessions_path, sessions_list, target_path)

        print(len(samples))

        save_dataset(f"{target_path}emotion.pkl", samples)

        samples_loaded = load_dataset(f"{target_path}emotion.pkl")

        print(len(samples_loaded))

