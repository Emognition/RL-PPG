import os.path
import pickle

import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm

from downstream_tasks.utils.constants import TIMESTAMP_FORMAT_ASSESSMENTS
from downstream_tasks.utils.assessments_utils import get_participant_assessments, get_evening_assessment_for_day, \
    get_assessments_from_next_day
from downstream_tasks.utils.physio_utils import load_processed_participant
from downstream_tasks.utils.helpers import save_dataset, load_dataset
from downstream_tasks.larfield.dataset_split.study_dates import get_study_dates

# TODO: set the path
DATA_PATH = "/home/emognition/Desktop/LarField"


def load_participant(dataset_dir: str, participant: str) -> ([], []):
    emotions_path = f"{dataset_dir}{participant}/assessments/emotion.pkl"
    return load_dataset(emotions_path)


def load_larfield_test(dataset_dir: str,
                       aggregate_samples: bool = True, if_load_demography: bool = False) -> ([], [], [],):
    x_test, y_test, participant_list = [], [], []

    dataset_dir = f"{dataset_dir}/datasets/larfield-downstream-tasks-fix"
    for participant in os.listdir(dataset_dir):
        if not "_y.pkl" in participant:
            participant = participant.split('_')[0] if '_' in participant else participant
            prcp_dataset = load_participant(dataset_dir, participant)

            if aggregate_samples:
                # 6 samples from a day as 1 input - all samples from a day as an input
                for day_samples, assessment in zip(x, y):
                    x_test.append(day_samples)
                    y_test.append(assessment)
            else:
                # 1 sample from a day as 1 input - each sample from a day as individual an input
                for day_samples, assessment in zip(x, y):
                    x_test += day_samples
                    y_test += [assessment] * len(day_samples)

            participant_list += [participant[:5]] * len(y)

            if demography_data is not None:
                demography_list += [demography_data[participant]] * len(y)
    return x_test, y_test, participant_list, demography_list


def get_participants_demography(dataset_dir: str) -> {}:
    # TODO: set the path
    demography_file_path = f"{dataset_dir}/demography.csv"
    if not os.path.exists(demography_file_path):
        # print("No file in ", demography_file_path)
        raise Exception(f"No file in {demography_file_path}")

    demography_df = pd.read_csv(demography_file_path, sep='\t', index_col=0)
    if demography_df.empty:
        return None

    demography_df = demography_df[
        ["UID", "iteration", "Jak określasz swoją płeć?", "Jaki jest Twój wiek (w latach, liczbą)?"]]
    demography_df["UID"] = demography_df["UID"].apply(lambda uid: uid[:5])
    demography_df = demography_df.set_index("UID")
    demography_df = demography_df.rename(
        columns={"Jak określasz swoją płeć?": "Gender", "Jaki jest Twój wiek (w latach, liczbą)?": "Age"})
    demography_df = demography_df.drop(columns=['iteration'])
    return demography_df.T.to_dict()


if __name__ == '__main__':

    test_set_dir = DATA_PATH + "/test_ppg"
    test_idx = pd.read_csv("test_ids.csv").set_index('ID')['iteration'].T.to_dict()
    counter = 0

    IF_ADD_MORNING_ASSESSMENT_FROM_TOMORROW = True

    for participant, iteration in test_idx.items():
        start_time = datetime.now()
        print(f"Processing participant: {participant} ")
        print(f"From iteration: {iteration} ")
        print(f"Processed/In iteration: {counter}/{len(test_idx)} ")
        counter += 1

        p_path = f"{test_set_dir}/{participant}"
        a_path = f"{test_set_dir}/{participant}/assessments/emotion.pkl"

        i_study_dates = get_study_dates(iteration)
        x, y, uid_list = [], [], []

        emotion_a = load_dataset(a_path)
        """
        unique_dates = sorted(list(set([x.date() for
                                        x in p_physio['timestamp_ecg']
                                        if i_study_dates[0] <= x.date() <= i_study_dates[1]])))

        print(unique_dates)
        p_data = {x: [] for x in unique_dates}
        for ecg, acc, timestamp_ecg in zip(p_physio['ecg'], p_physio['acc'], p_physio['timestamp_ecg']):
            date_ecg = timestamp_ecg.date()
            if date_ecg not in p_data:
                continue
            p_data[date_ecg].append({'ts': timestamp_ecg, 'ecg': ecg, 'acc': acc})
        del p_physio

        for assessment_datetime, morning_assessment in tqdm(morning_a.items()):
            if morning_assessment["status"] != "FILLED":
                continue

            evening_assessment = get_evening_assessment_for_day(evening_a, assessment_datetime)
            if evening_assessment is None:
                print(f"No evening assessment - skipping the day: {assessment_datetime}")
                continue

            label = {
                "morning": morning_assessment["response"],
                "evening": evening_assessment["response"],
                "morning_tomorrow": None
            }
            label["morning"]['filledTimestamp'] = morning_assessment["filledTimestamp"]
            label["evening"]['filledTimestamp'] = evening_assessment["filledTimestamp"]

            if IF_ADD_MORNING_ASSESSMENT_FROM_TOMORROW:
                tomorrow_assessments = get_assessments_from_next_day(morning_a, assessment_datetime)
                if len(tomorrow_assessments) > 0:
                    label['morning_tomorrow'] = list(tomorrow_assessments.values())[0]['response']

            assessment_datetime = datetime.strptime(assessment_datetime.split('T')[0], '%Y-%m-%d').date()
            if assessment_datetime not in p_data:
                continue
            samples_from_date = p_data[assessment_datetime]
            num_of_samples = len(samples_from_date)
            print(assessment_datetime, num_of_samples)
            if not samples_from_date:
                continue

            # offset to skip ~%5 samples from beginning and end of the session(s)
            offset = int(num_of_samples / 5)
            # simple version of collecting processed samples
            samples_from_date = sorted(samples_from_date, key=lambda element: element['ts'])
            # skipping the first index and last indexes
            data_samples_idx = np.linspace(start=offset,
                                           stop=num_of_samples + 1 - offset, num=8).astype(int)[1:-1]
            data_samples = [samples_from_date[idx] for idx in data_samples_idx]
            session_start, session_end = samples_from_date[0]['ts'].time(), samples_from_date[-1]['ts'].time()
            del samples_from_date

            for sample in data_samples:
                del sample['ts']

            x.append(data_samples)
            y.append(label)  # assessments, not just label - specific values could be easily excluded later
            print(f"{iteration}_{participant}")
        del p_data
        print(f"Processed participants from iteration {iteration}")
        os.makedirs(f"{DATA_PATH}/datasets/larfield-downstream-tasks-fix/", exist_ok=True)
        save_dataset(f"{DATA_PATH}/datasets/larfield-downstream-tasks-fix/{participant[:5]}_x.pkl", x)
        save_dataset(f"{DATA_PATH}/datasets/larfield-downstream-tasks-fix/{participant[:5]}_y.pkl", y)
        print(f"Data saved from test set of {participant} saved to: {DATA_PATH}/datasets/larfield-downstream-tasks-fix")
        print(f"Processing and saving time: {datetime.now() - start_time}")
        """