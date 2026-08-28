import json
import logging
import math
import pickle
import os
import shutil
from datetime import datetime, timedelta, date
from distutils.dir_util import copy_tree
from json import JSONDecodeError, JSONEncoder

from tqdm import tqdm
from shutil import make_archive


def get_login_from_uid(iteration, uid):
    # TODO: temporary wokraround
    if uid in ['qAaU22i5t2fAttjw4odyWsCEOjP2', 'Lqn3Z8NPuUWD5foFgUsQ9i9N3933', ]:
        return None  # TODO: DO NOT COMMIT original code
    if uid in ['11LKIbbe5Zfn13Ok64WdUsG1sJ72', 'AEyLGMcfzMW4NyQsZQ4BTJ33P002']:
        return None  # TODO: DO NOT COMMIT original code

    map = {}  # TODO: DO NOT COMMIT original code
    if isinstance(map[uid], str) and '_' in map[uid]:
        return map[uid].split('_')[1]
    return str(map[uid])


# todo this script contains too much mess - exclude some functions
def zip_session(participant_path, zipped_path, session):
    return make_archive(base_name=zipped_path, format='zip',
                        root_dir=participant_path, base_dir=session)


def zip_sessions_participants(participant_path, participant, zipped_path):
    tqdm_it = tqdm(os.listdir(participant_path))
    for session in tqdm_it:
        result = zip_session(participant_path, zipped_path, session)
        # print(result)
    return participant


# todo add info to shell about lacking metadata.json
def get_participant_sessions_meta(participant_path: str, min_num_of_batches: int = 3) -> [{}]:
    all_sessions = [load_json_file(f"{participant_path}/{session}/metadata.json")
                    for session in os.listdir(participant_path)
                    if session not in ["assessments", "database"] and os.path.exists(
            f"{participant_path}/{session}/metadata.json")]
    return [session for session in all_sessions if session['batches_num'] >= min_num_of_batches]


# todo exclude to dash_data_utils.py or something like this
def get_all_participants_sessions_meta(project_path: str, participant_list: []) -> {}:
    all_p_metadata = {}
    for participant in participant_list:
        if os.path.isdir(f"{project_path}/{participant}"):
            all_p_metadata[participant] = get_participant_sessions_meta(
                f"{project_path}/{participant}")
        else:
            all_p_metadata[participant] = None
    return all_p_metadata


# todo exclude to dash_data_utils.py or something like this
def get_all_participants_sessions_meta_from_raw(project_path: str, participant_list: []) -> {}:
    all_p_metadata = {}
    for participant in participant_list:
        if os.path.isdir(f"{project_path}/{participant}"):
            all_p_metadata[participant] = get_participant_sessions_meta(
                f"{project_path}/{participant}")
        else:
            all_p_metadata[participant] = None
    return all_p_metadata


def get_all_participants_daily_assessments(project_path: str, participant_list: []) -> {}:
    all_participant_daily_assessments, all_participant_emo_assessments = {}, {}

    for participant in participant_list:
        all_participant_daily_assessments[participant], all_participant_emo_assessments[participant] = {}, {}
        if os.path.isdir(f'{project_path}/{participant}/assessments/emotion'):
            all_participant_daily_assessments[participant]['emotion'] = {
                filename[:-5]: load_json_file(f'{project_path}/{participant}/assessments/emotion/{filename}')
                for filename in os.listdir(f'{project_path}/{participant}/assessments/emotion')
            }
        else:
            all_participant_daily_assessments[participant]['emotion'] = {}

        if os.path.isdir(f'{project_path}/{participant}/assessments/morning'):
            all_participant_daily_assessments[participant]['morning'] = {
                filename[:-5]: load_json_file(f'{project_path}/{participant}/assessments/morning/{filename}')
                for filename in os.listdir(f'{project_path}/{participant}/assessments/morning')
            }
        else:
            all_participant_daily_assessments[participant]['morning'] = {}

        if os.path.isdir(f'{project_path}/{participant}/assessments/evening'):
            all_participant_daily_assessments[participant]['evening'] = {
                filename[:-5]: load_json_file(f'{project_path}/{participant}/assessments/evening/{filename}')
                for filename in os.listdir(f'{project_path}/{participant}/assessments/evening')
            }
        else:
            all_participant_daily_assessments[participant]['evening'] = {}

    return all_participant_emo_assessments, all_participant_daily_assessments


def load_dataset(path):
    logging.info(f"Loading dataset from {path}")
    with open(path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def load_json_file(path: str) -> dict:
    if path[-5:] != '.json':
        path += '.json'

    with open(path, encoding="utf8") as json_file:
        json_data = json.load(json_file)

    return json_data


def save_json_file(data_to_save, path):
    # todo* why it doesn't work for dict keys?
    class TimeDeltaEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, timedelta):
                return str(o)
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            return super().default(o)

    if path[-5:] != '.json':
        path += '.json'

    with open(path, 'w') as json_file:
        json.dump(data_to_save, json_file, cls=TimeDeltaEncoder)
        # default=TimeDeltaEncoder)


def validate_json_file(file_path):
    try:
        with open(file_path) as f:
            json.load(f)
            return True
    except JSONDecodeError as e:
        return False


def save_dataset(path, dataset):
    logging.info(f"Saving dataset to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(dataset, file)


def print_processing_log(session_path, message):
    # print(f'Session: {session_path}', '\n',
    #       message)
    print(f'{message}, for session: {session_path}\n')


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def nice_percent(value):
    return round(value, 2) * 100


def mean(values: list):
    return sum(values) / len(values)


def get_raw_session_size(session_path: str) -> int:
    """
    returns size of raw session in bytes
    :param session_path:
    :return:
    """

    session_size = 0
    for batch_num in os.listdir(session_path):
        # a file(s) retrieved directly from the watch
        if batch_num == 'meta.json':
            continue

        session_size += sum(
            [os.path.getsize(f"{session_path}/{batch_num}/{x}") for x in os.listdir(f"{session_path}/{batch_num}")])
    return session_size


def get_merged_session_size(session_path: str) -> int:
    """
    returns size of merged session in bytes
    :param session_path:
    :return:
    """
    return sum([os.path.getsize(f"{session_path}/{file}") for file in os.listdir(f"{session_path}")])


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def copy_all_assessments(participant_path, target_path):
    copy_tree(participant_path, target_path)