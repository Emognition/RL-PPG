from datetime import timedelta

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S:%f'
TIMESTAMP_FORMAT_NO_MICROSECONDS = '%Y-%m-%dT%H:%M:%S'
TIMESTAMP_FORMAT_DASH = '%Y-%m-%dT%H-%M-%S-%f'
TIMESTAMP_FORMAT_ASSESSMENTS = '%Y-%m-%dT%H:%M:%S.%f'
TIMESTAMP_FORMAT_METADATA = '%Y-%m-%dT%H:%M:%S'

VERSIONS = {
    # "sensorServiceTest": {
    #     "signals": ["ACC", "LINEAR_ACC", "GYR", "LIGHT", "PRESSURE", "GRAV", "HRM", "HRM_RAW"]
    # },
    "_v6_": {
        "signals": ["ACC", "GRAV", "GYR", "HRM", "HRM_RAW", "LIGHT", "PEDOMETER", "PRESSURE"]
    },
}

VERSIONS_POLAR = {
    "_v1_": {
        "signals": ['ACC', 'ECG', 'HR'],
    },
}

# physio data constants
PHYSIO_FILES = {
    'SAMSUNG_WATCH': ["ACC", "GYR", "LIGHT", "PRESSURE", "GRAV", "HRM", "HRM_RAW", "PEDOMETER"],
    'SAMSUNG': ["ACC", "GYR", "LIGHT", "PRESSURE", "GRAV", "HRM", "HRM_RAW", "PEDOMETER"],
    'POLAR': ['ACC', 'ECG', 'HR'],
}

SIGNAL_FILES_COL_NUM = {
    "ACC": 4,
    "GRAV": 4,
    "GYR": 4,
    "HRM": 5,
    "HRM_RAW": 2,
    "LIGHT": 2,
    "PEDOMETER": 9,
    "PRESSURE": 5,
    "ECG": 2,
    "HR": 2
}

# without ts col
ALL_DATA_COLUMNS = {
    'ACC': ['x', 'y', 'z'],
    'ECG': ['ecg'],
    'HR': ['hr'],
    'GRAV': ['x', 'y', 'z'],
    'GYR': ['x', 'y', 'z'],
    'HRM': ['v1', 'v2', 'v3', 'v4'],
    'HRM_RAW': ['light_intensity'],
    'LIGHT': ['light_level'],
    'PEDOMETER': ['steps', 'walking_steps', 'running_steps', 'distance', 'calories', 'speed', 'freq', 'state'],
    'PRESSURE': ['v1', 'v2', 'v3', 'v4'],
}

HRM_VALUES = {
    'V1': 'HR',
    'V2': 'NO_INFO',
    'V3': 'RR_INTERVAL',
}

# TODO check the key values/mapping
PEDOMETER_STATES = {
    None: "SENSOR_PEDOMETER_STATE_UNKNOWN",
    None: "SENSOR_PEDOMETER_STATE_STOP",
    None: "SENSOR_PEDOMETER_STATE_WALK",
    None: "SENSOR_PEDOMETER_STATE_RUN"
}

signal_length = {
    'light_intensity': 3000,
    'HR': 3000,
    'ACC': 6000
}

MODEL_HALF_WINDOW = {
    "SKLEARN_V1": timedelta(seconds=5),
    "SKLEARN_V2": timedelta(seconds=5),
    "SKLEARN_V3": timedelta(seconds=5),
    "SKLEARN_V4": timedelta(seconds=15),
    "SKLEARN_V5": timedelta(seconds=15),
    "SKLEARN_V6": timedelta(seconds=5),
    "SKLEARN_V8": timedelta(seconds=60),
    "SKLEARN_V9": timedelta(seconds=30),
    "SKLEARN_V10": timedelta(seconds=30),
    "SKLEARN_V11": timedelta(seconds=30),
    "SKLEARN_V12": timedelta(seconds=10),
    "SKLEARN_V13": timedelta(seconds=10),
    "SKLEARN_V14": timedelta(seconds=10),
    "SKLEARN_V15": timedelta(seconds=10),
    "SKLEARN_V16": timedelta(seconds=10),
    "SKLEARN_V17": timedelta(seconds=60),
    "SKLEARN_V18": timedelta(seconds=30),
    "SKLEARN_V20": timedelta(seconds=30),
    "SKLEARN_V23": timedelta(seconds=30),
    "SKLEARN_V24": timedelta(seconds=30),
    "SKLEARN_V25": timedelta(seconds=30),
    "SKLEARN_V26": timedelta(seconds=30),
    "SKLEARN_V27": timedelta(seconds=30),
    "SKLEARN_V28": timedelta(seconds=30),
    "SKLEARN_V29": timedelta(seconds=30),
    "SKLEARN_V30": timedelta(seconds=30),
    "SKLEARN_V31": timedelta(seconds=30),
    "SKLEARN_V32": timedelta(seconds=30),
    "SKLEARN_V33": timedelta(seconds=30),
    "SKLEARN_V34": timedelta(seconds=30),
    "TFLITE_V1": timedelta(seconds=5),
    "TFLITE_V2": timedelta(seconds=60),
    "TF_LITE_V1": timedelta(seconds=5),
    "TF_LITE_V2": timedelta(seconds=60),
    # presonalized
    "SKLEARN_i07grouped": timedelta(seconds=30),
    # TODO: DO NOT COMMIT
}

# assessments constants
ASSESSMENTS_SOURCES = ('ML_MODEL', 'SCHEDULED', 'SELF_TRIGGERED')
ASSESSMENTS_STATUSES = ('FILLED', 'EXPIRED', 'CANCELLED')
ASSESSMENTS_ANSWERS = ('YES', 'NO', 'DONT_KNOW')
ASSESSMENTS_TYPES = ("EMOTION", "EVENING", "MORNING")

STACKED_COLORS = (
    'rgb(131, 90, 241)',
    'rgb(111, 231, 219)',
    'rgb(184, 247, 212)',
)

ANSWERS_COLORS = (
    'rgb(27,158,119)',
    'rgb(217,95,2)',
    'rgb(69,117,180)',
)
PATTERNS = ('\\', '/', 'x')

# todo check and fill it; sk learn 12?
# todo change it to dict, where key is app version and value is model_id
#   this will force checking if the bundled model changed
# BUNDLED_ML_MODEL = "SKLEARN_V11"
BUNDLED_ML_MODEL = "SKLEARN_V18"

# constants for ml model threshold/logs computation
EXPECTED_SESSION_LENGTH = 8
MIN_SESSION_LENGTH = 1
CHUNK_LENGTH = 2
BATCH_LENGTH = 120