"""Central configuration: paths, column names, and constants."""

TRAIN_PATH = "data/train_FD001.txt"
TEST_PATH = "data/test_FD001.txt"
RUL_PATH = "data/RUL_FD001.txt"

INDEX_FEATURES = ["unit", "time"]
SETTING_FEATURES = ["setting_1", "setting_2", "setting_3"]
SENSOR_FEATURES = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]

COLUMN_NAMES = INDEX_FEATURES + SETTING_FEATURES + SENSOR_FEATURES

# Cap for the RUL target (piecewise-linear degradation assumption)
RUL_CLIP_UPPER = 125

RANDOM_STATE = 42