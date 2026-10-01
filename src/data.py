"""Loading, cleaning, and feature/target preparation for FD001."""

import pandas as pd

from config import (
    TRAIN_PATH, TEST_PATH, RUL_PATH,
    COLUMN_NAMES, RUL_CLIP_UPPER,
)


def load_raw_data():
    """Load the raw FD001 train/test/RUL text files."""
    train_df = pd.read_csv(
        TRAIN_PATH, sep=r"\s+", names=COLUMN_NAMES, header=None
    )
    test_df = pd.read_csv(
        TEST_PATH, sep=r"\s+", names=COLUMN_NAMES, header=None
    )
    rul_df = pd.read_csv(RUL_PATH, sep=r"\s+", header=None)
    return train_df, test_df, rul_df


def drop_constant_columns(df):
    """Return a copy of df without columns that never vary."""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=constant_cols), constant_cols


def add_rul(df):
    """Add a Remaining Useful Life column (RUL = t_max - t) in place."""
    t_max = df.groupby("unit")["time"].transform("max")
    df["RUL"] = t_max - df["time"]
    return df


def prepare_train_data():
    """Load, clean, and label the training set. Returns (X_train, y_train, train_df)."""
    train_df, _, _ = load_raw_data()
    train_df, _ = drop_constant_columns(train_df)
    train_df = add_rul(train_df)

    features = [c for c in train_df.columns if c not in ("unit", "time", "RUL")]
    X_train = train_df[features]
    y_train = train_df["RUL"].clip(upper=RUL_CLIP_UPPER)
    return X_train, y_train, train_df, features


def prepare_test_data(features):
    """Load the test set, take each unit's last cycle, and pair it with true RUL.

    Returns (X_test, y_test, test_last_cycle_df).
    """
    _, test_df, rul_df = load_raw_data()
    test_df, _ = drop_constant_columns(test_df)

    last_cycle = test_df.groupby("unit").last().reset_index()
    X_test = last_cycle[features]
    y_test = rul_df.squeeze().clip(upper=RUL_CLIP_UPPER)
    return X_test, y_test, last_cycle