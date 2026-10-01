"""Evaluation metrics and diagnostic plots."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_predictions(y_true, y_pred):
    """Return MAE, RMSE, and R² for the given predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def print_metrics(mae, rmse, r2):
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")


def plot_actual_vs_predicted(y_true, y_pred):
    """Line plot of actual vs predicted RUL across test units."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.to_numpy(), color="r", label="Actual values")
    plt.plot(y_pred, color="y", linestyle="--", label="Predicted values")
    plt.xlabel("Unit")
    plt.ylabel("RUL (cycles)")
    plt.legend()
    plt.show()


def plot_unit_rul(model, train_df, features, unit_id=1):
    """Actual vs predicted RUL over time for a single training unit."""
    unit_data = train_df[train_df["unit"] == unit_id]
    X_unit = unit_data[features]

    y_actual = unit_data["RUL"].clip(upper=125)
    y_pred = model.predict(X_unit)

    plt.figure(figsize=(12, 5))
    plt.plot(unit_data["time"], y_actual, label="Actual RUL")
    plt.plot(unit_data["time"], y_pred, "--", label="Predicted RUL")
    plt.title(f"Unit {unit_id} - Actual vs Predicted RUL")
    plt.xlabel("Time Cycles")
    plt.ylabel("RUL (cycles)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()