"""Entry point: prepare data, tune LightGBM, evaluate on the test set."""

from data import prepare_train_data, prepare_test_data
from model import tune_lgbm
from evaluate import (
    evaluate_predictions, print_metrics,
    plot_actual_vs_predicted, plot_unit_rul,
)


def main():
    # 1. Prepare data
    X_train, y_train, train_df, features = prepare_train_data()
    X_test, y_test, _ = prepare_test_data(features)
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    # 2. Hyperparameter search — engine-grouped CV, no leakage across units
    best_model, best_cv_rmse = tune_lgbm(X_train, y_train, groups=train_df["unit"])
    print(f"Best CV RMSE (grouped): {best_cv_rmse:.2f}")

    # 3. Evaluate on the held-out test set
    y_pred = best_model.predict(X_test)
    mae, rmse, r2 = evaluate_predictions(y_test, y_pred)
    print_metrics(mae, rmse, r2)

    # 4. Diagnostic plots
    plot_actual_vs_predicted(y_test, y_pred)
    plot_unit_rul(best_model, train_df, features, unit_id=1)


if __name__ == "__main__":
    main()