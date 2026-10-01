"""LightGBM model definition and hyperparameter search."""

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from lightgbm import LGBMRegressor

from config import RANDOM_STATE


def make_lgbm():
    """Base LightGBM regressor with project defaults.

    n_jobs=1 so it doesn't collide with the search's parallelism.
    """
    return LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


PARAM_DIST = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(3, 12),
    "num_leaves": randint(15, 100),
    "min_child_samples": randint(10, 50),
    "learning_rate": loguniform(0.01, 0.2),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
}


def tune_lgbm(X_train, y_train, groups, n_iter=20, n_splits=3):
    """Random search with engine-grouped CV (no leakage across units).

    `groups` must be the unit id per training row, e.g. train_df['unit'].
    Returns (best_estimator, best_cv_rmse).
    """
    search = RandomizedSearchCV(
        estimator=make_lgbm(),
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=GroupKFold(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, -search.best_score_