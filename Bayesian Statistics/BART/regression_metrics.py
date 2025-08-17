import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def print_regression_metrics(y_true, y_pred, prefix=""):
    """
    Compute and print common regression metrics.

    Parameters
    ----------
    y_true : array‐like or torch.Tensor, shape (n_samples,) or (n_samples, n_targets)
    y_pred : array‐like or torch.Tensor, same shape as y_true
    prefix : str, optional
        If provided, prints this before each metric (e.g. "Test: MSE=...").

    Prints
    ------
    MSE, RMSE, MAE, R²
    """

    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mse)
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)

    fmt = "{prefix}{name}: {value:.4f}"
    for name, value in [
        ("MSE", mse),
        ("RMSE", rmse),
        ("MAE", mae),
        ("R²", r2),
    ]:
        print(fmt.format(prefix=(prefix + " ") if prefix else "", name=name, value=value))