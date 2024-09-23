import numpy as np
from typing import Union, Sequence

def accuracy(y_true: Sequence[Union[int, float]], y_pred: Sequence[Union[int, float]]) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (Sequence[Union[int, float]]): True labels.
        y_pred (Sequence[Union[int, float]]): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision(y_true: Sequence[Union[int, float]], y_pred: Sequence[Union[int, float]], positive_label: Union[int, float] = 1) -> float:
    """
    Calculate the precision score.

    Args:
        y_true (Sequence[Union[int, float]]): True labels.
        y_pred (Sequence[Union[int, float]]): Predicted labels.
        positive_label (Union[int, float]): The label of the positive class. Default is 1.

    Returns:
        float: Precision score.
    """
    true_positives = np.sum((np.array(y_true) == positive_label) & (np.array(y_pred) == positive_label))
    predicted_positives = np.sum(np.array(y_pred) == positive_label)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_true: Sequence[Union[int, float]], y_pred: Sequence[Union[int, float]], positive_label: Union[int, float] = 1) -> float:
    """
    Calculate the recall score.

    Args:
        y_true (Sequence[Union[int, float]]): True labels.
        y_pred (Sequence[Union[int, float]]): Predicted labels.
        positive_label (Union[int, float]): The label of the positive class. Default is 1.

    Returns:
        float: Recall score.
    """
    true_positives = np.sum((np.array(y_true) == positive_label) & (np.array(y_pred) == positive_label))
    actual_positives = np.sum(np.array(y_true) == positive_label)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true: Sequence[Union[int, float]], y_pred: Sequence[Union[int, float]], positive_label: Union[int, float] = 1) -> float:
    """
    Calculate the F1 score.

    Args:
        y_true (Sequence[Union[int, float]]): True labels.
        y_pred (Sequence[Union[int, float]]): Predicted labels.
        positive_label (Union[int, float]): The label of the positive class. Default is 1.

    Returns:
        float: F1 score.
    """
    prec = precision(y_true, y_pred, positive_label)
    rec = recall(y_true, y_pred, positive_label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0