import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay
import numpy as np
from numpy.typing import ArrayLike


def predict_and_plot(
    model_param: ClassifierMixin,
    inputs: ArrayLike,
    targets: ArrayLike,
    name: str = ''
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a classification model by computing key metrics and visualizing
    the confusion matrix.

    Args:
        model_param: A trained scikit-learn classifier with `predict` and
                     `predict_proba` methods.
        inputs: Feature matrix used for prediction.
        targets: True class labels corresponding to `inputs`.
        name: Optional label used in plot titles and printed output for
              identification purposes.

    Returns:
        A tuple of (probs, preds) where:
            - probs: Predicted probabilities for the positive class, shape (n_samples,).
            - preds: Predicted class labels, shape (n_samples,).
    """
    # 1. Predict probabilities for the positive class
    probs = model_param.predict_proba(inputs)[:, 1]

    # 2. Predict the class labels
    preds = model_param.predict(inputs)

    # 3. Calculate and display the Confusion Matrix
    conf_matrix = confusion_matrix(targets, preds, normalize='true')
    print("Confusion Matrix ({name}):\n", conf_matrix)

    # 4. Calculate and display the F1 Score
    f1 = f1_score(targets, preds)
    print("F1 score: {:.2f}%".format(f1 * 100))

    # 5. Calculate and display the AUROC score
    roc_auc = roc_auc_score(targets, probs)
    print(f"AUROC Score ({name}): {roc_auc:.4f}")

    # 6. Confusion Matrix plot
    plt.figure()
    sns.heatmap(conf_matrix, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))

    return probs, preds


def roc_curve_train_vs_validation(
    model_param: ClassifierMixin,
    inputs_train: ArrayLike,
    targets_train: ArrayLike,
    inputs_val: ArrayLike,
    targets_val: ArrayLike
) -> None:
    """
    Plots the ROC curves for training and validation sets on the same axes,
    allowing visual comparison of model performance across both splits.

    Args:
        model_param: A trained scikit-learn classifier compatible with
                     `RocCurveDisplay.from_estimator`.
        inputs_train: Feature matrix for the training set.
        targets_train: True class labels for the training set.
        inputs_val: Feature matrix for the validation set.
        targets_val: True class labels for the validation set.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    RocCurveDisplay.from_estimator(model_param, inputs_train, targets_train, ax=ax, name='Train Data ROC')
    RocCurveDisplay.from_estimator(model_param, inputs_val, targets_val, ax=ax, name='Validation Data ROC')

    plt.title('ROC Curve - Train vs Validation Data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()