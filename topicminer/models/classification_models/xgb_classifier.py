# topicminer/models/classification_models/classification_modles/xgb_classifier.py
"""
This module defines and manages operations related to the XBGoost classification, 
including model training, parameter optimization, and performance evaluation.
"""

import numpy as np

from sklearn.preprocessing import LabelEncoder

# from topicminer.models.classification_models.xgb_classifier import XGBClassifier
from xgboost import XGBClassifier

# Bayesian Optimization library
from bayes_opt import BayesianOptimization

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import parallel_backend
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize

import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # Ignore deprecation warnings

def train_xgb_classifier(
    X_train_bal, y_train_bal, params, eval_metric="mlogloss", workers=None
):

    if workers and workers > 8:
        workers = 8
        print("Max workers is 8, setting workers to 8")

    # Initialize the XGBoost classifier with provided parameters

    xgb_clf = XGBClassifier(
        use_label_encoder=False,  # Add this line
        verbosity=0,  # zero turns off XGBoost logging
        n_estimators=int(params["params"]["n_estimators"]),
        max_depth=int(params["params"]["max_depth"]),
        learning_rate=float(params["params"]["learning_rate"]),
        min_child_weight=int(params["params"]["min_child_weight"]),
        subsample=params["params"]["subsample"],
        colsample_bytree=params["params"]["colsample_bytree"],
        gamma=params["params"]["gamma"],
        eval_metric=eval_metric,
        n_jobs=workers,
    )

    # Fit the model on the balanced training data
    xgb_clf.fit(X_train_bal, y_train_bal)

    return xgb_clf


def xgb_classifier_cross_val(params, data, targets, scoring="f1", workers=None):
    if workers and workers > 8:
        workers = 8
        print("Max workers is 8, setting workers to 8")

    estimator = XGBClassifier(
        use_label_encoder=False,  # Add this line
        verbosity=0,  # Turning off XGBoost logging
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=int(params["min_child_weight"]),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        eval_metric="mlogloss",
        n_jobs=workers,
    )

    if scoring not in ["accuracy", "roc_auc"]:
        if scoring == "precision":
            scorer = make_scorer(precision_score, average="weighted")
        elif scoring == "recall":
            scorer = make_scorer(recall_score, average="weighted")
        elif scoring == "f1":
            scorer = make_scorer(f1_score, average="weighted")
        else:
            raise ValueError("Unsupported scoring method")
    else:
        scorer = scoring

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    try:
        results = cross_val_score(
            estimator, data, targets, scoring=scorer, cv=skf, n_jobs=workers
        )
    except Exception as e:
        print("An error occurred during model training:", e)
        return -np.inf  # Return negative infinity to allow optimizer to continue

    return np.mean(results)


def optimize_xgb(data, targets, scoring="f1", workers=None):
    param_bounds = {
        "n_estimators": (100, 200),
        "max_depth": (3, 6),
        "learning_rate": (0.05, 0.1),
        "min_child_weight": (1, 10),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "gamma": (0, 5),
    }

    optimizer = BayesianOptimization(
        f=lambda n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma: xgb_classifier_cross_val(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "gamma": gamma,
            },
            data,
            targets,
            scoring=scoring,
            workers=workers,
        ),
        pbounds=param_bounds,
        random_state=1,
        verbose=2,  # You may want to set verbose=0 to further reduce output
    )

    optimizer.maximize(init_points=2, n_iter=10)
    return optimizer.max

def plot_multi_class_roc_auc(y_true, y_scores, n_classes, title='Multi-class ROC'):
    # Binarize the output labels for each class
    y_test = label_binarize(y_true, classes=[*range(n_classes)])
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average and compute AUC for the macro-average
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"], _ = all_fpr, mean_tpr, None
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'yellow', 'green', 'red', 'purple', 'pink', 'brown', 'grey', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

def score_xgb_classifier(X_train_bal, X_test, y_train_bal, y_test, xgb_clf):
    # Predict and evaluate on the training set
    y_train_pred = xgb_clf.predict(X_train_bal)
    train_accuracy = accuracy_score(y_train_bal, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train_bal, y_train_pred, average="weighted"
    )

    # Predict and evaluate on the test set
    y_test_pred = xgb_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted"
    )

    print("==== Training Metrics ====")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")

    print("\n==== Testing Metrics ====")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

