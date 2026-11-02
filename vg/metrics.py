import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .common import CLASSES


def classification_metrics(targets, predictions):
    report = classification_report(targets, predictions, labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    return {'accuracy': float(np.mean(np.asarray(targets) == np.asarray(predictions))),
            'macro_f1': report['macro avg']['f1-score'], 'per_class': report,
            'confusion_matrix': confusion_matrix(targets, predictions, labels=list(range(len(CLASSES)))).tolist(),
            'count': len(targets)}


def select_threshold(targets, scores, minimum_accuracy=.90, minimum_coverage=.50):
    """Validation-only empirical selection; this does not calibrate probabilities."""
    confidence, predicted = scores.max(axis=1), scores.argmax(axis=1)
    for threshold in np.linspace(0, .95, 20):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage >= minimum_coverage and np.mean(predicted[accepted] == targets[accepted]) >= minimum_accuracy:
            return float(threshold)
    return None


def selective_metrics(targets, scores, threshold):
    if threshold is None:
        return {'threshold': None, 'coverage': None, 'accepted_accuracy': None,
                'note': 'Validation target was not met; selective prediction is disabled.'}
    accepted = scores.max(axis=1) >= threshold
    return {'threshold': threshold, 'coverage': float(accepted.mean()),
            'accepted_count': int(accepted.sum()),
            'accepted_accuracy': float(np.mean(scores.argmax(axis=1)[accepted] == targets[accepted])) if accepted.any() else None}
