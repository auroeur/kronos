import sklearn.metrics as metrics
import warnings

def classification_scores(y_true, y_pred, suppress_warning=True ):
    scores = {}

    with warnings.catch_warnings():
        if suppress_warning:
            warnings.simplefilter("ignore")
        scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        # Average of recalls obtained for each class
        scores['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true, y_pred)
        scores['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)
        # Suppress 'UserWarning: y_pred contains classes not in y_true' in console output
        scores['classification_report'] = metrics.classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    return scores
