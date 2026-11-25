"""Unified LASSO predictions and cohort metrics."""
def linear_predictions(bundle, frame):
    """Return class probabilities or survival probabilities on one finite grid."""
    from biomarker.downstream.lasso.model import survival_on_times
    x = bundle['preprocessor'].transform(frame[bundle['features']].to_numpy())
    probabilities = bundle['model'].predict_proba(x)
    if bundle['task_type'] == 'survival':
        return survival_on_times(probabilities,bundle['cuts'],bundle['eval_times'])
    return probabilities


def linear_metric(frame, prediction, task_type, labels):
    from sklearn.metrics import roc_auc_score
    from biomarker.downstream.utils.survival import concordance_index
    if task_type == 'survival':
        return float(concordance_index(frame.event_time,frame.event_observed,-prediction.mean(axis=1)))
    y = frame.label.astype(str).map({s:i for i,s in enumerate(labels)}).to_numpy(int)
    if task_type == 'binary':
        return float(roc_auc_score(y,prediction[:,1]))
    return float(roc_auc_score(y,prediction,multi_class='ovr',average='macro',labels=list(range(len(labels)))))
