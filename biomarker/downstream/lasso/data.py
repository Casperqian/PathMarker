"""Case aggregation and feature-panel discovery."""
import numpy as np
from biomarker.downstream.utils.tme_table import matrix_from_samples


def pool_case_frame(frame, features, outcomes):
    """Average raw slide features within each case, preserving its outcomes."""
    groups = frame.groupby("case_id", sort=True)
    if any(groups[column].nunique(dropna=False).gt(1).any() for column in outcomes):
        raise ValueError("Inconsistent outcomes within an internal case")
    return groups[list(features)].mean().join(groups[list(outcomes)].first()).reset_index()


def linear_case_frame(table, features, task_type):
    """Aggregate raw slide features before fitting any transformations."""
    import pandas as pd

    frame = pd.DataFrame(matrix_from_samples(table.samples, list(features)), columns=features)
    frame['case_id'] = [str(s.case_id) for s in table.samples]
    outcomes = ['event_time', 'event_observed'] if task_type == 'survival' else ['label']
    for column in outcomes:
        frame[column] = [getattr(s, column) for s in table.samples]
    return pool_case_frame(frame, features, outcomes)


def _linear_effect(frame, feature, contrast, task_type):
    from biomarker.downstream.utils.association import fit_cox_feature, fit_logistic_feature

    if task_type == 'survival':
        result = fit_cox_feature(frame[feature], frame.event_time, frame.event_observed,
                                 minimum_event_count=5)
        effect = result['log_hazard_ratio']
    else:
        result = fit_logistic_feature(frame[feature], frame.label.astype(str).eq(str(contrast)).astype(int))
        effect = result['log_odds_ratio']
    valid = (result['status'] in ('ok', 'optimizer_warning')
             and np.isfinite(effect) and np.isfinite(result['standard_error'])
             and result['standard_error'] > 0)
    return float(effect) if valid else np.nan


def rank_linear_features(frame, features, task_type, labels=(), maximum=100):
    """Rank training-frame IQR effects; preserve the winning OVR contrast."""
    comparisons = [None] if task_type == 'survival' else (list(labels)[-1:] if len(labels) == 2 else list(labels))
    rows = []
    for feature in features:
        effects = [(contrast, _linear_effect(frame, feature, contrast, task_type)) for contrast in comparisons]
        valid = [(contrast, effect) for contrast, effect in effects if np.isfinite(effect) and effect != 0]
        if valid:
            contrast, effect = max(valid, key=lambda item: abs(item[1]))
            rows.append((str(feature), contrast, effect))
    rows.sort(key=lambda row: (-abs(row[2]), row[0]))
    chosen = rows[:maximum]
    return ([row[0] for row in chosen], {row[0]: row[1] for row in chosen},
            {row[0]: row[2] for row in chosen})


def load_current_cohort(task, spec):
    """Load normalized current slide readouts for a cohort, before case pooling."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from biomarker.downstream.lasso.settings import stage_label, NANFANG_LABEL_MAPS
    from biomarker.downstream.utils import READOUT_ROOT
    from biomarker.downstream.utils.tme_table import build_task_table

    d = pd.read_csv(spec.label_csv, dtype={'case': str, 'slide': str, 'WSI': str}, low_memory=False)
    original_cases = d.case.nunique()
    if task.task_type == 'survival':
        if spec.normalization == 'path_survival':
            d['Event'] = pd.to_numeric(d['time (months)'], errors='coerce') * spec.time_scale
            d['Status'] = pd.to_numeric(d.status.astype(str).str.split(':').str[0], errors='coerce')
        else:
            d['Event'] = pd.to_numeric(d.Event, errors='coerce') * spec.time_scale
            d['Status'] = pd.to_numeric(d.Status, errors='coerce')
        d = d[d.Event.ge(0) & np.isfinite(d.Event) & d.Status.isin([0, 1])].copy()
        outcome = ['Event', 'Status']
    else:
        if spec.normalization in NANFANG_LABEL_MAPS:
            d['label']=d['class'].str.strip().map(NANFANG_LABEL_MAPS[spec.normalization])
            if d.label.isna().any():raise ValueError(f'{task.name}/{spec.name}: unmapped original class')
        elif spec.normalization == 'stage':
            d['label'] = d['class' if 'class' in d else 'label'].map(stage_label)
            d = d[d.label.notna()].copy()
        elif spec.normalization == 'brca_TNM-T':
            values = d['class'].str.upper().str.strip()
            d['label'] = values.map(lambda v: 'T1' if v=='T1' else 'T1+' if v in ('T2','T3','T4') else None)
            d = d[d.label.notna()].copy()
        elif spec.normalization == 'brca_TNM-N':
            values = d['class'].str.upper().str.strip()
            d['label'] = values.map(lambda v: 'N0' if v=='N0' else 'N+' if v in ('N+','N1','N2','N3') else None)
            d = d[d.label.notna()].copy()
        d['label'] = d.label.astype(str)
        if not set(d.label).issubset(task.labels):
            raise ValueError(f'{task.name}/{spec.name}: incompatible labels {set(d.label)-set(task.labels)}')
        outcome = ['label']
    if any(d.groupby('case')[c].nunique().gt(1).any() for c in outcome):
        raise ValueError(f'{task.name}/{spec.name}: conflicting case outcomes')
    d['slide'] = d.slide.fillna('').str.replace('/', ';', regex=False)
    d['WSI'] = d.slide
    with tempfile.TemporaryDirectory(prefix='lasso_labels_') as directory:
        label = Path(directory)/'labels.csv'
        d[['case','slide','WSI',*outcome]].to_csv(label,index=False)
        table = build_task_table(label,READOUT_ROOT,spec.readout,task_type=task.task_type,sample_level='slide')
    if not table.samples:
        raise ValueError(f'{task.name}/{spec.name}: no matching TME readouts')
    source = dict(label_csv=str(spec.label_csv), readout=spec.readout,
                  normalization=spec.normalization, time_scale=spec.time_scale,
                  label_cases=int(original_cases), valid_label_cases=int(d.case.nunique()),
                  matched_cases=len({str(s.case_id) for s in table.samples}),
                  matched_slides=len(table.samples))
    return table, source


def load_current_task(task):
    """Read current labels/readouts; no historical manifests or disk caches."""
    import pandas as pd
    frames, sources, universe = {}, {}, None
    for spec in task.cohorts:
        table, source = load_current_cohort(task, spec)
        if universe is None:
            universe = list(table.feature_names)
        # Feature names come from TCGA's schema, never from external outcomes.
        frame = linear_case_frame(table,universe,task.task_type)
        counts = pd.Series([s.case_id for s in table.samples]).value_counts()
        frame['n_slides'] = frame.case_id.map(counts).astype(int)
        frame['patient_id'] = frame.case_id.astype(str)
        frame['case_id'] = spec.name+'::'+frame.patient_id
        frame['cohort'] = spec.name
        frames[spec.name] = frame
        sources[spec.name] = source
        print('COHORT',task.name,spec.name,len(frame),'slides',len(table.samples),flush=True)
    return frames, universe, sources
