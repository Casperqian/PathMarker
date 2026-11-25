"""Shared LASSO-default populations and case splits for downstream methods."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratification(frame, task_type):
    return frame['event_observed' if task_type == 'survival' else 'label'].astype(str)


def split_internal(frame, task_type, seed=42):
    """The original LASSO holdout: 15% of all internal, drawn from non-TCGA."""
    frame = frame.sort_values('case_id').reset_index(drop=True)
    if not frame.case_id.is_unique:
        raise ValueError('Duplicate cohort-case identifier')
    pool = frame.loc[~frame.cohort.str.startswith('TCGA')]
    if pool.empty:
        return frame, frame.iloc[:0].copy(), 'no_non_tcga_internal'
    n_test = int(np.ceil(.15 * len(frame)))
    if n_test >= len(pool):
        raise ValueError(f'15% of all internal requires {n_test} test cases, but only {len(pool)} non-TCGA cases are available for stratified splitting')
    y = stratification(pool, task_type)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return frame, frame.iloc[:0].copy(), 'insufficient_outcomes_for_stratified_test'
    _, test_ids = train_test_split(pool.case_id.to_numpy(), test_size=n_test,
                                  stratify=y.to_numpy(), random_state=seed)
    mask = frame.case_id.isin(test_ids)
    return (frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True),
            'fixed_total_internal_15pct_non_tcga_only')


def development_folds(frame, task_type, seed=42):
    """Return identical ordered cohort-case manifests for every method."""
    frame = frame.sort_values('case_id').reset_index(drop=True)
    if not frame.case_id.is_unique:
        raise ValueError('Duplicate cohort-case identifier')
    y = stratification(frame, task_type)
    if y.nunique() < 2 or y.value_counts().min() < 5:
        raise ValueError('Development outcomes cannot support five stratified folds')
    manifests = []
    for _, vi in StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, y):
        manifest = frame[['case_id', 'cohort', 'patient_id']].copy()
        manifest['role'] = 'train'
        manifest.loc[vi, 'role'] = 'validation'
        manifests.append(manifest)
    return manifests


def prepare_experiment(setting_id, seed=42, loaded=None):
    """Use current LASSO labels/readouts as the common reference population."""
    from biomarker.downstream.lasso.settings import SETTINGS, TASKS
    from biomarker.downstream.lasso.holdout import prepare_setting

    setting = SETTINGS[setting_id]
    prepared = prepare_setting(setting, TASKS[setting.task], loaded=loaded, seed=seed)
    prepared['folds'] = development_folds(prepared['train'], prepared['cfg']['task_type'], seed)
    return prepared


def experiment_dir(method, setting_id, seed=42):
    from biomarker.downstream.utils import FINAL_ROOT
    from biomarker.downstream.lasso.settings import SETTINGS

    setting = SETTINGS[setting_id]
    root = FINAL_ROOT / method / setting.task
    if method == 'lasso':
        return root / f'seed_{seed}'
    root = root / setting.external_cohort
    return root if seed == 42 else root / f'seed_{seed}'


def require_cases(frame, case_ids, context):
    """Apply reference membership without dropping or reassigning missing cases."""
    ids = list(case_ids)
    if not frame.case_id.is_unique:
        raise ValueError(f'{context}: duplicate case IDs')
    indexed = frame.set_index('case_id', drop=False)
    missing = [case for case in ids if case not in indexed.index]
    if missing:
        raise ValueError(f'{context}: missing {len(missing)} reference cases, examples: {missing[:5]}')
    return indexed.loc[ids].reset_index(drop=True)


def save_splits(prepared, destination):
    """Save inspectable memberships alongside each method's fitted outputs."""
    root = Path(destination)
    folder = root / 'manifest'
    folder.mkdir(parents=True, exist_ok=True)
    cfg = prepared['cfg']
    folds = prepared.get('folds')
    if folds is None:
        folds = development_folds(prepared['train'], cfg['task_type'], cfg['seed'])
    for i, frame in enumerate(folds):
        frame.to_csv(folder / f'fold_{i}.csv', index=False)
    outcomes = ['event_time', 'event_observed'] if cfg['task_type'] == 'survival' else ['label']
    columns = ['case_id', 'cohort', 'patient_id', *outcomes]
    prepared['internal_test'][columns].assign(role='internal_test').to_csv(folder / 'internal_test.csv', index=False)
    pd.concat(list(prepared['external'].values()), ignore_index=True)[columns].assign(role='external').to_csv(folder / 'external.csv', index=False)
    (root / 'split_config.json').unlink(missing_ok=True)
