"""Summarize fixed-test LASSO results for one seed."""
import argparse
from pathlib import Path
import pandas as pd
from biomarker.downstream.lasso.settings import DEFAULT_SEED, SETTINGS


def read_result(root):
    root=Path(root)
    report=pd.read_csv(root/'performance_report.csv')
    task=report.task.iloc[0]
    setting=next(s for s in SETTINGS.values() if s.task==task)
    features=pd.read_csv(root/'feature.csv')
    if list(features.columns)!=['feature'] or not report.n_features.eq(len(features)).all():
        raise ValueError(f'{root}: inconsistent final panel')
    for name in [*(f'fold_{f}' for f in range(5)),'final']:
        if not (root/name/'model.joblib').exists():raise FileNotFoundError(root/name/'model.joblib')
    report.insert(1,'external_setting',setting.external_cohort)
    report.insert(2,'seed',int(root.name.removeprefix('seed_')))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', help='Comma-separated tasks; default: all configured tasks')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Report seed (default: %(default)s)')
    args = parser.parse_args()
    tasks = set(args.tasks.split(',')) if args.tasks else None
    frames = []
    for key, setting in SETTINGS.items():
        if tasks is not None and setting.task not in tasks:
            continue
        run = Path(setting.output_dir) / f'seed_{args.seed}'
        if not (run / 'performance_report.csv').exists():
            print('NOT_COMPLETED', key)
            continue
        frames.append(read_result(run))
    if frames:
        print(pd.concat(frames, ignore_index=True).to_string(index=False))


if __name__=='__main__':main()
