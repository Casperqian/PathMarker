"""Fixed internal holdout and development-ranked LASSO with a frozen final model."""
from __future__ import annotations

import argparse
import time
import torch

from biomarker.downstream.lasso.settings import DEFAULT_SEED


def run_setting(setting_id, seed=DEFAULT_SEED):
    from biomarker.downstream.lasso.settings import SETTINGS, TASKS
    from biomarker.downstream.lasso.holdout import prepare_setting, fit_setting, save_setting
    start=time.monotonic();torch.set_num_threads(1)
    setting=SETTINGS[setting_id]
    print('START',setting_id,flush=True)
    prepared=prepare_setting(setting,TASKS[setting.task],seed=seed)
    result=fit_setting(prepared)
    from biomarker.downstream.utils.experiment import experiment_dir
    report=save_setting(prepared,result,experiment_dir('lasso',setting_id,seed))
    print('COMPLETE',setting_id,'seconds',round(time.monotonic()-start,1),flush=True)
    return setting_id


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from biomarker.downstream.lasso.settings import SETTINGS, UNAVAILABLE
    parser=argparse.ArgumentParser(description='LASSO with one fixed external per task, configured heldout policy, and a frozen development model.')
    parser.add_argument('--settings',nargs='+',choices=list(SETTINGS),help='Explicit task/external setting identifiers; default: all')
    parser.add_argument('--tasks',help='Comma-separated task names; default: all configured tasks')
    parser.add_argument('--panel-mode',choices=['default'],default='default',help='Development-only feature and model selection')
    parser.add_argument('--workers',type=int,default=4,choices=[1,2,3,4])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Case holdout and CV split seed (default: %(default)s)')
    args=parser.parse_args()
    selected=list(args.settings or SETTINGS)
    if args.tasks:
        tasks=set(args.tasks.split(','));unknown=tasks-{s.task for s in SETTINGS.values()}-set(UNAVAILABLE)
        if unknown:parser.error('Unknown tasks: '+','.join(sorted(unknown)))
        selected=[k for k in selected if SETTINGS[k].task in tasks]
    for task,reason in UNAVAILABLE.items():print('UNAVAILABLE',task,reason,flush=True)
    failures=[]
    if args.workers==1:
        for key in selected:
            try:run_setting(key,seed=args.seed)
            except Exception as exc:
                import traceback
                traceback.print_exc();failures.append((key,str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(run_setting,key,args.seed):key for key in selected}
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:
                    import traceback
                    traceback.print_exc();failures.append((futures[future],str(exc)))
    for key,error in failures:print('FAILED',key,error,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':
    main()
