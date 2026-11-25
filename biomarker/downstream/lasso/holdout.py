"""Case-level fixed test, development screening, and one frozen final model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from biomarker.downstream.utils.experiment import (
    stratification, split_internal, development_folds, require_cases, save_splits,
)
from biomarker.downstream.lasso.data import rank_linear_features
from biomarker.downstream.lasso.model import fit_linear, survival_targets
from biomarker.downstream.lasso.evaluate import linear_predictions, linear_metric
from biomarker.downstream.utils.tme_table import TabularPreprocessor


def prepare_setting(setting, task, loaded=None, *, seed=42):
    from biomarker.downstream.lasso.data import load_current_task
    frames,features,sources=loaded if loaded is not None else load_current_task(task)
    all_internal=pd.concat([frames[c] for c in setting.internal_cohorts],ignore_index=True)
    if setting.internal_holdout:
        dev,test,status=split_internal(all_internal,task.task_type,seed=seed)
    else:
        if not all_internal.case_id.is_unique:raise ValueError('Duplicate cohort-case identifier')
        dev=all_internal.sort_values('case_id').reset_index(drop=True)
        test=dev.iloc[:0].copy()
        status='all_internal_development'
    external=frames[setting.external_cohort]
    assert not set(external.case_id)&set(all_internal.case_id)
    return dict(cfg=dict(task=task.name,task_type=task.task_type,labels=list(task.labels),seed=seed,time_unit=task.time_unit,
                         internal_holdout=setting.internal_holdout),
                train=dev,internal_test=test,external={setting.external_cohort:external},feature_names=features,
                sources=sources,split_status=status,setting_id=task.name+'/'+setting.external_cohort)


def select_candidate(summary):
    good=summary[np.isfinite(summary.score)].copy()
    if good.empty:raise ValueError('No candidate has five finite validation metrics')
    eligible=good[good.score>=good.score.max()-.005-1e-12]
    return eligible.sort_values(['k','score','l1_strength'],ascending=[True,False,False]).iloc[0].to_dict()


def metric(frame,prediction,kind,labels):
    if len(frame)==0:return float('nan')
    if kind!='survival' and set(frame.label)!=set(labels):return float('nan')
    return linear_metric(frame,prediction,kind,labels)


def targets(frame,kind,labels):
    if kind=='survival':
        cuts=np.quantile(frame.loc[frame.event_observed.eq(1),'event_time'],[.25,.5,.75,.9])
        return survival_targets(frame.event_time,frame.event_observed,cuts),cuts
    return frame.label.map({s:i for i,s in enumerate(labels)}).to_numpy(int),None


def fit_bundle(train,validation,features,kind,labels,lam,max_iter,fold,cache=None,seed=42):
    if cache is None:
        pre=TabularPreprocessor().fit(train[features].to_numpy(),features)
        x=pre.transform(train[features].to_numpy())
        target,cuts=targets(train,kind,labels)
    else:pre,x,target,cuts=cache
    if not pre.feature_names:raise ValueError('No nonconstant features in training fold')
    hp=dict(optimizer='proximal',lr=1.,l1_strength=float(lam),class_weight='none' if kind=='survival' else 'balanced')
    model,info=fit_linear(x,target,task_type=kind,n_outputs=4 if kind=='survival' else 1 if kind=='binary' else len(labels),
                          max_iter=max_iter,device='cpu',**hp)
    return dict(model=model,preprocessor=pre,features=list(features),labels=list(labels),task_type=kind,
        cuts=cuts,eval_times=cuts,fold=fold,hyperparameters=hp,training=info,seed=seed,
        train_cases=train.case_id.tolist(),validation_cases=validation.case_id.tolist())


def fit_setting(prepared,*,max_iter=2000,maximum=100):
    cfg=prepared['cfg'];seed=cfg['seed'];frame=prepared['train'].sort_values('case_id').reset_index(drop=True)
    labels=cfg['labels'];kind='survival' if cfg['task_type']=='survival' else 'binary' if len(labels)==2 else 'multiclass'
    y=stratification(frame,cfg['task_type'])
    if y.nunique()<2 or y.value_counts().min()<5:raise ValueError('Development outcomes cannot support five stratified folds')
    ranked,_,_=rank_linear_features(frame,prepared['feature_names'],cfg['task_type'],labels,maximum=maximum)
    limit=min(50,maximum,len(ranked))
    if not limit:raise ValueError('No estimable features in development')
    folds=[]
    for manifest in development_folds(frame,cfg['task_type'],seed):
        tr=require_cases(frame,manifest.loc[manifest.role.eq('train'),'case_id'],'LASSO train')
        va=require_cases(frame,manifest.loc[manifest.role.eq('validation'),'case_id'],'LASSO validation')
        target,cuts=targets(tr,kind,labels)
        folds.append((tr,va,target,cuts))
    summaries=[];best_by_k={}
    for k in range(1,limit+1):
        features=ranked[:k]
        caches=[]
        for tr,va,target,cuts in folds:
            pre=TabularPreprocessor().fit(tr[features].to_numpy(),features)
            caches.append((pre,pre.transform(tr[features].to_numpy()),target,cuts))
        for lam in [1e-4,1e-3,1e-2]:
            models=[];scores=[]
            for fold,((tr,va,_,_),cache) in enumerate(zip(folds,caches)):
                bundle=fit_bundle(tr,va,features,kind,labels,lam,max_iter,fold,cache,seed=seed)
                score=metric(va,linear_predictions(bundle,va),kind,labels)
                bundle['validation_score']=score;models.append(bundle);scores.append(score)
            mean=float(np.mean(scores));summaries.append(dict(k=k,l1_strength=lam,score=mean))
            old=best_by_k.get(k)
            if np.isfinite(mean) and (old is None or (mean,lam)>old[:2]):best_by_k[k]=(mean,lam,models)
    choice=select_candidate(pd.DataFrame(summaries));k=int(choice['k']);models=best_by_k[k][2]
    final=fit_bundle(frame,frame.iloc[:0],ranked[:k],kind,labels,choice['l1_strength'],max_iter,'final',seed=seed)
    final['fixed_test_cases']=prepared['internal_test'].case_id.tolist()
    final['case_assignments']=pd.concat([frame[['case_id','cohort','patient_id']].assign(role='development'),
        prepared['internal_test'][['case_id','cohort','patient_id']].assign(role='internal_test')],ignore_index=True).to_dict('records')
    return dict(models=models,final=final,features=ranked[:k],choice=choice,max_iter=max_iter)


def nonzero(bundle):
    w=bundle['model'].linear.weight.detach().cpu().numpy()
    return {f for f,yes in zip(bundle['preprocessor'].feature_names,np.any(np.abs(w)>1e-8,axis=0)) if yes}


def population(frame,kind,labels):
    return dict(n_cases=len(frame),n_slides=int(frame.n_slides.sum()),
        n_events=int(frame.event_observed.sum()) if kind=='survival' else None,
        class_counts=None if kind=='survival' else json.dumps(frame.label.value_counts().sort_index().to_dict()))


def save_setting(prepared,result,destination):
    import joblib
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);(dest/'predictions').mkdir(exist_ok=True)
    save_splits(prepared,dest)
    final=result['final'];kind=final['task_type'];labels=final['labels'];cfg=prepared['cfg'];models=result['models']
    common=set.intersection(*(nonzero(b) for b in models));final_nz=nonzero(final)
    pd.DataFrame({'feature':result['features']}).to_csv(dest/'feature.csv',index=False)
    training=[];coefficients=[];report=[]
    scores=[b['validation_score'] for b in models]
    common_info=dict(task=cfg['task'],metric='cindex' if kind=='survival' else 'auc',n_features=len(result['features']),
        final_nonzero=len(final_nz),cv_nonzero_intersection=len(common),cv_converged_folds=sum(b['training']['converged'] for b in models),
        final_converged=final['training']['converged'],l1_strength=result['choice']['l1_strength'])
    report.append(dict(**common_info,cohort='development',score=float(np.mean(scores)),fold_sd=float(np.std(scores,ddof=1)),
                       **population(prepared['train'],kind,labels)))
    for bundle in [*models,final]:
        name='final' if bundle['fold']=='final' else 'fold_'+str(bundle['fold']);folder=dest/name;folder.mkdir(exist_ok=True)
        joblib.dump(bundle,folder/'model.joblib',compress=3)
        restored=joblib.load(folder/'model.joblib');sample=prepared['train'].iloc[:8]
        np.testing.assert_allclose(linear_predictions(restored,sample),linear_predictions(bundle,sample),rtol=0,atol=0)
        tr=prepared['train'][prepared['train'].case_id.isin(bundle['train_cases'])]
        va=prepared['train'][prepared['train'].case_id.isin(bundle['validation_cases'])]
        training.append(dict(model=name,score=bundle.get('validation_score',np.nan),**(bundle['hyperparameters'] | bundle['training']),
            train_counts=json.dumps(population(tr,kind,labels)),validation_counts=json.dumps(population(va,kind,labels))))
        w=bundle['model'].linear.weight.detach().numpy();bias=bundle['model'].linear.bias.detach().numpy()
        for output in range(len(w)):
            for j,feature in enumerate(bundle['preprocessor'].feature_names):coefficients.append(dict(model=name,output=output,feature=feature,coefficient=float(w[output,j]),intercept=float(bias[output])))
    evaluations=[]
    if len(prepared['internal_test']):
        evaluations.append(('internal_test',prepared['internal_test']))
    evaluations.extend(prepared['external'].items())
    for co,frame in evaluations:
        prediction=linear_predictions(final,frame);score=metric(frame,prediction,kind,labels)
        saved=frame[['case_id','patient_id','cohort','n_slides',*(['event_time','event_observed'] if kind=='survival' else ['label'])]].copy()
        if kind=='survival':
            saved['risk']=-prediction.mean(axis=1)
            for j,t in enumerate(final['eval_times']):saved[f'survival_at_{t:.8g}']=prediction[:,j]
        else:
            for j,label in enumerate(labels):saved['prob_'+label]=prediction[:,j]
        saved.to_csv(dest/'predictions'/f'{co}.csv',index=False)
        report.append(dict(**common_info,cohort=co,score=score,fold_sd=np.nan,**population(frame,kind,labels)))
        if co=='internal_test':
            for center,f in frame.groupby('cohort'):
                idx=np.flatnonzero(frame.cohort.eq(center))
                report.append(dict(**common_info,cohort=center,score=metric(f,prediction[idx],kind,labels),fold_sd=np.nan,**population(f,kind,labels)))
    pd.DataFrame(training).to_csv(dest/'training.csv',index=False)
    pd.DataFrame(coefficients).to_csv(dest/'coefficients.csv',index=False)
    report=pd.DataFrame(report)
    report['metric_status']=np.where(np.isfinite(report.score),'ok','not_estimable')
    report.to_csv(dest/'performance_report.csv',index=False)
    (dest/'performance.csv').unlink(missing_ok=True)
    if prepared['internal_test'].empty:
        (dest/'predictions/internal_test.csv').unlink(missing_ok=True)
    (dest/'config.json').unlink(missing_ok=True)
    return report
