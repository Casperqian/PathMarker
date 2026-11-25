"""Offline preparation of external labels; LASSO training reads only prepared CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
from biomarker.downstream.lasso.settings import LABEL, NANFANG_LABEL_MAPS, stage_label

SPECS = [{'task': 'BRCA-TNM-T', 'kind': 'classification', 'labels': ['T1', 'T1+'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_TNM-T/data/SDPH_Breast_TNM-T.csv', 'destination': LABEL / 'BRCA/SDPH_BRCA-TNM-T.csv', 'normalization': 'brca_TNM-T', 'scale': 1.0, 'unit': '', 'cohort': 'SDPH_BRCA'}, {'task': 'BRCA-TNM-N', 'kind': 'classification', 'labels': ['N0', 'N+'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_TNM-N/data/SDPH_Breast_TNM-N.csv', 'destination': LABEL / 'BRCA/SDPH_BRCA-TNM-N.csv', 'normalization': 'brca_TNM-N', 'scale': 1.0, 'unit': '', 'cohort': 'SDPH_BRCA'}, {'task': 'CRC-BRAF', 'kind': 'classification', 'labels': ['0', '1'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/BRAF_V600E/Nanfang_CRC_retrospective_BRAF_V600E.csv', 'destination': LABEL / 'CRC/NanFang_CRC-BRAF.csv', 'normalization': 'nanfang_BRAF', 'scale': 1.0, 'unit': '', 'cohort': 'NanFang_CRC'}, {'task': 'CRC-MMR', 'kind': 'classification', 'labels': ['msi', 'mss'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/MMR/data/Nanfang_CRC_retrospective_MMR.csv', 'destination': LABEL / 'CRC/NanFang_CRC-MMR.csv', 'normalization': 'nanfang_MMR', 'scale': 1.0, 'unit': '', 'cohort': 'NanFang_CRC'}, {'task': 'CRC-TNM-T', 'kind': 'classification', 'labels': ['T1/T2', 'T3/T4'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_T/data/Nanfang_CRC_retrospective_T_Staging_Coarse.csv', 'destination': LABEL / 'CRC/NanFang_CRC-TNM-T.csv', 'normalization': 'nanfang_TNM-T', 'scale': 1.0, 'unit': '', 'cohort': 'NanFang_CRC'}, {'task': 'CRC-TNM-N', 'kind': 'classification', 'labels': ['N0', 'N+'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_N/data/Nanfang_CRC_retrospective_N_Staging_Coarse.csv', 'destination': LABEL / 'CRC/NanFang_CRC-TNM-N.csv', 'normalization': 'nanfang_TNM-N', 'scale': 1.0, 'unit': '', 'cohort': 'NanFang_CRC'}, {'task': 'CRC-TNM', 'kind': 'classification', 'labels': ['I-II', 'III-IV'], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/TNM_Staging/data/Nanfang_CRC_retrospective_TNM_Staging.csv', 'destination': LABEL / 'CRC/NanFang_CRC-TNM.csv', 'normalization': 'stage', 'scale': 1.0, 'unit': '', 'cohort': 'NanFang_CRC'}, {'task': 'BRCA-OS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_OS/data/SDPH_Breast_OS.csv', 'destination': LABEL / 'BRCA/SDPH_BRCA-OS.csv', 'normalization': 'path_survival', 'scale': 1.0, 'unit': 'months', 'cohort': 'SDPH_BRCA'}, {'task': 'BRCA-OS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_OS/data/SRRSH_Breast_OS.csv', 'destination': LABEL / 'BRCA/SRRSH_BRCA-OS.csv', 'normalization': 'path_survival', 'scale': 1.0, 'unit': 'months', 'cohort': 'SRRSH_BRCA'}, {'task': 'BRCA-OS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_OS/data/ZJ1_Breast_OS.csv', 'destination': LABEL / 'BRCA/ZJ1_C2-OS.csv', 'normalization': 'path_survival', 'scale': 1.0, 'unit': 'months', 'cohort': 'ZJ1_C2'}, {'task': 'BRCA-DFS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_DFS/data/SDPH_Breast_DFS.csv', 'destination': LABEL / 'BRCA/SDPH_BRCA-DFS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'SDPH_BRCA'}, {'task': 'BRCA-DFS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_DFS/data/SRRSH_Breast_DFS.csv', 'destination': LABEL / 'BRCA/SRRSH_BRCA-DFS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'SRRSH_BRCA'}, {'task': 'BRCA-DFS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/breast/SurPost_DFS/data/ZJ1_Breast_DFS.csv', 'destination': LABEL / 'BRCA/ZJ1_C2-DFS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'ZJ1_C2'}, {'task': 'CRC-DFS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/colon/DFS/data/ARGO_Colon_DFS.csv', 'destination': LABEL / 'CRC/ARGO_CRC-DFS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'ARGO_CRC'}, {'task': 'THCA-DFS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/thyroid/DFS/data/Zhefuyi-Thyroid_DFS.csv', 'destination': LABEL / 'THCA/Zhefuyi_Throid-DFS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'Zhefuyi_Throid'}, {'task': 'THCA-OS', 'kind': 'survival', 'labels': [], 'source': '/jhcnas4/Pathology/code/PathTasks/data/thyroid/OS/data/Zhefuyi-Thyroid_OS.csv', 'destination': LABEL / 'THCA/Zhefuyi_Throid-OS.csv', 'normalization': 'path_survival', 'scale': 30.0, 'unit': 'days', 'cohort': 'Zhefuyi_Throid'}]


def normalize_labels(spec):
    d=pd.read_csv(spec['source'],dtype={'case':str,'slide':str,'WSI':str},low_memory=False)
    mode=spec['normalization']
    if spec['kind']=='survival':
        d['Event']=pd.to_numeric(d['time (months)'],errors='coerce')*spec['scale']
        d['Status']=pd.to_numeric(d.status.astype(str).str.split(':').str[0],errors='coerce')
        d=d[d.Event.ge(0)&np.isfinite(d.Event)&d.Status.isin([0,1])].copy()
        outcomes=['Event','Status']
    else:
        if mode in NANFANG_LABEL_MAPS:
            d['label']=d['class'].str.strip().map(NANFANG_LABEL_MAPS[mode])
            if d.label.isna().any():raise ValueError('Unmapped class: '+spec['source'])
        elif mode=='stage':
            d['label']=d['class' if 'class' in d else 'label'].map(stage_label)
        elif mode=='brca_TNM-T':
            d['label']=d['class'].str.upper().str.strip().map({'T1':'T1','T2':'T1+','T3':'T1+','T4':'T1+'})
        elif mode=='brca_TNM-N':
            d['label']=d['class'].str.upper().str.strip().map({'N0':'N0','N+':'N+','N1':'N+','N2':'N+','N3':'N+'})
        d=d[d.label.notna()].copy();d['label']=d.label.astype(str)
        assert set(d.label)<=set(spec['labels'])
        outcomes=['label']
    if any(d.groupby('case')[c].nunique().gt(1).any() for c in outcomes):
        raise ValueError('Conflicting case labels: '+spec['source'])
    d['slide']=d.slide.fillna('').str.replace('/',';',regex=False);d['WSI']=d.slide
    out=d[['case','slide',*outcomes,'WSI']].copy()
    if spec['kind']=='survival':out['time_unit']=spec['unit']
    out['label_source']=spec['source']
    return out.drop_duplicates().reset_index(drop=True)


def keep_available_readouts(frame, readouts):
    from biomarker.downstream.utils.tme_table import parse_slide_ids
    rows=[]
    for row in frame.fillna('').to_dict('records'):
        slides=[s for s in parse_slide_ids(row) if readouts.get(s) is not None]
        if slides:
            row['slide']=';'.join(slides)
            row['WSI']=row['slide']
            rows.append(row)
    return pd.DataFrame(rows,columns=frame.columns).drop_duplicates().reset_index(drop=True)


if __name__=='__main__':
    from biomarker.downstream.utils import READOUT_ROOT
    from biomarker.downstream.utils.tme_table import build_readout_index
    nanfang_readouts=build_readout_index(READOUT_ROOT,'NanFang_CRC')
    for spec in SPECS:
        d=normalize_labels(spec);path=Path(spec['destination']);path.parent.mkdir(parents=True,exist_ok=True)
        if spec['cohort']=='NanFang_CRC':d=keep_available_readouts(d,nanfang_readouts)
        d.to_csv(path,index=False)
        print(spec['task'],spec['cohort'],d['case'].nunique(),flush=True)
