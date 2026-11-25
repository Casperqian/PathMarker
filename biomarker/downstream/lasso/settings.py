"""Explicit inputs and destinations for the fixed-test LASSO experiments."""
from dataclasses import dataclass
from pathlib import Path
import re

from biomarker.downstream.utils import FINAL_ROOT, LABEL_ROOT

DEFAULT_SEED = 42
LABEL = LABEL_ROOT

@dataclass(frozen=True)
class Cohort:
    name: str
    label_csv: Path
    readout: str
    normalization: str = 'canonical'
    time_scale: float = 1.

@dataclass(frozen=True)
class Task:
    name: str
    task_type: str
    labels: tuple
    cohorts: tuple
    time_unit: str = ''

@dataclass(frozen=True)
class Setting:
    task: str
    external_cohort: str
    internal_cohorts: tuple
    output_dir: str
    internal_holdout: bool = True


def canonical(cancer, cohort, filename):
    return Cohort(cohort, LABEL/cancer/filename, 'TCGA_COAD,TCGA_READ' if cohort=='TCGA_CRC' else cohort)


def stage_label(value):
    value=str(value).strip().upper().replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III').replace('Ⅳ','IV')
    value=re.sub(r'^STAGE\s*','',value)
    if re.fullmatch(r'(III|IV)[ABC]?[123]?',value):return 'III-IV'
    if re.fullmatch(r'(I|II)[ABC]?[123]?',value):return 'I-II'
    return None

NANFANG_LABEL_MAPS = {'nanfang_TNM-T': {'T1+T2': 'T1/T2', 'T3+T4': 'T3/T4'}, 'nanfang_TNM-N': {'N0': 'N0', 'N+': 'N+'}, 'nanfang_BRAF': {'-': '0', '+': '1'}, 'nanfang_MMR': {'pmmr': 'mss', 'dmmr': 'msi'}}

TASKS = {
    'BRCA-ER': Task('BRCA-ER', 'classification', ('Negative', 'Positive'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-ER.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-ER.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZFY_Pros', LABEL/'BRCA/ZFY-Pros-ER.csv', 'ZFY_Pros', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-ER.csv', 'ZJ1_C2', 'canonical'),
    ), ''),
    'BRCA-PR': Task('BRCA-PR', 'classification', ('Negative', 'Positive'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-PR.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-PR.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZFY_Pros', LABEL/'BRCA/ZFY-Pros-PR.csv', 'ZFY_Pros', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-PR.csv', 'ZJ1_C2', 'canonical'),
    ), ''),
    'BRCA-HER2': Task('BRCA-HER2', 'classification', ('Negative', 'Positive'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-HER2.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-HER2.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZFY_Pros', LABEL/'BRCA/ZFY-Pros-HER2.csv', 'ZFY_Pros', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-HER2.csv', 'ZJ1_C2', 'canonical'),
    ), ''),
    'BRCA-molecular-4': Task('BRCA-molecular-4', 'classification', ('Basal', 'Her2', 'LumA', 'LumB'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-molecular-4.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-molecular-4.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-molecular-4.csv', 'ZJ1_C2', 'canonical'),
    ), ''),
    'BRCA-TNM-T': Task('BRCA-TNM-T', 'classification', ('T1', 'T1+'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-TNM-T.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-TNM-T.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZFY_Pros', LABEL/'BRCA/ZFY-Pros-TNM-T.csv', 'ZFY_Pros', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-TNM-T.csv', 'ZJ1_C2', 'canonical'),
        Cohort('SDPH_BRCA', LABEL/'BRCA/SDPH_BRCA-TNM-T.csv', 'SDPH_BRCA', 'canonical'),
    ), ''),
    'BRCA-TNM-N': Task('BRCA-TNM-N', 'classification', ('N0', 'N+'), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-TNM-N.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-TNM-N.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('ZFY_Pros', LABEL/'BRCA/ZFY-Pros-TNM-N.csv', 'ZFY_Pros', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-BRCA-TNM-N.csv', 'ZJ1_C2', 'canonical'),
        Cohort('SDPH_BRCA', LABEL/'BRCA/SDPH_BRCA-TNM-N.csv', 'SDPH_BRCA', 'canonical'),
    ), ''),
    'CRC-BRAF': Task('CRC-BRAF', 'classification', ('0', '1'), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-BRAF.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-BRAF.csv', 'CPTAC_COAD', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-BRAF.csv', 'ARGO_CRC', 'canonical'),
        Cohort('Fuzhou_CRC', LABEL/'CRC/Fuzhou_CRC-BRAF.csv', 'Fuzhou_CRC', 'canonical'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_CRC-BRAF.csv', 'NanFang_CRC', 'canonical'),
    ), ''),
    'CRC-MMR': Task('CRC-MMR', 'classification', ('msi', 'mss'), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-MMR.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-MMR.csv', 'CPTAC_COAD', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-MMR.csv', 'ARGO_CRC', 'canonical'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_CRC-MMR.csv', 'NanFang_CRC', 'canonical'),
    ), ''),
    'CRC-TNM-T': Task('CRC-TNM-T', 'classification', ('T1/T2', 'T3/T4'), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-TNM-T.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-TNM-T.csv', 'CPTAC_COAD', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-TNM-T.csv', 'ARGO_CRC', 'canonical'),
        Cohort('Fuzhou_CRC', LABEL/'CRC/Fuzhou_CRC-TNM-T.csv', 'Fuzhou_CRC', 'canonical'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_CRC-TNM-T.csv', 'NanFang_CRC', 'canonical'),
    ), ''),
    'CRC-TNM-N': Task('CRC-TNM-N', 'classification', ('N0', 'N+'), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-TNM-N.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-TNM-N.csv', 'CPTAC_COAD', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-TNM-N.csv', 'ARGO_CRC', 'canonical'),
        Cohort('Fuzhou_CRC', LABEL/'CRC/Fuzhou_CRC-TNM-N.csv', 'Fuzhou_CRC', 'canonical'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_CRC-TNM-N.csv', 'NanFang_CRC', 'canonical'),
    ), ''),
    'CRC-TNM': Task('CRC-TNM', 'classification', ('I-II', 'III-IV'), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-TNM.csv', 'TCGA_COAD,TCGA_READ', 'stage'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-TNM.csv', 'CPTAC_COAD', 'stage'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-TNM.csv', 'ARGO_CRC', 'stage'),
        Cohort('Fuzhou_CRC', LABEL/'CRC/Fuzhou_CRC-TNM.csv', 'Fuzhou_CRC', 'stage'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_CRC-TNM.csv', 'NanFang_CRC', 'canonical'),
    ), ''),
    'BRCA-OS': Task('BRCA-OS', 'survival', (), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/BRCA-OS.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('CPTAC_BRCA', LABEL/'BRCA/CPTAC_BRCA-OS.csv', 'CPTAC_BRCA', 'canonical'),
        Cohort('SDPH_BRCA', LABEL/'BRCA/SDPH_BRCA-OS.csv', 'SDPH_BRCA', 'canonical'),
        Cohort('SRRSH_BRCA', LABEL/'BRCA/SRRSH_BRCA-OS.csv', 'SRRSH_BRCA', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-OS.csv', 'ZJ1_C2', 'canonical'),
    ), 'months'),
    'CRC-OS': Task('CRC-OS', 'survival', (), (
        Cohort('TCGA_CRC', LABEL/'CRC/CRC-OS.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('CPTAC_COAD', LABEL/'CRC/CPTAC_COAD-OS.csv', 'CPTAC_COAD', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-OS.csv', 'ARGO_CRC', 'canonical'),
        Cohort('NanFang_CRC', LABEL/'CRC/NanFang_Colon_OS.csv', 'NanFang_CRC', 'canonical'),
    ), 'months'),
    'BRCA-DFS': Task('BRCA-DFS', 'survival', (), (
        Cohort('TCGA_BRCA', LABEL/'BRCA/TCGA_BRCA-DFS.csv', 'TCGA_BRCA', 'canonical'),
        Cohort('SDPH_BRCA', LABEL/'BRCA/SDPH_BRCA-DFS.csv', 'SDPH_BRCA', 'canonical'),
        Cohort('SRRSH_BRCA', LABEL/'BRCA/SRRSH_BRCA-DFS.csv', 'SRRSH_BRCA', 'canonical'),
        Cohort('ZJ1_C2', LABEL/'BRCA/ZJ1_C2-DFS.csv', 'ZJ1_C2', 'canonical'),
    ), 'days'),
    'CRC-DFS': Task('CRC-DFS', 'survival', (), (
        Cohort('TCGA_CRC', LABEL/'CRC/TCGA_CRC-DFS.csv', 'TCGA_COAD,TCGA_READ', 'canonical'),
        Cohort('ARGO_CRC', LABEL/'CRC/ARGO_CRC-DFS.csv', 'ARGO_CRC', 'canonical'),
    ), 'days'),
    'THCA-DFS': Task('THCA-DFS', 'survival', (), (
        Cohort('TCGA_THCA', LABEL/'THCA/TCGA_THCA-DFS.csv', 'TCGA_THCA', 'canonical'),
        Cohort('Zhefuyi_Throid', LABEL/'THCA/Zhefuyi_Throid-DFS.csv', 'Zhefuyi_Throid', 'canonical'),
    ), 'days'),
    'THCA-OS': Task('THCA-OS', 'survival', (), (
        Cohort('TCGA_THCA', LABEL/'THCA/TCGA_THCA-OS.csv', 'TCGA_THCA', 'canonical'),
        Cohort('Zhefuyi_Throid', LABEL/'THCA/Zhefuyi_Throid-OS.csv', 'Zhefuyi_Throid', 'canonical'),
    ), 'days'),
}

for endpoint in ('OS', 'DFS'):
    TASKS[f'STAD-{endpoint}'] = Task(f'STAD-{endpoint}', 'survival', (), (
        Cohort('TCGA_STAD', LABEL/f'STAD/TCGA_STAD-{endpoint}.csv', 'TCGA_STAD'),
        Cohort('YNCH-YNFirst_Gastric', LABEL/f'STAD/YNCH-YNFirst_Gastric-{endpoint}.csv',
               'YNCH_Gastric,YNFirst_Gastric'),
        Cohort('Nanfang_Gastric', LABEL/f'STAD/Nanfang_Gastric-{endpoint}.csv',
               'Nanfang_Gastric', time_scale=30.),
    ), 'days')

# One fixed external cohort per task.
SETTINGS={}
def add(task,external,internal,*,internal_holdout=True):
    output_dir=str(FINAL_ROOT / 'lasso' / task)
    SETTINGS[task+'/'+external]=Setting(task,external,tuple(internal.split(',')),output_dir,internal_holdout)
add('BRCA-ER','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA,ZFY_Pros')
add('BRCA-PR','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA,ZFY_Pros')
add('BRCA-HER2','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA,ZFY_Pros')
add('BRCA-molecular-4','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA')
add('BRCA-TNM-T','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA,ZFY_Pros,SDPH_BRCA')
add('BRCA-TNM-N','ZJ1_C2','TCGA_BRCA,CPTAC_BRCA,ZFY_Pros,SDPH_BRCA')
add('BRCA-OS','ZJ1_C2','TCGA_BRCA,SDPH_BRCA',internal_holdout=False)
add('BRCA-DFS','ZJ1_C2','TCGA_BRCA,SDPH_BRCA',internal_holdout=False)
add('CRC-BRAF','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC,Fuzhou_CRC')
add('CRC-MMR','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC')
add('CRC-TNM-T','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC,Fuzhou_CRC')
add('CRC-TNM-N','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC,Fuzhou_CRC')
add('CRC-TNM','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC,Fuzhou_CRC')
add('CRC-OS','NanFang_CRC','TCGA_CRC,CPTAC_COAD,ARGO_CRC')
add('CRC-DFS','ARGO_CRC','TCGA_CRC')
add('THCA-OS','Zhefuyi_Throid','TCGA_THCA')
add('THCA-DFS','Zhefuyi_Throid','TCGA_THCA')
add('STAD-OS','Nanfang_Gastric','TCGA_STAD,YNCH-YNFirst_Gastric')
add('STAD-DFS','Nanfang_Gastric','TCGA_STAD,YNCH-YNFirst_Gastric')

UNAVAILABLE={'BRCA-TNM':'No verified overall-stage labels for an eligible external cohort.'}
