import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd


# =========================
# 配置区：按你当前工程代码保持一致
# =========================
from sam3.train.data.dataset_paths import PRETRAIN_ROOTS, ENTITY_TABLE_DATASETS

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# 你若只接受这三类 level，就保留这个集合
ALLOWED_LEVELS = {"tissue", "cell", "nuclei"}


# =========================
# 数据结构
# =========================
@dataclass
class LabelInfo:
    sample_id: str
    organ_name: str
    level_name: str
    object_name: str


@dataclass
class BadCase:
    dataset_name: str
    row_idx: int
    image_stem: str
    label_stem: str
    issue_type: str
    issue_detail: str
    parsed_organ: str = ""
    parsed_level: str = ""
    parsed_object: str = ""


# =========================
# 与你当前 dataset 代码一致的辅助函数
# =========================
def ds_base_dir(dataset_name: str) -> Path:
    return PRETRAIN_ROOTS[dataset_name]


def metadata_path(dataset_name: str) -> Path:
    return PRETRAIN_ROOTS[dataset_name] / "all" / "metadata.csv"


def normalize_image_stem(dataset_name: str, image_stem: str) -> str:
    if dataset_name in ENTITY_TABLE_DATASETS:
        return image_stem.replace("_pathology_unspecified", "")
    return image_stem


def find_image_path(dataset_name: str, image_stem: str) -> Optional[Path]:
    base_dir = ds_base_dir(dataset_name)
    candidate_dirs = [
        base_dir / "all",
        base_dir / "train",
        base_dir / "images",
    ]

    for image_dir in candidate_dirs:
        if not image_dir.exists():
            continue

        for ext in IMG_EXTS:
            exact_path = image_dir / f"{image_stem}{ext}"
            if exact_path.exists():
                return exact_path

            fuzzy_matches = sorted(image_dir.glob(f"{image_stem}*{ext}"))
            if fuzzy_matches:
                return fuzzy_matches[0]

    return None


def parse_label(label: str) -> LabelInfo:
    """
    按你当前代码的规则解析：
    <sample>_pathology_<organ>_<level>+<object>
    """
    if "_pathology_" not in label:
        raise ValueError(f"Invalid label format: {label}")

    left, right = label.split("_pathology_", 1)
    sample_id = f"{left}_pathology"

    fields = right.split("_")
    if len(fields) < 2:
        raise ValueError(f"Invalid label format: {label}")

    organ_token = fields[0]
    rest = "_".join(fields[1:])

    organ_name = " ".join(organ_token.split("+"))
    rest_parts = rest.split("+")

    level_name = rest_parts[0].strip()
    object_name = " ".join(rest_parts[1:]).strip() if len(rest_parts) > 1 else ""

    return LabelInfo(
        sample_id=sample_id,
        organ_name=organ_name,
        level_name=level_name,
        object_name=object_name,
    )


def is_suspicious_name(text: str) -> Tuple[bool, str]:
    """
    比较保守的异常名字检查，只抓明显不对的情况
    """
    if text is None:
        return True, "is None"

    s = str(text).strip()
    if s == "":
        return True, "empty string"

    if s.lower() in {"nan", "none", "null"}:
        return True, f"invalid literal: {s}"

    if "  " in s:
        return True, "contains double spaces"

    if s.startswith(("+", "_", "-", ";")) or s.endswith(("+", "_", "-", ";")):
        return True, "starts or ends with unusual delimiter"

    # 只含标点或分隔符
    if re.fullmatch(r"[\W_]+", s):
        return True, "contains only punctuation"

    return False, ""


def load_standard_samples(dataset_name: str, metadata: pd.DataFrame):
    required_cols = {"image", "label"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"{dataset_name} metadata missing columns: {missing}")

    samples = []
    for row_idx, row in metadata.iterrows():
        samples.append(
            {
                "row_idx": int(row_idx),
                "dataset_name": dataset_name,
                "image_stem": str(row["image"]),
                "label_stem": str(row["label"]),
            }
        )
    return samples


def load_entity_table_samples(dataset_name: str, metadata: pd.DataFrame):
    required_cols = {"filename", "tissue", "cell", "nuclei"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"{dataset_name} metadata missing columns: {missing}")

    samples = []
    for row_idx, row in metadata.iterrows():
        file_stem = Path(str(row["filename"])).stem

        for level in ("tissue", "cell", "nuclei"):
            items = row.get(level)
            if pd.isna(items) or not items or not isinstance(items, str):
                continue

            entities = {entity.strip() for entity in items.split(";") if entity.strip()}
            for entity in entities:
                image_id = f"{file_stem}_pathology_unspecified"
                label_id = f"{image_id}_{level}+{entity.replace(' ', '+')}"
                samples.append(
                    {
                        "row_idx": int(row_idx),
                        "dataset_name": dataset_name,
                        "image_stem": image_id,
                        "label_stem": label_id,
                    }
                )
    return samples


# def load_samples(dataset_name: str):
#     md_path = metadata_path(dataset_name)
#     if not md_path.exists():
#         raise FileNotFoundError(f"Metadata not found: {md_path}")

#     metadata = pd.read_csv(md_path)

#     if dataset_name in ENTITY_TABLE_DATASETS:
#         return load_entity_table_samples(dataset_name, metadata), metadata

#     return load_standard_samples(dataset_name, metadata), metadata

def load_samples(dataset_name: str):
    md_path = metadata_path(dataset_name)
    if not md_path.exists():
        raise FileNotFoundError(f"Metadata not found: {md_path}")

    metadata = pd.read_csv(md_path)

    if dataset_name in ENTITY_TABLE_DATASETS:
        metadata = metadata.head(100).copy()   # 只检查前 100 行
        return load_entity_table_samples(dataset_name, metadata), metadata

    return load_standard_samples(dataset_name, metadata), metadata


# =========================
# 核心检查逻辑
# =========================
def audit_dataset(dataset_name: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, metadata = load_samples(dataset_name)

    organ_counter = Counter()
    level_counter = Counter()
    object_counter = Counter()
    image_missing_count = 0
    parse_error_count = 0

    bad_cases: List[BadCase] = []
    parsed_rows = []

    seen_pairs = Counter()

    for sample in samples:
        row_idx = sample["row_idx"]
        image_stem_raw = sample["image_stem"]
        label_stem = sample["label_stem"]

        seen_pairs[(image_stem_raw, label_stem)] += 1

        # 1) parse label
        try:
            info = parse_label(label_stem)
        except Exception as e:
            parse_error_count += 1
            bad_cases.append(
                BadCase(
                    dataset_name=dataset_name,
                    row_idx=row_idx,
                    image_stem=image_stem_raw,
                    label_stem=label_stem,
                    issue_type="parse_error",
                    issue_detail=str(e),
                )
            )
            continue

        # 2) 统计
        organ_counter[info.organ_name] += 1
        level_counter[info.level_name] += 1
        object_counter[info.object_name] += 1

        parsed_rows.append(
            {
                "dataset_name": dataset_name,
                "row_idx": row_idx,
                "image_stem_raw": image_stem_raw,
                "label_stem": label_stem,
                "sample_id": info.sample_id,
                "organ_name": info.organ_name,
                "level_name": info.level_name,
                "object_name": info.object_name,
            }
        )

        # 3) 检查 organ / level / object 命名
        for field_name, field_value in [
            ("organ", info.organ_name),
            ("level", info.level_name),
            ("object", info.object_name),
        ]:
            bad, detail = is_suspicious_name(field_value)
            if bad:
                bad_cases.append(
                    BadCase(
                        dataset_name=dataset_name,
                        row_idx=row_idx,
                        image_stem=image_stem_raw,
                        label_stem=label_stem,
                        issue_type=f"{field_name}_suspicious",
                        issue_detail=detail,
                        parsed_organ=info.organ_name,
                        parsed_level=info.level_name,
                        parsed_object=info.object_name,
                    )
                )

        # 4) 检查 level 是否在预期集合里
        if info.level_name not in ALLOWED_LEVELS:
            bad_cases.append(
                BadCase(
                    dataset_name=dataset_name,
                    row_idx=row_idx,
                    image_stem=image_stem_raw,
                    label_stem=label_stem,
                    issue_type="level_out_of_vocab",
                    issue_detail=f"level={info.level_name} not in {sorted(ALLOWED_LEVELS)}",
                    parsed_organ=info.organ_name,
                    parsed_level=info.level_name,
                    parsed_object=info.object_name,
                )
            )

        # 5) object 里若出现明显异常词，可单独提示
        if info.object_name.lower() in {"tissue", "cell", "nuclei"}:
            bad_cases.append(
                BadCase(
                    dataset_name=dataset_name,
                    row_idx=row_idx,
                    image_stem=image_stem_raw,
                    label_stem=label_stem,
                    issue_type="object_too_generic",
                    issue_detail=f"object={info.object_name} is too generic",
                    parsed_organ=info.organ_name,
                    parsed_level=info.level_name,
                    parsed_object=info.object_name,
                )
            )

        if "pathology" in info.object_name.lower():
            bad_cases.append(
                BadCase(
                    dataset_name=dataset_name,
                    row_idx=row_idx,
                    image_stem=image_stem_raw,
                    label_stem=label_stem,
                    issue_type="object_contains_pathology",
                    issue_detail="object contains the word 'pathology'",
                    parsed_organ=info.organ_name,
                    parsed_level=info.level_name,
                    parsed_object=info.object_name,
                )
            )

        # 6) 检查图片是否存在
        image_stem = normalize_image_stem(dataset_name, image_stem_raw)
        img_path = find_image_path(dataset_name, image_stem)
        if img_path is None:
            image_missing_count += 1
            bad_cases.append(
                BadCase(
                    dataset_name=dataset_name,
                    row_idx=row_idx,
                    image_stem=image_stem_raw,
                    label_stem=label_stem,
                    issue_type="image_missing",
                    issue_detail=f"image not found for normalized stem: {image_stem}",
                    parsed_organ=info.organ_name,
                    parsed_level=info.level_name,
                    parsed_object=info.object_name,
                )
            )

    # 7) 重复样本检查
    duplicate_rows = []
    for (image_stem, label_stem), cnt in seen_pairs.items():
        if cnt > 1:
            duplicate_rows.append(
                {
                    "dataset_name": dataset_name,
                    "image_stem": image_stem,
                    "label_stem": label_stem,
                    "duplicate_count": cnt,
                }
            )

    # 8) 导出 csv
    parsed_df = pd.DataFrame(parsed_rows)
    bad_df = pd.DataFrame([asdict(x) for x in bad_cases])
    dup_df = pd.DataFrame(duplicate_rows)

    organ_df = pd.DataFrame(
        [{"organ_name": k, "count": v} for k, v in organ_counter.most_common()]
    )
    level_df = pd.DataFrame(
        [{"level_name": k, "count": v} for k, v in level_counter.most_common()]
    )
    object_df = pd.DataFrame(
        [{"object_name": k, "count": v} for k, v in object_counter.most_common()]
    )

    # parsed_df.to_csv(out_dir / f"{dataset_name}_parsed.csv", index=False)
    # bad_df.to_csv(out_dir / f"{dataset_name}_bad_cases.csv", index=False)
    # dup_df.to_csv(out_dir / f"{dataset_name}_duplicates.csv", index=False)
    # organ_df.to_csv(out_dir / f"{dataset_name}_organ_stats.csv", index=False)
    # level_df.to_csv(out_dir / f"{dataset_name}_level_stats.csv", index=False)
    # object_df.to_csv(out_dir / f"{dataset_name}_object_stats.csv", index=False)

    summary = {
        "dataset_name": dataset_name,
        "metadata_rows": int(len(metadata)),
        "expanded_samples": int(len(samples)),
        "parse_error_count": int(parse_error_count),
        "image_missing_count": int(image_missing_count),
        "bad_case_count": int(len(bad_cases)),
        "duplicate_pair_count": int(len(duplicate_rows)),
        "unique_organs": int(len(organ_counter)),
        "unique_levels": int(len(level_counter)),
        "unique_objects": int(len(object_counter)),
        "top_levels": level_counter.most_common(20),
        "top_organs": organ_counter.most_common(20),
        "top_objects": object_counter.most_common(50),
    }

    with open(out_dir / f"{dataset_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="all",
        help="单个数据集名字，或 all",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./check_pretrain",
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_name == "all":
        dataset_list = list(PRETRAIN_ROOTS)
    else:
        dataset_list = [args.dataset_name]

    all_summary = []
    for ds_name in dataset_list:
        print(f"\n=== Auditing {ds_name} ===")
        try:
            summary = audit_dataset(ds_name, out_dir)
            all_summary.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[ERROR] {ds_name}: {e}")
            all_summary.append(
                {
                    "dataset_name": ds_name,
                    "error": str(e),
                }
            )

    pd.DataFrame(all_summary).to_csv(out_dir / "all_dataset_summary.csv", index=False)
    with open(out_dir / "all_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()