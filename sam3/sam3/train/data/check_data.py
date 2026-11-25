import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tqdm import tqdm

# ================= 配置部分 (直接复用你的逻辑) =================
from sam3.train.data.dataset_paths import (
    DATA_ROOT, DATASETS, EXTERNAL_DATASETS, SEMANTIC_ROOTS, WSI_DATASETS,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
WSI_EXTS = (".svs", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ================= 工具函数 =================
def _ds_base_dir(ds_name: str) -> Path:
    return SEMANTIC_ROOTS[ds_name]

def _find_image_path(image_dir: Path, sample_id: str) -> Path | None:
    for ext in IMG_EXTS:
        p = image_dir / f"{sample_id}{ext}"
        if p.exists(): return p
        matches = sorted(image_dir.glob(f"{sample_id}*{ext}"))
        if matches: return matches[0]
    return None

def _find_path(dir: Path, sample_id: str, exts: list) -> Path | None:
    for ext in exts:
        p = dir / f"{sample_id}{ext}"
        if p.exists(): return p
        matches = sorted(dir.glob(f"{sample_id}*{ext}"))
        if matches: return matches[0]
    return None

def _parse_mask_filename(mask_file: str):
    stem = mask_file.rsplit(".", 1)[0]
    left, right = stem.split("_pathology_", 1)
    sample_id = left + "_pathology"
    return sample_id

# ================= 核心校验逻辑 =================
def main():
    standard_datasets = EXTERNAL_DATASETS + DATASETS
    wsi_datasets = WSI_DATASETS
    splits = ["train", "test"]
    
    missing_count = 0
    error_count = 0

    print("开始校验 Standard Datasets (Patch 级别)...")
    for ds in standard_datasets:
        print(f"\n正在校验数据集: {ds}")
        base_dir = _ds_base_dir(ds)
        for split in splits:
            image_dir = base_dir / split
            mask_dir = base_dir / f"{split}_mask"
            
            if not mask_dir.exists():
                # print(f"  [跳过] 找不到 mask 目录: {mask_dir}")
                continue
                
            mask_files = [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            
            for mask_path in tqdm(mask_files, desc=f"Processing {ds}/{split}"):
                try:
                    sample_id = _parse_mask_filename(mask_path.name)
                except Exception as e:
                    print(f"[命名错误] {ds}/{split}: 无法解析 {mask_path.name} -> {e}")
                    error_count += 1
                    continue
                
                img_path = _find_image_path(image_dir, sample_id)
                if img_path is None:
                    print(f"[缺失Image] {ds}/{split}: Mask为 '{mask_path.name}'，但找不到对应的 image (使用 sample_id: '{sample_id}' 在 {image_dir} 中匹配失败)")
                    missing_count += 1

    # print("\n开始校验 WSI Datasets...")
    # for ds in wsi_datasets:
    #     base_dir = _ds_base_dir(ds)
    #     for split in splits:
    #         wsi_dir = base_dir / split
    #         mask_dir = base_dir / f"{split}_mask"
    #         coord_dir = base_dir / f"{split}_coord"
            
    #         if not mask_dir.exists():
    #             continue
                
    #         mask_files = [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in WSI_EXTS]
            
    #         for mask_path in mask_files:
    #             try:
    #                 sample_id = _parse_mask_filename(mask_path.name)
    #             except Exception as e:
    #                 print(f"[命名错误] {ds}/{split}: 无法解析 {mask_path.name} -> {e}")
    #                 error_count += 1
    #                 continue
                
    #             # 校验 WSI Image
    #             wsi_path = _find_path(wsi_dir, sample_id, WSI_EXTS)
    #             if wsi_path is None:
    #                 print(f"[缺失WSI Image] {ds}/{split}: Mask为 '{mask_path.name}'，找不到对应的WSI (匹配 {sample_id} in {wsi_dir})")
    #                 missing_count += 1
                
    #             # 校验 h5 coord 文件
    #             coord_path = _find_path(coord_dir, sample_id, [".h5"])
    #             if coord_path is None:
    #                 print(f"[缺失h5坐标] {ds}/{split}: Mask为 '{mask_path.name}'，找不到对应的h5坐标文件 (匹配 {sample_id} in {coord_dir})")
    #                 missing_count += 1

    print("\n" + "="*40)
    print("校验完成！")
    print(f"总计无法解析文件名的 Mask 数量: {error_count}")
    print(f"总计找不到对应 Image/WSI/h5 的 Mask 数量: {missing_count}")
    if error_count == 0 and missing_count == 0:
        print("完美！所有数据集的 Mask 都成功找到了对应的图片和坐标文件。")

if __name__ == "__main__":
    main()