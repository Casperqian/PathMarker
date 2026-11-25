"""Replace these example paths with local dataset directories."""
from pathlib import Path

DATA_ROOT = Path("/path/to/your/data")
SEMANTIC_ROOTS = {
    "DatasetA": DATA_ROOT / "DatasetA",
    "DatasetB": DATA_ROOT / "DatasetB",
    "DatasetC": DATA_ROOT / "DatasetC",
}
DATASETS = ["DatasetA"]
EXTERNAL_DATASETS = ["DatasetB"]
WSI_DATASETS = ["DatasetC"]

# Each directory contains all/metadata.csv. Entity-table images live under images/.
PRETRAIN_ROOTS = {
    "DatasetD": DATA_ROOT / "DatasetD",
    "DatasetE": DATA_ROOT / "DatasetE",
}
ENTITY_TABLE_DATASETS = {"DatasetE"}
INSTANCE_ROOT = DATA_ROOT / "instances"
INSTANCE_VAL_DATASETS = {"DatasetA"}
