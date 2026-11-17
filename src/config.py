from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

PUBTATOR_URL = (
    "https://github.com/chanzuckerberg/ChemDisGene/raw/master/"
    "data/curated/abstracts.txt.gz"
)
RAW_FILE = RAW_DIR / "chemdisgene_curated_abstracts.txt.gz"

SEED = 42
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
ENTITY_TYPES = ("CHEMICAL", "DISEASE", "GENE")
TOKEN_PATTERN = r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[^\w\s]"
CRF_PARAMS = {
    "algorithm": "lbfgs",
    "c1": 0.1,
    "c2": 0.1,
    "max_iterations": 150,
    "all_possible_transitions": True,
}
