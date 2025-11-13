# archivo: scripts/_quick_snapshot_check.py (temporal para debug)
import glob
from src.data.datasets import SnapshotLPDataset

files = glob.glob("data/raw/*.csv")
assert files, "No hay CSVs en data/raw/*.csv"

ds = SnapshotLPDataset([files[0]], step=1, negative_k=5)
print("Snapshots generados:", len(ds))
pyg, meta, pos, neg, mask = ds[0]
print("pyg:", pyg)
print("meta:", meta)
print("pos:", pos.shape, "neg:", neg.shape, "mask:", mask.item())