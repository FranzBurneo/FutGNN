from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.graph_builder import load_passes, _infer_column_map

csv = sys.argv[1]
df = load_passes(csv, cm=None, near_dist=18, lookahead_rows=40)
miss = df[df["to_id"].isna()].copy()
cm = df.attrs["column_map"]
cols = [c for c in [cm.event_type, cm.player_from, cm.player_to, cm.team_id, cm.minute, cm.second, cm.x_start, cm.y_start, cm.x_end, cm.y_end, cm.event_id] if c]
print("Total pases:", len(df), " | Sin receptor:", len(miss))
print(miss[cols].head(20).to_string(index=False))