import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--split-file", required=True)
args = ap.parse_args()
with open(args.split_file, "r", encoding="utf-8") as f:
    s = json.load(f)
for k in ["train","val","test"]:
    print(k.upper(), len(s.get(k, [])))
    for p in s.get(k, []):
        print("  -", p)