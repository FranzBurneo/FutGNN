from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Carpeta raíz con HTML (recursivo)")
    args = ap.parse_args()

    root = Path(args.folder)
    files = list(root.rglob("*.htm")) + list(root.rglob("*.html"))
    files = sorted(files)

    ok, bad = 0, 0
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        if "matchCentreData" in txt and "events" in txt:
            ok += 1
        else:
            bad += 1
            print("[BAD]", f)

    print(f"\nTotal: {len(files)} | OK: {ok} | BAD: {bad}")

if __name__ == "__main__":
    main()