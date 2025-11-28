from __future__ import annotations
import argparse, glob, json, os, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Genera splits Train/Val/Test a partir de CSVs")
    ap.add_argument("--glob", default="data/raw/*.csv", help="Patrón de archivos")
    ap.add_argument("--out", default="splits/split_v1.json", help="Ruta del JSON de salida")
    ap.add_argument("--train", type=float, default=0.6, help="Proporción train")
    ap.add_argument("--val", type=float, default=0.2, help="Proporción val")
    ap.add_argument("--test", type=float, default=0.2, help="Proporción test")
    ap.add_argument("--seed", type=int, default=42, help="Semilla")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    assert files, f"No se encontraron CSVs con: {args.glob}"

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    n_train = max(1, int(round(n * args.train)))
    n_val   = max(0, int(round(n * args.val)))
    n_test  = max(1, n - n_train - n_val)

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": args.seed,
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # también guardamos listados .txt para inspección rápida
    (out_path.parent / (out_path.stem + ".train.txt")).write_text("\n".join(train_files), encoding="utf-8")
    (out_path.parent / (out_path.stem + ".val.txt")).write_text("\n".join(val_files), encoding="utf-8")
    (out_path.parent / (out_path.stem + ".test.txt")).write_text("\n".join(test_files), encoding="utf-8")

    print(f"[OK] Split guardado en: {out_path}")
    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    main()