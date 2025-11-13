from __future__ import annotations

import argparse
import csv
import glob
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP
from src.train.train_lp import eval_hits_at_k, train_one_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Entrenamiento de Link Prediction con GNN sobre snapshots de redes de pases"
    )
    ap.add_argument("--glob", type=str, default="data/raw/*.csv",
                    help="Patrón glob para cargar partidos (CSV)")
    ap.add_argument("--epochs", type=int, default=10, help="Épocas de entrenamiento")
    ap.add_argument("--batch", type=int, default=16, help="Tamaño de batch (n° snapshots)")
    ap.add_argument("--step", type=int, default=1, help="Paso temporal para snapshots")
    ap.add_argument("--neg", type=int, default=10, help="Negatives por positivo")
    ap.add_argument("--neg-k", type=int, help="Alias de --neg (si se usa, sobrescribe --neg)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--hidden", type=int, default=64, help="Dimensión oculta del GNN")
    ap.add_argument("--layers", type=int, default=2, help="Número de capas GNN")
    ap.add_argument("--out", type=int, default=64, help="Dimensión de salida del cabezal LP")
    ap.add_argument("--device", type=str, default=None, help='"cpu" o "cuda". Si se omite, se detecta automáticamente.')
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    ap.add_argument("--workers", type=int, default=0, help="num_workers del DataLoader")
    ap.add_argument("--val_ratio", type=float, default=0.33, help="Proporción para validación [0,1)")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="Ruta de checkpoint a guardar")
    ap.add_argument("--runs_file", type=str, default="runs/lp_metrics.csv", help="Ruta del CSV de métricas")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Semilla
    set_seed(args.seed)

    # Negatives por positivo (alias)
    neg_k = args.neg_k if args.neg_k is not None else args.neg

    # Archivos y split
    files: List[str] = sorted(glob.glob(args.glob))
    assert files, f"No hay CSVs para el patrón: {args.glob}"
    random.shuffle(files)

    cut = max(1, int(len(files) * (1 - args.val_ratio))) if len(files) > 1 else 1
    train_files = files[:cut]
    val_files = files[cut:] if cut < len(files) else []

    print(f"[INFO] Partidos: {len(files)} | patrón: {args.glob}")
    print(f"[INFO] Train: {len(train_files)} | Val: {len(val_files)}")

    # Datasets
    ds_tr = SnapshotLPDataset(train_files, step=args.step, negative_k=neg_k)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda batch: list(zip(*batch)),  # (pyg, pos, neg, mask, meta)
    )

    dl_va: Optional[DataLoader] = None
    if len(val_files) > 0:
        ds_va = SnapshotLPDataset(val_files, step=args.step, negative_k=neg_k)
        dl_va = DataLoader(
            ds_va,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=lambda batch: list(zip(*batch)),
        )

    print(f"[INFO] Snapshots Train: {len(ds_tr)} | Val: {len(ds_va) if dl_va else 0}")

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Device: {device}")

    # Modelo y optimizador
    model = GNNSimpleLP(in_dim=2, hidden=args.hidden, layers=args.layers, out=args.out).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[INFO] Modelo: GNNSimpleLP(hidden={args.hidden}, layers={args.layers}, out={args.out}) | lr={args.lr}")

    # Logging CSV
    runs_path = Path(args.runs_file)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    if not runs_path.exists():
        with runs_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "hits5_tr", "hits5_val"])

    # Checkpoints
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("-inf")

    # Loop de entrenamiento
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dl_tr, optim, device)
        hits5_tr = eval_hits_at_k(model, dl_tr, K=5, device=device)

        if dl_va is not None:
            hits5_va = eval_hits_at_k(model, dl_va, K=5, device=device)
            print(f"[Epoch {e:02d}] loss={loss:.4f} | Hits@5(TR)={hits5_tr:.3f} | Hits@5(VAL)={hits5_va:.3f}")
        else:
            hits5_va = float("nan")
            print(f"[Epoch {e:02d}] loss={loss:.4f} | Hits@5(TR)={hits5_tr:.3f}")

        # Append métricas
        with runs_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([e, f"{loss:.6f}", f"{hits5_tr:.6f}", "" if torch.isnan(torch.tensor(hits5_va)) else f"{hits5_va:.6f}"])

        # Guardar mejor checkpoint por val
        if dl_va is not None and hits5_va > best_val:
            best_val = hits5_va
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "val_hits5": best_val},
                ckpt_path.as_posix()
            )
            print(f"[CKPT] Guardado mejor modelo en {ckpt_path} (Hits@5 VAL={best_val:.3f})")


if __name__ == "__main__":
    main()