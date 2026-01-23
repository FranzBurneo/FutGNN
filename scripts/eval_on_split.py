from __future__ import annotations

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SnapshotLPDataset
from src.models.gnn_lp import GNNSimpleLP
from src.train.train_lp import eval_hits_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--neg", type=int, default=5)

    # device: si el usuario pide cuda pero no está disponible, caemos a cpu.
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # NUEVO: permite elegir conv (gcn/sage/gat). Si no se pasa, usa el del ckpt;
    # si el ckpt no lo trae, por defecto usamos "sage" (tu ckpt es curve30_sage_v2).
    ap.add_argument("--conv-type", choices=["gcn", "sage", "gat"], default=None)

    args = ap.parse_args()

    # Normaliza device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] Pediste CUDA pero torch.cuda.is_available() es False. Usando CPU.")
        device = "cpu"

    # Lee split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_files = split.get("test") or split.get("TEST") or []
    assert test_files, "No hay archivos en TEST."

    # Dataset / DataLoader
    ds_te = SnapshotLPDataset(test_files, step=args.step, negative_k=args.neg)
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: list(zip(*b)),
    )

    # Carga checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

    # Args guardados en el checkpoint (si existen)
    h = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # Determina conv_type: prioridad CLI > ckpt > default "sage"
    conv_type = args.conv_type or h.get("conv_type", "sage")

    # Construye modelo con los mismos hiperparámetros del ckpt (si están)
    model = GNNSimpleLP(
        in_dim=h.get("in_dim", 2),
        hidden=h.get("hidden", 64),
        layers=h.get("layers", 2),
        out=h.get("out", 64),
        conv_type=conv_type,
    ).to(device)

    # Carga pesos
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint no tiene la clave 'model'. Revisa el formato del .pt")

    model.load_state_dict(ckpt["model"])

    # Eval
    hits5 = eval_hits_at_k(model, dl_te, K=5, device=device)
    print(
        f"[EVAL] Split={os.path.basename(args.split_file)} | "
        f"CKPT={os.path.basename(args.ckpt)} | "
        f"conv_type={conv_type} | device={device} | "
        f"Hits@5(TEST)={hits5:.3f}"
    )


if __name__ == "__main__":
    main()