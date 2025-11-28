import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Lista de CSVs de runs")
    ap.add_argument("--labels", nargs="+", required=True, help="Etiquetas para cada CSV")
    ap.add_argument("--out-prefix", default="runs/compare")
    args = ap.parse_args()

    assert len(args.runs) == len(args.labels), "Debe coincidir runs y labels"
    dfs = []
    for csv_path, label in zip(args.runs, args.labels):
        df = pd.read_csv(csv_path)
        df["label"] = label
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Loss
    for label, d in all_df.groupby("label"):
        plt.plot(d["epoch"], d["loss"], label=label)
    plt.title("Loss por época")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_loss.png"); plt.clf()

    # Hits@5 Val
    for label, d in all_df.groupby("label"):
        if "hits5_val" in d:
            plt.plot(d["epoch"], d["hits5_val"], label=label)
    plt.title("Hits@5 (Validación)")
    plt.xlabel("Época"); plt.ylabel("Hits@5"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_hits5_val.png"); plt.clf()

    print(f"[OK] Figuras guardadas en {args.out_prefix}_loss.png y {args.out_prefix}_hits5_val.png")

if __name__ == "__main__":
    main()