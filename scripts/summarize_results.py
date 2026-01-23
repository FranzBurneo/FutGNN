import re, argparse, pandas as pd, pathlib as P

def parse_eval_lines(text: str) -> pd.DataFrame:
    rows = []

    # hits es lo único obligatorio
    hits_pat = re.compile(r"Hits@5\(TEST\)\s*=\s*(\d+(?:\.\d+)?)", re.I)

    # candidatos flexibles para split/ckpt/conv
    split_pat = re.compile(r"(?:Split|SPLIT)\s*[:=]\s*([^\|\-]+)", re.I)
    ckpt_pat  = re.compile(r"(?:CKPT|Checkpoint)\s*[:=]\s*([^\|\-]+)", re.I)
    conv_pat  = re.compile(r"(?:conv_type|conv)\s*[:=]\s*([^\|\-]+)", re.I)

    for line in text.splitlines():
        mh = hits_pat.search(line)
        if not mh:
            continue

        h5 = float(mh.group(1))

        ms = split_pat.search(line)
        mc = ckpt_pat.search(line)
        mv = conv_pat.search(line)

        rows.append({
            "split": (ms.group(1).strip() if ms else "UNKNOWN"),
            "ckpt":  (mc.group(1).strip() if mc else "UNKNOWN"),
            "conv":  (mv.group(1).strip() if mv else "UNKNOWN"),
            "Hits@5_TEST": h5
        })

    return pd.DataFrame(rows)

def best_val_from_runs(csv_path):
    df=pd.read_csv(csv_path)
    # asume columnas: epoch, hits5_val, loss_tr, ...
    c=[c for c in df.columns if c.lower().startswith("hits") and "val" in c.lower()]
    if not c: return None
    col=c[0]
    idx=df[col].idxmax()
    return float(df.loc[idx,col])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval-log", required=True, help="pega aquí el contenido de tus EVAL o pasa un archivo .txt")
    ap.add_argument("--runs", nargs="*", default=[], help="CSV de runs para obtener mejor Hits@5_val")
    args=ap.parse_args()

    # eval-log puede ser ruta o texto pegado
    p=P.Path(args.eval_log)
    txt=p.read_text(encoding="utf-8") if p.exists() else args.eval_log
    df_eval=parse_eval_lines(txt)

    if df_eval.empty:
        print("\n[ERROR] No se pudieron parsear líneas del eval-log.")
        print("Asegúrate de que evals.txt tenga líneas tipo:")
        print("[EVAL] Split=... | CKPT=... | conv_type=... | ... Hits@5(TEST)=0.123")
        raise SystemExit(1)

    # mapea ckpt->best_val
    best={}
    for r in args.runs:
        name=P.Path(r).stem
        try:
            best[name]=best_val_from_runs(r)
        except Exception:
            best[name]=None

    # arma columnas bonitas
    def label_split(s): 
        return "curve_30" if "30" in s else ("curve_70" if "70" in s else s)
    df_eval["split"]=df_eval["split"].map(label_split)
    df_eval["Modelo"]=df_eval["conv"].str.upper()
    df_eval=df_eval.rename(columns={"split":"Split","Hits@5_TEST":"Hits@5 (Test)","ckpt":"Checkpoint"})

    # salida ordenada
    df_out=df_eval[["Modelo","Split","Hits@5 (Test)","Checkpoint"]].sort_values(["Modelo","Split"])

    print("\n=== Resultados finales (EVAL) ===")
    print(df_out.to_string(index=False))

    # ejemplo de markdown
    print("\nMarkdown:\n")
    md="| Modelo | Split | Hits@5 (Test) |\n|---|---|---|\n"
    for _,r in df_out.iterrows():
        md+=f"| {r['Modelo']} | {r['Split']} | {r['Hits@5 (Test)']:.3f} |\n"
    print(md)

if __name__=="__main__":
    main()