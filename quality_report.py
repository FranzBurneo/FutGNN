# quality_report.py
from pathlib import Path
import pandas as pd, pickle, json
import networkx as nx

DEF_THRESH = {"coverage_min": 65.0, "intra_team_min": 95.0}

def read_gpickle(path: Path):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def pct_intra_team(G: nx.DiGraph):
    if G.number_of_edges() == 0: return 0.0
    same = 0
    for u,v in G.edges():
        tu, tv = G.nodes[u].get("team"), G.nodes[v].get("team")
        if tu is not None and tv is not None and tu == tv:
            same += 1
    return 100.0 * same / G.number_of_edges()

def dist_stats(G: nx.DiGraph):
    import math
    dists = []
    for u,v,d in G.edges(data=True):
        da = d.get("dist_sum", 0.0); w = max(1, d.get("weight", 1))
        dists.append(da / w)
    if not dists: return (0.0, 0.0, 0.0)
    s = pd.Series(dists)
    return (float(s.mean()), float(s.median()), float(s.quantile(0.95)))

def top_pairs(G: nx.DiGraph, k=3, min_attempts=5):
    rows = []
    for u,v,d in G.edges(data=True):
        w = int(d.get("weight",1))
        if w < min_attempts: continue
        succ = int(d.get("success_count", 0))
        rate = succ / w if w else 0.0
        rows.append((u,v,w,rate))
    if not rows: return ""
    df = pd.DataFrame(rows, columns=["u","v","w","succ_rate"]).sort_values(["w","succ_rate"], ascending=[False, False]).head(k)
    return "; ".join([f"{r.u}->{r.v} w={r.w} p={r.succ_rate:.2f}" for r in df.itertuples(index=False)])

def main():
    processed = Path("data/processed")
    rep = pd.read_csv(processed / "report.csv")
    out_rows = []
    for r in rep.itertuples(index=False):
        nx_path = Path(r.nx_path)
        if not nx_path.exists():
            print(f"[WARN] Falta {nx_path}")
            continue
        G = read_gpickle(nx_path)
        intra = pct_intra_team(G)
        mean_d, med_d, p95_d = dist_stats(G)
        tops = top_pairs(G, k=3, min_attempts=5)
        out_rows.append({
            "csv": r.csv,
            "nodes": r.nodes,
            "edges": r.edges,
            "coverage_pct": r.coverage_pct,
            "intra_team_pct": round(intra,2),
            "dist_mean": round(mean_d,2),
            "dist_median": round(med_d,2),
            "dist_p95": round(p95_d,2),
            "top_pairs": tops,
            "nx_path": r.nx_path
        })
    out = pd.DataFrame(out_rows).sort_values(["edges","coverage_pct"], ascending=[False,False])
    out.to_csv(processed / "report_enhanced.csv", index=False)
    # banderas de calidad
    fails = out[(out["coverage_pct"] < DEF_THRESH["coverage_min"]) | (out["intra_team_pct"] < DEF_THRESH["intra_team_min"])]
    print(f"[OK] Guardado: data/processed/report_enhanced.csv")
    if not fails.empty:
        print("\n[ALERTA] Partidos bajo umbral:")
        print(fails[["csv","coverage_pct","intra_team_pct"]].to_string(index=False))
        # +++ IMPRIMIR team_coverage_pct SI VIENE EN EL GRAFO +++
        team_cov = G.graph.get("team_coverage_pct")
        if team_cov is not None:
            print(f"    team_coverage_pct: {team_cov:.2f}")

if __name__ == "__main__":
    main()