import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/lp_metrics.csv")

ax = df[["loss"]].plot(title="Loss")
ax.get_figure().savefig("runs/loss.png", dpi=150, bbox_inches="tight"); plt.close()

ax = df[["hits5_tr","hits5_val"]].plot(title="Hits@5")
ax.get_figure().savefig("runs/hits5.png", dpi=150, bbox_inches="tight"); plt.close()

print("Saved: runs/loss.png, runs/hits5.png")