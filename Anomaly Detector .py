#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd

def detect_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()  # Select only numeric columns from the DataFrame
    if not numeric_cols:
        raise ValueError("No numeric columns available for anomaly detection.")
        
    # Compute z-scores for each numeric column
    # Formula: (value - column_mean) / column_std
    zscores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0)
    # For each row, find the largest absolute z-score across all numeric features
    max_abs_z = zscores.abs().max(axis=1)

    scored = df.copy()
    # Add anomaly score (max absolute z-score per row)
    scored["anomaly_score"] = max_abs_z
    scored["is_anomaly"] = max_abs_z > threshold
    return scored.sort_values("anomaly_score", ascending=False)


# In[73]:


df = pd.read_csv(r"C:\Users\My PC\Downloads\optometry.csv")
scored = detect_anomalies(df, threshold=2.5)
scored["excel_row"] = scored.index + 2  # +1 to switch to 1-based, +1 for the header row
anomalies = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)
display(anomalies)


# In[79]:


import matplotlib.pyplot as plt
import seaborn as sns

threshold = 2.5  # adjust sd threshold

scored = detect_anomalies(df, threshold=threshold).copy()
scored["excel_row"] = scored.index + 2  # keeps spreadsheet row numbers handy
anomalies = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)

sns.set_theme(style="whitegrid", context="talk")

fig = plt.figure(figsize=(16, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.25)

ax_timeline = fig.add_subplot(gs[:, :2])
ax_hist = fig.add_subplot(gs[:, 2])

# Timeline: all points in blue, flagged anomalies in red, threshold band for context
ax_timeline.scatter(
    scored.index,
    scored["anomaly_score"],
    s=55,
    color="#4C78A8",
    alpha=0.55,
    label="All records",
)
ax_timeline.scatter(
    anomalies.index,
    anomalies["anomaly_score"],
    s=115,
    color="#E45756",
    edgecolor="black",
    linewidth=1,
    label="Flagged anomalies",
)
ax_timeline.axhspan(
    threshold,
    scored["anomaly_score"].max() * 1.05,
    color="#F58518",
    alpha=0.12,
    label=f"Threshold ≥ {threshold}",
)
ax_timeline.set_title("Anomaly Score Timeline")
ax_timeline.set_xlabel("Record index")
ax_timeline.set_ylabel("Anomaly score")
ax_timeline.legend(loc="upper right", frameon=True)    #can move outside because could block anomaly points

# Distribution: histogram of scores with a dotted vertical threshold
ax_hist.hist(
    scored["anomaly_score"],
    bins=30,
    color="#4C78A8",
    alpha=0.85,
    edgecolor="white",
)
ax_hist.axvline(
    threshold,
    color="#E45756",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold}",
)
ax_hist.set_title("Score Distribution")
ax_hist.set_xlabel("Anomaly score")
ax_hist.set_ylabel("Record count")
ax_hist.legend()

plt.show()

