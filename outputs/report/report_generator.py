import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 0. Setup base directory
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths
classification_report_path = os.path.join(BASE_DIR, "classification_report.csv")
extra_metrics_path = os.path.join(BASE_DIR, "extra_metrics.csv")

# ==============================
# 1. Load Classification Report
# ==============================
df = pd.read_csv(classification_report_path)

# Detect the first column name dynamically
first_col = df.columns[0]
df = df.rename(columns={first_col: "Class"})

# Remove summary rows (accuracy, macro avg, weighted avg)
class_df = df[~df["Class"].isin(["accuracy", "macro avg", "weighted avg"])].copy()

# Convert relevant columns to numeric safely
for col in ["precision", "recall", "f1-score"]:
    class_df.loc[:, col] = pd.to_numeric(class_df[col], errors="coerce")

# ==============================
# 2. Plot: Classification Report
# ==============================
plt.figure(figsize=(10, 6))
x = np.arange(len(class_df["Class"]))
width = 0.25

plt.bar(x - width, class_df["precision"], width, label="Precision")
plt.bar(x, class_df["recall"], width, label="Recall")
plt.bar(x + width, class_df["f1-score"], width, label="F1-Score")

plt.xticks(x, class_df["Class"].tolist(), rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Metrics by Emotion Class", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "classification_report.png"), dpi=300)
plt.close()

print("✅ Saved classification_report.png")

# ==============================
# 3. Load Extra Metrics
# ==============================
extra_df = pd.read_csv(extra_metrics_path)

# Detect the first column name dynamically
first_col_extra = extra_df.columns[0]
extra_df = extra_df.rename(columns={first_col_extra: "Model"})

# Melt for plotting
melted = extra_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

# ==============================
# 4. Plot: Extra Metrics
# ==============================
plt.figure(figsize=(8, 5))
plt.barh(melted["Metric"], melted["Score"], color="skyblue", edgecolor="black")
plt.xlim(0, 1)
plt.xlabel("Score")
plt.title("Overall Model Performance Metrics", fontsize=14, fontweight="bold")

# Add labels on bars
for i, v in enumerate(melted["Score"]):
    plt.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "extra_metrics.png"), dpi=300)
plt.close()

print("✅ Saved extra_metrics.png")
