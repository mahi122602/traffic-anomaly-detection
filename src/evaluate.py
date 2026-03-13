import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

print("Starting model evaluation...")

# File paths
scores_path = "results/anomaly_scores.npy"
labels_path = "results/labels.npy"

# Check if files exist
if not os.path.exists(scores_path) or not os.path.exists(labels_path):
    print("Error: Required files not found.")
    print("Run detect_anomaly.py first to generate anomaly scores.")
    exit()

# Load saved scores and labels
scores = np.load(scores_path)
labels = np.load(labels_path)

print("Scores loaded:", len(scores))
print("Labels loaded:", len(labels))

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)

# Compute AUC score
roc_auc = auc(fpr, tpr)

print(f"AUC Score: {roc_auc:.4f}")

# Create ROC plot
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Anomaly Detection")

plt.legend(loc="lower right")
plt.grid(True)

# Save figure
os.makedirs("results", exist_ok=True)
save_path = "results/roc_curve.png"

plt.savefig(save_path)
print("ROC curve saved to:", save_path)

plt.show()

print("Evaluation complete.")