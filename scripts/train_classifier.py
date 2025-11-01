# scripts/train_classifier.py (fixed, robust, saves metrics as CSV and better styled images)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import argparse
from collections import Counter
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker # Added for table formatting

def load_features(npz_path):
    """Load features, labels, and class names from a .npz file."""
    a = np.load(npz_path, allow_pickle=True)
    X = a['features']
    y = a['labels']
    class_names_raw = list(a['class_names'])
    class_names = [str(name) for name in class_names_raw]
    return X, y, class_names

def save_table_as_image(df, filename, title="Metrics Table", col_width=2.0):
    """Render a DataFrame as a nicely formatted image (PNG)."""
    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna('N/A')

    # Prepare cell text with formatting
    cell_text = []
    for row in df_clean.values:
        cell_text.append([f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x) for x in row])

    # --- Improved Figure Size Calculation ---
    # Base height + height per row, Base width + width per column
    fig_height = 0.5 + (len(df_clean) + 1) * 0.4 # Adjust 0.4 for row height
    fig_width = 1.0 + len(df_clean.columns) * col_width # Use col_width parameter

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Create the table
    the_table = ax.table(
        cellText=cell_text,
        colLabels=df_clean.columns,
        rowLabels=df_clean.index,
        cellLoc='center',
        loc='center',
        colWidths=[col_width]*len(df_clean.columns) # Set column widths
    )

    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.2) # Adjust vertical scale slightly

    # --- Style Cells ---
    for (i, j), cell in the_table.get_celld().items():
        cell.set_edgecolor('grey') # Add cell borders
        if i == 0: # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark blue header
            cell.set_height(0.6) # Increase header height slightly
        elif j == -1: # Index column
             cell.set_text_props(weight='bold')
             cell.set_facecolor('#f2f2f2') # Light grey index
        else: # Data cells
            cell.set_facecolor('white')

    plt.title(title, fontsize=14, weight='bold', pad=20)
    # plt.tight_layout() # tight_layout often interferes with table rendering

    try:
        # Use bbox_inches='tight' and pad_inches to ensure everything fits
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved table image: {filename}")
    except Exception as e:
        print(f"Error saving table image {filename}: {e}")
    finally:
        plt.close(fig) # Ensure the figure is closed

# --- Main script execution (rest is the same as before) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to the .npz features file")
    parser.add_argument("--out_model", default="outputs/classifier_final.joblib", help="Where to save the model")
    parser.add_argument("--metrics_dir", default="outputs/metrics", help="Directory to save metrics files")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)
    print(f"Metrics will be saved to: {metrics_path.resolve()}")

    X, y, class_names = load_features(args.features)
    print("Loaded features:", X.shape)
    print("Loaded labels:", y.shape)

    counter = Counter(y.tolist())
    print("Label distribution:", counter)

    present_labels = sorted(list(counter.keys()))
    target_names_filtered = [class_names[i] if i < len(class_names) else f"class_{i}" for i in present_labels]
    print(f"Class names used for metrics: {target_names_filtered}")

    if len(present_labels) < 2:
        raise SystemExit(f"ERROR: Need at least 2 classes to train classifier. Found: {present_labels}")

    min_count = min(counter.values())
    stratify_arg = y if min_count >= 2 else None
    if stratify_arg is None:
        print("Warning: some classes have <2 samples; proceeding without stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify_arg, random_state=42
    )

    print("Training linear SVM classifier...")
    svc = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    # --- Classification Report ---
    report_dict = classification_report(
        y_test, y_pred, labels=present_labels,
        target_names=target_names_filtered, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    if 'support' in report_df.columns:
         report_df['support'] = report_df['support'].astype(int)

    report_csv = metrics_path / "classification_report.csv"
    report_df.round(4).to_csv(report_csv)
    print(f"Saved classification report CSV: {report_csv}")

    report_img = metrics_path / "classification_report.png"
    # Adjust column width for classification report image
    save_table_as_image(report_df.round(4), report_img, "Classification Report", col_width=1.5)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=target_names_filtered, columns=target_names_filtered)
    cm_csv = metrics_path / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv)
    print(f"Saved confusion matrix CSV: {cm_csv}")

    # Confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", xticklabels=target_names_filtered, yticklabels=target_names_filtered)
    plt.title("Confusion Matrix", fontsize=14, weight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_png = metrics_path / "confusion_matrix.png"
    plt.savefig(cm_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved confusion matrix image: {cm_png}")

    # --- Extra Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    extra_metrics = {
        "Accuracy": [accuracy],
        "Precision (Macro Avg)": [precision_macro],
        "Recall (Macro Avg)": [recall_macro],
        "F1-Score (Macro Avg)": [f1_macro],
    }
    extra_df = pd.DataFrame(extra_metrics, index=["Overall Performance"])

    extra_csv = metrics_path / "extra_metrics.csv"
    extra_df.round(4).to_csv(extra_csv)
    print(f"Saved extra metrics CSV: {extra_csv}")

    extra_img = metrics_path / "extra_metrics.png"
    # Adjust column width for extra metrics image
    save_table_as_image(extra_df.round(4), extra_img, "Model Performance Summary", col_width=2.5)

    # --- Save model bundle ---
    joblib.dump({'model': svc, 'present_labels': present_labels, 'class_names': target_names_filtered}, args.out_model)
    print(f"Saved trained SVM model: {args.out_model}")

    print("\n✅ All metrics and model successfully saved.")