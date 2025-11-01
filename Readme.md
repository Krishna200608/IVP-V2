# **Thermal Emotion Recognition Project**

This project uses **Vision Transformer (ViT) embeddings + a balanced Support Vector Machine (SVM)** to detect emotions from thermal facial images. It includes **preprocessing with a robust face detector and data augmentation**, **feature extraction**, **classifier training with class balancing**, **batch inference**, **detailed metrics saving (CSV and Images)**, and a **Gradio web app** for interactive predictions.

---

## **Datasets Used**

This project combines two primary thermal emotion datasets to create a more robust and diverse training set.

### **1. Original Project Dataset (Zenodo)**
This is the foundational dataset for the project, providing thermal images (`.jpg` format) for the six core emotion classes.
* **Link:** [https://zenodo.org/records/15830552](https://zenodo.org/records/15830552)

### **2. Comprehensive Facial Thermal Dataset (Mendeley)**
This dataset was integrated to increase the sample size and data variety. It contains thermal images (`.bmp` format) categorized by emotion, with nested sub-folders for different thermal palettes (e.g., `ICEBLUE`, `IRNBOW`, `RAINBOW`).
* **Link:** [https://data.mendeley.com/datasets/8885sc9p4z/1](https://data.mendeley.com/datasets/8885sc9p4z/1)

The `scripts/flatten_dataset.py` script is used to merge the nested Mendeley dataset into the primary `data/` folder, creating a single, unified data source for the preprocessing pipeline.

---

## **Project Structure**
```

IVP-Thermal-Emotion-f82dffcae4647b0b4790f5dc9263932494f1e25c/
├── scripts/
│   ├── preprocess.py          \# Preprocesses images (detect face, resize, augment)
│   ├── extract\_features.py    \# Extracts ViT features
│   ├── train\_classifier.py    \# Trains the SVM and saves metrics
│   ├── infer.py               \# Predicts emotion for a single image
│   ├── batch\_infer.py         \# Predicts emotions for multiple images
│   └── flatten\_dataset.py     \# Merges external datasets (run once if needed)
├── data/                      \# Input thermal images (organized by emotion)
├── outputs/
│   ├── preprocessed\_augmented.npz \# Output of preprocess.py
│   ├── features\_augmented.npz     \# Output of extract\_features.py
│   ├── classifier\_final.joblib    \# Trained SVM model
│   ├── predictions\_final.csv      \# Output of batch\_infer.py
│   └── metrics/                   \# Training metrics output
│       ├── classification\_report.csv
│       ├── classification\_report.png  \# Image of report
│       ├── confusion\_matrix.csv
│       ├── confusion\_matrix.png
│       ├── extra\_metrics.csv          \# Summary metrics CSV
│       └── extra\_metrics.png          \# Image of summary metrics
├── app.py                     \# Gradio web app for predictions
├── README.md
└── requirements.txt

````

---

## **Setup Environment**

```powershell
# Create a virtual environment using Python 3.11
py -3.11 -m venv thermal_env311

# Activate the environment in PowerShell
.\thermal_env311\Scripts\activate
# --- OR --- Activate in Command Prompt (cmd) / Git Bash
# thermal_env311\Scripts\activate.bat
# --- OR --- Activate in macOS/Linux bash/zsh
# source thermal_env311/bin/activate

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
````

**`requirements.txt`** should include:

```
torch
transformers
numpy
scikit-learn
opencv-python
Pillow
joblib
gradio
pyngrok
matplotlib
seaborn
pandas
```

-----

## **Step-by-Step Usage**

### **0. (Optional) Merge External Dataset**

If you have downloaded the external dataset (Mendeley) and it is not yet merged, run this script *once* to organize and copy the images into your `data/` folder. **Replace the `--source_dir` path with the actual path to the downloaded `Facial emotion` folder.**

```powershell
# Example for Windows:
python scripts/flatten_dataset.py --source_dir "C:/path/to/downloaded/Facial emotion" --dest_dir data

# Example for macOS/Linux:
python scripts/flatten_dataset.py --source_dir "/path/to/downloaded/Facial emotion" --dest_dir data
```

-----

### **1. Preprocess Images (with Augmentation)**

This script uses an improved face detector and applies data augmentation, especially for underrepresented classes like 'fear'.

```powershell
python scripts/preprocess.py --data_dir data --save_npz outputs/preprocessed_augmented.npz --augment --augment_factor 3
```

  * `--augment`: Enables data augmentation.
  * `--augment_factor 3`: Creates multiple augmented versions per image (more for 'fear').

-----

### **2. Extract Features (ViT Embeddings)**

Extract features from the preprocessed and augmented data.

```powershell
python scripts/extract_features.py --npz_input outputs/preprocessed_augmented.npz --out_npz outputs/features_augmented.npz --batch_size 8 --num_threads 6
```

-----

### **3. Train Classifier (Balanced SVM & Save Metrics)**

Train the SVM using the augmented features. This script uses `class_weight='balanced'` and saves performance metrics as both CSV and styled PNG images.

```powershell
python scripts/train_classifier.py --features outputs/features_augmented.npz --out_model outputs/classifier_final.joblib --metrics_dir outputs/metrics
```

**Output:**

  * `outputs/classifier_final.joblib`: The trained model.
  * `outputs/metrics/classification_report.csv`: Detailed performance metrics.
  * `outputs/metrics/classification_report.png`: Image of the classification report table.
  * `outputs/metrics/confusion_matrix.csv`: Confusion matrix data.
  * `outputs/metrics/confusion_matrix.png`: Visual heatmap of the confusion matrix.
  * `outputs/metrics/extra_metrics.csv`: Summary metrics (Accuracy, Macro Precision/Recall/F1).
  * `outputs/metrics/extra_metrics.png`: Image of the summary metrics table.

-----

### **4. Test Single Image**

Test the final model on a single image. Note the updated output format.

```powershell
python scripts/infer.py --image "data/happy/0.jpg" --classifier outputs/classifier_final.joblib
```

**Example Output:**

```
Final output : [happy (91.32%)]
Other compositions : ['happy', (91.32%)], ['surprise', (6.40%)], ['anger', (1.28%)], ['neutral', (0.44%)], ['sad', (0.42%)], ['fear', (0.14%)]
```

-----

### **5. Batch Inference**

Run predictions on all images in the `data` directory using the final model. **Note the use of `python -m`** to correctly handle imports within the `scripts` package.

```powershell
# Ensure batch_infer.py defaults are updated or specify classifier/out_csv
python -m scripts.batch_infer --data_dir data
# OR (if defaults not updated)
# python -m scripts.batch_infer --data_dir data --classifier outputs/classifier_final.joblib --out_csv outputs/predictions_final.csv
```

**Output:** `outputs/predictions_final.csv` containing predicted labels and probabilities for all images.

-----

### **6. Run Gradio Web App**

Launch the interactive web application (ensure `app.py` loads `classifier_final.joblib`).

```powershell
python app.py
```

  * Opens a local web interface (e.g., `http://127.0.0.1:7860`).
  * Upload a thermal image to get the predicted emotion.

-----

## **7. Performance Notes**

The `train_classifier.py` script automatically saves performance metrics (classification report, confusion matrix, summary stats) as both CSV files and PNG images to the `outputs/metrics/` directory.

You can still measure average inference time using `batch_infer.py` (ensure it uses the final classifier and is run with `python -m`).

-----

## **8. Optional Improvements**

  * Compare **SVM on ViT embeddings** vs **fine-tuned ViT head** for performance improvement.
  * Experiment with **different kernels (RBF, polynomial)** in SVM.
  * Integrate **real-time webcam input** in the Gradio app.
  * **Fine-tune the Face Detector:** For even better face detection on thermal images, consider fine-tuning the deep learning face detector (`res10_300x300_ssd_iter_140000.caffemodel`) on a thermal face dataset with bounding box annotations.

-----

## **Author**

**IIT2023139 — Krishna Sikheriya**

-----

```
```