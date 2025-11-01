# scripts/infer.py (Formatted Output)
from pathlib import Path
import sys

# Make sure 'scripts' directory is importable when executing this file directly
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import argparse
import joblib
import numpy as np
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor

# Local import (preprocess.py must be in the same scripts/ folder)
from preprocess import preprocess_image

def predict(image_path, model_name, classifier_path):
    # Load processor and ViT (feature extractor)
    proc = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name).eval()

    # Load classifier
    try:
        clf_bundle = joblib.load(classifier_path)
        clf = clf_bundle['model']
        # --- Ensure class_names are strings ---
        class_names_raw = clf_bundle.get('class_names') or clf_bundle.get('classes') or clf_bundle.get('present_labels')
        if class_names_raw is None:
             raise ValueError("Class names/labels not found in the classifier bundle.")
        # Convert potential numpy strings or other types to standard Python strings
        class_names = [str(name) for name in class_names_raw]
        # --- End Ensure ---
        if isinstance(class_names, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in class_names):
            class_names = [f"class_{i}" for i in class_names] # Fallback if names are just indices
    except Exception as e:
        print(f"ERROR loading classifier: {e}")
        return

    img = preprocess_image(image_path, target_size=(224, 224))
    if img is None:
        print("No face detected or could not read image.")
        return

    pil = Image.fromarray(img.astype('uint8'))
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = vit(**inputs)
        feat = outputs.pooler_output.numpy()

    # --- Start Output Formatting ---
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feat)[0]
        # Create a list of (emotion, probability) tuples and sort by probability descending
        prob_list = sorted(zip(class_names, probs), key=lambda item: item[1], reverse=True)

        # Get the top prediction
        top_emotion, top_prob = prob_list[0]
        print(f"Final output : [{top_emotion} ({top_prob*100:.2f}%)]")

        # Format the rest
        other_compositions = [f"['{emo}', ({prob*100:.2f}%)]" for emo, prob in prob_list]
        print(f"Other compositions : {', '.join(other_compositions)}")

    else:
        # Fallback if predict_proba is not available
        pred_idx = clf.predict(feat)[0]
        label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
        print(f"Final output : [{label} (Probability not available)]")
    # --- End Output Formatting ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    # Make sure this default points to your final model
    parser.add_argument("--classifier", default="outputs/classifier_final.joblib")
    args = parser.parse_args()
    predict(args.image, args.model_name, args.classifier)