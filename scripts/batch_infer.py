# scripts/batch_infer.py
import argparse
from pathlib import Path
import joblib
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
from scripts.preprocess import preprocess_image
import csv
import time


def predict_image(img_path, vit, proc, clf, class_names):
    img = preprocess_image(img_path, target_size=(224,224))
    if img is None:
        return None, None
    pil = Image.fromarray(img.astype('uint8'))
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = vit(**inputs)
        feat = outputs.pooler_output.numpy()
    pred_idx = int(clf.predict(feat)[0])
    probs = clf.predict_proba(feat)[0] if hasattr(clf, "predict_proba") else None
    # Ensure class_names are strings before indexing
    class_names_str = [str(name) for name in class_names]
    label = class_names_str[pred_idx] if pred_idx < len(class_names_str) else f"class_{pred_idx}"
    return label, probs


def main(args):
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    model_name = args.model_name
    classifier_path = args.classifier

    print("Loading ViT and processor (CPU)...")
    proc = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name).eval()

    print("Loading classifier...")
    bundle = joblib.load(classifier_path)
    clf = bundle['model']
    # Load class names, ensuring they are strings
    class_names_raw = bundle.get('class_names') or bundle.get('classes') or bundle.get('present_labels')
    if class_names_raw is None:
        raise ValueError("Class names/labels not found in the classifier bundle.")
    class_names = [str(name) for name in class_names_raw] # Ensure strings
    if all(isinstance(x, (int, np.integer)) for x in class_names): # Fallback if only indices
        class_names = [f"class_{i}" for i in class_names]

    # --- MODIFIED LINE: Added '.bmp' ---
    img_paths = sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in ('.jpg','.jpeg','.png', '.bmp')])
    # --- END MODIFICATION ---

    print(f"Found {len(img_paths)} images") # This count should now be higher

    times = []
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','pred_label','pred_probabilities'])
        # Use tqdm for progress bar
        from tqdm import tqdm
        for p in tqdm(img_paths, desc="Batch Inferencing"):
            t0 = time.time()
            label, probs = predict_image(p, vit, proc, clf, class_names)
            t1 = time.time()
            times.append(t1 - t0)

            if label is None:
                writer.writerow([str(p), 'NO_FACE', ''])
            else:
                probs_str = ''
                if probs is not None:
                    # Ensure using string class names for consistent output
                    probs_str = ';'.join([f"{class_names[i]}:{probs[i]:.4f}" for i in range(len(class_names))])
                writer.writerow([str(p), label, probs_str])

    print("Saved results to", out_csv)
    total_images = len(times)
    if total_images > 0:
        total_time = sum(times)
        print(f"Processed {total_images} images in {total_time:.2f} seconds")
        print(f"Average inference time per image: {total_time/total_images:.3f} seconds")
        print(f"Max per image: {max(times):.3f}, Min per image: {min(times):.3f}")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="root folder with class subfolders or images")
    parser.add_argument("--classifier", default="outputs/classifier_final.joblib", help="Path to the trained classifier .joblib file")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--out_csv", default="outputs/predictions_final.csv", help="Path to save the output predictions CSV file")
    args = parser.parse_args()
    main(args)