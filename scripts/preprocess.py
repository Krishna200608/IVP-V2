# scripts/preprocess.py (with Data Augmentation)
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random

# --- Face Detector from Step 1 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
models_dir = PROJECT_ROOT / "models"

proto_path = str(models_dir / "deploy.prototxt")
model_path = str(models_dir / "res10_300x300_ssd_iter_140000.caffemodel")
try:
    face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
except cv2.error as e:
    raise IOError(
        f"Could not load Caffe model for face detection. "
        f"Files looked for: {proto_path} and {model_path}. "
        f"Make sure to download them (see README)."
    ) from e
    
# --- New: Data Augmentation Function ---
def augment_image(image):
    """Applies random augmentations to an image."""
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = random.uniform(-10, 10)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Random brightness/contrast adjustment
    alpha = random.uniform(0.8, 1.2) # contrast
    beta = random.uniform(-10, 10)   # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image


def preprocess_image(image_path, target_size=(224, 224)):
    # ... (rest of the function is the same as in Step 1) ...
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Use the deep learning model for detection
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Find the best detection (highest confidence)
    best_detection = None
    max_confidence = 0.0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            best_detection = detections[0, 0, i, 3:7]

    # confidence level
    if best_detection is None or max_confidence < 0.25:  # Confidence threshold
        return None

    # Extract bounding box and crop the face
    box = best_detection * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # Ensure the bounding box is within the image bounds
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    if startX >= endX or startY >= endY:
        return None

    face = img[startY:endY, startX:endX]
    
    # Convert to grayscale for thermal-like processing
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize to the model's expected input size (ViT uses 224x224)
    face_resized = cv2.resize(gray_face, target_size, interpolation=cv2.INTER_AREA)

    # Convert back to 3 channels as expected by ViT
    face_3_channel = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

    return face_3_channel


def build_dataset(data_dir, out_images_dir=None, target_size=(224,224), augment=False, augment_factor=2):
    data_dir = Path(data_dir)
    images = []
    labels = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    for idx, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        image_paths = list(cls_dir.glob("*.*"))
        
        
        # --- START OF MODIFICATION ---
        # Set a higher augmentation factor for the 'fear' class
        current_augment_factor = augment_factor
        if cls == 'fear':
            # Increase this number if the imbalance is very large
            current_augment_factor = 10 
            print(f"\nApplying aggressive augmentation to '{cls}' class (factor: {current_augment_factor}).")
        # --- END OF MODIFICATION ---
        
        for p in tqdm(image_paths, desc=f"Processing {cls}"):
            if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
                
            proc_img = preprocess_image(p, target_size)
            
            if proc_img is not None:
                # Add original image
                images.append(proc_img)
                labels.append(idx)
                
                # Add augmented versions
                if augment:
                    for _ in range(augment_factor):
                        augmented = augment_image(proc_img)
                        images.append(augmented)
                        labels.append(idx)

                if out_images_dir:
                    out_path = Path(out_images_dir) / cls
                    out_path.mkdir(parents=True, exist_ok=True)
                    fn = out_path / p.name
                    cv2.imwrite(str(fn), cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR))

    if not images:
        return np.zeros((0, *target_size, 3), dtype=np.uint8), np.array([], dtype=np.int64), []
        
    return np.stack(images), np.array(labels, dtype=np.int64), class_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_images_dir", default=None)
    parser.add_argument("--target_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--save_npz", default="outputs/preprocessed_augmented.npz")
    parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
    parser.add_argument("--augment_factor", type=int, default=2, help="Number of augmented versions per image")
    args = parser.parse_args()

    images, labels, class_names = build_dataset(
        args.data_dir, 
        args.out_images_dir, 
        tuple(args.target_size),
        augment=args.augment,
        augment_factor=args.augment_factor
    )
    
    print("Processed images (with augmentation):", images.shape, "labels:", labels.shape)
    
    Path(args.save_npz).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(args.save_npz, images=images, labels=labels, class_names=class_names)
    print("Saved to", args.save_npz)