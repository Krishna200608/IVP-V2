# scripts/extract_features.py
import numpy as np
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

def load_images_from_npz(npz_path):
    a = np.load(npz_path, allow_pickle=True)
    images = a['images']  # shape (N, H, W, 3)
    labels = a['labels']
    class_names = a['class_names'].tolist()
    return images, labels, class_names

def batch_iter(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i+batch_size]

def extract_and_save(npz_input, out_npz, model_name='google/vit-base-patch16-224-in21k', batch_size=16, num_threads=6):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model = model.to(device)  # type: ignore
    model.eval()

    images, labels, class_names = load_images_from_npz(npz_input)
    features = []
    for batch in tqdm(batch_iter(images, batch_size), total=(len(images)+batch_size-1)//batch_size):
        # convert numpy batch to PIL images
        pil_images = [Image.fromarray(x.astype('uint8')) for x in batch]
        inputs = processor(images=pil_images, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # use pooled output (CLS-like) or last_hidden_state[:,0,:]
            vecs = outputs.pooler_output.cpu().numpy()
            features.append(vecs)
    features = np.vstack(features)
    np.savez_compressed(out_npz, features=features, labels=labels, class_names=class_names)
    print("Saved features to", out_npz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_input", required=True)
    parser.add_argument("--out_npz", default="outputs/features.npz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=6)
    args = parser.parse_args()
    extract_and_save(args.npz_input, args.out_npz, batch_size=args.batch_size, num_threads=args.num_threads)
