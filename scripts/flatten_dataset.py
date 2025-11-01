# scripts/flatten_dataset.py
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def flatten_and_merge(source_dir, dest_dir):
    """
    Copies image files from a nested source directory structure like
    'source/<emotion>/<palette>/image.bmp' to a flat destination
    structure like 'dest/<emotion>/image.bmp'.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' not found.")
        return

    # Find emotion folders (e.g., angry, happy) in the source directory
    emotion_folders = [d for d in source_path.iterdir() if d.is_dir()]

    for emotion_folder in tqdm(emotion_folders, desc="Processing emotions"):
        emotion_name = emotion_folder.name
        
        # Create the corresponding emotion folder in the destination if it doesn't exist
        dest_emotion_path = dest_path / emotion_name
        dest_emotion_path.mkdir(parents=True, exist_ok=True)

        # Find all image files within this emotion folder, searching recursively
        image_files = list(emotion_folder.rglob("*.bmp")) + \
                      list(emotion_folder.rglob("*.png")) + \
                      list(emotion_folder.rglob("*.jpg"))
        
        if not image_files:
            print(f"Warning: No images found in {emotion_folder}")
            continue

        # Copy each image to the flattened destination
        for img_file in tqdm(image_files, desc=f"  -> Merging '{emotion_name}'", leave=False):
            # Use a new name to avoid overwriting files with the same name from different subfolders
            new_filename = f"{img_file.parent.name}_{img_file.name}"
            shutil.copy(img_file, dest_emotion_path / new_filename)

    print(f"\nSuccessfully merged new dataset into '{dest_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a nested image dataset and merge it into another directory.")
    parser.add_argument("--source_dir", required=True, help="Path to the downloaded, nested dataset directory (e.g., 'C:/Users/YourUser/Downloads/Facial emotion').")
    parser.add_argument("--dest_dir", default="data", help="Path to your project's data directory.")
    
    args = parser.parse_args()
    
    flatten_and_merge(args.source_dir, args.dest_dir)