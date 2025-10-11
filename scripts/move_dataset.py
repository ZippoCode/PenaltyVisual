import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Merge datasets")
parser.add_argument("--source_dir", help="Source dataset path")
parser.add_argument("--target_dir", help="Target dataset path")
args = parser.parse_args()

source_dir = args.source_dir
target_dir = args.target_dir

subsets = ["train", "valid", "test"]

for subset in subsets:
    source_images = os.path.join(source_dir, subset, "images")
    source_labels = os.path.join(source_dir, subset, "labels")
    
    target_images = os.path.join(target_dir, subset, "images")
    target_labels = os.path.join(target_dir, subset, "labels")
    
    if not os.path.exists(source_images):
        print(f"⚠️  Skipping {subset}: source images not found")
        continue
    
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    image_count = 0
    for file in os.listdir(source_images):
        src = os.path.join(source_images, file)
        dst = os.path.join(target_images, file)
        shutil.copy2(src, dst)
        image_count += 1
    
    label_count = 0
    if os.path.exists(source_labels):
        for file in os.listdir(source_labels):
            src = os.path.join(source_labels, file)
            dst = os.path.join(target_labels, file)
            shutil.copy2(src, dst)
            label_count += 1
    
    print(f"✅ {subset}: copied {image_count} images, {label_count} labels")

print(f"✅ Dataset merge completed!")