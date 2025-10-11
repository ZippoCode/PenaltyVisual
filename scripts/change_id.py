import os
import yaml
import argparse

parser = argparse.ArgumentParser(description="Reassign class ID in dataset")
parser.add_argument("--dataset", help="Path to dataset directory")
parser.add_argument("--class_name", default="ball", help="Class name to reassign (default: ball)")
parser.add_argument("--new_id", type=int, default=1, help="New ID to assign (default: 1)")

args = parser.parse_args()

dataset_dir = args.dataset
class_name = args.class_name
new_id = args.new_id

yaml_path = os.path.join(dataset_dir, "data.yaml")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

old_names = data["names"]

if len(old_names) != 1:
    raise ValueError(f"Expected 1 class, found {len(old_names)}: {old_names}")

if old_names[0] != class_name:
    raise ValueError(f"Expected class '{class_name}', found '{old_names[0]}'")

print(f"📌 Current: {class_name} (ID 0)")
print(f"🔄 Changing to: {class_name} (ID {new_id})")

for subset in ["train", "valid", "test"]:
    labels_dir = os.path.join(dataset_dir, subset, "labels")
    if not os.path.exists(labels_dir):
        continue
    
    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue
        
        path = os.path.join(labels_dir, file)
        new_lines = []
        
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                parts[0] = str(new_id)
                new_lines.append(" ".join(parts))
        
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

new_names = [None] * (new_id + 1)
new_names[new_id] = class_name

data["names"] = new_names
with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print(f"✅ Class ID changed successfully!")
print(f"🔤 Final classes: {new_names}")