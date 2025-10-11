import os
import yaml
import argparse

parser = argparse.ArgumentParser(description="Filter dataset classes")
parser.add_argument("--dataset", help="Path to dataset directory")
parser.add_argument("--keep", nargs="+", default=["ball"], help="Classes to keep (default: ball)")
args = parser.parse_args()

dataset_dir = args.dataset
keep_classes = args.keep

yaml_path = os.path.join(dataset_dir, "data.yaml")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

old_names = data["names"]

keep_ids = {}
for cls in keep_classes:
    if cls not in old_names:
        raise ValueError(f"Class '{cls}' not found in data.yaml")
    keep_ids[old_names.index(cls)] = cls

print(f"✅ Keeping only: {keep_classes}")
print(f"📌 Original IDs: {dict(sorted(keep_ids.items()))}")

id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_ids.keys()))}
print(f"🔄 ID mapping: {id_mapping}")

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
                
                old_id = int(parts[0])
                if old_id not in keep_ids:
                    continue
                
                new_id = id_mapping[old_id]
                new_lines.append(f"{new_id} {' '.join(parts[1:])}")
        
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

new_names = [keep_ids[old_id] for old_id in sorted(keep_ids.keys())]
data["names"] = new_names
with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print(f"✅ Dataset filtered successfully!")
print(f"🔤 Final classes: {new_names}")