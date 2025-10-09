import os
import yaml

# === CONFIGURA QUI ===
dataset_dir = ""   # <-- metti qui il percorso della cartella principale
class_a = "ball"                  # nome attuale con ID 0
class_b = "kicker"                # nome attuale con ID 1
# ======================

yaml_path = os.path.join(dataset_dir, "data.yaml")

# Leggi le classi originali dal file YAML
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

old_names = data["names"]

if class_a not in old_names or class_b not in old_names:
    raise ValueError("Una delle classi specificate non esiste in data.yaml")

id_a = old_names.index(class_a)
id_b = old_names.index(class_b)

print(f"🔄 Invertendo classi: {class_a} (ID {id_a}) ↔ {class_b} (ID {id_b})")

# Funzione per invertire le etichette nei file .txt
def swap_labels(labels_dir):
    if not os.path.exists(labels_dir):
        return
    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue
        file_path = os.path.join(labels_dir, file)
        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                # swap degli ID
                if cls == id_a:
                    cls = id_b
                elif cls == id_b:
                    cls = id_a
                parts[0] = str(cls)
                new_lines.append(" ".join(parts))
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

# Applica alle sottocartelle
for subset in ["train", "valid", "test"]:
    labels_path = os.path.join(dataset_dir, subset, "labels")
    if os.path.exists(labels_path):
        print(f"✏️  Aggiornando {labels_path} ...")
        swap_labels(labels_path)

# Inverti anche i nomi nel data.yaml
new_names = old_names.copy()
new_names[id_a], new_names[id_b] = new_names[id_b], new_names[id_a]
data["names"] = new_names

with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print("✅ Classi invertite correttamente!")
print("🔤 Nuove classi in data.yaml:", new_names)