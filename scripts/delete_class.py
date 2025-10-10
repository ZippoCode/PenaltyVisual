import os
import yaml

# === CONFIGURA QUI ===
dataset_dir = ""   # <-- cambia qui con il percorso della cartella principale
remove_class = "player"             # <-- nome della classe da eliminare
# ======================

yaml_path = os.path.join(dataset_dir, "data.yaml")

# Carica le classi originali
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

old_names = data["names"]

# Trova l'ID della classe da rimuovere
if remove_class not in old_names:
    raise ValueError(f"La classe '{remove_class}' non è presente in data.yaml")

remove_id = old_names.index(remove_class)
print(f"🗑️  Classe da rimuovere: '{remove_class}' (ID {remove_id})")

# Crea nuova lista di nomi (senza la classe da eliminare)
new_names = [n for n in old_names if n != remove_class]

# Funzione per aggiornare i file di label
def process_labels(labels_dir):
    if not os.path.exists(labels_dir):
        return
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
                if old_id == remove_id:
                    continue  # scarta le annotazioni della classe rimossa
                # Se la classe eliminata aveva ID minore di altre, scala gli ID successivi di -1
                new_id = old_id - 1 if old_id > remove_id else old_id
                parts[0] = str(new_id)
                new_lines.append(" ".join(parts))
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

# Applica a tutte le sottocartelle
for subset in ["train", "valid", "test"]:
    labels_path = os.path.join(dataset_dir, subset, "labels")
    if os.path.exists(labels_path):
        print(f"✏️  Aggiornando {labels_path} ...")
        process_labels(labels_path)

# Aggiorna il file data.yaml
data["names"] = new_names
with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print("✅ Classe rimossa correttamente!")
print("🔤 Nuove classi:", new_names)