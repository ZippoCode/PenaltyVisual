import os
import random
import cv2
import yaml
import time

# === CONFIGURA QUI ===
dataset_dir = ""   # <-- cambia qui
# ======================

yaml_path = os.path.join(dataset_dir, "data.yaml")

# Carica le classi dal file data.yaml
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]

# Percorsi alle sottocartelle
splits = ["train", "valid", "test"]

def get_random_image():
    split = random.choice(splits)
    img_dir = os.path.join(dataset_dir, split, "images")
    label_dir = os.path.join(dataset_dir, split, "labels")

    imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    if not imgs:
        return None, None, None

    img_file = random.choice(imgs)
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    return img_path, label_path, split

def draw_yolo_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return img  # nessuna annotazione

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_center, y_center, width, height = map(float, parts[:5])
            cls_id = int(cls_id)

            # Converti da YOLO normalized a pixel
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            color = (0, 255, 0)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img

# === LOOP PRINCIPALE ===
print("🔁 Mostrando immagini casuali con label YOLO (Ctrl+C per uscire)")
try:
    while True:
        img_path, label_path, split = get_random_image()
        if not img_path:
            print("Nessuna immagine trovata.")
            break

        img = draw_yolo_boxes(img_path, label_path)
        if img is None:
            continue

        cv2.imshow(f"{split.upper()} — {os.path.basename(img_path)}", img)
        key = cv2.waitKey(0)  # aspetta tasto per passare all'immagine successiva
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\n🛑 Interrotto manualmente.")
    cv2.destroyAllWindows()