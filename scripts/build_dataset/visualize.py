import os
import random
import cv2
import yaml

# Configuration
dataset_dir = "<Path>"  # Set your dataset path here

yaml_path = os.path.join(dataset_dir, "data.yaml")

with open(yaml_path, "r") as f:
    class_names = yaml.safe_load(f)["names"]

splits = ["train", "valid", "test"]

def get_random_image():
    split = random.choice(splits)
    img_dir = os.path.join(dataset_dir, split, "images")
    label_dir = os.path.join(dataset_dir, split, "labels")
    
    imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    if not imgs:
        return None, None, None
    
    img_file = random.choice(imgs)
    return (os.path.join(img_dir, img_file),
            os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt"),
            split)

def draw_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        return img
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id, x_c, y_c, box_w, box_h = map(float, parts[:5])
            cls_id = int(cls_id)
            
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)
            
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img

# Main loop
print("🔁 Displaying random images (press any key for next, Ctrl+C to exit)")
try:
    while True:
        img_path, label_path, split = get_random_image()
        if not img_path:
            print("No images found")
            break
        
        img = draw_boxes(img_path, label_path)
        if img is None:
            continue
        
        cv2.imshow(f"{split.upper()} - {os.path.basename(img_path)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\n🛑 Stopped")
    cv2.destroyAllWindows()