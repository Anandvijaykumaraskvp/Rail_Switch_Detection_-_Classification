!pip install ultralytics

import os
import yaml
from ultralytics import YOLO

#######################################
# 1. CREATE data.yaml
#######################################
dataset_path = "/kaggle/input/railsem19-yolo-split/RailSem19_yolo_split"

data_dict = {
    "train": os.path.join(dataset_path, "images/train"),
    "val": os.path.join(dataset_path, "images/val"),
    "nc": 1,  # Number of classes (since merging into single "switch" class)
    "names": ["switch"]  # Unified class name
}

# Save the YAML file
data_yaml_path = os.path.join(os.getcwd(), "data.yaml")
with open(data_yaml_path, "w") as f:
    yaml.dump(data_dict, f)

print(f"✅ data.yaml created at: {data_yaml_path}")

#######################################
# 2. TRAIN YOLO MODEL
#######################################

# Load YOLOv8 model
model = YOLO("yolov8l.pt")  # You are using YOLOv8 Large model

model.train(
    data=data_yaml_path,  # ✅ Ensure data_yaml_path is correctly defined
    epochs=300,  # More epochs for better learning
    batch=8,
    save_period=5,  # Save checkpoint every 5 epochs
    patience=50,  # Stop training if no improvement in 50 epochs
    imgsz=960,  # Large image size for better accuracy
    lr0=1e-4,  # Lower initial learning rate for fine-tuning
    lrf=0.01,  # Final LR = lr0 * lrf
    warmup_epochs=5.0,  # Longer warmup for stable training
    momentum=0.95,  # Higher momentum for smoother optimization
    weight_decay=0.001,  # Regularization
    fliplr=0.5,  # Flip augmentation
    mosaic=1.0,  # Mosaic augmentation
    mixup=0.2,  # Mixup augmentation
    scale=0.9,  # Image scale factor
    device="cuda",  # Ensure it's running on GPU
    workers=2,  # Number of DataLoader workers
    project="improved_exp",
    name="yolov8l_long_run",
    exist_ok=True
)

print("✅ Training Started! Check logs in 'improved_exp/yolov8l_long_run'")

