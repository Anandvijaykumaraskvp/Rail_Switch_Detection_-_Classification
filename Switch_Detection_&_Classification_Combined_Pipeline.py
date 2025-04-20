import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ensure inline display in Jupyter/Colab
%matplotlib inline

# ------------------------- #
# Paths to Trained Models   #
# ------------------------- #
YOLO_MODEL_PATH = "/kaggle/input/yolo_detect/other/default/1/135 epoch_yolo_L.pt"
CLASSIFICATION_MODEL_PATH = "/kaggle/input/newkeras/other/default/1/switch_classification_model.keras"

# ------------------------- #
# Load YOLOv8 Detection Model #
# ------------------------- #
print("🔄 Loading YOLOv8 Detection Model...")
detection_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLOv8 model loaded successfully!")

# ------------------------- #
# Load MobileNetV2 Classification Model #
# ------------------------- #
print("🔄 Loading MobileNetV2 Classification Model...")
classification_model = load_model(CLASSIFICATION_MODEL_PATH, safe_mode=False)
print("✅ MobileNetV2 classification model loaded successfully!")

# Class labels for classification
CLASS_NAMES = ["switch-left", "switch-right"]

# ------------------------- #
# Preprocessing Function for Classification #
# ------------------------- #
def preprocess_image(image, target_size=(224, 224)):
    """Resizes, normalizes, and preprocesses the image for MobileNetV2 classification."""
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("⚠️ Skipping empty or invalid image region!")
        return None
    
    image = cv2.resize(image, target_size)  # Resize to 224x224
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image, axis=0)  # Expand dimensions for model input

# ------------------------- #
# Detection + Classification Pipeline #
# ------------------------- #
def detect_and_classify(image_path):
    """
    1. Displays **original test image**.
    2. Detects railway switches using YOLOv8.
    3. Extracts detected regions.
    4. Classifies switches as 'switch-left' or 'switch-right'.
    5. Sorts switches from **bottom to top** and assigns names (SWITCH-1, SWITCH-2,...).
    6. Displays detected switches and their classification.
    7. Asks user for expected switch position.
    8. Compares results and prints PASS/FAIL.
    9. 🚨 Shows WARNING if mismatch is found.
    10. Displays final processed image.
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Error loading image: {image_path}")

    # ------------------- #
    # Show Original Test Image
    # ------------------- #
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Original Test Image")
    plt.show(block=True)

    print("🔄 Running YOLOv8 detection...")
    results = detection_model(image)

    # Get detected objects
    detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to numpy
    if len(detections) == 0:
        print("❌ No switch detected in the image!")
        return

    print(f"✅ {len(detections)} switches detected!")

    detected_switches = []  # List to store detected switch images and info

    # Process each detected switch
    for (x1, y1, x2, y2, conf, cls) in detections:
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Extract region of interest (ROI)
        detected_switch = image[y1:y2, x1:x2]

        # Classify only if valid image region exists
        processed_img = preprocess_image(detected_switch)
        if processed_img is None:
            continue  # Skip invalid detections

        # Run classification
        prediction = classification_model.predict(processed_img)
        class_index = np.argmax(prediction)
        class_label = CLASS_NAMES[class_index]
        confidence = prediction[0][class_index]

        # Store detection details
        detected_switches.append({
            "image": cv2.cvtColor(detected_switch, cv2.COLOR_BGR2RGB),  # Convert to RGB for display
            "label": class_label,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })

    # ------------------- #
    # Sorting by Y-coordinates (Bottom to Top)
    # ------------------- #
    detected_switches.sort(key=lambda x: x["bbox"][1], reverse=True)  # Sort by Y1 (bottom to top)

    # ------------------- #
    # Assign Switch Names
    # ------------------- #
    for idx, switch in enumerate(detected_switches):
        switch["name"] = f"SWITCH-{idx+1}"

    # ------------------- #
    # Print Results
    # ------------------- #
    print("\n🔹 **Detection Results (Sorted from Bottom to Top):**")
    for switch in detected_switches:
        print(f"{switch['name']}: {switch['label']} ({switch['confidence']:.2f} confidence)")

    # ------------------- #
    # User Input for Expected Positions
    # ------------------- #
    expected_positions = {}
    for switch in detected_switches:
        while True:
            expected_label = input(f"👉 Enter expected position for {switch['name']} (switch-left/switch-right): ").strip().lower()
            if expected_label in CLASS_NAMES:
                expected_positions[switch["name"]] = expected_label
                break
            else:
                print("⚠️ Invalid input! Please enter 'switch-left' or 'switch-right'.")

    # ------------------- #
    # Compare Results and Print PASS/FAIL
    # ------------------- #
    mismatch_detected = False
    print("\n🔍 **Validation Results:**")
    for switch in detected_switches:
        actual_label = switch["label"]
        expected_label = expected_positions[switch["name"]]
        status = "✅ PASS" if actual_label == expected_label else "❌ FAIL - MISMATCH"

        print(f"{switch['name']}: Actual={actual_label}, Expected={expected_label} ➝ {status}")

        # 🚨 Alert if a mismatch is found
        if actual_label != expected_label:
            mismatch_detected = True
            print("\033[1;31m🚨 WARNING: SWITCH MISMATCH DETECTED! STOP THE TRAIN! 🚨\033[0m")
            print(f"\033[1;31m❌ {switch['name']} is misaligned and not in the expected position! \033[0m")

    # ------------------- #
    # Show Detected Switches
    # ------------------- #
    for switch in detected_switches:
        plt.figure(figsize=(5, 5))
        plt.imshow(switch["image"])
        plt.title(f"{switch['name']}: {switch['label']} ({switch['confidence']:.2f})")
        plt.axis("off")
        plt.show(block=True)

    # ------------------- #
    # Draw Final Bounding Boxes & Labels
    # ------------------- #
    for switch in detected_switches:
        x1, y1, x2, y2 = switch["bbox"]
        class_label = switch["label"]
        confidence = switch["confidence"]
        switch_name = switch["name"]

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image,
            f"{switch_name}: {class_label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # ------------------- #
    # Show Final Image with Detections
    # ------------------- #
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLOv8 Detection + Classification Results")
    plt.show(block=True)

# ------------------------- #
# Test Pipeline on an Image #
# ------------------------- #
TEST_IMAGE_PATH = "/kaggle/input/final-test-images/Final_Test_Images/2.jpg"  # Change this to your test image
detect_and_classify(TEST_IMAGE_PATH)
