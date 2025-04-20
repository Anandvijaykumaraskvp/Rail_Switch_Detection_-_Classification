import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import random

# Fixed Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Must be divisible by 2 (for class balancing)
AUGMENTATION_COUNT = 3  # Generates 3 augmented images per original
CLASS_NAMES = ["switch-left", "switch-right"]

##############################################################################
#                             STEP 1: Define Dataset Paths                   #
##############################################################################
base_dir = "/kaggle/input/classification-dataset-resized-gap/classification_dataset_resized"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

##############################################################################
#                         STEP 2: Compute Class Weights                      #
##############################################################################
train_labels = []
for label in CLASS_NAMES:
    folder = os.path.join(train_dir, label)
    if os.path.exists(folder):
        num_samples = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_labels.extend([CLASS_NAMES.index(label)] * num_samples)

if len(train_labels) == 0:
    raise ValueError("No training images found!")

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Computed Class Weights:", class_weight_dict)

##############################################################################
#                        STEP 3: Custom Preprocessing Function                #
##############################################################################
def preprocess_image(image_path, target_size=IMG_SIZE):
    """Loads an image, applies denoising, sharpens, and resizes while preserving aspect ratio."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping corrupted image: {image_path}")
        return None
    
    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Sharpening Kernel
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen_kernel)
    
    # Adaptive Resize while preserving aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Padding to maintain target size
    delta_w, delta_h = target_size[0] - new_w, target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = image.astype(np.float32) / 255.0  # Normalize
    return image

##############################################################################
#                        STEP 4: Data Augmentation Functions                  #
##############################################################################
def augment_image(image):
    """Applies multiple augmentations without flipping or zooming."""
    # Random Brightness
    brightness_factor = 0.8 + np.random.rand() * 0.4  
    image_aug = np.clip(image * brightness_factor, 0, 1)

    # Random Contrast
    contrast_factor = 0.8 + np.random.rand() * 0.4
    mean = np.mean(image_aug, axis=(0, 1), keepdims=True)
    image_aug = np.clip(mean + (image_aug - mean) * contrast_factor, 0, 1)

    # Random Rotation (≤15°)
    angle = -15 + np.random.rand() * 30  
    h, w = image_aug.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine((image_aug * 255).astype(np.uint8), M, (w, h))
    rotated = rotated.astype(np.float32) / 255.0

    return rotated

##############################################################################
#                    STEP 5: Custom Data Generator                           #
##############################################################################
class AnnotatedDataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, batch_size=32, target_size=IMG_SIZE, class_names=CLASS_NAMES, augment=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_names = class_names
        self.augment = augment

        # Load all image files per class
        self.image_files = {label: [] for label in class_names}
        for label in class_names:
            folder = os.path.join(image_dir, label)
            if os.path.exists(folder):
                self.image_files[label] = [
                    os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

        # Shuffle the dataset
        for label in class_names:
            random.shuffle(self.image_files[label])

    def __len__(self):
        total_images = sum(len(files) for files in self.image_files.values())
        return int(np.ceil(total_images * (1 + AUGMENTATION_COUNT) / self.batch_size))

    def __getitem__(self, idx):
        batch_images, batch_labels = [], []
        
        batch_size_per_class = self.batch_size // len(self.class_names)

        for label in self.class_names:
            label_files = self.image_files[label]
            selected_files = random.sample(label_files, min(batch_size_per_class, len(label_files)))

            for image_path in selected_files:
                image = preprocess_image(image_path)
                if image is None:
                    continue

                label_index = self.class_names.index(label)
                batch_images.append(image)
                batch_labels.append(label_index)

                # Augmentation (1 original + 3 augmented)
                for _ in range(AUGMENTATION_COUNT):
                    batch_images.append(augment_image(image))
                    batch_labels.append(label_index)

        batch_images = np.array(batch_images)
        batch_labels = keras.utils.to_categorical(batch_labels, num_classes=len(self.class_names))
        return batch_images, batch_labels

##############################################################################
#             STEP 6: Build & Train the Model (MobileNetV2 Base)             #
##############################################################################
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output_layer = Dense(len(CLASS_NAMES), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss=CategoricalFocalCrossentropy(),
    metrics=["accuracy"]
)

train_generator = AnnotatedDataGenerator(image_dir=train_dir, batch_size=BATCH_SIZE, augment=True)
val_generator = AnnotatedDataGenerator(image_dir=val_dir, batch_size=BATCH_SIZE, augment=False)

print("Training model with optimized settings...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=[ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.3, min_lr=1e-6, verbose=1)]
)
model.save("switch_classification_model.h5")
print("Model saved as switch_classification_model.h5")


##############################################################################
# STEP 12: Evaluate the Model
##############################################################################
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
print(f"Training Accuracy: {train_accuracy:.2%}")

val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.2%}")

# saving in different format
model.save("/kaggle/working/switch_classification_model.keras")


from tensorflow.keras.models import load_model

model = load_model("/kaggle/working/switch_classification_model.keras")

print("Model loaded successfully!")


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Sharpening
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen_kernel)
    
    # Resize while keeping aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Padding to maintain target size
    delta_w, delta_h = target_size[0] - new_w, target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = image.astype(np.float32) / 255.0  # Normalize
    return image


# Test Evaluation

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ["switch-left", "switch-right"]
TEST_DIR = "/kaggle/input/d/padmaaskvp/classification-dataset-resized-gap/classification_dataset_resized_gap/classification_dataset_resized/test"
MODEL_PATH = "/kaggle/input/newkeras/other/default/1/switch_classification_model.keras"

##############################################################################
#                    Preprocessing Function (Same as Training)               #
##############################################################################
def preprocess_image(image_path, target_size=IMG_SIZE):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping corrupted image: {image_path}")
        return None
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen_kernel)
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    delta_w, delta_h = target_size[0] - new_w, target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = image.astype(np.float32) / 255.0
    return image

##############################################################################
#                 Load Test Images and Ground Truth Labels                   #
##############################################################################
def load_test_data(test_dir):
    images = []
    labels = []
    for label in CLASS_NAMES:
        folder = os.path.join(test_dir, label)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder, fname)
                image = preprocess_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(CLASS_NAMES.index(label))
    return np.array(images), np.array(labels)

print("Loading test images...")
X_test, y_true = load_test_data(TEST_DIR)
print(f"Loaded {len(X_test)} test images.")

##############################################################################
#                          Load Trained Model                                #
##############################################################################
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

##############################################################################
#                          Make Predictions                                  #
##############################################################################
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

##############################################################################
#                      Calculate Evaluation Metrics                          #
##############################################################################
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\nEvaluation Metrics on Test Set:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
