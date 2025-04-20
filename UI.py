import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import pygame
from tensorflow.keras.applications import MobileNetV2

# === Load Models ===
yolo_model = YOLO("C:/Users/anand/Desktop/Working notebooks/YoloV8/135 epoch_yolo_L.pt")
model = MobileNetV2(weights=None, input_shape=(224, 224, 3), classes=2)
model.load_weights("C:/Users/anand/Desktop/Working notebooks/MobieNetV2/switch_classification_model.h5", by_name=True)

# === Sound Alert Setup ===
alert_sound_path = "C:/Users/anand/Downloads/emergency-alarm-with-reverb-29431.mp3"
pygame.mixer.init()

# === Global Variables ===
uploaded_image = None
original_image = None
boxes = None
confidence_scores = []  # To store YOLO confidence scores
manual_route = {}
last_prediction = {}
switch_frame = None
switch_buttons = {}
cropped_switches_data = [] # To store cropped images and classification results
flash_state = True # For flashing text

# === GUI Setup ===
root = tk.Tk()
root.title("Train Monitor and Dispatch System")
root.geometry("800x900") # Increased window size for better visualization

# === Style Configuration ===
button_font = ("Arial", 10, "bold")
label_font = ("Arial", 10)
header_font = ("Helvetica", 16, "bold")
section_title_font = ("Arial", 12, "bold")

# === Create Canvas and Scrollbar ===
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# === Create Frame inside Canvas ===
main_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=main_frame, anchor="nw")

# === Header Section ===
header_bg = "#f0f0f0" # Light gray background
header_frame = tk.Frame(main_frame, bg=header_bg, padx=10, pady=10)
header_frame.pack(pady=10, fill="x")

try:
    logo_img = Image.open("C:/Users/anand/Desktop/Working notebooks/GUI/logo.jpg")
    logo_img = logo_img.resize((80, 80))
    logo_img = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(header_frame, image=logo_img, bg=header_bg)
    logo_label.image = logo_img
    logo_label.pack(side=tk.LEFT, padx=10)
except:
    logo_label = tk.Label(header_frame, text="[Logo Missing]", bg=header_bg)
    logo_label.pack(side=tk.LEFT)

title_label = tk.Label(header_frame, text="Train Monitor and Dispatch System", font=header_font, bg=header_bg)
title_label.pack(side=tk.LEFT, padx=10)

# === Dispatcher and Engineer Info ===
info_frame = tk.Frame(main_frame, padx=10, pady=5)
info_frame.pack(fill="x")
dispatcher_label = tk.Label(info_frame, text="Dispatcher: [Anand V]", font=label_font)
dispatcher_label.pack(side="left")
engineer_label = tk.Label(info_frame, text="Locomotive Engineer: [Alexander]", font=label_font)
engineer_label.pack(side="right")

welcome_label = tk.Label(main_frame, text="Welcome to the Switch Detection Interface!", font=label_font)
welcome_label.pack(pady=10)

# === Image Panel Section ===
image_panel_frame = LabelFrame(main_frame, text="Detected Switches Image", font=section_title_font, padx=10, pady=10)
image_panel_frame.pack(pady=10, padx=10, fill="both", expand=True)
panel_title = tk.Label(image_panel_frame, text="Original Image", font=label_font)
panel_title.pack(pady=5)
panel = tk.Label(image_panel_frame)
panel.pack(pady=5)

# === Switch Configuration Section ===
switch_config_frame = LabelFrame(main_frame, text="Switch Configuration", font=section_title_font, padx=10, pady=10)
switch_config_frame.pack(pady=10, padx=10, fill="x")
switch_frame = switch_config_frame

# === Core Functions ===
def open_image():
    global uploaded_image, original_image, confidence_scores, cropped_switches_data
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        uploaded_image = cv2.imread(file_path)
        original_image = uploaded_image.copy()
        confidence_scores = []  # Reset confidence scores for a new image
        cropped_switches_data = [] # Reset cropped switches data
        detect_switches()

def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    img = img.resize((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk
    panel_title.config(text="Detected Switches")

def detect_switches():
    global uploaded_image, boxes, switch_frame, confidence_scores
    if uploaded_image is None:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    rgb_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(rgb_image, conf=0.25)
    boxes = []
    switch_number = 1  # Initialize switch number counter
    annotated_image = uploaded_image.copy() # Create a copy for drawing

    for result in results:
        for box_data in result.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy.cpu().numpy()[0])
            confidence = box_data.conf.cpu().numpy()[0]
            boxes.append((x1, y1, x2, y2))
            confidence_scores.append(confidence)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add switch number text to the image
            text = f"SWITCH-{switch_number}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (255, 0, 0)  # Red color for the number
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y2 + text_size[1] + 10 # Adjust position if near top edge

            cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            switch_number += 1

    if not boxes:
        messagebox.showinfo("Info", "No switches detected.")
    else:
        show_image(annotated_image)
        list_switches()
        update_scroll_region() # Update scroll region after listing switches

def list_switches():
    global boxes, switch_buttons, manual_route, confidence_scores
    switch_buttons = {}
    manual_route.clear()

    for widget in switch_frame.winfo_children():
        widget.destroy()

    total_label = tk.Label(switch_frame, text=f"Total Switches Detected: {len(boxes)}", font=section_title_font)
    total_label.grid(row=0, column=0, columnspan=4, pady=(5, 10))

    for i, box in enumerate(boxes):
        switch_label_text = f"SWITCH-{i + 1}"
        #switch_label_text = f"SWITCH-{i + 1} (Conf: {confidence_scores[i]:.2f})"
        switch_label = tk.Label(switch_frame, text=switch_label_text, font=label_font)
        switch_label.grid(row=i + 1, column=0, padx=5, pady=3, sticky="w") # Left align

        switch_left_button = tk.Button(switch_frame, text="Switch-Left", font=button_font, relief="raised", borderwidth=2,
                                        command=lambda idx=i: set_route(idx, "Switch-Left"))
        switch_left_button.grid(row=i + 1, column=1, padx=5, pady=3)

        switch_right_button = tk.Button(switch_frame, text="Switch-Right", font=button_font, relief="raised", borderwidth=2,
                                         command=lambda idx=i: set_route(idx, "Switch-Right"))
        switch_right_button.grid(row=i + 1, column=2, padx=5, pady=3)

        status_label = tk.Label(switch_frame, text="", font=label_font)
        status_label.grid(row=i + 1, column=3, padx=10, pady=3, sticky="w") # Left align

        manual_route[i] = {
            "route": None,
            "left_btn": switch_left_button,
            "right_btn": switch_right_button
        }

        switch_buttons[i] = {
            "left": switch_left_button,
            "right": switch_right_button,
            "status": status_label
        }

def set_route(switch_idx, route):
    buttons = manual_route[switch_idx]
    buttons["route"] = route

    if route == "Switch-Left":
        buttons["left_btn"].config(state=tk.DISABLED, bg="lightgreen")
        buttons["right_btn"].config(state=tk.NORMAL, bg="SystemButtonFace")
    else:
        buttons["right_btn"].config(state=tk.DISABLED, bg="lightgreen")
        buttons["left_btn"].config(state=tk.NORMAL, bg="SystemButtonFace")

    messagebox.showinfo("Route Set", f"Manual route for SWITCH-{switch_idx + 1} set to {route}")
    update_scroll_region() # Update scroll region after setting route

def classify_switches():
    global last_prediction, cropped_switches_data
    last_prediction = {}
    cropped_switches_data = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = original_image[y1:y2, x1:x2]
        resized = cv2.resize(crop, (224, 224)) / 255.0
        input_tensor = np.expand_dims(resized, axis=0)
        prediction = model.predict(input_tensor)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = "Switch-Left" if predicted_class_index == 0 else "Switch-Right"
        confidence = prediction[predicted_class_index] # Get the confidence score for the predicted class
        last_prediction[i] = predicted_class
        cropped_switches_data.append((crop, predicted_class, confidence, i + 1)) # Store index + 1 as switch number

    messagebox.showinfo("Classification Complete", "All switches classified.")
    show_cropped_switches()
    update_scroll_region() # Update scroll region after classification

def show_cropped_switches():
    if not cropped_switches_data:
        messagebox.showinfo("Info", "No switches have been classified yet.")
        return

    cropped_switches_window = tk.Toplevel(root)
    cropped_switches_window.title("Cropped Switches Classification")

    for i, (cropped_img_cv2, predicted_class, confidence, switch_number) in enumerate(cropped_switches_data):
        switch_group = LabelFrame(cropped_switches_window, text=f"SWITCH-{switch_number}", font=section_title_font, padx=10, pady=10)
        switch_group.pack(pady=10, padx=10, fill="x")

        cropped_img_rgb = cv2.cvtColor(cropped_img_cv2, cv2.COLOR_BGR2RGB)
        cropped_img_pil = Image.fromarray(cropped_img_rgb)
        cropped_img_pil = cropped_img_pil.resize((100, 100)) # Resize for better display
        cropped_img_tk = ImageTk.PhotoImage(cropped_img_pil)

        img_label = tk.Label(switch_group, image=cropped_img_tk)
        img_label.image = cropped_img_tk # Keep a reference
        img_label.pack(side="left", padx=10)

        info_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}"
        info_label = tk.Label(switch_group, text=info_text, font=label_font, justify="left")
        info_label.pack(side="left", padx=10)

def flash_stop_text(label):
    global flash_state
    if flash_state:
        label.config(fg="red")
    else:
        label.config(fg="black") # Or any other non-red color
    flash_state = not flash_state
    label.after(500, flash_stop_text, label) # Flash every 500 milliseconds

def check_route_match():
    global flash_state
    if not last_prediction:
        messagebox.showwarning("Warning", "No classification results available. Please classify first.")
        return

    report = ""
    all_match = True

    print(f"Number of detected boxes: {len(boxes)}")
    print(f"Manual routes: {manual_route}")
    print(f"Last predictions: {last_prediction}")

    for idx in range(len(boxes)):
        manual = manual_route.get(idx, {}).get("route", None)
        predicted = last_prediction.get(idx, None)
        status_label = switch_buttons[idx]["status"]

        if manual is None:
            result = "No manual route set."
            emoji = "❌"
            all_match = False
        elif predicted is None:
            result = "No prediction available."
            emoji = "❌"
            all_match = False
        elif manual == predicted:
            result = f"Match ✅ ({manual})"
            emoji = "✅"
        else:
            result = f"Mismatch ❌ (Manual: {manual}, Predicted: {predicted})"
            emoji = "❌"
            all_match = False

        status_label.config(text=emoji)
        report += f"SWITCH-{idx + 1}: {result}\n"

    print(f"Generated report:\n{report}")

    if all_match:
        messagebox.showinfo("All Routes Match", report)
    else:
        pygame.mixer.music.load(alert_sound_path)
        pygame.mixer.music.play()

        # Display error popup
        error_popup = tk.Toplevel(root)
        error_popup.title("ERROR: Route Mismatch!")
        try:
            error_popup.iconbitmap("error.ico")
        except tk.TclError:
            pass # Handle case where icon file is not found
        error_popup.geometry("400x250")

        stop_label = tk.Label(error_popup, text="STOP THE TRAIN", font=("Arial", 24, "bold"), fg="red")
        stop_label.pack(pady=20)
        flash_stop_text(stop_label)

        report_label = tk.Label(error_popup, text=report, font=label_font, justify="left")
        report_label.pack(padx=10, pady=10)

        # Optionally add a button to close the error message
        close_button = tk.Button(error_popup, text="Acknowledge", font=button_font, command=error_popup.destroy)
        close_button.pack(pady=10)

    update_scroll_region() # Update scroll region after checking match

def update_scroll_region():
    """Update the scroll region of the canvas."""
    main_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

# === Buttons Section ===
button_frame = tk.Frame(main_frame, padx=10, pady=10)
button_frame.pack(pady=10, fill="x")

upload_btn = tk.Button(button_frame, text="UPLOAD FIELD IMAGE", font=button_font, relief="raised", borderwidth=2, command=open_image)
upload_btn.pack(side="left", padx=5)

classify_btn = tk.Button(button_frame, text="LOCK THE ROUTE", font=button_font, relief="raised", borderwidth=2, command=classify_switches)
classify_btn.pack(side="left", padx=5)

check_btn = tk.Button(button_frame, text="VALIDATE", font=button_font, relief="raised", borderwidth=2, command=check_route_match)
check_btn.pack(side="left", padx=5)

# === Update scroll region initially and when window is resized ===
main_frame.bind("<Configure>", lambda event: update_scroll_region())
update_scroll_region()

# === Start GUI ===
root.mainloop()