import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import streamlit as st # Import Streamlit for caching

# --- PATH CORRECTION ---
# Get the base directory of this script (model_inference.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to your trained model relative to the BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'mask_detection_model.h5')

# Define the path to your Haar Cascade XML file
HAARCASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')


# --- Model and Cascade Loading (Cached for Streamlit) ---
@st.cache_resource # This decorator tells Streamlit to run this function only once
def load_resources():
    """
    Loads the TensorFlow model and Haar Cascade classifier.
    Uses st.cache_resource to ensure these heavy resources are loaded only once.
    """
    model = None
    cascade = None

    # Load TensorFlow Model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            st.success(f"Model loaded successfully from {MODEL_PATH}")
            print(f"Model loaded successfully from {MODEL_PATH}") # Also print to console for debugging
        else:
            st.error(f"Error: Model file not found at: {MODEL_PATH}")
            print(f"Error: Model file not found at: {MODEL_PATH}")
            # Raising an error here will stop the app if the model isn't found
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    except Exception as e:
        st.error(f"Error loading machine learning model: {e}")
        print(f"Error loading machine learning model: {e}")


    # Load Haar Cascade Classifier
    try:
        cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if cascade.empty():
            # Fallback to OpenCV's default data path if not found in project root
            st.warning(f"Warning: Haar Cascade not found at {HAARCASCADE_PATH}. Attempting OpenCV default path.")
            print(f"Warning: Haar Cascade not found at {HAARCASCADE_PATH}. Attempting OpenCV default path.")
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades/haarcascade_frontalface_default.xml') # Corrected path syntax for cv2.data.haarcascades
            if cascade.empty():
                st.error(f"Error: Haar Cascade classifier not loaded. Ensure 'haarcascade_frontalface_default.xml' is in your project root or accessible.")
                print(f"Error: Haar Cascade classifier not loaded. Ensure 'haarcascade_frontalface_default.xml' is in your project root or accessible.")
                raise IOError(f"Haar Cascade classifier not loaded from {HAARCASCADE_PATH} or OpenCV default path.")
        st.success("Haar Cascade classifier loaded successfully.")
        print("Haar Cascade classifier loaded successfully.")
    except Exception as e:
        st.error(f"Error loading face detection classifier: {e}")
        print(f"Error loading face detection classifier: {e}")

    return model, cascade

# Call the cached function to get the model and cascade globally
FACES_MODEL, FACE_CASCADE = load_resources()

# Define the class names your model predicts
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# --- NMS Helper Function ---
def non_max_suppression_fast(boxes, overlapThresh):
    """
    Applies Non-Maximum Suppression to a list of bounding boxes.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


# Function to preprocess a single image for prediction
def preprocess_image(pil_img_object):
    """
    Preprocesses a PIL Image object for the mask detection model.
    """
    img_rgb = pil_img_object.convert("RGB")
    resized_img = img_rgb.resize((35, 35))
    img_array = np.array(resized_img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# Function to detect faces and predict mask status
def detect_and_predict_faces(image_path):
    """
    Detects faces in an image, applies Non-Maximum Suppression,
    and predicts mask status for each *distinct* detected face.
    """
    # Use the globally loaded model and cascade
    if FACES_MODEL is None or FACE_CASCADE is None:
        return [], "Error: ML resources not loaded. Please check logs for details."

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return [], "Error: Could not read image from path. Check file integrity or permissions."
    
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    raw_faces = FACE_CASCADE.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(raw_faces) == 0:
        return [], "No faces detected in the image."

    boxes_for_nms = []
    for (x, y, w, h) in raw_faces:
        boxes_for_nms.append([int(x), int(y), int(x + w), int(y + h)])
    
    picked_boxes = non_max_suppression_fast(np.array(boxes_for_nms), overlapThresh=0.3)
    
    results = []
    if len(picked_boxes) == 0:
        return [], "No distinct faces detected in the image after filtering. Try a different image or adjust NMS threshold."

    for (x1, y1, x2, y2) in picked_boxes:
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        x_shift = (w) * 0.1
        y_shift = (h) * 0.1

        crop_x_min = int(max(0, x - x_shift))
        crop_y_min = int(max(0, y - y_shift))
        crop_x_max = int(min(img_cv.shape[1], x + w + x_shift))
        crop_y_max = int(min(img_cv.shape[0], y + h + y_shift))

        # Corrected order: [rows (height), cols (width)]
        cropped_face_cv = img_cv[crop_y_min:crop_y_max, crop_x_min:crop_x_max] 

        if cropped_face_cv.size == 0 or cropped_face_cv.shape[0] == 0 or cropped_face_cv.shape[1] == 0:
            print(f"Warning: Empty or invalid crop for face at original box ({x},{y},{w},{h}). Skipping.")
            continue

        cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_cv, cv2.COLOR_BGR2RGB))
        
        processed_face_array = preprocess_image(cropped_face_pil)
        
        predictions_raw = FACES_MODEL.predict(processed_face_array)
        
        predicted_class_index = np.argmax(predictions_raw[0])
        confidence = predictions_raw[0][predicted_class_index] * 100
        
        predicted_label = CLASS_NAMES[predicted_class_index]
        
        results.append({
            'box': [int(x), int(y), int(w), int(h)], 
            'prediction': predicted_label,
            'confidence': f"{confidence:.2f}%"
        })
    
    return results, None