import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont # For drawing on image
import numpy as np
import uuid # For unique temp filenames

# Import your core ML logic (model_inference.py should be in the same directory)
from model_inference import detect_and_predict_faces 

st.set_page_config(
    page_title="Face Mask Detection App",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("😷 Face Mask Detection App")
st.markdown("Upload an image below to detect faces and determine if masks are worn correctly.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    # Use a unique ID for the session's image to prevent re-running unnecessarily
    # and to ensure a unique filename for temp storage
    file_id = uploaded_file.file_id
    if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.predictions_data = None # Clear previous predictions
        st.session_state.analyzed_image_display = None # Clear previous analyzed image

    # Read image as PIL Image for display and processing
    pil_image = Image.open(uploaded_file)
    
    st.image(pil_image, caption="Uploaded Image.", use_column_width=True)
    st.markdown("---")

    if st.button("Analyze Image", key="analyze_button"):
        with st.spinner("Detecting faces and predicting mask status..."):
            # Create a temporary file path to save the uploaded image for OpenCV
            temp_dir = "temp_streamlit_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_filepath = os.path.join(temp_dir, str(uuid.uuid4()) + os.path.splitext(uploaded_file.name)[1])
            
            # Write the uploaded file to disk for OpenCV to read
            pil_image.save(temp_filepath)

            # Call your existing detection function
            predictions, error_message = detect_and_predict_faces(temp_filepath)

            # Clean up the temporary file immediately
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            st.session_state.predictions_data = predictions
            st.session_state.error_message = error_message
            st.session_state.analyzed_image_base = pil_image # Store base image to draw on it

    if st.session_state.get('predictions_data') is not None:
        predictions = st.session_state.predictions_data
        error_message = st.session_state.error_message
        analyzed_image_base = st.session_state.analyzed_image_base

        if error_message:
            st.error(f"Error: {error_message}")
        elif not predictions:
            st.info("No faces detected in the image.")
        else:
            st.success("Detection complete!")
            st.subheader("Detection Results:")
            
            # Prepare image for drawing
            img_with_boxes = analyzed_image_base.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Try to load a font for text.
            try:
                # Prioritize system fonts common on Linux/Windows/macOS or provide specific path
                font_path_linux = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                font_path_windows = "C:/Windows/Fonts/arial.ttf"
                font_path_macos = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

                if os.path.exists(font_path_linux):
                    font = ImageFont.truetype(font_path_linux, 15)
                elif os.path.exists(font_path_windows):
                    font = ImageFont.truetype(font_path_windows, 15)
                elif os.path.exists(font_path_macos):
                    font = ImageFont.truetype(font_path_macos, 15)
                else:
                    font = ImageFont.load_default() # Fallback to default PIL font
                    st.warning("Could not find common Arial/DejaVu fonts. Using default font for labels.")
            except IOError:
                font = ImageFont.load_default()
                st.warning("Could not load a custom font, using default font for labels.")


            # Colors for bounding boxes and text
            COLOR_MAP = {
                "with_mask": "green",
                "mask_weared_incorrect": "orange",
                "without_mask": "red"
            }

            for i, result in enumerate(predictions):
                x, y, w, h = result['box']
                prediction_label = result['prediction'] # Renamed for clarity
                confidence_score = result['confidence'] # Renamed for clarity
                
                # Get color based on prediction
                color = COLOR_MAP.get(prediction_label, "blue")

                # Draw bounding box
                draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=3)
                
                # --- CORRECTED POSITION AND VARIABLE NAMES ---
                # Define text_label AFTER 'prediction_label' and 'confidence_score' are assigned
                text_label = f"{prediction_label} ({confidence_score})" 
                
                # Define text position for calculation (top-left corner of text)
                text_x_pos = x
                text_y_pos = y - 5 

                # Get the bounding box of the text. Returns (left, top, right, bottom)
                bbox = draw.textbbox((text_x_pos, text_y_pos), text_label, font=font)
                
                # Calculate actual text width and height from the bounding box
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Background for text
                bg_x1 = x
                bg_y1 = y - text_height - 5
                bg_x2 = x + text_width + 5
                bg_y2 = y

                draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=color)
                draw.text((x + 2, y - text_height - 3), text_label, fill="white", font=font)


                st.write(f"**Face {i+1}:**")
                st.write(f"  - **Prediction:** `{prediction_label}`")
                st.write(f"  - **Confidence:** `{confidence_score}`")
            
            st.markdown("---")
            st.subheader("Image with Detections:")
            st.image(img_with_boxes, caption="Image with detected faces and mask status", use_column_width=True)
            st.session_state.analyzed_image_display = img_with_boxes

else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.caption("Developed with Streamlit, TensorFlow, and OpenCV.")