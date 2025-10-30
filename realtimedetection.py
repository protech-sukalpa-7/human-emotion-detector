import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, Dense, Flatten
import keras
from PIL import Image
import tempfile


# Page configuration
st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# Register Sequential for backward compatibility
keras.utils.get_custom_objects().update({"Sequential": Sequential})

# ---- Load model (with caching) ----
@st.cache_resource
def load_emotion_model():
    with open("facialemotionmodel.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("facialemotionmodel.h5")
    return model

@st.cache_resource
def load_face_cascade():
    # Try multiple possible paths for the Haar cascade file
    paths = [
        'haarcascade_frontalface_default.xml',  # Current directory
        cv2.__file__[:-11] + 'data/haarcascade_frontalface_default.xml',  # OpenCV installation directory
    ]
    
    for haar_file in paths:
        cascade = cv2.CascadeClassifier(haar_file)
        if not cascade.empty():
            return cascade
    
    raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml. Please make sure OpenCV is installed correctly.")

# ---- Feature Extraction ----
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ---- Process Image ----
def detect_emotion(image, model, face_cascade):
    labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
        4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        
        pred = model.predict(img, verbose=0)
        emotion = labels[pred.argmax()]
        confidence = np.max(pred) * 100
        
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'position': (x, y, w, h)
        })
        
        cv2.putText(image, f"{emotion} ({confidence:.1f}%)", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    return image, results

# ---- Streamlit UI ----
def main():
    st.title("😊 Facial Emotion Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application detects facial emotions in real-time using deep learning. "
        "Upload an image or use your webcam to detect emotions like Happy, Sad, Angry, and more!"
    )
    st.sidebar.markdown("### Supported Emotions")
    st.sidebar.markdown("- 😠 Angry\n- 🤢 Disgust\n- 😨 Fear\n- 😊 Happy\n- 😐 Neutral\n- 😢 Sad\n- 😲 Surprise")
    
    # Load models
    try:
        model = load_emotion_model()
        face_cascade = load_face_cascade()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Mode selection
    mode = st.radio("Select Mode", ["Upload Image", "Webcam"], horizontal=True)
    
    if mode == "Upload Image":
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to load image. Please try another file.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(rgb_image, use_container_width=True)
            
            with st.spinner("Detecting emotions..."):
                processed_image, results = detect_emotion(image.copy(), model, face_cascade)
            
            with col2:
                st.subheader("Detected Emotions")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display results
            if results:
                st.success(f"✅ Detected {len(results)} face(s)")
                for i, result in enumerate(results, 1):
                    st.metric(
                        f"Face {i}",
                        result['emotion'],
                        f"{result['confidence']:.2f}% confidence"
                    )
            else:
                st.warning("⚠️ No faces detected in the image")
    
    else:  # Webcam mode
        st.subheader("📹 Webcam Emotion Detection")
        st.info("Click 'Start Webcam' to begin real-time emotion detection")
        
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        
        if run:
            webcam = cv2.VideoCapture(0)
            
            if not webcam.isOpened():
                st.error("❌ Unable to access webcam. Please check your camera permissions.")
                st.stop()
            
            stop_button = st.button('Stop Webcam')
            
            while run and not stop_button:
                ret, frame = webcam.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break
                
                processed_frame, results = detect_emotion(frame, model, face_cascade)
                
                # Convert BGR to RGB for display
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(processed_frame)
                
                if stop_button:
                    break
            
            webcam.release()
            st.success("Webcam stopped")

if __name__ == "__main__":
    main()