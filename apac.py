import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import os
import glob
import time
import pyperclip
import zipfile
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ===========================
# Install gdown if not already installed
# ===========================
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

# ===========================
# Helper Functions
# ===========================
def download_file_from_drive(drive_url, output_path):
    """Download a file from Google Drive given a sharable URL."""
    file_id = None
    if "id=" in drive_url:
        file_id = drive_url.split("id=")[-1]
    elif "file/d/" in drive_url:
        file_id = drive_url.split("file/d/")[1].split("/")[0]

    if file_id:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    else:
        st.error("Invalid Google Drive URL!")

def download_and_unzip(drive_url, extract_to):
    """Download ZIP file from Drive and unzip it."""
    zip_path = "temp.zip"
    download_file_from_drive(drive_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

# ===========================
# Load Custom CSS for Styling
# ===========================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styling.")

local_css("style.css")

st.markdown("""
    <style>
        textarea {
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# Initialize MediaPipe
# ===========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===========================
# Load Trained Models
# ===========================
@st.cache_resource
# Load the Keras model
def load_sign_model(model_path):
    model = tf.keras.models.load_model(model_path)  # Load the .keras file model
    return model

# Load the Scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)  # Load the scaler from .pkl file
    return scaler

# File paths to your model and scaler
live_model_path = 'final_model.keras'  # Path to your model file
l_model_path = "L_model.h5"
n_model_path = "N_model.h5"
w_model_path = "W_model.h5"
scaler_path = 'scaler.pkl'  # Path to your scaler file

# Load model and scaler
live_model = load_sign_model(live_model_path)
letter_model = load_sign_model(l_model_path)
number_model = load_sign_model(n_model_path)
word_model = load_sign_model(w_model_path)
scaler = load_scaler(scaler_path)

with open('sign_language_features.pkl', 'rb') as f:
    data = pickle.load(f)
    labels = data['labels']

# ===========================
# Labels
# ===========================
letter_labels = {i: chr(65+i) for i in range(26)}  # A-Z
number_labels = {i: str(i) for i in range(10)}     # 0-9
word_labels = {
    0: "afraid", 1: "agree", 2: "assistance", 3: "bad", 4: "become", 5: "college",
    6: "doctor", 7: "from", 8: "pain", 9: "pray", 10: "secondary", 11: "skin",
    12: "small", 13: "specific", 14: "stand", 15: "today", 16: "warn", 17: "which",
    18: "work", 19: "you", 20: "are", 21: "is", 22: "do"
}

class_labels = {
    0: 'are', 1: 'did', 2: 'doing', 3: 'eat', 
    4: 'going', 5: 'How', 6: 'is', 7: 'name',
    8: 'What', 9: 'Where', 10: 'Which', 11: 'you', 12: 'your'
}

# ===========================
# Preprocessing Functions
# ===========================
def extract_keypoints_from_image(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints).reshape(1, 1, 63).astype(np.float32)
    return None

def extract_keypoints_from_image_words(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        if len(keypoints) == 63:
            keypoints += [0.0] * 63
        if len(keypoints) == 126:
            return np.array(keypoints).reshape(1, 126).astype(np.float32)  # <== Fix here


    return None

def extract_keypoints_from_frame(frame):
    # Use MediaPipe to process the frame and extract hand landmarks
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Iterate through the detected hands and extract their landmarks
        for hand_landmarks in sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x):
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        # If only one hand is detected, pad the keypoints for the second hand
        if len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * (63))  # Padding for the second hand
    else:
        # If no hands are detected, return None
        return None

    # Ensure the keypoints list has 126 values (63 per hand)
    if len(keypoints) != 126:
        return None  # In case of unexpected results

    # Reshape and return keypoints as a numpy array
    return np.array(keypoints).reshape(1, 126).astype(np.float32)

# ===========================
# Streamlit UI Setup
# ===========================
st.sidebar.image("apac_logo.jpg", width=120)
st.sidebar.title("APAC")

st.title("Introducing APAC – AI-Powered Accessibility Chatbot")
st.write(
    "APAC is a chatbot that responds to sign language, using AI to convert sign language gestures "
    "to text and vice versa. Whether you use sign language, text, or commands, APAC ensures seamless interaction."
)

# ===========================
# Sidebar Input Selection
# ===========================
input_choice = st.selectbox("Select Input Type", ("Upload Image", "Live Webcam", "Text to Sign"))
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

# ===========================
# Upload Image Handling
# ===========================
if input_choice == "Upload Image":
    choice = st.selectbox("Select Input Type", ("Letters/Numbers", "Words"))
    if "sentence" not in st.session_state:
        st.session_state.sentence = ""

    if choice == "Letters/Numbers":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            keypoints = extract_keypoints_from_image(image)

            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)

            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            img_array = img_array / 255.0
            img_input = np.expand_dims(img_array, axis=0)

            if keypoints is not None:
                keypoints_input = np.array(keypoints).reshape(1, -1)

                pred_letter = letter_model.predict(img_input)
                pred_number = number_model.predict(img_input)

                conf_letter = np.max(pred_letter)
                conf_number = np.max(pred_number)

                label_letter = letter_labels.get(np.argmax(pred_letter), "Unknown")
                label_number = number_labels.get(np.argmax(pred_number), "Unknown")

                confidences = [conf_letter, conf_number]
                labels = [label_letter, label_number]
                best_index = np.argmax(confidences)
                best_label = labels[best_index]
                best_confidence = confidences[best_index]

                predicted_label = best_label if best_confidence >= 0.7 else "Uncertain"
                st.text_area("Translated Output", value=predicted_label, height=150, key="output_box", disabled=True)

                if st.button("Copy Text", key="copy_text_upload"):
                    pyperclip.copy(predicted_label)
                    st.success("Text copied to clipboard!")
            else:
                st.warning("No hand detected or unable to extract keypoints.")

    if choice == "Words":
        uploaded_file = st.file_uploader("Upload a sign language image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            keypoints = extract_keypoints_from_image_words(image)
            if keypoints is not None:
                keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))  # Important
                prediction = live_model.predict(keypoints_scaled, verbose=0)
                predicted_class = int(np.argmax(prediction, axis=1)[0])
                predicted_label = class_labels.get(predicted_class, "Unknown")
                st.text_area("Translated Output", value=predicted_label, height=100, key="output_box", disabled=True)
                if st.button("Copy Text", key="copy_text_upload"):
                    pyperclip.copy(predicted_label)
                    st.success("Text copied to clipboard!")
            else:
                st.warning("No hand detected or unable to extract keypoints.")

# ===========================
# Live Webcam Handling (Updated with Chatbot)
# ===========================
if input_choice == "Live Webcam":
    if st.button("Start Webcam for Live Translation", key="start_webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        sentence = ""

        if "stop_webcam" not in st.session_state:
            st.session_state.stop_webcam = False

        if st.session_state.stop_webcam:
            cap.release()
            st.info("Webcam stopped.")


        if st.button("Reset", key="reset"):
            sentence = ""
            st.session_state.sentence = ""

        st.info("Webcam started! Show your signs.")

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        last_prediction_time = time.time()
        previous_word = None
        sentence_output = st.empty()
        chatbot_response = st.empty()

        def extract_keypoints_2hands(multi_hand_landmarks):
            keypoints = []
            for hand_landmarks in multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if len(multi_hand_landmarks) == 1:
                keypoints.extend([0.0] * 63)
            return np.array(keypoints).reshape(1, 126).astype(np.float32)  # <== Fix here


        def get_chatbot_response(sentence):
            query = sentence.strip().lower()
            responses = {
                "how are you": "I'm doing great! 😊",
                "what are you doing": "I'm helping you translate signs!",
                "did you eat": "Yes, I had some data bytes! 😄",
                "where are you going": "I'm staying right here to assist you!",
                "what is your name": "I'm APAC, your sign language assistant!"
            }
            return responses.get(query, None)

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or st.session_state.stop_webcam:
                    st.info("Webcam Stopped.")
                    break


                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                current_time = time.time()

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if current_time - last_prediction_time >= 4:
                        keypoints_array = extract_keypoints_2hands(results.multi_hand_landmarks)
                        if keypoints_array is not None and keypoints_array.shape == (1, 126):
                            keypoints_flatten = keypoints_array.reshape(1, -1)
                            keypoints_scaled = scaler.transform(keypoints_flatten)
                            prediction = live_model.predict(keypoints_scaled, verbose=0)

                            predicted_class = np.argmax(prediction)  # <== Add this
                            predicted_label = labels[predicted_class]  # <== Add this

                            if predicted_label != previous_word:
                                if previous_word in ["How", "how", "What", "what", "where", "Where"] and predicted_label == "your":
                                    sentence += " are"
                                    previous_word = "are"  # Update previous_word also
                                else:
                                    sentence += " " + predicted_label
                                    previous_word = predicted_label

                                last_prediction_time = current_time


                                # Check for chatbot response
                                cleaned_sentence = sentence.lower().strip()
                                if any(q in cleaned_sentence for q in [
                                    "how are you", "what are you doing", "did you eat", "where are you going", "what is your name"
                                ]):
                                    response = get_chatbot_response(cleaned_sentence)
                                    if response:
                                        chatbot_response.success(f"🤖 APAC: {response}")

                stframe.image(frame, channels="BGR")
                sentence_output.text_area(f"Translated Output {st.session_state.counter}",
                                          value=sentence.strip(), height=150,
                                          key=f"sentence_output_{st.session_state.counter}", disabled=True)

                st.session_state.counter += 1
                st.session_state.sentence = sentence.strip()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    st.text_area("Final Translated Output", value=st.session_state.sentence,
                 height=150, key="final_sentence_output", disabled=True)

    if st.button("Copy Text ", key="copy_text_live"):
        if st.session_state.sentence.strip():
            pyperclip.copy(st.session_state.sentence.strip())
            st.success("Text copied to clipboard!")
        else:
            st.warning("No text to copy!")

# ===========================
# Text-to-Sign Handling
# ===========================
elif input_choice == "Text to Sign":
    text_input = st.text_input("Enter text to translate to sign language:")
    image_directory = "Data"

    if text_input:
        words = text_input.split()
        st.write("Corresponding Sign Language Images:")

        for word in words:
            folder_path = os.path.join(image_directory, word.lower())
            if os.path.exists(folder_path):
                image_files = glob.glob(os.path.join(folder_path, "*.*"))
                if image_files:
                    st.image(image_files[0], caption=word.capitalize(), use_column_width=True)
                else:
                    st.warning(f"No images found inside '{word}' folder.")
            else:
                st.warning(f"No folder found for '{word}'.")




