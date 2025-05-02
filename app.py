import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
import time
import pyperclip
import pickle
import tensorflow as tf
from werkzeug.utils import secure_filename
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg','PNG','JPG','JPEG'}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load models and scaler
def load_models():
    models = {
        'live_model': load_model('models/final_model.keras'),
        'letter_model': load_model('models/L_model.h5'),
        'number_model': load_model('models/N_model.h5'),
        'word_model': load_model('models/W_model.h5')
    }
    
    with open('models/sign_language_features.pkl', 'rb') as f:
        data = pickle.load(f)
        labels = data['labels']
    
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return models, labels, scaler

models, labels, scaler = load_models()

# Labels
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

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
            return np.array(keypoints).reshape(1, 126).astype(np.float32)

    return None

def extract_keypoints_from_frame(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x):
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        if len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * (63))
    else:
        return None

    if len(keypoints) != 126:
        return None

    return np.array(keypoints).reshape(1, 126).astype(np.float32)

def extract_keypoints_2hands(multi_hand_landmarks):
    keypoints = []
    for hand_landmarks in multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    if len(multi_hand_landmarks) == 1:
        keypoints.extend([0.0] * 63)
    return np.array(keypoints).reshape(1, 126).astype(np.float32)

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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Debug: Print received request data
    app.logger.info("Received upload request")
    app.logger.info(f"Request files: {request.files}")
    app.logger.info(f"Request form: {request.form}")

    # Check if file was uploaded
    if 'file' not in request.files:
        app.logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error("Empty filename received")
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, PNG allowed'}), 400
    
    try:
        # Get and normalize choice parameter
        choice = request.form.get('choice', '').strip().lower()
        app.logger.info(f"Processing choice: {choice}")
        
        # Validate choice parameter
        valid_choices = {
            'letters': 'letters',
            'letters/numbers': 'letters',
            'lettersnumbers': 'letters',
            'numbers': 'letters',
            'words': 'words'
        }
        
        if choice not in valid_choices:
            app.logger.error(f"Invalid choice parameter: {choice}")
            return jsonify({
                'error': 'Invalid choice parameter. Must be "Letters/Numbers" or "Words"',
                'received': choice
            }), 400
        
        processing_type = valid_choices[choice]
        
        # Save the uploaded file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f"File saved to: {filepath}")
        
        # Process the image
        with Image.open(filepath) as img:
            # Verify and convert image
            img.verify()
            img = Image.open(filepath)  # Reopen after verify
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if processing_type == 'letters':
                app.logger.info("Processing as Letters/Numbers")
                keypoints = extract_keypoints_from_image(img)
                
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                
                # Handle different image formats
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                img_array = img_array / 255.0
                img_input = np.expand_dims(img_array, axis=0)
                
                if keypoints is None:
                    app.logger.warning("No hand keypoints detected")
                    return jsonify({'error': 'No hand detected'}), 400
                
                # Make predictions
                pred_letter = models['letter_model'].predict(img_input)
                pred_number = models['number_model'].predict(img_input)
                
                # Process results
                conf_letter = float(np.max(pred_letter))
                conf_number = float(np.max(pred_number))
                
                label_letter = letter_labels.get(np.argmax(pred_letter), "Unknown")
                label_number = number_labels.get(np.argmax(pred_number), "Unknown")
                
                # Determine best prediction
                if conf_letter >= conf_number:
                    predicted_label = label_letter
                    confidence = conf_letter
                else:
                    predicted_label = label_number
                    confidence = conf_number
                
                if confidence < 0.7:
                    predicted_label = "Uncertain"
                
                app.logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.2f})")
                
                return jsonify({
                    'success': True,
                    'image_url': f'/static/images/{filename}',
                    'prediction': predicted_label,
                    'confidence': confidence,
                    'type': 'letter' if predicted_label in letter_labels.values() else 'number'
                })
            
            elif processing_type == 'words':
                app.logger.info("Processing as Words")
                keypoints = extract_keypoints_from_image_words(img)
                
                if keypoints is None:
                    app.logger.warning("No hand keypoints detected for words")
                    return jsonify({'error': 'No hand detected'}), 400
                
                # Scale keypoints and predict
                keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))
                prediction = models['live_model'].predict(keypoints_scaled, verbose=0)
                predicted_class = int(np.argmax(prediction, axis=1)[0])
                predicted_label = class_labels.get(predicted_class, "Unknown")
                confidence = float(np.max(prediction))
                
                app.logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.2f})")
                
                return jsonify({
                    'success': True,
                    'image_url': f'/static/images/{filename}',
                    'prediction': predicted_label,
                    'confidence': confidence,
                    'type': 'word'
                })
    
    except Exception as e:
        app.logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        # Clean up if file was saved but processing failed
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500
    
    
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            keypoints_array = extract_keypoints_2hands(results.multi_hand_landmarks)
            if keypoints_array is not None and keypoints_array.shape == (1, 126):
                keypoints_flatten = keypoints_array.reshape(1, -1)
                keypoints_scaled = scaler.transform(keypoints_flatten)
                prediction = models['live_model'].predict(keypoints_scaled, verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_label = labels[predicted_class]
                
                return jsonify({
                    'success': True,
                    'prediction': predicted_label,
                    'has_hands': True
                })
        
        return jsonify({
            'success': True,
            'prediction': '',
            'has_hands': False
        })

@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    text_input = request.form.get('text', '')
    words = text_input.split()
    image_paths = []
    
    for word in words:
        folder_path = os.path.join('static/Data', word.lower())
        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_paths.append({
                    'word': word.capitalize(),
                    'path': f'/static/Data/{word.lower()}/{image_files[0]}'
                })
    
    return jsonify({
        'success': True,
        'images': image_paths
    })

if __name__ == '__main__':
    # Ensure we're running on the correct host and port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
