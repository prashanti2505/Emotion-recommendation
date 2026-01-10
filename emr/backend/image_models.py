import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

# Emotion labels for 5-class model
EMOTION_LABELS = ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral']

# ======================================================
# 1️⃣ Basic CNN Model (Built from scratch)
# ======================================================
def build_basic_cnn(input_shape=(48, 48, 1), num_classes=5):
    """
    Builds a basic CNN for facial emotion recognition.
    Architecture: 4 Conv blocks + Dense layers
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ======================================================
# 2️⃣ Load Transfer Learning Model (Pre-trained)
# ======================================================
def load_transfer_learning_model(model_path):
    """
    Loads the pre-trained transfer learning model (MobileNetV2/VGG16+Transformer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    
    model = load_model(model_path)
    return model

# ======================================================
# 3️⃣ Preprocess Image for Basic CNN
# ======================================================
def preprocess_image_basic_cnn(image_bytes):
    """
    Preprocess image for Basic CNN model (48x48 grayscale)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect face using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # No face detected, use entire image
        face_img = gray
    else:
        # Use first detected face
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
    
    # Resize to 48x48
    face_img = cv2.resize(face_img, (48, 48))
    
    # Normalize pixel values
    face_img = face_img / 255.0
    
    # Reshape for model input (1, 48, 48, 1)
    face_img = face_img.reshape(1, 48, 48, 1)
    
    return face_img

# ======================================================
# 4️⃣ Preprocess Image for Transfer Learning Model
# ======================================================
def preprocess_image_transfer_learning(image_bytes):
    """
    Preprocess image for Transfer Learning model (128x128 RGB)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face using Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # No face detected, use entire image
        face_img = img_rgb
    else:
        # Use first detected face
        (x, y, w, h) = faces[0]
        face_img = img_rgb[y:y+h, x:x+w]
    
    # Resize to 128x128
    face_img = cv2.resize(face_img, (128, 128))
    
    # Normalize using ImageNet preprocessing (for transfer learning models)
    # MobileNetV2/VGG16 preprocessing: scale to [-1, 1]
    face_img = face_img.astype(np.float32)
    face_img = (face_img / 127.5) - 1.0
    
    # Reshape for model input (1, 128, 128, 3)
    face_img = face_img.reshape(1, 128, 128, 3)
    
    return face_img

# ======================================================
# 5️⃣ Predict Emotion
# ======================================================
def predict_emotion(model, preprocessed_image, model_type='basic_cnn'):
    """
    Predict emotion using the loaded model
    Returns: (emotion_label, confidence_score)
    """
    # Get prediction probabilities
    predictions = model.predict(preprocessed_image, verbose=0)
    
    # Get the class with highest probability
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Map to emotion label
    emotion = EMOTION_LABELS[predicted_class_idx]
    
    return emotion, confidence

# ======================================================
# 6️⃣ Initialize Models (Called once at startup)
# ======================================================
class ImageEmotionDetector:
    """
    Wrapper class to manage multiple CNN models
    """
    def __init__(self, models_dir='../models'):
        """
        Load all available models from the models directory
        
        Args:
            models_dir: Directory containing model files
        """
        self.models = {}
        self.models_dir = os.path.join(os.path.dirname(__file__), models_dir)
        
        # Build Basic CNN (always available)
        print("📦 Loading Basic CNN...")
        self.basic_cnn_model = build_basic_cnn()
        self.models['basic_cnn'] = {
            'model': self.basic_cnn_model,
            'name': 'Basic CNN',
            'description': 'Fast, lightweight model'
        }
        print("   ✅ Basic CNN loaded")
        
        # Load all transfer learning models from the models directory
        if os.path.exists(self.models_dir):
            model_files = {
                'mobilenetv2': 'mobilenetv2.keras',
                'mobilenet': 'mobilenetmodel.keras',
                'model': 'model.keras'  # Generic fallback
            }
            
            for model_key, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        print(f"📦 Loading {filename}...")
                        model = load_transfer_learning_model(filepath)
                        
                        # Assign friendly names
                        if model_key == 'mobilenetv2':
                            name = 'MobileNetV2'
                            desc = 'Balanced speed and accuracy'
                        elif model_key == 'mobilenet':
                            name = 'MobileNet'
                            desc = 'Fast and efficient'
                        else:
                            name = 'Transfer Learning Model'
                            desc = 'Pre-trained model'
                        
                        self.models[model_key] = {
                            'model': model,
                            'name': name,
                            'description': desc
                        }
                        print(f"   ✅ {name} loaded successfully")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load {filename}: {e}")
        
        if len(self.models) == 1:
            print("⚠️  Warning: Only Basic CNN is available. Add .keras files to models/ folder.")
        else:
            print(f"✅ {len(self.models)} models loaded successfully!")
    
    def get_available_models(self):
        """
        Get list of available models
        
        Returns:
            dict: {model_key: {name, description}}
        """
        return {
            key: {'name': info['name'], 'description': info['description']}
            for key, info in self.models.items()
        }
    
    def predict(self, image_bytes, model_type='mobilenetv2'):
        """
        Predict emotion from image bytes
        
        Args:
            image_bytes: Raw image bytes
            model_type: Model key (e.g., 'basic_cnn', 'mobilenetv2', 'vgg16_transformer')
        
        Returns:
            emotion (str), confidence (float)
        """
        # Default to basic_cnn if model not found
        if model_type not in self.models:
            print(f"⚠️  Model '{model_type}' not found, using Basic CNN")
            model_type = 'basic_cnn'
        
        model_info = self.models[model_type]
        model = model_info['model']
        
        # Use appropriate preprocessing
        if model_type == 'basic_cnn':
            preprocessed = preprocess_image_basic_cnn(image_bytes)
        else:
            preprocessed = preprocess_image_transfer_learning(image_bytes)
        
        # Predict
        emotion, confidence = predict_emotion(model, preprocessed, model_type)
        
        return emotion, confidence