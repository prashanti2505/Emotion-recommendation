🏗️ System Architecture
Overview
The Emotion-Based Music Recommendation System follows a 3-tier architecture:

Presentation Layer (Frontend - Streamlit)
Application Layer (Backend - FastAPI)
Data Layer (Models & Dataset)
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Streamlit Frontend)                      │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐          │
│  │Text Input  │  │Image Upload│  │Webcam Capture│          │
│  └────────────┘  └────────────┘  └─────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP REST API
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     BACKEND SERVER                          │
│                      (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Endpoints                           │  │
│  │  • POST /predict/text                                │  │
│  │  • POST /predict/image                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Text Models    │  │ Image Models │  │Music Recomm. │  │
│  │  (NLP/BERT)     │  │  (CNN/TL)    │  │  (Algorithm) │  │
│  └─────────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│  ┌──────────────┐  ┌──────────────────────────────────┐   │
│  │ ML Models    │  │       Dataset                     │   │
│  │ model.keras  │  │  Music Info.csv (50K+ songs)      │   │
│  └──────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
Component Details
1. Frontend Layer (Streamlit)
File: frontend/app.py

Responsibilities:

User interface rendering
Input collection (text, image, webcam)
Model selection
API communication
Results display
Key Features:

Responsive design
Real-time webcam capture
Interactive song cards with Spotify links
Emotion visualization with color coding
Technology Stack:

Streamlit 1.28.1
Pillow (image processing)
Requests (HTTP client)
2. Backend Layer (FastAPI)
File: backend/main.py

Responsibilities:

RESTful API endpoints
Request validation
Model orchestration
Response formatting
Error handling
API Endpoints:

python
GET  /                    # Health check
POST /predict/text        # Text emotion prediction
POST /predict/image       # Image emotion prediction
GET  /emotions           # List available emotions
Request Flow:

Client Request
    ↓
FastAPI Router
    ↓
Input Validation
    ↓
Model Prediction
    ↓
Music Recommendation
    ↓
JSON Response
Technology Stack:

FastAPI 0.104.1
Uvicorn (ASGI server)
Pydantic (data validation)
3. ML Models Layer
3.1 Text Emotion Detection
File: backend/text_models.py

Model: BERT-Emotions-Classifier (HuggingFace)

Architecture:

Input Text
    ↓
BERT Tokenizer
    ↓
BERT Model (12 layers)
    ↓
Classification Head
    ↓
11 Emotion Classes
    ↓
Emotion Mapping
    ↓
6 Final Classes
Emotion Mapping:

python
{
    'joy', 'love', 'excitement', 'amusement' → 'Happy'
    'interest', 'satisfaction', 'calmness'   → 'Neutral'
    'sadness'                                → 'Sad'
    'anger', 'disgust'                       → 'Angry'
    'fear'                                   → 'Fearful'
    'surprise'                               → 'Surprised'
}
3.2 Image Emotion Detection
File: backend/image_models.py

Two Models Available:

A. Basic CNN
Architecture:

Input (48×48×1 grayscale)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout
    ↓
Conv2D(256) → BatchNorm → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense(512) → Dropout → Dense(256) → Dropout
    ↓
Dense(5, softmax)
Parameters: ~2M Input: 48×48 grayscale Output: 5 emotion classes

B. Transfer Learning Model
Architecture (MobileNetV2):

Input (128×128×3 RGB)
    ↓
MobileNetV2 (pre-trained ImageNet)
    ↓
Conv2D(64) → BatchNorm
    ↓
Conv2D(128) → BatchNorm → MaxPool
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, relu)
    ↓
Dense(5, softmax)
Parameters: ~3-15M (depending on base model) Input: 128×128 RGB Output: 5 emotion classes

Preprocessing Pipeline:

Raw Image Bytes
    ↓
Decode to NumPy Array
    ↓
Face Detection (Haar Cascade)
    ↓
Extract Face ROI
    ↓
Resize (48×48 or 128×128)
    ↓
Normalize Pixels
    ↓
Model Prediction
    ↓
Softmax Probabilities
    ↓
Argmax → Emotion Label
4. Music Recommendation Engine
File: backend/music_recommender.py

Algorithm:

Detected Emotion + Mode
    ↓
Emotion-to-Audio-Features Mapping
    ↓
Filter Dataset by Valence & Energy
    ↓
Random Sampling (n songs)
    ↓
Format Output (name, artist, link)
Audio Features Used:

Valence: Musical positivity (0-1)
Energy: Intensity/activity (0-1)
Emotion Mapping Rules:

Mood-Matching Mode:

Sad       → valence < 0.4, energy < 0.5
Happy     → valence > 0.6, energy > 0.5
Angry     → valence < 0.4, energy > 0.7
Surprised → 0.4 < valence < 0.7, energy > 0.6
Fearful   → valence < 0.4, 0.6 < energy < 1.0
Neutral   → 0.4 < valence < 0.6, 0.4 < energy < 0.6
Mood-Uplifting Mode:

Sad       → valence > 0.6, 0.4 < energy < 0.7
Angry     → valence > 0.5, energy < 0.5
Fearful   → valence > 0.6, 0.3 < energy < 0.6
Happy     → valence > 0.6, energy > 0.5 (maintain)
Surprised → 0.5 < valence < 0.8, energy > 0.6
Neutral   → 0.4 < valence < 0.6, 0.4 < energy < 0.6
Data Flow
Text Input Flow
User types text
    ↓
Streamlit sends POST /predict/text
    ↓
FastAPI receives request
    ↓
TEXT_CLASSIFIER(text)
    ↓
Map 11 emotions → 6 emotions
    ↓
recommend_songs_by_emotion(emotion)
    ↓
Filter dataset by valence/energy
    ↓
Return songs with Spotify links
    ↓
Streamlit displays results
Image Input Flow
User uploads/captures image
    ↓
Streamlit sends POST /predict/image
    ↓
FastAPI receives image bytes
    ↓
Preprocess image (resize, normalize)
    ↓
Face detection (Haar Cascade)
    ↓
Extract face ROI
    ↓
Model.predict(face_image)
    ↓
Get emotion + confidence
    ↓
recommend_songs_by_emotion(emotion)
    ↓
Return songs with Spotify links
    ↓
Streamlit displays results
Security Considerations
Input Validation:
Text length limits
Image size limits
File type validation
Error Handling:
Try-catch blocks
Graceful degradation
User-friendly error messages
CORS:
Configured for localhost
Should be restricted in production
Rate Limiting:
Not implemented (add in production)
Scalability Considerations
Current Limitations
Single-threaded model inference
In-memory model loading
No caching mechanism
No load balancing
Recommended Improvements
Model Serving:
Use TensorFlow Serving
Implement model versioning
Add GPU acceleration
Caching:
Redis for frequent queries
Cache face detection results
Load Balancing:
Multiple backend instances
Nginx/HAProxy
Database:
PostgreSQL for user data
MongoDB for song metadata
Async Processing:
Celery task queue
Batch processing
Performance Metrics
Model Inference Time (CPU)
Text Model: ~200-300ms
Basic CNN: ~50-100ms
Transfer Learning: ~200-400ms
Song Recommendation: ~10-20ms
Memory Usage
Text Model: ~500MB
Basic CNN: ~200MB
Transfer Learning: ~1-2GB
Total System: ~2-3GB
Throughput
Current: ~5-10 requests/second
Recommended for production: 100+ requests/second
Technology Stack Summary
Component	Technology	Version
Backend Framework	FastAPI	0.104.1
Frontend Framework	Streamlit	1.28.1
ML Framework	TensorFlow	2.15.0
NLP Library	Transformers	4.35.0
Computer Vision	OpenCV	4.8.1
Data Processing	Pandas	2.0.3
Server	Uvicorn	0.24.0
Deployment Architecture (Recommended)
Internet
    ↓
Load Balancer (Nginx)
    ↓
┌─────────────────────────────────┐
│   Frontend Instances (3x)       │
│   Streamlit on port 8501-8503   │
└─────────────────────────────────┘
    ↓
API Gateway
    ↓
┌─────────────────────────────────┐
│   Backend Instances (5x)        │
│   FastAPI on port 8001-8005     │
└─────────────────────────────────┘
    ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Redis   │  │PostgreSQL│  │   S3     │
│  Cache   │  │   DB     │  │  Models  │
└──────────┘  └──────────┘  └──────────┘
Monitoring & Logging
Recommended Tools:

Logging: Python logging, ELK stack
Metrics: Prometheus + Grafana
Tracing: Jaeger
Alerts: PagerDuty
Key Metrics to Monitor:

Request latency
Error rates
Model inference time
Memory usage
API response codes
This architecture is designed to be modular, scalable, and maintainable for production deployment.

