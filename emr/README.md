🎵 Emotion-Based Music Recommendation System
An end-to-end AI-powered web application that detects human emotions from text and facial images and recommends music accordingly. The system leverages deep learning techniques from Natural Language Processing and Computer Vision to understand a user’s emotional state and suggest songs that either match their mood or help uplift it. Built with a FastAPI backend and a Streamlit frontend, this project demonstrates a scalable, production-ready architecture for personalized emotion-aware music recommendation.

🌟 Features
Multi-Modal Emotion Detection:
📝 Text-based emotion analysis using NLP (BERT)
🖼️ Image-based facial emotion recognition using CNN
📸 Real-time webcam emotion detection
Dual AI Models:
Basic CNN (custom-built architecture)
Transfer Learning (MobileNetV2 + Transformer)
Support for multiple models - users can switch between models
Real-time model selection in UI

Smart Music Recommendations:
Mood-matching mode (songs that reflect your emotion)
Mood-uplifting mode (songs that improve your mood)

Integration with Spotify tracks
Production-Ready Architecture:
FastAPI backend with RESTful endpoints

Streamlit frontend with modern UI
Modular, scalable codebase

🚀 Quick Start
Prerequisites
Python 3.8 - 3.10
pip package manager
4GB+ RAM (for ML models)
Installation
Clone or download the project:
bash
   cd Emotion-Music-Recommender
Install dependencies:
bash
   pip install -r requirements.txt
Prepare the dataset:
Ensure data/Music Info.csv exists
Ensure at least one model exists in models/ folder:
models/mobilenetv2.keras (recommended)
models/mobilenetmodel.keras (alternative)

Running the Application
Step 1: Start the Backend Server

Open a terminal and run:

bash
cd backend
python main.py
You should see:

✅ All ML models loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
The backend API will be available at http://localhost:8000

Step 2: Start the Frontend

Open a new terminal (keep the backend running) and run:

bash
cd frontend
streamlit run app.py
The web interface will automatically open in your browser at http://localhost:8501

🧠 Machine Learning Models
Text Emotion Detection
Model: BERT-Emotions-Classifier (HuggingFace)
Input: Raw text
Output: 11 emotions mapped to 6 classes
Emotions: Happy, Sad, Angry, Fearful, Surprised, Neutral
Image Emotion Detection
1. Basic CNN
Architecture: 4 Conv blocks + Dense layers
Input: 48×48 grayscale images
Parameters: ~2M
Speed: Fast (~50ms)
Accuracy: Good for basic use cases (~60-65%)
2. MobileNetV2 (Recommended)
Architecture: MobileNetV2 + Custom layers
Input: 128×128 RGB images
Parameters: ~3M
Speed: Fast (~200ms)
Accuracy: Good (~70%)
Best For: Production use
3. MobileNet (Alternative)
Architecture: MobileNet + Custom layers
Input: 128×128 RGB images
Parameters: ~3M
Speed: Fast (~150ms)
Accuracy: Good (~68-70%)
Best For: Mobile/low-resource environments
Users can select their preferred model through the UI!

Song Recommendation Logic
Uses Spotify audio features:

Valence: Musical positivity (0-1)
Energy: Intensity and activity (0-1)
Mood-Matching Rules:

Sad: Low valence, low energy
Happy: High valence, high energy
Angry: Low valence, high energy
Surprised: Medium valence, high energy
Fearful: Low valence, medium-high energy
Neutral: Balanced valence and energy
Mood-Uplifting Rules:

Sad → Positive, mid-energy tracks
Angry → Calm, pleasant songs
Fearful → Soothing tracks
Others → Maintain/enhance positive vibes

📊 Supported Emotions
Happy 😊: Joyful, excited, amused
Sad 😢: Depressed, melancholic
Angry 😠: Frustrated, irritated
Fearful 😨: Anxious, worried
Surprised 😲: Shocked, amazed
Neutral 😐: Calm, balanced
🎯 Model Performance
Text Model: ~85% accuracy on emotion classification
Transfer Learning: ~70-75% accuracy on FER2013 dataset
Basic CNN: ~60-65% accuracy (faster inference)