🎵 Emotion-Based Music Recommendation System

An end-to-end AI-powered web application that detects human emotions from text and facial images and recommends music accordingly. The system leverages deep learning techniques from Natural Language Processing and Computer Vision to understand a user’s emotional state and suggest songs that either match their mood or help uplift it. Built with a FastAPI backend and a Streamlit frontend, this project demonstrates a scalable, production-ready architecture for personalized emotion-aware music recommendation.

🌟 Features
1) Multi-Modal Emotion Detection:

         📝 Text-based emotion analysis using NLP (BERT)
         
         🖼️ Image-based facial emotion recognition using CNN
         
         📸 Real-time webcam emotion detection


2) Dual AI Models:

         Basic CNN (custom-built architecture)
         
         Transfer Learning (MobileNetV2 + Transformer)
         
         Support for multiple models - users can switch between models
         
         Real-time model selection in UI


3) Smart Music Recommendations:

         Mood-matching mode (songs that reflect your emotion)
         
         Mood-uplifting mode (songs that improve your mood)
         
         Integration with Spotify tracks


4) Production-Ready Architecture:

         FastAPI backend with RESTful endpoints
         
         Streamlit frontend with modern UI
         
         Modular, scalable codebase

************************************************************************************************************************************************
🚀 Quick Start

Prerequisites

         Python 3.8 - 3.10
         
         pip package manager
         
         4GB+ RAM (for ML models)


Installation

      1) Clone or download the project
      
         cd Emotion-Music-Recommender

      2) Install dependencies:
      
         pip install -r requirements.txt
   
      3) Prepare the dataset:
      
         Ensure data/Music Info.csv exists
         
         Ensure at least one model exists in models/ folder:
         
         models/mobilenetv2.keras (recommended)
         
         models/mobilenetmodel.keras (alternative)


Running the Application

      Step 1: Start the Backend Server
      
      Open a terminal and run:
      
      cd backend
      
      python main.py
      
      Step 2: Start the Frontend
      
      Open a new terminal (keep the backend running) and run:
     
      cd frontend
      
      streamlit run app.py
      

The web interface will automatically open in your browser at http://localhost:8501

************************************************************************************************************************************************
🧠 Machine Learning Models

1) Text Emotion Detection

   Model: BERT-Emotions-Classifier (HuggingFace)

2) Image Emotion Detection

   1. Basic CNN
      
   2. MobileNetV2
      
   3. MobileNet (Alternative)

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
