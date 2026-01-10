from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

# Import custom modules
from text_models import TEXT_CLASSIFIER, EMOTION_MAPPING
from image_models import ImageEmotionDetector
from music_recommender import recommend_songs_by_emotion

# ======================================================
# 1️⃣ Initialize FastAPI App
# ======================================================
app = FastAPI(
    title="Emotion-Based Music Recommendation API",
    description="Detect emotions from text/images and recommend music",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 2️⃣ Load ML Models at Startup (ONCE)
# ======================================================
# Initialize image emotion detector with all available models
try:
    image_detector = ImageEmotionDetector(models_dir='../models')
    print("✅ All ML models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    image_detector = None

# ======================================================
# 3️⃣ Pydantic Models for Request/Response
# ======================================================
class TextEmotionRequest(BaseModel):
    text: str
    uplift_mode: bool = False

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    songs: list

# ======================================================
# 4️⃣ Health Check Endpoint
# ======================================================
@app.get("/")
async def root():
    return {
        "message": "Emotion-Based Music Recommendation API",
        "status": "running",
        "endpoints": {
            "text": "/predict/text",
            "image": "/predict/image"
        }
    }

# ======================================================
# 5️⃣ Text Emotion Prediction Endpoint
# ======================================================
@app.post("/predict/text", response_model=EmotionResponse)
async def predict_text_emotion(request: TextEmotionRequest):
    """
    Predict emotion from text input using NLP model
    
    Args:
        text: Input text to analyze
        uplift_mode: Whether to use mood-uplifting recommendations
    
    Returns:
        Detected emotion, confidence, and song recommendations
    """
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Predict emotion using text classifier
        result = TEXT_CLASSIFIER(request.text)[0]
        
        # Map to 5-class emotion system
        raw_emotion = result['label'].lower()
        mapped_emotion = EMOTION_MAPPING.get(raw_emotion, 'Neutral')
        confidence = float(result['score'])
        
        # Get song recommendations
        songs = recommend_songs_by_emotion(
            emotion=mapped_emotion,
            n=7,
            uplift=request.uplift_mode
        )
        
        return EmotionResponse(
            emotion=mapped_emotion,
            confidence=confidence,
            songs=songs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ======================================================
# 6️⃣ Image Emotion Prediction Endpoint
# ======================================================
@app.post("/predict/image", response_model=EmotionResponse)
async def predict_image_emotion(
    file: UploadFile = File(...),
    model_type: str = Form("mobilenetv2"),
    uplift_mode: bool = Form(False)
):
    """
    Predict emotion from image using CNN models
    
    Args:
        file: Uploaded image file
        model_type: Model key (e.g., 'basic_cnn', 'mobilenetv2', 'vgg16_transformer')
        uplift_mode: Whether to use mood-uplifting recommendations
    
    Returns:
        Detected emotion, confidence, and song recommendations
    """
    try:
        # Check if models are loaded
        if image_detector is None:
            raise HTTPException(
                status_code=500,
                detail="Image models not loaded. Please check model files."
            )
        
        # Validate model type
        available_models = image_detector.get_available_models()
        if model_type not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Available: {list(available_models.keys())}"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Predict emotion
        emotion, confidence = image_detector.predict(image_bytes, model_type)
        
        # Get song recommendations
        songs = recommend_songs_by_emotion(
            emotion=emotion,
            n=7,
            uplift=uplift_mode
        )
        
        return EmotionResponse(
            emotion=emotion,
            confidence=confidence,
            songs=songs
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ======================================================
# 7️⃣ Get Available Emotions Endpoint
# ======================================================
@app.get("/emotions")
async def get_emotions():
    """
    Get list of available emotions
    """
    return {
        "emotions": ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Neutral"]
    }

# ======================================================
# 8️⃣ Get Available Models Endpoint
# ======================================================
@app.get("/models")
async def get_models():
    """
    Get list of available models for image prediction
    """
    if image_detector is None:
        return {"models": {}}
    
    return {
        "models": image_detector.get_available_models()
    }

# ======================================================
# 9️⃣ Run Server
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)