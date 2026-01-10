import streamlit as st
import requests
from PIL import Image
import io
import base64

# ======================================================
# Configuration
# ======================================================
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# Custom CSS Styling
# ======================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        border: none;
    }
    .song-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1DB954;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .happy { background-color: #FFD700; color: #000; }
    .sad { background-color: #4169E1; color: #fff; }
    .angry { background-color: #DC143C; color: #fff; }
    .surprised { background-color: #FF69B4; color: #fff; }
    .fearful { background-color: #800080; color: #fff; }
    .neutral { background-color: #808080; color: #fff; }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# Helper Functions
# ======================================================
def get_emotion_color(emotion):
    """Map emotion to CSS class"""
    emotion_map = {
        "Happy": "happy",
        "Sad": "sad",
        "Angry": "angry",
        "Surprised": "surprised",
        "Fearful": "fearful",
        "Neutral": "neutral"
    }
    return emotion_map.get(emotion, "neutral")

def display_songs(songs, emotion):
    """Display song recommendations"""
    if not songs:
        st.warning("No songs found. Please try again.")
        return
    
    st.success(f"🎵 Found {len(songs)} songs for your {emotion} mood!")
    
    for idx, song in enumerate(songs, 1):
        st.markdown(f"""
        <div class="song-card">
            <h4>{idx}. {song['name']}</h4>
            <p><strong>Artist:</strong> {song['artist']}</p>
            <a href="{song['link']}" target="_blank">
                🎧 Play on Spotify
            </a>
        </div>
        """, unsafe_allow_html=True)

def predict_text_emotion(text, uplift_mode):
    """Call backend API for text emotion prediction"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict/text",
            json={"text": text, "uplift_mode": uplift_mode},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def predict_image_emotion(image_bytes, model_type, uplift_mode):
    """Call backend API for image emotion prediction"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"model_type": model_type, "uplift_mode": uplift_mode}
        
        response = requests.post(
            f"{BACKEND_URL}/predict/image",
            files=files,
            data=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ======================================================
# Main App
# ======================================================
def main():
    # Header
    st.title("🎵 Emotion-Based Music Recommendation System")
    st.markdown("### Discover music that matches your mood!")
    
    # Fetch available models from backend
    try:
        models_response = requests.get(f"{BACKEND_URL}/models", timeout=5)
        available_models = models_response.json().get("models", {})
    except:
        available_models = {
            "basic_cnn": {"name": "Basic CNN", "description": "Fast, lightweight"},
            "mobilenetv2": {"name": "MobileNetV2", "description": "Balanced speed and accuracy"},
            "mobilenet": {"name": "MobileNet", "description": "Fast and efficient"}
        }
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Input method selection
        input_method = st.radio(
            "Choose Input Method:",
            ["📝 Text", "🖼️ Upload Image", "📸 Webcam"]
        )
        
        # Model selection (only for image inputs)
        if input_method in ["🖼️ Upload Image", "📸 Webcam"]:
            if available_models:
                # Create formatted options
                model_options = {
                    key: f"{info['name']} - {info['description']}"
                    for key, info in available_models.items()
                }
                
                selected_model = st.selectbox(
                    "Select Model:",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options[x]
                )
            else:
                st.warning("⚠️ No models available. Check backend.")
                selected_model = "basic_cnn"
        else:
            selected_model = None
        
        # Mood mode selection
        uplift_mode = st.checkbox(
            "🚀 Mood Uplifting Mode",
            value=False,
            help="Enable this to get songs that improve your mood"
        )
        
        st.markdown("---")
        st.markdown("#### About")
        st.info("""
        This app uses AI to detect your emotion and recommends music from Spotify.
        
        - **Text**: NLP-based emotion detection
        - **Image**: CNN-based facial emotion recognition
        """)
        
        # Display available models
        if input_method in ["🖼️ Upload Image", "📸 Webcam"] and available_models:
            st.markdown("---")
            st.markdown("#### Available Models")
            for key, info in available_models.items():
                st.markdown(f"**{info['name']}**")
                st.caption(info['description'])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📥 Input")
        
        # Text Input
        if input_method == "📝 Text":
            text_input = st.text_area(
                "How are you feeling today?",
                placeholder="Type your thoughts here...",
                height=150
            )
            
            if st.button("🎯 Analyze Emotion"):
                if text_input.strip():
                    with st.spinner("Analyzing your emotion..."):
                        result = predict_text_emotion(text_input, uplift_mode)
                        
                        if result:
                            st.session_state['result'] = result
                else:
                    st.warning("Please enter some text.")
        
        # Image Upload
        elif input_method == "🖼️ Upload Image":
            uploaded_file = st.file_uploader(
                "Upload your photo",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear photo of your face"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width='stretch')
                
                if st.button("🎯 Analyze Emotion"):
                    with st.spinner("Detecting emotion from image..."):
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        result = predict_image_emotion(img_byte_arr, selected_model, uplift_mode)
                        
                        if result:
                            st.session_state['result'] = result
        
        # Webcam Capture
        elif input_method == "📸 Webcam":
            st.info("📸 Click the button below to capture a photo from your webcam")
            
            camera_photo = st.camera_input("Take a picture")
            
            if camera_photo:
                if st.button("🎯 Analyze Emotion"):
                    with st.spinner("Detecting emotion from webcam..."):
                        # Get image bytes
                        img_bytes = camera_photo.getvalue()
                        
                        result = predict_image_emotion(img_bytes, selected_model, uplift_mode)
                        
                        if result:
                            st.session_state['result'] = result
    
    with col2:
        st.header("📊 Results")
        
        # Display results if available
        if 'result' in st.session_state:
            result = st.session_state['result']
            
            # Display emotion
            emotion = result['emotion']
            confidence = result['confidence']
            emotion_class = get_emotion_color(emotion)
            
            st.markdown(f"""
            <div class="emotion-badge {emotion_class}">
                {emotion} ({confidence:.1%} confidence)
            </div>
            """, unsafe_allow_html=True)
            
            # Display mode
            if uplift_mode:
                st.info("🚀 Showing mood-uplifting songs")
            else:
                st.info("🎵 Showing mood-matching songs")
            
            # Display songs
            display_songs(result['songs'], emotion)
            
            # Clear button
            if st.button("🔄 Clear Results"):
                del st.session_state['result']
                st.rerun()
        else:
            st.info("👈 Select an input method and analyze your emotion to get music recommendations!")

# ======================================================
# Run App
# ======================================================
if __name__ == "__main__":
    main()