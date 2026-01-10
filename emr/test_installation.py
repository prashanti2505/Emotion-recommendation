"""
Test script to verify the installation and setup of the Emotion-Based Music Recommender
Run this before starting the application to ensure everything is configured correctly.
"""

import os
import sys

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8 and version.minor <= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
        print("   Required: Python 3.8 - 3.10")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = {
        'tensorflow': '2.15.0',
        'transformers': '4.35.0',
        'fastapi': '0.104.1',
        'streamlit': '1.28.1',
        'cv2': 'opencv-python',
        'pandas': '2.0.3',
        'numpy': '1.24.3',
        'requests': '2.31.0'
    }
    
    all_installed = True
    
    for package, version in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ opencv-python - Installed")
            else:
                module = __import__(package)
                print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def check_folder_structure():
    """Check if folder structure is correct"""
    print("\n🔍 Checking folder structure...")
    
    required_folders = [
        'backend',
        'frontend',
        'models',
        'data'
    ]
    
    required_files = [
        'backend/main.py',
        'backend/text_models.py',
        'backend/music_recommender.py',
        'backend/image_models.py',
        'frontend/app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    # Check folders
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ {folder}/ - Found")
        else:
            print(f"❌ {folder}/ - Missing")
            all_good = False
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            all_good = False
    
    return all_good

def check_data_files():
    """Check if required data files exist"""
    print("\n🔍 Checking data files...")
    
    data_file = 'data/Music Info.csv'
    model_files = {
        'mobilenetv2.keras': 'MobileNetV2',
        'vgg16_transformer.keras': 'VGG16 + Transformer',
        'mobilenetmodel.keras': 'MobileNet (Alternative)'
    }
    
    all_good = True
    models_found = 0
    
    # Check music dataset
    if os.path.exists(data_file):
        print(f"✅ {data_file} - Found")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            required_columns = ['name', 'artist', 'spotify_id', 'valence', 'energy']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
                all_good = False
            else:
                print(f"   ✅ All required columns present ({len(df)} songs)")
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            all_good = False
    else:
        print(f"❌ {data_file} - Not found")
        print(f"   Please add your song dataset as '{data_file}'")
        all_good = False
    
    # Check model files
    print("\n🔍 Checking model files...")
    model_files = {
        'mobilenetv2.keras': 'MobileNetV2',
        'mobilenetmodel.keras': 'MobileNet'
    }
    
    for filename, model_name in model_files.items():
        model_path = f'models/{filename}'
        if os.path.exists(model_path):
            print(f"✅ {model_path} - Found ({model_name})")
            models_found += 1
            
            # Try to load it
            try:
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                print(f"   ✅ Model loaded successfully")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
                all_good = False
        else:
            print(f"⚠️  {model_path} - Not found ({model_name})")
    
    if models_found == 0:
        print("\n⚠️  Warning: No models found in models/ folder")
        print("   Basic CNN will be used as fallback (lower accuracy)")
        print("   Recommended: Add at least mobilenetv2.keras")
    elif models_found < 2:
        print(f"\n✅ {models_found} model(s) found (Basic CNN + {models_found} transfer learning)")
        print(f"   Optional: Add remaining model for more options")
    else:
        print(f"\n✅ All 2 models found! Users can select from 3 models total (including Basic CNN)")
    
    return all_good

def check_text_model():
    """Check if text model can be loaded"""
    print("\n🔍 Checking text emotion model...")
    
    try:
        from transformers import pipeline
        print("⏳ Loading BERT model (this may take a few minutes on first run)...")
        classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")
        
        # Test prediction
        result = classifier("I am so happy today!")[0]
        print(f"✅ Text model loaded successfully")
        print(f"   Test prediction: {result['label']} ({result['score']:.2%})")
        return True
    except Exception as e:
        print(f"❌ Error loading text model: {e}")
        return False

def test_backend_import():
    """Test if backend modules can be imported"""
    print("\n🔍 Testing backend imports...")
    
    try:
        sys.path.append('backend')
        from text_models import TEXT_CLASSIFIER, EMOTION_MAPPING
        print("✅ text_models.py - Imported successfully")
        
        from music_recommender import recommend_songs_by_emotion
        print("✅ music_recommender.py - Imported successfully")
        
        from image_models import ImageEmotionDetector
        print("✅ image_models.py - Imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("🧪 EMOTION-BASED MUSIC RECOMMENDER - INSTALLATION TEST")
    print("=" * 60)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Folder Structure", check_folder_structure()))
    results.append(("Data Files", check_data_files()))
    results.append(("Text Model", check_text_model()))
    results.append(("Backend Imports", test_backend_import()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 60)
        print("\n✅ Your installation is ready!")
        print("\n📝 Next steps:")
        print("   1. Open Terminal 1: cd backend && python main.py")
        print("   2. Open Terminal 2: cd frontend && streamlit run app.py")
        print("   3. Open browser to http://localhost:8501")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\n❌ Please fix the issues above before running the application.")
        print("\n📖 See SETUP_GUIDE.md for detailed instructions.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)