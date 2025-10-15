#!/usr/bin/env python3
"""
Prediction Script for Multimodal Pneumonia Diagnosis Model
Tests trained model on new chest X-ray and EMR inputs
"""

import argparse
import sys
import json
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
import joblib
from torchvision import transforms

from models import create_model
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP, PROC_DIR

def ensure_rgb(image):
    """Ensure image is in RGB mode"""
    return image.convert('RGB') if image.mode != 'RGB' else image

def preprocess_image(image_path, img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Preprocess input image to match training pipeline
    
    Args:
        image_path: Path to input image
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Ensure RGB
        image = ensure_rgb(image)
        
        # Apply same transforms as validation/test
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Transform and add batch dimension
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess image '{image_path}': {e}")

def preprocess_emr(age, sex, fever_temp, cough_score, WBC_count, SpO2, 
                  scaler_path, label_map_path):
    """
    Preprocess EMR input to match training pipeline
    
    Args:
        age: Patient age
        sex: Patient sex (M/F)
        fever_temp: Fever temperature
        cough_score: Cough severity score (0-5)
        WBC_count: White blood cell count
        SpO2: Oxygen saturation
        scaler_path: Path to fitted scaler
        label_map_path: Path to label map
    
    Returns:
        Preprocessed EMR tensor
    """
    try:
        # Load label map
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        # Validate inputs
        if sex.upper() not in ['M', 'F']:
            raise ValueError(f"Sex must be 'M' or 'F', got '{sex}'")
        
        if not (0 <= cough_score <= 5):
            raise ValueError(f"Cough score must be 0-5, got {cough_score}")
        
        if not (70 <= SpO2 <= 100):
            raise ValueError(f"SpO2 must be 70-100, got {SpO2}")
        
        # Create EMR array in training order: [age, fever_temp, cough_score, WBC_count, SpO2, sex_F, sex_M]
        emr_data = np.array([
            age,
            fever_temp,
            cough_score,
            WBC_count,
            SpO2,
            1 if sex.upper() == 'F' else 0,  # sex_F
            1 if sex.upper() == 'M' else 0   # sex_M
        ]).reshape(1, -1)
        
        # Scale continuous features (first 5 columns)
        emr_scaled = emr_data.copy()
        emr_scaled[:, :5] = scaler.transform(emr_data[:, :5])
        
        # Convert to tensor
        emr_tensor = torch.tensor(emr_scaled, dtype=torch.float32)
        
        return emr_tensor, label_map
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess EMR data: {e}")

def load_model(model_path, device):
    """
    Load trained model
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, checkpoint
        
    except Exception as e:
        raise ValueError(f"Failed to load model from '{model_path}': {e}")

def predict_pneumonia(model, image_tensor, emr_tensor, device):
    """
    Run prediction on input data
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        emr_tensor: Preprocessed EMR tensor
        device: Device to run inference on
    
    Returns:
        Prediction probability and label
    """
    try:
        # Move tensors to device
        image_tensor = image_tensor.to(device)
        emr_tensor = emr_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            logits, probabilities = model(image_tensor, emr_tensor)
            probability = probabilities.item()
        
        # Convert to label
        label = "pneumonia" if probability > 0.5 else "normal"
        
        return probability, label
        
    except Exception as e:
        raise ValueError(f"Failed to run prediction: {e}")

def format_emr_display(age, sex, fever_temp, cough_score, WBC_count, SpO2):
    """Format EMR data for display"""
    return f"{{age: {age}, sex: {sex.upper()}, fever_temp: {fever_temp}, cough_score: {cough_score}, WBC_count: {WBC_count}, SpO2: {SpO2}}}"

def save_prediction_to_log(image_path, age, sex, fever_temp, cough_score, WBC_count, SpO2, 
                          prediction, confidence, log_file_path="outputs/logs/report.txt"):
    """
    Save prediction result to log file
    
    Args:
        image_path: Path to input image
        age, sex, fever_temp, cough_score, WBC_count, SpO2: EMR data
        prediction: Prediction result (pneumonia/normal)
        confidence: Prediction confidence (0-1)
        log_file_path: Path to log file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Format log entry
        log_entry = f"""
========================
Patient Prediction Log
Date: {timestamp}
Image: {Path(image_path).name}
Prediction: {prediction.title()}
Confidence: {confidence:.2f}
Input EMR:
  Age = {age}
  Sex = {sex}
  Fever Temp = {fever_temp} °C
  Cough Score = {cough_score}
  WBC Count = {WBC_count}
  SpO2 = {SpO2} %
========================
"""
        
        # Append to log file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to save log: {e}")
        return False

def get_user_input(prompt, input_type=str, default=None, choices=None, min_val=None, max_val=None):
    """
    Get user input with validation and error handling
    
    Args:
        prompt: Input prompt string
        input_type: Type to convert input to (str, int, float)
        default: Default value if user presses enter
        choices: List of valid choices (for str inputs)
        min_val: Minimum value (for numeric inputs)
        max_val: Maximum value (for numeric inputs)
    
    Returns:
        Validated user input
    """
    while True:
        try:
            # Show prompt with default
            if default is not None:
                full_prompt = f"{prompt} (default: {default}): "
            else:
                full_prompt = f"{prompt}: "
            
            user_input = input(full_prompt).strip()
            
            # Use default if empty
            if not user_input and default is not None:
                return default
            
            # Convert to requested type
            if input_type == str:
                value = user_input
            elif input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                value = user_input
            
            # Validate choices
            if choices is not None and value not in choices:
                print(f"❌ Invalid choice. Please choose from: {', '.join(choices)}")
                continue
            
            # Validate numeric ranges
            if min_val is not None and value < min_val:
                print(f"❌ Value must be >= {min_val}")
                continue
            
            if max_val is not None and value > max_val:
                print(f"❌ Value must be <= {max_val}")
                continue
            
            return value
            
        except ValueError:
            print(f"❌ Invalid input. Please enter a valid {input_type.__name__}")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def interactive_prediction():
    """Interactive prediction mode"""
    print("🔍 Multimodal Pneumonia Diagnosis Prediction")
    print("="*50)
    print("Welcome! I'll guide you through entering patient data for prediction.")
    print()
    
    # Set device (always use CPU for compatibility)
    device = torch.device('cpu')
    print(f"🔧 Using device: {device}")
    print()
    
    # Get model path
    print("📁 Model Configuration")
    print("-" * 30)
    model_path = get_user_input(
        "Enter path to trained model",
        input_type=str,
        default="outputs/checkpoints/best_model.pt"
    )
    
    # Check model file exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Image not found. Please try again.")
        return
    
    # Get image path
    print("\n🖼️ Chest X-ray Image")
    print("-" * 30)
    image_path = get_user_input(
        "Enter path to chest X-ray image (PNG/JPG)",
        input_type=str
    )
    
    # Check image file exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Image not found. Please try again.")
        return
    
    # Get EMR data
    print("\n📊 Patient EMR Data")
    print("-" * 30)
    
    age = get_user_input(
        "Enter patient age",
        input_type=int,
        min_val=1,
        max_val=90
    )
    
    sex = get_user_input(
        "Enter patient sex",
        input_type=str,
        choices=['M', 'F', 'm', 'f']
    ).upper()
    
    fever_temp = get_user_input(
        "Enter fever temperature (°C)",
        input_type=float,
        min_val=36.5,
        max_val=41.0
    )
    
    cough_score = get_user_input(
        "Enter cough severity score (0-5)",
        input_type=int,
        min_val=0,
        max_val=5
    )
    
    WBC_count = get_user_input(
        "Enter WBC count (/µL)",
        input_type=int,
        min_val=4000,
        max_val=20000
    )
    
    SpO2 = get_user_input(
        "Enter SpO2 (%)",
        input_type=int,
        min_val=70,
        max_val=100
    )
    
    # Check required files exist
    print("\n🔍 Validating required files...")
    scaler_path = PROC_DIR / "emr_scaler.joblib"
    label_map_path = PROC_DIR / "label_map.json"
    
    if not scaler_path.exists():
        print(f"❌ Error: EMR scaler not found: {scaler_path}")
        print("   Run data preprocessing first: python scripts/prepare_data.py --emr")
        return
    
    if not label_map_path.exists():
        print(f"❌ Error: Label map not found: {label_map_path}")
        print("   Run data preprocessing first: python scripts/prepare_data.py --emr")
        return
    
    print("   ✅ All required files found")
    
    try:
        # Load model
        print(f"\n📁 Loading model from: {model_path}")
        model, checkpoint = load_model(model_path, device)
        print(f"   Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Preprocess image
        print(f"\n🖼️ Preprocessing image: {image_path}")
        image_tensor = preprocess_image(image_path)
        print(f"   Image shape: {image_tensor.shape}")
        
        # Preprocess EMR
        print("\n📊 Preprocessing EMR data...")
        emr_tensor, label_map = preprocess_emr(
            age, sex, fever_temp, cough_score, 
            WBC_count, SpO2, scaler_path, label_map_path
        )
        print(f"   EMR shape: {emr_tensor.shape}")
        
        # Run prediction
        print("\n🔮 Running prediction...")
        probability, label = predict_pneumonia(model, image_tensor, emr_tensor, device)
        
        # Display results in the requested format
        print("\n" + "="*32)
        print("Patient Prediction Result")
        print("-" * 30)
        print(f"Prediction: {label.title()}")
        print(f"Confidence: {probability:.2f}")
        print("Input EMR: ")
        print(f"  Age = {age}")
        print(f"  Sex = {sex}")
        print(f"  Fever Temp = {fever_temp} °C")
        print(f"  Cough Score = {cough_score}")
        print(f"  WBC Count = {WBC_count}")
        print(f"  SpO2 = {SpO2} %")
        print("="*32)
        print("✅ Prediction completed successfully!")
        
        # Ask if user wants to save results
        print("\n📝 Report Logging")
        print("-" * 20)
        save_log = get_user_input(
            "Do you want to save this result to report.txt?",
            input_type=str,
            choices=['Y', 'N', 'y', 'n'],
            default='N'
        ).upper()
        
        if save_log == 'Y':
            success = save_prediction_to_log(
                image_path, age, sex, fever_temp, cough_score, 
                WBC_count, SpO2, label, probability
            )
            if success:
                print("✅ Result saved to outputs/logs/report.txt")
            else:
                print("⚠️ Failed to save result to log file")
        else:
            print("📝 Result not saved to log file")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main prediction function with both CLI and interactive modes"""
    # Check if any arguments were provided
    if len(sys.argv) > 1:
        # CLI mode (original functionality)
        parser = argparse.ArgumentParser(
            description="Predict pneumonia from chest X-ray and EMR data",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python scripts/predict.py --image_path sample_xray.png --age 62 --sex M --fever_temp 38.9 --cough_score 4 --WBC_count 14500 --SpO2 88
  
  python scripts/predict.py --model_path custom_model.pt --image_path xray.jpg --age 45 --sex F --fever_temp 37.2 --cough_score 2 --WBC_count 8000 --SpO2 95
            """
        )
        
        # Model arguments
        parser.add_argument('--model_path', type=str, default='outputs/checkpoints/best_model.pt',
                           help='Path to trained model checkpoint')
        
        # Image arguments
        parser.add_argument('--image_path', type=str, required=True,
                           help='Path to chest X-ray image (PNG/JPG)')
        
        # EMR arguments
        parser.add_argument('--age', type=int, required=True,
                           help='Patient age (1-90)')
        parser.add_argument('--sex', type=str, required=True, choices=['M', 'F', 'm', 'f'],
                           help='Patient sex (M/F)')
        parser.add_argument('--fever_temp', type=float, required=True,
                           help='Fever temperature (36.5-41.0)')
        parser.add_argument('--cough_score', type=int, required=True,
                           help='Cough severity score (0-5)')
        parser.add_argument('--WBC_count', type=int, required=True,
                           help='White blood cell count (4000-20000)')
        parser.add_argument('--SpO2', type=int, required=True,
                           help='Oxygen saturation (70-100)')
        
        # Optional arguments
        parser.add_argument('--device', type=str, default='auto',
                           help='Device to use (auto, cpu, cuda, mps)')
        parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose output')
        
        args = parser.parse_args()
        
        print("🔍 Multimodal Pneumonia Diagnosis Prediction (CLI Mode)")
        print("="*60)
        
        # Set device
        if args.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(args.device)
        
        print(f"🔧 Using device: {device}")
        
        # Validate inputs
        print("\n📋 Validating inputs...")
        
        # Check image file exists
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"❌ Error: Image file not found: {image_path}")
            sys.exit(1)
        
        # Check model file exists
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Error: Model file not found: {model_path}")
            sys.exit(1)
        
        # Check required files exist
        scaler_path = PROC_DIR / "emr_scaler.joblib"
        label_map_path = PROC_DIR / "label_map.json"
        
        if not scaler_path.exists():
            print(f"❌ Error: EMR scaler not found: {scaler_path}")
            print("   Run data preprocessing first: python scripts/prepare_data.py --emr")
            sys.exit(1)
        
        if not label_map_path.exists():
            print(f"❌ Error: Label map not found: {label_map_path}")
            print("   Run data preprocessing first: python scripts/prepare_data.py --emr")
            sys.exit(1)
        
        print("   ✅ All required files found")
        
        # Validate EMR ranges
        if not (1 <= args.age <= 90):
            print(f"❌ Error: Age must be 1-90, got {args.age}")
            sys.exit(1)
        
        if not (36.5 <= args.fever_temp <= 41.0):
            print(f"❌ Error: Fever temperature must be 36.5-41.0, got {args.fever_temp}")
            sys.exit(1)
        
        if not (4000 <= args.WBC_count <= 20000):
            print(f"❌ Error: WBC count must be 4000-20000, got {args.WBC_count}")
            sys.exit(1)
        
        print("   ✅ Input validation passed")
        
        try:
            # Load model
            print(f"\n📁 Loading model from: {model_path}")
            model, checkpoint = load_model(model_path, device)
            print(f"   Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            
            # Preprocess image
            print(f"\n🖼️ Preprocessing image: {image_path}")
            image_tensor = preprocess_image(image_path)
            print(f"   Image shape: {image_tensor.shape}")
            
            # Preprocess EMR
            print("\n📊 Preprocessing EMR data...")
            emr_tensor, label_map = preprocess_emr(
                args.age, args.sex, args.fever_temp, args.cough_score, 
                args.WBC_count, args.SpO2, scaler_path, label_map_path
            )
            print(f"   EMR shape: {emr_tensor.shape}")
            
            # Run prediction
            print("\n🔮 Running prediction...")
            probability, label = predict_pneumonia(model, image_tensor, emr_tensor, device)
            
            # Display results
            print("\n" + "="*50)
            print("📊 PREDICTION RESULTS")
            print("="*50)
            print(f"Prediction: {label.title()}")
            print(f"Confidence: {probability:.2f}")
            print(f"Input EMR: {format_emr_display(args.age, args.sex, args.fever_temp, args.cough_score, args.WBC_count, args.SpO2)}")
            
            if args.verbose:
                print(f"\n📈 Detailed Results:")
                print(f"   Raw probability: {probability:.4f}")
                print(f"   Threshold: 0.5")
                print(f"   Confidence level: {'High' if probability > 0.8 or probability < 0.2 else 'Medium'}")
                print(f"   Model epoch: {checkpoint.get('epoch', 'unknown')}")
            
            print("="*50)
            print("✅ Prediction completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during prediction: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    else:
        # Interactive mode
        interactive_prediction()

if __name__ == "__main__":
    main()
