import tensorflow as tf
import cv2
import numpy as np
import sys
import os

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Check if file exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    try:
        # Try to import various image format handlers
        try:
            from pillow_avif import AvifImagePlugin  # For AVIF support
        except ImportError:
            print("AVIF support not available. Install with: pip install pillow-avif")
        
        try:
            from pillow_heif import register_heif_opener  # For HEIC support
            register_heif_opener()
        except ImportError:
            print("HEIC support not available. Install with: pip install pillow-heif")
        
        try:
            from PIL import Image, ImageOps
            import warnings
            warnings.filterwarnings('ignore')  # Ignore PIL warnings
            
            # Try to open the image
            img = Image.open(image_path)
            
            # Convert to RGB (handles grayscale, RGBA, etc.)
            img = img.convert('RGB')
            
            # Auto-orient image based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Resize maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with padding if needed
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Paste the image in the center
            offset = ((target_size[0] - img.size[0]) // 2,
                     (target_size[1] - img.size[1]) // 2)
            new_img.paste(img, offset)
            
            # Convert to numpy array
            img = np.array(new_img)
            
            # Normalize to [0,1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            # Try OpenCV as fallback
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                return img
            else:
                raise ValueError("OpenCV fallback also failed to load the image")
                
    except Exception as e:
        raise ValueError(f"Error loading image: {e}\n"
                        f"Supported formats: JPEG, PNG, WebP, AVIF*, HEIC*\n"
                        f"(*requires additional packages)\n"
                        f"Install support for all formats with:\n"
                        f"pip install pillow pillow-avif pillow-heif")

def predict_chonk(image_path):
    try:
        # Print current working directory and image path for debugging
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to load image from: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Check if model exists
        model_path = 'models/best_model.keras'
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
            
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess image
        img = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img)[0][0]
        
        # Get class and confidence
        is_chonky = prediction > 0.5
        confidence = prediction if is_chonky else 1 - prediction
        
        # Print result
        cat_type = "CHONKY" if is_chonky else "NORMAL"
        print(f"\nResult: {cat_type} CAT")
        print(f"Confidence: {confidence*100:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage tips:")
        print("1. Make sure the image file exists and is a valid image format")
        print("2. Use the full path if the image is not in the current directory")
        print("3. Make sure the trained model exists at 'models/best_model.keras'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py input.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_chonk(image_path) 