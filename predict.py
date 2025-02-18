import os
import tensorflow as tf
from utils import load_and_preprocess_image
from feedback import get_user_feedback, fine_tune_model, save_feedback_data

def predict_chonk(image_path, enable_feedback=True):
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
        
        # Get feedback if enabled
        if enable_feedback:
            correct = get_user_feedback(image_path, is_chonky)
            if not correct:
                # User says prediction was wrong
                correct_label = 0 if is_chonky else 1  # Opposite of prediction
                
                # Fine-tune model
                loss = fine_tune_model(model, img, correct_label)
                print(f"Model updated (loss: {loss:.4f})")
                
                # Save feedback data
                save_feedback_data(image_path, correct_label)
                
                # Save updated model
                model.save('models/best_model.keras')
                print("Model saved with updates")
        
        return is_chonky, confidence
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage tips:")
        print("1. Make sure the image file exists and is a valid image format")
        print("2. Use the full path if the image is not in the current directory")
        print("3. Make sure the trained model exists at 'models/best_model.keras'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_chonk(sys.argv[1]) 