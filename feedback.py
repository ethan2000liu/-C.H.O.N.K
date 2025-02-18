import os
import numpy as np
import tensorflow as tf
from utils import load_and_preprocess_image

def get_user_feedback(image_path, model_prediction):
    """Get feedback from user about the prediction"""
    print(f"\nModel predicted: {'CHONKY' if model_prediction else 'NORMAL'} CAT")
    while True:
        response = input("Was this prediction correct? (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'

def fine_tune_model(model, new_image, correct_label, learning_rate=0.00001):
    """Fine-tune the model with a single image"""
    # Compile model with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create small dataset with the new image
    X = np.array([new_image[0]])  # Remove batch dimension
    y = np.array([correct_label])
    
    # Fine-tune for a few steps
    history = model.fit(X, y, epochs=5, verbose=0)
    return history.history['loss'][-1]

def save_feedback_data(image_path, correct_label):
    """Save feedback data for future retraining"""
    feedback_dir = 'feedback_data'
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Copy image to feedback directory
    import shutil
    label_dir = os.path.join(feedback_dir, 'normal_cat' if correct_label == 0 else 'chonky_cat')
    os.makedirs(label_dir, exist_ok=True)
    
    new_path = os.path.join(label_dir, os.path.basename(image_path))
    shutil.copy2(image_path, new_path) 