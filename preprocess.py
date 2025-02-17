import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_preprocess_images(folder_path, target_size=(224, 224)):
    """Load, resize and normalize images from a folder"""
    images = []
    labels = []
    
    # Determine if it's a normal or chonky cat based on folder name
    is_normal = 'normal_cat' in folder_path
    label = 0 if is_normal else 1
    
    # List all jpg files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    print(f"Processing {folder_path}...")
    for img_file in tqdm(image_files):
        try:
            # Read image
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load {img_file}")
                continue
                
            # Resize
            img = cv2.resize(img, target_size)
            
            # Convert to RGB (from BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0,1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    return np.array(images), np.array(labels)

def main():
    # Parameters
    NORMAL_PATH = "normal_cat"
    CHONKY_PATH = "chonky_cat"
    TARGET_SIZE = (224, 224)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Load and preprocess images
    normal_images, normal_labels = load_and_preprocess_images(NORMAL_PATH, TARGET_SIZE)
    chonky_images, chonky_labels = load_and_preprocess_images(CHONKY_PATH, TARGET_SIZE)
    
    # Combine datasets
    X = np.concatenate([normal_images, chonky_images])
    y = np.concatenate([normal_labels, chonky_labels])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Create output directory
    os.makedirs('processed_data', exist_ok=True)
    
    # Save processed data
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    
    print("\nDataset statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")

if __name__ == "__main__":
    main() 