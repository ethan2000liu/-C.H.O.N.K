import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

def create_model():
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add our custom top layers with more complexity
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Store base_model as an attribute to access it later
    model.base_model = base_model
    
    # First: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model

def main():
    # Load preprocessed data
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Create more aggressive data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.4),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.GaussianNoise(0.1),
    ])
    
    # Create training dataset with augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Setup callbacks with longer patience
    checkpoint = ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
    
    # First phase: train only the top layers
    print("Phase 1: Training top layers...")
    history1 = model.fit(
        train_dataset,
        epochs=30,
        validation_data=test_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Second phase: fine-tune more ResNet layers
    print("\nPhase 2: Fine-tuning ResNet layers...")
    # Unfreeze more layers of the base model
    for layer in model.base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_dataset,
        epochs=50,
        validation_data=test_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_auc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    main() 