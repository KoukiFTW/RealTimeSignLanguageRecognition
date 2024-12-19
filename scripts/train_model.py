from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os

# Load augmented training data and preprocessed test data
def load_data():
    # Load augmented training data
    X_train = np.load("data/augmented_X_train.npy")
    y_train = np.load("data/augmented_y_train.npy")

    # Load test data
    test_data = np.load("data/preprocessed_data.npz")
    X_test, y_test = test_data['X_test'], test_data['y_test']

    return X_train, y_train, X_test, y_test

# Build the model (with fine-tuning)
def build_model(input_shape, num_classes):
    # Load MobileNetV2 as the base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Unfreeze the base model for fine-tuning

    # Add custom classification head
    model = Sequential([
        base_model,
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Train the model
def train_model():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Build model
    input_shape = X_train.shape[1:]  # (224, 224, 3)
    num_classes = y_train.shape[1]  # Number of output classes
    model = build_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-5),  # Small learning rate for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    # Save the final model in TensorFlow SavedModel format
    model.save("models/sign_language_model_final.h5")

    print("Training complete. Final model saved to 'models/sign_language_model_final'")
    return history

if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Train the model
    train_model()
