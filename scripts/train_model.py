from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
import os

# Create directories if not exist
os.makedirs("models", exist_ok=True)

# Load preprocessed data
def load_data(npz_file):
    data = np.load(npz_file)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

# Build the model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Train the model
def train_model():
    # Load data
    X_train, y_train, X_test, y_test = load_data("data/preprocessed_data.npz")

    # Build model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )

    # Save the final model in TensorFlow SavedModel format
    model.save("models/sign_language_model_final.h5")

    print("Training complete. Final model saved to 'models/sign_language_model_final'")
    return history

if __name__ == "__main__":
    train_model()