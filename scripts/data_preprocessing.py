import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical # type: ignore

# Load and preprocess images
def preprocess_images(data_dir, csv_file, image_size=(224, 224), num_classes=11):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for _, row in data.iterrows():
        img_path = os.path.join(data_dir, row['filename'])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            continue

        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)

        # Encode label
        labels.append(row['class'])

    images = np.array(images, dtype="float32") / 255.0
    labels = to_categorical(labels, num_classes=num_classes)
    return images, labels

# Save preprocessed data to .npz
def save_data(output_file, X_train, y_train, X_test, y_test):
    np.savez_compressed(output_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Preprocessed data saved to {output_file}")

# Main function
def main():
    train_dir = "data/train"
    test_dir = "data/test"
    train_csv = "data/train_labels.csv"
    test_csv = "data/test_labels.csv"
    output_file = "data/preprocessed_data.npz"

    # Preprocess train and test data
    X_train, y_train = preprocess_images(train_dir, train_csv)
    X_test, y_test = preprocess_images(test_dir, test_csv)

    # Save to file
    save_data(output_file, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()