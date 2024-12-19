import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,        # Rotate images randomly by ±15 degrees
    zoom_range=0.2,           # Randomly zoom by 20%
    brightness_range=[0.8, 1.2],  # Adjust brightness randomly
    horizontal_flip=True,     # Randomly flip images horizontally
    fill_mode='nearest'       # Fill in missing pixels after transformations
)

def augment_data(X, y, augment_times=2, batch_size=32):
    """
    Augment data using ImageDataGenerator.
    :param X: Original images (numpy array).
    :param y: Original labels (numpy array).
    :param augment_times: Number of augmented samples per image.
    :param batch_size: Batch size for augmentation.
    :return: Augmented images and labels.
    """
    augmented_images = []
    augmented_labels = []

    for i in range(len(X)):
        # Expand dimensions to fit ImageDataGenerator
        img = X[i].reshape((1,) + X[i].shape)

        # Generate augmented images
        aug_iter = datagen.flow(img, batch_size=1)
        for _ in range(augment_times):
            aug_img = next(aug_iter)[0]  # Extract the augmented image
            augmented_images.append(aug_img)
            augmented_labels.append(y[i])  # Duplicate the label

    return np.array(augmented_images), np.array(augmented_labels)

def main():
    # Load preprocessed training data
    data = np.load("data/preprocessed_data.npz")
    X_train = data['X_train']
    y_train = data['y_train']

    print(f"Original training data shape: {X_train.shape}, {y_train.shape}")

    # Augment training data
    augmented_X_train, augmented_y_train = augment_data(X_train, y_train)

    print(f"Augmented training data shape: {augmented_X_train.shape}, {augmented_y_train.shape}")

    # Combine original and augmented data
    combined_X_train = np.concatenate((X_train, augmented_X_train), axis=0)
    combined_y_train = np.concatenate((y_train, augmented_y_train), axis=0)

    # Save augmented dataset
    np.save("data/augmented_X_train.npy", combined_X_train)
    np.save("data/augmented_y_train.npy", combined_y_train)
    print("Augmented dataset saved!")

if __name__ == "__main__":
    main()