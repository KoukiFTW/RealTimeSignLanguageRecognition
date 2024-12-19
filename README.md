# RealTimeSignLanguageRecognition

This project is a real-time sign language recognition system that detects hand gestures using MediaPipe Hands and classifies them using a TensorFlow-based deep learning model trained on hand landmarks. The system can recognize gestures representing numbers from `0` to `10` and provides accurate predictions in real-time.

---

## Project Features
- **Real-Time Gesture Detection**: Tracks the user's hand and recognizes gestures dynamically.
- **Hand Landmark-Based Model**: Utilizes MediaPipe's hand landmarks for efficient and accurate classification.
- **Optimized with TensorFlow Lite**: Ensures fast inference during real-time detection.
- **Data Augmentation**: Improves model robustness by augmenting training data.
- **Confidence Filtering**: Filters predictions based on confidence thresholds to reduce noise.

---

## Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd RealTimeSignLanguageRecognition
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
RealTimeSignLanguageRecognition/
│
├── data/
│   ├── train/                      # Training images (optional, if raw images are used)
│   ├── test/                       # Testing images (optional, if raw images are used)
│   ├── train_labels.csv            # Labels for training data
│   ├── test_labels.csv             # Labels for testing data
│   ├── preprocessed_data.npz       # Preprocessed training and testing data
│   ├── augmented_X_train.npy       # Augmented training images
│   ├── augmented_y_train.npy       # Augmented training labels
│   ├── landmarks_train.npz         # Hand landmarks for training
│   ├── landmarks_test.npz          # Hand landmarks for testing
│
├── models/
│   ├── landmark_model_final/       # Final trained model directory (TensorFlow format)
│   ├── best_landmark_model.keras   # Best model checkpoint
│   ├── landmark_model_final.tflite # TensorFlow Lite model for real-time inference
│
├── scripts/
│   ├── data_preprocessing.py       # Script to preprocess raw images
│   ├── data_augmentation.py        # Script for augmenting training data
│   ├── extract_landmarks.py        # Script to extract hand landmarks
│   ├── train_landmark_model.py     # Script to train the landmark-based model
│   ├── convert_to_tflite.py        # Script to convert model to TensorFlow Lite
│   ├── real_time_landmark_detection.py # Real-time gesture recognition script
│
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Dependencies
└── LICENSE                         # License for the project
```

---

## Step-by-Step Usage

### 1. Data Preprocessing (Optional)
If you have raw images, preprocess them into `.npz` format.
```bash
python scripts/data_preprocessing.py
```

### 2. Data Augmentation
Augment the training data, especially for gestures like `0` and `5`.
```bash
python scripts/data_augmentation.py
```

### 3. Extract Hand Landmarks
Convert preprocessed images into hand landmarks using MediaPipe.
```bash
python scripts/extract_landmarks.py
```

### 4. Train the Model
Train a deep learning model using the extracted hand landmarks.
```bash
python scripts/train_landmark_model.py
```

### 5. Convert the Model to TensorFlow Lite
Optimize the trained model for real-time inference.
```bash
python scripts/convert_to_tflite.py
```

### 6. Real-Time Gesture Recognition
Run the real-time detection system.
```bash
python scripts/real_time_landmark_detection.py
```

---

## Troubleshooting

### Common Issues
1. **Webcam Not Opening**:
   - Ensure your webcam is connected and not used by another application.
   - Check permissions if running on a restricted system.

2. **Low Accuracy for Specific Gestures**:
   - Augment the dataset for poorly detected gestures.
   - Ensure balanced data for all classes.

3. **Slow Detection**:
   - Use the TensorFlow Lite model for faster inference.
   - Ensure the system meets the recommended hardware requirements.

---

## Future Improvements
- Add support for more gestures or sign languages.
- Enhance robustness to handle different lighting conditions and backgrounds.
- Integrate with speech synthesis for gesture-to-speech translation.

---

## Contributors
- **ABDUL ALIF BCS22090018** - Initial development
