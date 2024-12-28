# Automatic Indian Sign Language Recognition Using MediaPipe Holistic and LSTM Network  

## Overview  

This project focuses on building a robust system for recognizing Indian Sign Language (ISL) gestures using the **MediaPipe Holistic** framework for keypoint extraction and **Long Short-Term Memory (LSTM)** networks for gesture classification. By leveraging cutting-edge machine learning and computer vision techniques, this project bridges communication barriers and fosters inclusivity for the deaf and hard-of-hearing community.  

## Features  

- **Keypoint Detection**: Uses MediaPipe Holistic to detect and extract keypoints of the hands, face, and body for detailed gesture analysis.  
- **Temporal Gesture Recognition**: Utilizes LSTM networks to analyze sequential keypoint data, recognizing the temporal dynamics of sign language gestures.  
- **Multi-Class Gesture Support**: Accurately classifies a wide range of ISL gestures, supporting real-world applications.  
- **Preprocessing Pipelines**: Implements padding, normalization, and encoding to prepare data for effective model training.  
- **Performance Metrics**: Includes precision, recall, F1-score, and confusion matrix to evaluate and refine model performance.  

## Workflow  

### 1. **Data Preparation**  
- Extracts keypoints using MediaPipe Holistic from videos of sign language gestures.  
- Encodes gestures into feature-label pairs and applies preprocessing steps like zero-padding for uniform sequence lengths.  

### 2. **Model Training**  
- Trains LSTM models on sequential keypoint data.  
- Incorporates dropout layers and hyperparameter tuning to reduce overfitting and improve generalization.  

### 3. **Evaluation**  
- Evaluates the trained model using accuracy, confusion matrix, and detailed classification reports.  
- Analyzes misclassifications to refine the gesture recognition process.  

### 4. **Deployment**  
- Saves the trained model in TensorFlow’s SavedModel and HDF5 formats, making it deployment-ready for integration into real-world systems.  

## Technologies Used  

- **MediaPipe Holistic**: For extracting keypoints of hands, face, and pose.  
- **TensorFlow/Keras**: For building and training LSTM-based neural networks.  
- **Python**: For data preprocessing, model training, and evaluation.  
- **NumPy & Pandas**: For handling and processing datasets.  
- **Matplotlib & Seaborn**: For visualizing results and performance metrics.  

## Applications  

- Real-time sign language translation for improved accessibility.  
- Educational tools to teach Indian Sign Language.  
- Enhancing inclusivity in customer service, healthcare, and public administration.  

## How to Use  

### Prerequisites  
- Install Python 3.8 or higher.  
- Install required libraries using:  
  ```bash  
  pip install -r requirements.txt  
