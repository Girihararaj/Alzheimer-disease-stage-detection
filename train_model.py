import os
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Initialize the pre-trained VGG16 model for feature extraction (without top layers)
model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features from an image using VGG16
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 as VGG16 expects this size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image for VGG16
    
    features = model_vgg.predict(img)  # Extract features from the image
    features = features.flatten()  # Flatten the output to a 1D vector
    return features

# Prepare the dataset
data = []
labels = []

# Folder paths for each stage (replace with the actual folder path)
base_dir = "D:\Alzheimer web\OriginalDataset"  # Set this to your dataset location
stages = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Map each stage to a numeric label
stage_map = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 'ModerateDemented': 3}

# Iterate over each folder and collect features and labels
for stage in stages:
    stage_path = os.path.join(base_dir, stage)
    for img_name in os.listdir(stage_path):
        img_path = os.path.join(stage_path, img_name)
        features = extract_features(img_path)
        data.append(features)
        labels.append(stage_map[stage])

# Convert data and labels to numpy arrays
X = np.array(data)
y = np.array(labels)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
model = xgb.XGBClassifier(eval_metric='mlogloss')

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save_model('alzheimer_model.json')
