import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import xgboost as xgb
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)

# Set the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'

# Ensure the 'uploads' folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('alzheimer_model.json')

# Load the VGG16 model for feature extraction (without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model with the base VGG16 model as the feature extractor
model_vgg = Model(inputs=base_model.input, outputs=base_model.output)

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image to be suitable for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract features using the VGG16 model
def extract_features(img_path):
    img_array = preprocess_image(img_path)
    features = model_vgg.predict(img_array)  # Get the feature map (7x7x512)
    features = features.flatten()  # Flatten the feature map to (25088,)
    return features

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            # Save the file to the server
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features from the image using VGG16
            extracted_features = extract_features(filepath)

            # Predict the Alzheimer's disease stage using the XGBoost model
            prediction = model.predict([extracted_features])

            # Map prediction result to corresponding Alzheimer's stage
            stage_map = {0: 'Non-demented', 1: 'Very Mild Demented', 2: 'Mild Demented', 3: 'Moderate Demented'}
            predicted_stage = stage_map.get(prediction[0], 'Unknown Stage')

            return render_template('index.html', predicted_stage=predicted_stage)

    return render_template('index.html', predicted_stage=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
