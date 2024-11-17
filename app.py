from flask import Flask, request, render_template, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

print("Current working directory:", os.getcwd())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Set up path for uploaded images and model file
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload to 16MB
model_path = r"C:\Users\dhruv\waste_classification_model.h5"  # Path to your model

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the pre-trained waste classification model
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define the mapping of model output to waste categories
labels = {0: "Non-recyclable", 1: "Recyclable"}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess and predict using the model
def predict_image(filepath):
    try:
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))  # Match your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(img_array)

        # Get the predicted label
        predicted_label = labels[int(prediction[0][0] > 0.5)]  # Binary classification
        return predicted_label
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return None

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            logging.warning("No file part")
            return jsonify({"success": False, "message": "No file part"}), 400

        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"success": False, "message": "No selected file"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logging.warning(f"Invalid file type: {file.filename}")
            return jsonify({"success": False, "message": "Invalid file type. Only jpg, jpeg, and png are allowed."}), 400

        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run the prediction function
        if model is None:
            logging.error("Model not loaded")
            return jsonify({"success": False, "message": "Model failed to load. Please try again later."}), 500
        
        result = predict_image(filepath)

        if result is None:
            return jsonify({"success": False, "message": "Error in image classification"}), 500

        # Pass result and image path to the template
        return render_template('Website.html', result=result, image_path='uploads/' + file.filename)
    
    return render_template('Website.html')

# Entry point for the application
if __name__ == "__main__":
    app.run(debug=True)
