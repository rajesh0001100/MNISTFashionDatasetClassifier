import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
print("Loading model...")
model = load_model("fashion_mnist_cnn.h5")
print("Model loaded successfully!")

# Fashion MNIST class labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None, image_path=None)

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image for the CNN model
        # Fashion MNIST images are 28x28 grayscale
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        # Invert colors if necessary (depending on uploaded image style vs training dataset)
        # Fashion MNIST features white items on black background. If uploading a dark-on-white image, uncomment below:
        # img_array = 1.0 - img_array
        
        # Reshape to match model input shape: (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = class_names[predicted_idx]
        confidence = float(np.max(tf.nn.softmax(predictions[0])) * 100)
        
        return render_template('index.html', 
                               prediction=predicted_label, 
                               confidence=f"{confidence:.2f}%", 
                               image_path=filepath)
        
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)