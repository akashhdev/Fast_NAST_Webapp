import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras import models, layers
import tensorflow as tf
import numpy as np
import io
from PIL import Image

# Initialize Flask
app = Flask(__name__)

# Set the upload folder and configure app
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit file size to 16 MB


import tensorflow as tf
from tensorflow.keras import layers, models

# Downsample function with proper initialization
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()  # Sequential model

    # Add Conv2D
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False
        )
    )
    
    # Add BatchNormalization if needed
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())  # Proper initialization

    # Add LeakyReLU
    result.add(tf.keras.layers.LeakyReLU())
    
    return result  # Ensure it returns a valid Sequential model

# Upsample function with correct initialization
def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()  # Sequential model
    
    # Add Conv2DTranspose
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False
        )
    )
    
    # Add BatchNormalization
    result.add(tf.keras.layers.BatchNormalization())
    
    # Add Dropout if needed
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))  # Proper initialization

    # Add ReLU
    result.add(tf.keras.layers.ReLU())
    
    return result  # Ensure it returns a valid Sequential model

# Recreate the generator model
def create_generator():
    inputs = layers.Input(shape=[256, 256, 3])  # Input layer

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    # Define the final output layer
    last = layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding='same',
        kernel_initializer='he_normal',
        activation='tanh',
    )

    # Build the model with proper connections
    x = inputs  # Start with the input
    skips = []  # Store skip connections

    # Downsampling through the model
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = list(reversed(skips[:-1]))  # Reverse the skips

    # Upsampling with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])  # Establish skip connections

    # Final output layer
    x = last(x)

    return models.Model(inputs, x)  # Return the model


# Recreate the generator model and load the weights
generator = create_generator()
generator.load_weights("weights/generator_model_e5.weights.h5")  # Adjust path to your model file

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    # Check if the file has an allowed extension
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def landing_page():
    return render_template("index.html")  # Use the landing page template

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            return "No file uploaded", 400
        
        if not allowed_file(uploaded_file.filename):
            return "Invalid file type", 400
        
        # Save the file to the upload folder with a secure name
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(file_path)
        
        # Process the image with the colorization model
        original_image = Image.open(file_path)  # Open the uploaded image
        
        # Ensure it has 3 color channels (RGB)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')  # Convert to RGB
        
        # Resize the image to 256x256
        resized_image = original_image.resize((256, 256))
        
        
        # Convert to numpy array and preprocess
        image_array = np.array(resized_image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Apply the colorization model
        colorized_image_array = generator(image_array)[0].numpy()  # Remove batch dimension
        colorized_image = Image.fromarray((colorized_image_array * 255).astype(np.uint8))  # Convert back to PIL
        
        # Save the colorized image to the upload folder with a new name
        colorized_filename = "colorized_" + filename
        colorized_file_path = os.path.join(app.config["UPLOAD_FOLDER"], colorized_filename)
        colorized_image.save(colorized_file_path)  # Save the colorized image
        
        # Redirect to the result page with the original and colorized filenames
        return redirect(url_for("display_result", filename=filename, colorized_filename=colorized_filename))
    
    # Show the upload form
    return render_template("upload.html")

@app.route("/result/<filename>/<colorized_filename>")
def display_result(filename, colorized_filename):
    return render_template("result.html", filename=filename, colorized_filename=colorized_filename)  # Pass both filenames to the template

@app.route("/uploads/<filename>")
def send_uploaded_file(filename):
    # Serve the uploaded file
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Function to delete all files in a specified folder
def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):  # List all files in the folder
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete file or symbolic link
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Delete empty directories
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Route to clean the upload folder
@app.route("/cleanup")
def cleanup():
    # Delete all files in the upload folder
    delete_all_files_in_folder(app.config["UPLOAD_FOLDER"])

    # Redirect back to the index page after cleanup
    return redirect(url_for("landing_page"))  # Adjust to your index route


if __name__ == "__main__":
    app.run(debug=True)
