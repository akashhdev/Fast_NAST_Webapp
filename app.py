import os
import glob
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from model import test_image  # Import the test_image function from model.py

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max file size

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def landing_page():
    return render_template("index.html")  # Create an index.html template


@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    style = request.args.get("style", "best_model")  # Get style from URL
    print(f"Received request for upload with style: {style}")  # Debugging log

    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            checkpoint_path = f"./checkpoints/{style}.pth"
            print(f"Using model: {style}, Checkpoint Path: {checkpoint_path}")  # Debug log

            test_image(image_path=file_path, checkpoint_model=checkpoint_path, save_path="uploads")

            styled_filename = f"{style}-output.jpg"
            return redirect(url_for("display_result", filename=filename, styled_filename=styled_filename))
        else:
            return "Invalid file type", 400

    return render_template("upload.html", style=style)







@app.route("/result/<filename>/<styled_filename>")
def display_result(filename, styled_filename):
    print("Displaying Result - Original:", filename, "Styled:", styled_filename)
    return render_template("result.html", filename=filename, styled_filename=styled_filename)


@app.route("/uploads/<path:filename>")
def send_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/clear_results")
def clear_results():
    results_folder = os.path.join(app.config["UPLOAD_FOLDER"], "results")

    # Ensure the folder exists before trying to delete files
    if os.path.exists(results_folder):
        for file in os.listdir(results_folder):
            file_path = os.path.join(results_folder, file)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    os.remove(file_path)  # Delete each file
                except PermissionError:
                    print(f"Permission denied: {file_path}")  # Log error if file is locked

    return redirect(url_for("landing_page"))

if __name__ == "__main__":
    app.run(debug=True)
