# Image Colorization Web App

## Overview
This project is a simple web application built with Flask that allows users to upload black-and-white images and receive colorized versions using a trained machine learning model. The application features a minimalist UI, allowing users to upload images, view colorized results, and clear uploaded files.

<!-- Demo Screenshot 1 -->
<img src="https://github.com/akashhdev/imageRestorization/assets/89295808/0e6bdad0-7a92-4e06-aa44-0dbee2924df6" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 2 -->
<img src="https://github.com/akashhdev/imageRestorization/assets/89295808/6b1ee30b-89e1-452a-a109-af2064f08d37" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 3 -->
<img src="https://github.com/akashhdev/imageRestorization/assets/89295808/e0f1e57d-724d-412b-9bcc-55dca9d198c8" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 4 -->
<img src="https://github.com/akashhdev/imageRestorization/assets/89295808/ad5492c2-b5c6-4d46-a4dd-42c533aba3fb" alt="Example Image" width="400" height="300"/>


## Features
- Upload black-and-white images and get colorized results.
- Display both the original and colorized images.
- Ability to restore more images, which deletes uploaded files and returns to the home page.

## Setup and Installation
To run this Flask application, follow these steps:

### Prerequisites
- Python 3.9 or later
- Flask
- TensorFlow (with appropriate support for your hardware)
- Other dependencies (e.g., numpy, PIL)

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/your-project.git

Navigate to the project directory:

cd your-project


Set up a virtual environment (optional, but recommended):

python -m venv venv

source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows

Install the required dependencies:

pip install -r requirements.txt

**Running the Application**

Start the Flask application:
python app.py
Open your web browser and navigate to http://127.0.0.1:5000/.

Use the application to upload images, view colorized results, and more.


**Additional Notes**
Cleanup: Clicking the "Restore More Images" button on the result page deletes all uploaded files in the uploads folder.
Error Handling: If you encounter errors, check the console output for detailed messages and consider adjusting batch size, resource allocation, or other configurations.

**Contributing**
If you'd like to contribute to this project, please open a pull request or create an issue on GitHub. Contributions, bug reports, and feature requests are welcome.

**License**
This project is licensed under the MIT Open license. See the LICENSE file for more details.
