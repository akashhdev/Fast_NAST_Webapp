# Image Colorization Web App

## Overview
This project is a simple web application built with Flask that allows users to upload any images of their choosing and transfer a choose art style to them using a trained machine learning model. The application features a minimalist UI, allowing users to upload images, view colorized results, and clear uploaded files.

<!-- Demo Screenshot 1 -->
<img src="https://github.com/user-attachments/assets/2c1b5096-7017-4633-9c57-b147ea92e07f" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 2 -->
<img src="https://github.com/user-attachments/assets/674ae6d4-8d64-4255-a243-6a9a4861652a" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 3 -->
<img src="https://github.com/user-attachments/assets/9026ace7-6491-416a-a83c-250a19009435" alt="Example Image" width="400" height="300"/>

<!-- Demo Screenshot 4 -->
<img src="https://github.com/akashhdev/imageRestorization/assets/89295808/ad5492c2-b5c6-4d46-a4dd-42c533aba3fb" alt="Example Image" width="400" height="300"/>


**Features**
Upload black-and-white images and apply artistic style transfer.

Display both the original and stylized images.

Option to process more images, which clears previous uploads and returns to the home page.

**Setup and Installation**
Prerequisites
Python 3.9 or later

Flask
PyTorch (with appropriate support for your hardware)
Other dependencies (e.g., numpy, PIL)

**Installation**
Clone the repository:

git clone https://github.com/akashhdev/Fast_NAST_Webapp.git  
cd Fast_NAST_Webapp  
Set up a virtual environment (optional but recommended):

nginx
python -m venv venv  

**Activate the virtual environment:**

On Linux/macOS:
source venv/bin/activate  

On Windows:
venv\Scripts\activate  

Install required dependencies:
pip install --upgrade pip  
pip install -r requirements.txt  

Running the Application
Start the Flask application:

nginx
python app.py  
Open a web browser and go to:

cpp
http://127.0.0.1:5000/  

**Additional Notes**
Clicking the "Try More Images" button clears the uploaded files and redirects to the homepage.

If errors occur, check the console output and adjust configurations such as batch size or resource allocation.

**Contributing**
If you'd like to contribute, open a pull request or create an issue on GitHub. Contributions, bug reports, and feature requests are welcome.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.
