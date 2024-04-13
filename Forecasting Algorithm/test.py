from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return 'No file part!'

    uploaded_file = request.files['file']

    # If the user does not select a file, the browser submits an empty part without filename
    if uploaded_file.filename == '':
        return 'No selected file!'

    # Check if the file has an allowed extension
    if not allowed_file(uploaded_file.filename):
        return 'Invalid file extension!'

    # Securely save the file
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(file_path)

    # Call your Python script here with the file_path as a parameter
    # Example:
    # os.system(f'python process_csv.py {file_path}')

    # Redirect the user to a success page
    return redirect(url_for('success'))

@app.route('/success')
def success():
    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run()
