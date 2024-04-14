import uuid
from datetime import datetime

import json

import numpy as np
from flask import jsonify, send_from_directory
from skforecast.model_selection import backtesting_forecaster

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, request, render_template, redirect, url_for, flash
from lightgbm import LGBMRegressor
from werkzeug.utils import secure_filename
import os
import pickle
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import firebase_admin
from firebase_admin import credentials, storage,auth, firestore
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)


#root folder
ROOT_FOLDER = "E:\\Data\\UP\\8vo Semestre\\Big Data\\Forecasting Algorithm"
# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'upload'
STATIC_FOLDER = "static"
DATASETS_FOLDER = "datasets"
KEYS_FOLDER = "keys"
MODELS_FOLDER = 'models'
RESULTS_FOLDER = "results"
FORECAST_FOLDER= "forecast"

app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_FOLDER, UPLOAD_FOLDER)
app.config['STATIC_FOLDER'] = os.path.join(ROOT_FOLDER, STATIC_FOLDER)
app.config["KEYS_FOLDER"] = os.path.join(ROOT_FOLDER, KEYS_FOLDER)
# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Initialize Firebase SDK
cred = credentials.Certificate(os.path.join(app.config['KEYS_FOLDER'], 'key.json'))
firebase_admin.initialize_app(cred,{'storageBucket': 'proyectobigdata-b170b.appspot.com'})
db = firestore.client()
bucket = storage.bucket()

class User(UserMixin):
    def __init__(self, username, name, email, password):
        self.username = username
        self.id = username
        self.name = name
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(username):
    # Retrieve user from Firebase based on username
    user_ref = db.collection('users').where('username', '==', username).limit(1)
    users = user_ref.get()

    if not users:
        flash('Invalid username or password', 'error')
        return redirect(url_for('login'))

    user_data = users[0].to_dict()
    # Return the user object
    return User(username, user_data['name'], user_data['email'], user_data['password'])

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Store user information in Firebase
        user_data = {
            'name': name,
            'username': username,
            'email': email,
            'password': hashed_password
        }
        db.collection('users').add(user_data)

        # Redirect to login page after signup
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Retrieve user from Firebase based on username
        user_ref = db.collection('users').where('username', '==', username).limit(1)
        users = user_ref.get()

        if not users:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))

        user = users[0].to_dict()

        # Check if the password matches
        if check_password_hash(user['password'], password):
            # Log in the user
            user_obj = User(username=username, name=user['name'], email=user['email'], password=user['password'])  # Create User object
            login_user(user_obj)  # Login the user
            return redirect(url_for('user_page'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

@app.route('/user_page')
@login_required
def user_page():
    return render_template('user_page.html', user=current_user)

@app.route('/datasets')
def datasets():
    datasets_ref = db.collection(DATASETS_FOLDER).document(current_user.username).collection('files')
    datasets = datasets_ref.get()
    return render_template('datasets.html', datasets=datasets)

@app.route('/refresh_datasets', methods=['GET'])
@login_required
def refresh_datasets():
    # Handle the refresh action here
    datasets_ref = db.collection(DATASETS_FOLDER).document(current_user.username).collection('files')
    datasets = datasets_ref.get()
    return render_template('datasets.html', datasets=datasets)

@app.route('/models', methods=['GET', 'POST'])
@login_required
def models():
    # Load the regressors from the JSON file
    with open(os.path.join(os.path.join(ROOT_FOLDER, MODELS_FOLDER), 'models.json')) as f:
        regressors = json.load(f)

    # Retrieve saved models for the current user from Firestore
    user_models_ref = db.collection(MODELS_FOLDER).document(current_user.username).collection('files')
    user_models = user_models_ref.get()

    # Process user's saved models
    user_saved_models = []
    for model in user_models:
        model_data = model.to_dict()
        user_saved_models.append(model_data)

    # Pass user's saved models and regressors to the template
    return render_template('models.html', regressors=regressors, user_saved_models=user_saved_models)

@app.route('/predict')
def predict():
    datasets_ref = db.collection(DATASETS_FOLDER).document(current_user.username).collection('files')
    datasets = datasets_ref.get()

    # Retrieve saved models for the current user from Firestore
    user_models_ref = db.collection(MODELS_FOLDER).document(current_user.username).collection('files')
    user_models = user_models_ref.get()

    # Process user's saved models
    user_saved_models = []
    for model in user_models:
        model_data = model.to_dict()
        user_saved_models.append(model_data)

    return render_template('predict.html', datasets=datasets, models=user_saved_models)

@app.route('/save_model', methods=['POST'])
@login_required
def save_model():
    regressor = request.form['regressor']
    model_name = request.form['model_name']

    # Extract parameters for the chosen regressor from models.json
    with open(os.path.join(ROOT_FOLDER,MODELS_FOLDER,'models.json')) as f:
        regressors_params = json.load(f)
    regressor_params = regressors_params[regressor]

    # Extract form data and validate parameters
    model_params = {}
    for param, info in regressor_params.items():
        value = request.form.get(param)
        if value is not None:
            model_params[param] = eval(info['type'])(value)
        elif info['required']:
            flash(f"Parameter '{param}' is required", 'error')
            return redirect(url_for('models'))
    # Create the model with the specified parameters
    # For demonstration purposes, let's assume a simple model creation function
    model = create_model(regressor, model_params)

    file_id = str(uuid.uuid4())
    model_id = model_name+"_"+regressor+"_"+file_id

    # Save the model as a file (you need to implement this function)
    model_file_url = save_model_as_file(model,username=current_user.username,model_id=model_id)

    internal_save_model(model_file_url,model_name,model_id,regressor,model_params, get_empty_train_params())
    os.remove(model_file_url)
    return redirect(url_for('models'))

def get_empty_train_params():
    return {
        'date_trained': None,
        'dataset_root': None,
        'temporal_column':None,
        'y_column':None,
        'group_by':None,
        "clave":None,
        'resampling_rule':None,
        'steps':None
    }

def internal_save_model(model_file_url, model_name, model_id, regressor, model_params, train_params):
    blob = bucket.blob(UPLOAD_FOLDER+"/"+MODELS_FOLDER+"/"+current_user.username+"/"+model_id)

    blob.metadata = {"Content-Type": "application/octet-stream"}

    blob.upload_from_filename(model_file_url, predefined_acl='publicRead')
    # Create URL for the uploaded file
    file_url = blob.public_url

    # Store model information in Firebase Firestore
    model_info = {
        'root': model_id,
        'name': model_name,
        'user': current_user.username,
        'regressor': regressor,
        'params': model_params,
        'file_url': file_url,  # Update with actual file path
        'upload_date': datetime.now(),
        'train_params' : train_params
    }
    db.collection('models').document(current_user.username).collection('files').document(model_id).set(model_info)
    flash('Model saved successfully', 'success')

@app.route('/results')
def results():
    # Retrieve forecasts from Firestore
    forecasts = db.collection(FORECAST_FOLDER).document(current_user.username).collection('files').get()

    # Initialize an empty list to store forecast data
    forecast_data = []

    # Iterate over the forecast documents
    for forecast in forecasts:
        # Convert each forecast document to a dictionary and append it to the list
        forecast_data.append(forecast.to_dict())

    # Render the results.html template with forecast data
    return render_template('results.html', forecasts=forecast_data)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/model_description')
def model_description():
    return render_template('model_description.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # Get form data
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())

        # Get the username
        username = current_user.username

        dataset_name_ = request.form['dataset_name']

        # Extract the original filename without extension
        filename_without_extension, original_extension = os.path.splitext(file.filename)

        # Concatenate the parts to form the filename
        _secret_file_name = f"{secure_filename(dataset_name_)}_{filename_without_extension}_{file_id}{original_extension}"

        subfolder_path = os.path.join(DATASETS_FOLDER, os.path.join(username, _secret_file_name))

        # Extract the directory path from the secret file name
        directory_path = os.path.dirname(subfolder_path)

        # Ensure that the directory structure exists or create it
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], directory_path), exist_ok=True)

        # Save file to the server with unique name
        file_url = os.path.join(app.config['UPLOAD_FOLDER'], subfolder_path)
        file.save(file_url)

        blob = bucket.blob(UPLOAD_FOLDER+"/"+DATASETS_FOLDER+"/"+username+"/"+_secret_file_name)
        blob.upload_from_filename(file_url,predefined_acl='publicRead')

        # Create URL for the uploaded file
        file_url = blob.public_url

        # Read CSV file with pandas to display metadata
        try:
            df = pd.read_csv(file_url)
        except Exception as e:
            flash('Error reading CSV file: ' + str(e))
            return redirect(request.url)

        # Get metadata
        rows, columns = df.shape

        # Save dataset info to Firestore
        dataset_info = {
            'root':_secret_file_name,
            'name': dataset_name_,
            'user': username,
            'file_id': file_id,
            'original_name': secure_filename(file.filename),
            'upload_date': datetime.now(),
            'rows': rows,
            'columns': columns,
            'file_url': file_url
            # Add more metadata if needed
        }
        db.collection(DATASETS_FOLDER).document(username).collection('files').document(_secret_file_name).set(dataset_info)
        flash('File uploaded successfully')
        return redirect(url_for('datasets'))
    else:
        flash('Invalid file type')
        return redirect(request.url)

@app.route('/download_csv/<dataset_id>')
def download_csv(dataset_id):
    dataset_ref = db.collection('datasets').document(dataset_id)
    dataset = dataset_ref.get()
    if not dataset.exists:
        flash('Dataset not found')
        return redirect(url_for('datasets'))
    dataset_info = dataset.to_dict()
    filename = dataset_info['name']
    blob = bucket.blob('uploads/' + filename)
    url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=300), method='GET')
    return redirect(url)

####################################Previous
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
    file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(file_url)

    # Redirect the user to the column mapping page
    return redirect(url_for('map_columns', filename=filename))

@app.route('/prepare_train_mapping', methods=['POST'])
def prepare_train_mapping():
    # Get form data
    model_root = request.form['model_root']
    dataset_root = request.form['dataset_root']

    model = db.collection(MODELS_FOLDER).document(current_user.username).collection('files').document(model_root).get().to_dict()
    dataset = db.collection(DATASETS_FOLDER).document(current_user.username).collection('files').document(dataset_root).get().to_dict()

    model_url = model['file_url']
    dataset_url = dataset['file_url']
    # Generate a random folder ID
    random_folder_id = str(uuid.uuid4())

    # Create a directory to store the files
    folder_path = os.path.join('train', random_folder_id)
    os.makedirs(folder_path, exist_ok=True)
    # Download and save the dataset file
    dataset_filename = os.path.basename(dataset_url)
    dataset_path = os.path.join(folder_path, dataset_filename)
    dataset_response = requests.get(dataset_url)
    with open(dataset_path, 'wb') as dataset_file:
        dataset_file.write(dataset_response.content)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(dataset_path)

    # Get the column names
    columns = df.columns.tolist()
    os.remove(dataset_path)
    os.rmdir(folder_path)
    # Render the column mapping template
    return render_template('train_model.html', columns=columns,dataset_root=dataset_root,model_root=model_root, model_url=model_url, dataset_url=dataset_url)

def custom_metric(y_true, y_pred):
    '''
    Calculate the MAE, MSE, RMSE, and R2
    '''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae":mae, "mse":mse,"rmse":rmse, "r2":r2}


@app.route('/train_model', methods=['POST'])
def train_model():
    # Get form data
    model_root = request.form['model_root']
    dataset_root = request.form['dataset_root']

    model_db = db.collection(MODELS_FOLDER).document(current_user.username).collection('files').document(model_root).get().to_dict()
    dataset_db = db.collection(DATASETS_FOLDER).document(current_user.username).collection('files').document(dataset_root).get().to_dict()

    model_url = request.form['model_url']
    dataset_url = request.form['dataset_url']

    # Generate a random folder ID
    random_folder_id = str(uuid.uuid4())

    # Create a directory to store the files
    folder_path = os.path.join(ROOT_FOLDER,'train', random_folder_id)
    os.makedirs(folder_path, exist_ok=True)

    try:
        # Download and save the dataset file
        dataset_filename = os.path.basename(dataset_url)
        dataset_path = os.path.join(folder_path, dataset_filename)
        dataset_response = requests.get(dataset_url)
        with open(dataset_path, 'wb') as dataset_file:
            dataset_file.write(dataset_response.content)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Get the column names
        columns = df.columns.tolist()

        # Download and save the model file
        model_filename = os.path.basename(model_url)
        model_path = os.path.join(folder_path, model_filename)
        model_response = requests.get(model_url)
        with open(model_path, 'wb') as model_file:
            model_file.write(model_response.content)

        # Load the model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        # Get the selected columns for date, y, and clave values
        date_column = request.form['date_column']
        y_column = request.form['y_column']
        filter_column = request.form['filter_column']  # Column for filtering
        clave = request.form['clave']  # Clave value
        resampling_rule = request.form['resampling_rule']  # Resampling rule
        steps = int(request.form['steps'])  # Number of steps

        # Filter the dataset
        if filter_column and clave:
            df = df[df[filter_column] == clave]

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Set date column as index
        df.set_index(date_column, inplace=True)

        # Resample the dataset using the selected resampling rule
        df_resampled = df[y_column].resample(resampling_rule).sum()

        # Create model
        forecaster = ForecasterAutoreg(
            regressor=model,
            lags=steps+1,
        )

        # Fit the forecaster to the resampled dataset
        forecaster.fit(y=df_resampled)

        # Save the model to the file
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        internal_save_model(model_path,model_db['name'],model_db['root'],model_db['regressor'], model_db["params"],{
            'date_trained': datetime.now(),
            'dataset_root': dataset_db['root'],
            'temporal_column': date_column,
            'y_column': y_column,
            'group_by': filter_column,
            "clave":clave,
            'resampling_rule':resampling_rule,
            'steps':steps
        })

        # Forecast future predictions
        forecast = forecaster.predict(steps=steps)  # Use user-specified steps

        # Get the start and end dates of the forecasted period
        forecast_start_date = df_resampled.index[-len(forecast)]
        forecast_end_date = forecast.index[-1]

        # Create subplot with shared x-axis
        fig = make_subplots(rows=1, cols=1)

        # Plot actual data
        actual_trace = go.Scatter(x=df_resampled.index, y=df_resampled.values, mode='lines', name='Actual', line=dict(color='blue'))
        fig.add_trace(actual_trace, row=1, col=1)

        # Plot forecast
        forecast_trace = go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast', line=dict(color='red', dash='dash'))
        fig.add_trace(forecast_trace, row=1, col=1)

        # Update layout
        fig.update_layout(title='Actual vs Forecast Plot', xaxis_title='Date', yaxis_title='Value', showlegend=True, height=600)

        # Update x-axis range to zoom in on forecasted period
        fig.update_xaxes(range=[forecast_start_date, forecast_end_date])

        # Create HTML file for the interactive plot
        # Generate a unique identifier
        unique_id = str(uuid.uuid4())[:8]  # Use the first 8 characters of the UUID

        # Create HTML file for the interactive plot
        plotly_output_filename = f'forecast_plot_{unique_id}.html'
        plotly_output_path = os.path.join(ROOT_FOLDER,app.config['STATIC_FOLDER'], plotly_output_filename)
        fig.write_html(plotly_output_path)

        # Upload the forecast results as a CSV file to the bucket
        forecast_csv_filename = f'forecast_{unique_id}.csv'
        forecast_csv_path = os.path.join(ROOT_FOLDER,app.config['STATIC_FOLDER'], forecast_csv_filename)
        forecast.to_csv(forecast_csv_path, index_label='Date')

        # Upload the CSV file to the bucket
        blob = bucket.blob(f'{UPLOAD_FOLDER}/{FORECAST_FOLDER}/{current_user.username}/{forecast_csv_filename}')
        blob.upload_from_filename(forecast_csv_path,predefined_acl='publicRead')
        csv_url = blob.public_url

        # Upload the HTML file to the bucket
        blob = bucket.blob(f'{UPLOAD_FOLDER}/{FORECAST_FOLDER}/{current_user.username}/{plotly_output_filename}')
        blob.upload_from_filename(plotly_output_path,predefined_acl='publicRead')
        plot_url = blob.public_url

        # Perform backtesting with the forecaster
        metric, prediction_backtesting = backtesting_forecaster(
            forecaster=forecaster,
            y=df_resampled,  # Assuming 'Demand' is the target column
            steps=steps,
            metric=custom_metric,  # Use mean_absolute_error as the metric
            initial_train_size=len(df_resampled[:forecast_start_date]),  # Use data up to forecast start date for training
            refit=False,  # Do not refit the forecaster during backtesting
            verbose=True,
        )
        # Store all relevant information in the database
        forecast_root = f'forecast_{unique_id}'
        forecast_data = {
            'forecast_root': forecast_root,
            'model_root': model_db["root"],
            'dataset_root': dataset_db['root'],
            'train_params': model_db["train_params"],
            'datetime': datetime.now(),
            'metrics': metric,
            'csv_url': csv_url,
            'plot_url': plot_url
        }

        db.collection(FORECAST_FOLDER).document(current_user.username).collection('files').document(forecast_root).set(forecast_data)

        # Delete temporary files and folder
        os.remove(plotly_output_path)
        os.remove(forecast_csv_path)
        os.remove(dataset_path)
        os.remove(model_path)
        os.rmdir(folder_path)

        # Redirect the user to the success page with the link to the interactive plot
        return redirect(url_for('success', forecast_root=forecast_root, csv_url=csv_url , plotly_plot_url=plot_url))

    except Exception as e:
        # Handle errors
        return render_template('error.html', error=str(e))

@app.route('/plotly_plot')
def plotly_plot():
    # Get the URL of the Plotly plot
    plot_url = request.args.get("plot_url")

    # Ensure the static folder exists
    static_folder = app.config['STATIC_FOLDER']
    os.makedirs(os.path.join(ROOT_FOLDER,static_folder), exist_ok=True)

    # Download the plot from the URL
    response = requests.get(plot_url)

    # Save the plot to the static folder
    filename = os.path.basename(plot_url)
    with open(os.path.join(static_folder, filename), 'wb') as f:
        f.write(response.content)

    # Return the plot from the static folder
    return send_from_directory(static_folder, filename)

@app.route('/success')
def success():
    return render_template('success.html',
                           forecast_root=request.args["forecast_root"],
                           csv_url=request.args["csv_url"],
                           plotly_plot_url=request.args["plotly_plot_url"]
                           )

def create_model(regressor, model_params):
    if regressor == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**model_params)
    elif regressor == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_params)
    elif regressor == 'LGBMRegressor':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")

    return model


def save_model_as_file(model, username, model_id):
    # Define the directory path for storing models
    models_dir = os.path.join(ROOT_FOLDER, 'models', username)
    os.makedirs(models_dir, exist_ok=True)

    # Define the file path for the model
    model_filename = f'{model_id}.pkl'
    model_filepath = os.path.join(models_dir, model_filename)

    # Save the model to the file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    return model_filepath

if __name__ == '__main__':
    app.run()


