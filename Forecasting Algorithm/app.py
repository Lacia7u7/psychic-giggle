from flask import Flask, request, render_template, redirect, url_for
from lightgbm import LGBMRegressor
from werkzeug.utils import secure_filename
import os
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg

app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

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

    # Redirect the user to the column mapping page
    return redirect(url_for('map_columns', filename=filename))

@app.route('/map_columns/<filename>', methods=['GET', 'POST'])
def map_columns(filename):
    # Read the CSV file into a pandas DataFrame
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    # Get the column names
    columns = df.columns.tolist()

    if request.method == 'POST':
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

        # Create and configure the LGBMRegressor
        regressor = LGBMRegressor()
        # Create model
        forecaster = ForecasterAutoreg(
            regressor=regressor,
            lags=steps+1
        )
        # Fit the forecaster to the resampled dataset
        forecaster.fit(y=df_resampled)

        # Forecast future predictions
        forecast = forecaster.predict(steps=steps)  # Use user-specified steps

        # Plot the forecast
        plt.figure(figsize=(10, 6))

        # Plot three times the steps back
        start_date = df_resampled.index[-1] - pd.DateOffset(days=3 * steps)
        end_date = df_resampled.index[-1]
        df_resampled[start_date:end_date].plot(label='Actual')

        # Plot the forecast
        forecast.plot(label='Forecast', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Forecast Plot')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/forecast_plot.png')

        # Redirect the user to the success page
        return redirect(url_for('success'))

    # Render the column mapping template
    return render_template('map_columns.html', columns=columns)

@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
