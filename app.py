import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

# Load the trained model from the file
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: model.pkl file not found. Please ensure it's in the same directory.")
    exit()

# Initialize the Flask application
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date string from the web form
        date_str = request.form['date']
        
        # Generate features from the date string
        date_to_predict = pd.to_datetime(date_str)
        
        # Create a DataFrame for the single data point.
        input_data = pd.DataFrame({
            'hour': [date_to_predict.hour],
            'day_of_week': [date_to_predict.dayofweek],
            'month': [date_to_predict.month],
            'quarter': [date_to_predict.quarter],
            'temperature_c': [15.0]  # Placeholder temperature
        })
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]
        
        return render_template('result.html', prediction=f"{prediction:.2f} kWh")
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)