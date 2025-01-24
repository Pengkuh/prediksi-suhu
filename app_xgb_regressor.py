from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
import xgboost as xgb
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Process the file
        data = pd.read_excel(filepath)
        
        if 'Date' not in data.columns or 'Temperature' not in data.columns:
            return "Dataset harus memiliki kolom 'Date' dan 'Temperature'."

        data['date'] = pd.to_datetime(data['Date'])
        data.sort_values('date', inplace=True)

        # Feature engineering
        data['day'] = (data['date'] - data['date'].min()).dt.days
        data['month'] = data['date'].dt.month
        data['weekday'] = data['date'].dt.weekday

        # Model training
        X = data[['day', 'month', 'weekday']]
        y = data['Temperature']
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X, y)

        # Prediction
        future_days = pd.DataFrame({'day': np.arange(data['day'].max() + 1, data['day'].max() + 31)})
        future_dates = [data['date'].max() + pd.Timedelta(days=i) for i in range(1, 31)]
        future_days['month'] = [d.month for d in future_dates]
        future_days['weekday'] = [d.weekday() for d in future_dates]

        future_temps = model.predict(future_days)

        future_data = pd.DataFrame({
            'date': future_dates,
            'predicted_temperature': future_temps
        })

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(data['date'], data['Temperature'], label='Historical Data')
        plt.plot(future_data['date'], future_data['predicted_temperature'], label='Predictions', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.title('Temperature Prediction')
        plt.legend()
        plt.grid()

        # Save plot to string
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Pass to output.html
        return render_template(
            'output.html',
            tables=future_data.to_html(classes='table table-bordered', index=False),
            plot_url=plot_url
        )

if __name__ == '__main__':
    app.run(debug=True)
