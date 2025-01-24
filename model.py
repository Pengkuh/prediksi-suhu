import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def process_data(data):
    data['date'] = pd.to_datetime(data['Date'])
    data.sort_values('date', inplace=True)
    data['day'] = (data['date'] - data['date'].min()).dt.days
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday
    return data

def train_and_predict(data):
    data = process_data(data)

    # Prepare features and target
    X = data[['day', 'month', 'weekday']]
    y = data['Temperature']

    # Train the model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # Predict for the next 30 days
    future_days = pd.DataFrame({'day': np.arange(data['day'].max() + 1, data['day'].max() + 31)})
    future_dates = [data['date'].max() + pd.Timedelta(days=i) for i in range(1, 31)]
    future_days['month'] = [d.month for d in future_dates]
    future_days['weekday'] = [d.weekday() for d in future_dates]
    future_temps = model.predict(future_days)

    # Prepare future data for display
    future_data = pd.DataFrame({
        'date': future_dates,
        'predicted_temperature': future_temps
    })

    # Generate plot
    plot_url = plot_results(data, future_data)
    return data, future_data, plot_url

def plot_results(data, future_data):
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
    return plot_url
