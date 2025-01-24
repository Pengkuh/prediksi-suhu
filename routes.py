from flask import request, render_template, current_app
import os
import pandas as pd
from model import process_data, train_and_predict, plot_results

def init_routes(app):
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
            # Save the uploaded file
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the file
            data = pd.read_excel(filepath)
            if 'Date' not in data.columns or 'Temperature' not in data.columns:
                return "Dataset harus memiliki kolom 'Date' dan 'Temperature'."

            # Process and predict
            data, future_data, plot_url = train_and_predict(data)

            # Pass results to output.html
            return render_template(
                'output.html',
                tables=future_data.to_html(classes='table table-bordered', index=False),
                plot_url=plot_url
            )
