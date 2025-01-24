from flask import Flask
import os
from routes import init_routes

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize routes
init_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
