from flask import Flask
from routes.api import api_bp
from routes.views import views_bp
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Register blueprints
app.register_blueprint(api_bp, url_prefix="/")
app.register_blueprint(views_bp, url_prefix="/")

if __name__ == "__main__":
    app.run(debug=True)

