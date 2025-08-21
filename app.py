import os
from flask import Flask


def create_app():
    app = Flask(__name__)

    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-for-sessions')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        from routes.views import views_bp
        from routes.api import api_bp

        app.register_blueprint(views_bp)
        app.register_blueprint(api_bp, url_prefix='/api')

        print("Blueprints registered successfully!")

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Check if routes/views.py and routes/api.py exist")

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)