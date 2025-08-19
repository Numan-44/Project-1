from flask import Flask


def create_app():
    app = Flask(__name__)

    # Configuration (only file uploads if needed)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Import and register routes
    # from routes.views import views_bp  # Frontend routes (HTML pages)
    from routes.api import api_bp  # Backend routes (API endpoints)

    # app.register_blueprint(views_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
