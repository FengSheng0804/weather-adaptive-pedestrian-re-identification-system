from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
import os

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = '请先登录以访问此页面。'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    import uuid
    app.config['SERVER_INSTANCE_ID'] = str(uuid.uuid4())

    db.init_app(app)
    login_manager.init_app(app)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_IMAGES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_VIDEOS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESTORED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESTORED_IMAGES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESTORED_VIDEOS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REID_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REID_IMAGES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REID_VIDEOS_FOLDER'], exist_ok=True)

    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.user import user_bp
    from app.routes.weather import weather_bp
    from app.routes.reid import reid_bp
    from app.routes.admin import admin_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(user_bp, url_prefix='/user')
    app.register_blueprint(weather_bp, url_prefix='/weather')
    app.register_blueprint(reid_bp, url_prefix='/reid')
    app.register_blueprint(admin_bp, url_prefix='/admin')

    # Add custom Jinja filter for Beijing Time
    from datetime import timedelta
    @app.template_filter('beijing_time')
    def beijing_time_filter(dt):
        if dt is None:
            return '-'
        # Assuming dt is a naive datetime object representing UTC
        beijing_dt = dt + timedelta(hours=8)
        return beijing_dt.strftime('%Y-%m-%d %H:%M:%S')

    return app
