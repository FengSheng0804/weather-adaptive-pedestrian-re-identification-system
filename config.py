import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'apris.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app', 'static', 'uploads')
    UPLOAD_IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
    UPLOAD_VIDEOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos')
    OUTPUT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app', 'static', 'output')
    RESTORED_FOLDER = os.path.join(OUTPUT_FOLDER, 'restored')
    RESTORED_IMAGES_FOLDER = os.path.join(RESTORED_FOLDER, 'images')
    RESTORED_VIDEOS_FOLDER = os.path.join(RESTORED_FOLDER, 'videos')
    REID_FOLDER = os.path.join(OUTPUT_FOLDER, 'reid')
    REID_IMAGES_FOLDER = os.path.join(REID_FOLDER, 'images')
    REID_VIDEOS_FOLDER = os.path.join(REID_FOLDER, 'videos')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload size
